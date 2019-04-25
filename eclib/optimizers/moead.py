'''
設計変数: 任意
目的関数: 1次元配列
解集団形状: 1次元配列, 2重配列(島モデル)

各個体にランク(自然数), 混雑度(実数)が割り当てられる
集団評価値(参考): ハイパーボリューム

解集団 =~ [個体]

選択
[個体] => [個体]

個体が生成されるのは次の3通りのいずれか
初期化: [] => 個体
交叉: [個体] => 個体 / [個体] => [個体]
突然変異: 個体 => 個体
'''

import argparse
import os
import pickle
import random
import shutil
from itertools import islice
from operator import attrgetter, itemgetter

import numpy as np

from ..base import Individual
from ..base import Population
from ..base import NondominatedSortIterator
from ..base import CrowdingDistanceCalculator
# from ..operations import SelectionIterator
# from ..operations import MatingIterator
from .iterators import SelectionIterator
from .iterators import MatingIterator

# デフォルト用
from ..operations import UniformInitializer
from ..operations import RandomSelection
from ..operations import RouletteSelection
from ..operations import TournamentSelection
from ..operations import TournamentSelectionStrict
from ..operations import TournamentSelectionDCD
from ..operations import BlendCrossover
from ..operations import SimulatedBinaryCrossover
from ..operations import PolynomialMutation

# default_selection = TournamentSelection(ksize=2)s
default_selection = TournamentSelectionStrict(ksize=2)
# default_selection = TournamentSelectionDCD()
# default_crossover = BlendCrossover(alpha=0.5)
default_crossover = SimulatedBinaryCrossover(rate=0.9, eta=20)
default_mutation = PolynomialMutation(rate=0.05, eta=20)


################################################################################
# スカラー化関数
################################################################################

def scalar_weighted_sum(indiv, weight, ref_point):
    return -np.sum(weight * np.abs(indiv.wvalue - ref_point))

def scalar_chebyshev(indiv, weight, ref_point):
    return -np.max(weight * np.abs(indiv.wvalue - ref_point))

def scalar_boundaryintersection(indiv, weight, ref_point):
    ''' norm(weight) == 1
    '''
    nweight = weight / np.linalg.norm(weight)

    bi_theta = 5.0
    d1 = np.abs(np.dot((indiv.wvalue - ref_point), nweight))
    d2 = np.linalg.norm(indiv.wvalue - (ref_point - d1 * nweight))
    return -(d1 + bi_theta * d2)


################################################################################

class MOEAD(object):
    ''' MOEADモデル(2D)
    '''
    name = 'MOEA/D'

    def __init__(self, popsize=None, problem=None, pool=None, ksize=3,
                 scalar=scalar_chebyshev,
                 selection=default_selection,
                 crossover=default_crossover,
                 mutation=default_mutation,
                 n_obj=2):
        self.popsize = popsize
        self.ksize = ksize
        self.problem = problem

        self.nobj = int(n_obj)
        print("set nobj = ",self.nobj)
        self.scalar = scalar

        self.n_parents = 2       # 1回の交叉の親個体の数
        self.n_cycle = 2         # 選択候補をリセットする周期(n_parentsの倍数にすること)
        self.alternation = 'new' # 世代交代方法

        self.select_it = SelectionIterator(selection=selection, pool=pool)
        self.mate_it = MatingIterator(crossover=crossover,
                                      mutation=mutation,
                                      pool=pool)
        self.sort_it = NondominatedSortIterator
        self.share_fn = CrowdingDistanceCalculator(key=attrgetter('data')) # Fitness -> Individual

        if not self.popsize:
            self.init_weight2d()
        # else:
        #     self.ref_point = np.full(self.nobj, 'inf', dtype=np.float64)

    def __call__(self, population):
        if not self.popsize:
            self.popsize = len(population)
            self.init_weight2d()
            population.sort(key=attrgetter('data'))

        next_population = self.advance(population)
        return self.alternate(population, next_population)

    def init_weight2d(self, popsize=None, ksize=None):
        ''' 重みベクトルと近傍テーブルの初期化
        '''
        if popsize:
            self.popsize = popsize
        if ksize:
            self.ksize = ksize
        if not self.popsize or not self.ksize:
            return

        def get_neighbor(index):
            imin = min(max(index-(self.ksize-1)//2, 0),
                       self.popsize-self.ksize)
            return list(range(imin, imin+self.ksize))

        # self.weight = np.array([[i+1, self.popsize-i]
        #                        for i in range(self.popsize)])
        self.weight = [[1,0],[0,1]]
        self.weight.extend([(i/(self.popsize-1.0), 1.0-i/(self.popsize-1.0)) for i in range(1, self.popsize-1)])
        self.weight = np.array(self.weight)
        self.table = np.array([get_neighbor(i) for i in range(self.popsize)])
        # self.nobj = len(self.weight)    #nobjを可変に(nomura)
        self.ref_point = np.full(self.nobj, 'inf', dtype=np.float64)

    def update_reference(self, indiv):
        try:
            self.ref_point = np.min([self.ref_point, np.array(indiv.wvalue)],axis=0)
            # print("updata ref point = ", self.ref_point)
        except:
            print(self.ref_point.dtype)
            print(self.ref_point)
            print(np.array(indiv.wvalue).dtype)
            print(np.array(indiv.wvalue))
            print([self.ref_point, np.array(indiv.wvalue)])
            raise
            exit()

    def init_population(self, creator, popsize=None):
        ''' 初期集団生成
        '''
        if popsize:
            self.popsize = popsize
            self.init_weight2d()

        population = Population(capacity=self.popsize, origin=self)

        while not population.filled():
            indiv = creator()
            fitness = indiv.evaluate(self.problem)
            population.append(fitness)

        # self.calc_fitness(population)
        # self.weight = np.array([[i ,w] for i,w in zip(range(self.popsize), population[0].data.wvalue)])
        # self.weight = np.array([population[i].data.wvalue for i in (range(self.popsize))])

        self.ref_point = np.min([fit.data.wvalue for fit in population], axis=0)
        return population

    def advance(self, population):
        ''' 選択→交叉→突然変異→評価→適応度計算→世代交代
        '''
        next_population = Population(capacity=self.popsize, origin=self)
        # select_it = self.select_it(self.population, reset_cycle=self.n_cycle)
        # select_it = iter(select_it) # Fixed

        for i in range(self.popsize):
            child_fit = self.get_offspring(i, population, self.table[i],
                                           self.weight[i])
            next_population.append(child_fit)

        return next_population

    def alternate(self, population, next_population):
        return next_population

    def get_offspring(self, index, population, table, weight):
        ''' 各個体の集団内における適応度を計算する
        1. スカラー化関数
        交叉，突然変異を行い最良個体を1つ返す
        * 2018.11.21 新しい解が古い解より良い場合に置き換える処理に変更
        '''
        subpopulation = [population[i] for i in table]

        for i, fit in enumerate(subpopulation):
            fit_value = self.scalar(fit.data, weight, self.ref_point)
            fit.set_fitness((fit_value,), 1)

        select_it = self.select_it(subpopulation, reset_cycle=self.n_cycle)
        paretns = list(islice(select_it, self.n_parents))

        # offspring = []

        child = random.choice(list(self.mate_it(paretns)))
        # for child in self.mate_it(paretns):
        child_fit = child.evaluate(self.problem)
        self.update_reference(child)
        fit_value = self.scalar(child_fit.data, weight, self.ref_point)
        child_fit.set_fitness((fit_value,), 1)
        # offspring.append(child_fit)

        if self.alternation == 'new':
            return max(population[index], child_fit)
        elif self.alternation == 'all':
            return max(*subpopulation, child_fit)
        else:
            print('Unexpected alternation type:', self.alternation)
            raise Exception('UnexpectedAlternation')

    def calc_rank(self, population, n=None):
        ''' 各個体の集団内におけるランクを計算して設定する
        外部から呼ぶ
        '''
        for i, front in enumerate(self.sort_it(population)):
            rank = i + 1
            for fit in front:
                fit.rank = rank
        return population

        ################################################################################
        # weight vector 
        ################################################################################
    def weight_generator(self, nobj=None, popsize=None):
        '''
         任意の次元の重みベクトルを作成
        '''
        import copy

        weights = []

        if not nobj:
            nobj = self.nobj
        if not popsize:
            popsize = self.popsize
        
        if nobj == 2:
            weights = [[1,0],[0,1]]
            weights.extend([(i/(self.popsize-1.0), 1.0-i/(self.popsize-1.0)) for i in range(1, self.popsize-1)])
            weights = np.array(weights)

        else:
            ele_candidate = np.array(list(range(popsize)))/popsize

            def weight_recursive(weights, weight, left, total, idx=0):

                if idx == nobj-1:
                    weight[idx] = float(left)/float(total)
                    weights.append(copy.copy(weight))
                    return weights

                else:
                    for i in range(left+1):
                        weight[idx] = float(i)/float(total)
                        weight_recursive(weights, weight, left-i, total, idx+1)

            weight_recursive(weights, [0.0]*nobj, popsize, popsize)
            weights = np.array(weights)

            print(f"nobj={nobj}, popsize={popsize}")
            print(ele_candidate)
            with open("temp.txt", "w") as f:
                print(weights, file=f)

        return weights
        



################################################################################

def __test__():
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.test:
        __test__()
        return


if __name__ == '__main__':
    main()
