# -*- coding: utf-8 -*-
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
import shutil
from itertools import islice
from operator import attrgetter, itemgetter

from multiprocessing import Pool
import multiprocessing as mp
# from mpi4py import MPI
# from mpi4py.futures import MPIPoolExecutor

import numpy as np

from ..base import Individual
from ..base import Population
from ..base import NondominatedSortIterator
from ..base import CrowdingDistanceCalculator
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

from .mypool import MyPool

# default_selection = TournamentSelection(ksize=2)
default_selection = TournamentSelectionStrict(ksize=2)
# default_selection = TournamentSelectionDCD()
# default_crossover = BlendCrossover(alpha=0.5)
default_crossover = SimulatedBinaryCrossover(rate=0.9, eta=20)
default_mutation = PolynomialMutation(rate=0.05, eta=20)


################################################################################

class NSGA2_para(object):
    ''' NSGA-IIモデル
    '''
    name = 'NSGA-II'

    def __init__(self, popsize=None, problem=None, pool=None,
                 selection=default_selection,
                 crossover=default_crossover,
                 mutation=default_mutation):
        self.popsize = popsize
        self.problem = problem

        self.n_parents = 2        # 1回の交叉の親個体の数
        self.n_cycle = 2          # 選択候補をリセットする周期(n_parentsの倍数にすること)
        self.alternation = 'join' # 世代交代方法

        self.select_it = SelectionIterator(selection=selection, pool=pool)
        self.mate_it = MatingIterator(crossover=crossover,
                                      mutation=mutation,
                                      pool=pool)
        self.sort_it = NondominatedSortIterator
        self.share_fn = CrowdingDistanceCalculator(key=attrgetter('data')) # Fitness -> Individual

    def __call__(self, population):
        if not self.popsize:
            self.popsize = len(population)

        next_population = self.advance(population)
        return self.alternate(population, next_population)

    def init_eval(self, creator, population):
        indiv = creator()
        fitness = indiv.evaluate(self.problem)
        # population.append(fitness)
        return fitness
        # return 100

    def wrap_eval(self, args):
        return self.init_eval(*args)

    def init_population(self, creator, popsize=None):
        ''' 初期集団生成
        '''
        if popsize:
            self.popsize = popsize
        # self.generation = 1

        population = Population(capacity=self.popsize,
                                origin=self)

        while not population.filled():
            indiv = creator()
            fitness = indiv.evaluate(self.problem)
            population.append(fitness)

        def eval_para(n_indiv):
            indiv = creator()
            fitness = indiv.evaluate(self.problem)
            population.append(fitness)

        # comm = MPI.COMM_WORLD
        # mpirank = comm.Get_rank()
        # size = comm.Get_size()

        # status = MPI.Status()
        # print("status", status)

        # p.map(self.wrap_eval, [(creator,population)])
        p = Pool(int(mp.cpu_count()))
        tmplist = [(creator,population) for i in range(popsize)]
        fitnesses = p.map(self.wrap_eval, tmplist)

        # print(fitnesses)
        for i in range(popsize):
            population.append(fitnesses[i])

        #     child_fit = child.evaluate(self.problem)
    
        return population

    def wrap_adv_eval(self, args):
        return self.adv_eval(*args)

    def advance(self, population):
        ''' 選択→交叉→突然変異→評価(→適応度計算→世代交代)
        '''
        next_population = Population(capacity=self.popsize,
                                     origin=self)
        select_it = self.select_it(population, reset_cycle=self.n_cycle)
        select_it = iter(select_it) # Fixed

        while not next_population.filled():
            parents_it = list(islice(select_it, self.n_parents)) # Fixed

            for child in self.mate_it(parents_it):
                child_fit = child.evaluate(self.problem)
                next_population.append(child_fit)

        # p = Pool(int(mp.cpu_count()))
        # tmplist = [(select_it,) for i in range(self.popsize)]
        # child_fits = p.map(self.wrap_adv_eval, tmplist)
        # print(child_fits)

        # for i in range(self.popsize):
        #     next_population.append(child_fits[i])

        return next_population

    def alternate(self, population, next_population):
        ''' 適応度計算 → 世代交代
        1. 親世代を子世代で置き換える
        2. 親世代と子世代の和からランクを求める
        '''
        if self.alternation == 'replace':
            self.calc_fitness(next_population)
            return next_population

        elif self.alternation == 'join':
            joined = population + next_population
            next_population = self.calc_fitness(joined, n=self.popsize)
            # print([fit.data.id for fit in next_population])
            # exit()
            return Population(next_population, capacity=self.popsize,
                              origin=self)

        else:
            print('Unexpected alternation type:', self.alternation)
            raise Exception('UnexpectedAlternation')

    def calc_fitness(self, population, n=None):
        ''' 各個体の集団内における適応度を計算する
        1. 比優越ソート
        2. 混雑度計算
        '''
        lim = len(population) if n is None else n
        selected = []

        for i, front in enumerate(self.sort_it(population)):
            # print('g:', self.generation, 'i:', i, 'l:', len(front))
            rank = i + 1
            fit_value = -i # TODO: 可変にする
            # if i == 0:
            #     print('len(i==0):', len(front), ' ')

            if self.share_fn:
                it = self.share_fn(front)
                try:
                    for fit, crowding in zip(front, it):
                        fitness = fit_value, crowding
                        # print(fitness)
                        fit.set_fitness(fitness, rank)
                except:
                    print('Error')
                    print(front)
                    print(it)
                    raise
            else:
                for fit in front:
                    fitness = fit_value,
                    fit.set_fitness(fitness, rank)

            lim -= len(front) # 個体追加後の余裕
            if lim >= 0:
                selected.extend(front)
                if lim == 0:
                    return selected
            # elif i == 0:
            #     return front
            else:
                front.sort(key=itemgetter(1), reverse=True) # 混雑度降順で並べ替え
                # print([itemgetter(1)(fit) for fit in front])
                # exit()
                selected.extend(front[:lim])
                return selected

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
