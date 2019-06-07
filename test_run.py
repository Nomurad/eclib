#! /usr/bin/env python3

'''
benchmark test for MOEA/D
'''

import argparse
import glob
import os
import shutil
import sys
from operator import attrgetter

import numpy as np

from eclib.benchmarks import rosenbrock, zdt1, zdt2, zdt3, zdt4, zdt6
from eclib.operations import UniformInitializer
from eclib.operations import RouletteSelection
from eclib.operations import TournamentSelection
from eclib.operations import TournamentSelectionStrict
from eclib.operations import TournamentSelectionDCD
from eclib.operations import BlendCrossover
from eclib.operations import SimulatedBinaryCrossover
from eclib.operations import PolynomialMutation
from eclib.optimizers import NSGA2
from eclib.optimizers import MOEAD
from eclib.optimizers import NSGA2_para
from eclib.base import Individual
from eclib.base import Environment
from eclib.base import Creator
from eclib.base.population import Normalization

import myutils as ut 

class Problem():
    def __init__(self):
        pass

    def __call__(self, x):
        return self.problem(x)

    def problem(self, x):
        # return x[0], x[0]**2
        return zdt3(x)
        # return rosenbrock(x)


def main(model, out):
    n_dim = 10
    popsize = 30
    epoch = 100

    problem = Problem()

    with Environment() as env:
        indiv_pool = env.register(Individual)
        initializer = UniformInitializer(n_dim)
        creator = Creator(initializer, indiv_pool)

        if model == 'moead':
            optimizer = MOEAD(problem=problem, pool=indiv_pool, ksize=5, normalization=True)
            optimizer.weight_generator(nobj=4, divisions=50)
            popsize = int(popsize)
            epoch = epoch
        elif model == 'nsga2':
            optimizer = NSGA2(problem=problem, pool=indiv_pool, normalization=True)
        elif model == 'para':
            optimizer = NSGA2_para(problem=problem, pool=indiv_pool)

        else:
            raise Exception('Unexpected model name')

        population = optimizer.init_population(creator, popsize=popsize)
        history = [population]

        for i in range(1,epoch+1):
            if i%10 == 0:
                print("epoch ", i)
            population = optimizer(population)
            history.append(population)

            if i == epoch:
                file = f'popsize{popsize}_epoch{epoch}_{ut.strnow("%Y%m%d_%H%M%S")}.pkl'
                file = os.path.join(out, file)
                if not os.path.exists(out):
                    os.makedirs(out)

                print('save:',file)
                ut.save(file, (env, optimizer, history))
        
        return env, optimizer, history


def get_model(out):
    # モデル読み込み
    # model_cls = {'nsga2':NSGA2, 'moead':MOEAD}[model]
    files = ut.fsort(glob.glob(os.path.join(out, f'*epoch*.pkl')))
    for i, file in enumerate(files):
        print(f'[{i}]', file)
    print('select file')
    n = int(input())
    if n == -1:
        pass
    elif n < 0:
        return
    file = files[n]
    print('file:', file)
    env, optimizer, history = ut.load(file)
    return env, optimizer, history


def get_gene_data(out):
    '''
    各世代の遺伝子と評価値を取得(世代数込み)
    out : datas, genomes

    dim1:世代数, dim2:Fitness_Series, dim3:Fitness_value
    '''
    env,opt,history = get_model(out)
    dat_size = 1 + len(history[0].data[0].get_indiv())
    gene_size = 1 + len(history[0].data[0].get_indiv().get_variable())
    #                 [history, series, indiv_value]
    datas = np.zeros((len(history), dat_size, len(history[0].data)) )
    genomes = np.zeros((len(history), gene_size, len(history[0].data)) )
    for i, pop in enumerate(history):
        datas[i,0,:] = i
        genomes[i,0,:] = i
        datas[i,1:,:] = (np.array([fit.data.value for fit in pop]).T)
        genomes[i,1:,:] = (np.array([indv.data.get_variable() for indv in pop]).T)

    return datas,genomes

def plt_result(out):
    import matplotlib.pyplot as plt
    
    datas, genomes = get_gene_data(out)

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(1,1,1)
    # ax.set_ylim(0,3.0)
    cm = plt.get_cmap("Blues")
    sc = ax.scatter( datas[:,-2], datas[:, -1], c=datas[:,0], cmap=cm)
    plt.colorbar(sc)

    plt.show()

def __test__(out, model='nsga2'):
    env,opt,history = get_model(out)
    # env = M.Optimize_ENV(model, popsize=len(history[0]), ksize=5).env
    # population = env.optimizer.init_population(env.creator, popsize=5)
    population = history[-1]
    normalizing = Normalization(population)
    norm = normalizing(population, max_ref=np.array([1,1]))
    print(normalizing.max_obj_val, normalizing.min_obj_val)

    # fits = population.data
    # indivs = [indiv.data for indiv in fits]
    # obj_dim = len(indivs[0].value)
    # for i, indiv in enumerate(indivs):
    #     for j in range(obj_dim):
    #         print( f'{indiv.value[j]:>13.3f}', end=" ")
    #     print("| ", end=" ")
    #     for j in range(obj_dim):
    #         print( f'{indiv.wvalue[j]:>7.3f}', end=" ")
        
    #     norm = normalizing(population, max_ref=np.array([1,1]))
    #     print(f'{normalizing.max_obj_val}')
        # print(f'{fits[i].value},{fits[i].rank}')
    
    for pop in history:
        normalizing(pop)
        print(normalizing.max_obj_val, normalizing.min_obj_val)


def get_args():
    '''
    docstring for get_args.
    '''
    default_optimizer = 'moead'

    parser = argparse.ArgumentParser()
    parser.add_argument('method', nargs='?', default='',
                        help='Main method type')
    parser.add_argument('--model', '-m', default=default_optimizer,
                        help='Model type')
    parser.add_argument('--out', '-o', default='',
                        help='Filename of the new script')
    parser.add_argument('--clear', '-c', action='store_true',
                        help='Remove output directory before start')
    parser.add_argument('--force', '-f', action='store_true',
                        help='force')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    out = args.out
    model = args.model 

    out = os.path.join('test_result', model, args.out)

    if args.method == 'r':
        plt_result(out)
    elif args.test:
        __test__(out)
    else:
        print("run test script.")
        main(model ,out)