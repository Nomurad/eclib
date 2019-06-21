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
import json

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
        x,y = zdt1(x)
        return x, y
        # return rosenbrock(x)


def main(model, out):
    n_dim = 10
    popsize = 30
    epoch = 100*5

    problem = Problem()

    with Environment() as env:
        indiv_pool = env.register(Individual)
        initializer = UniformInitializer(n_dim)
        creator = Creator(initializer, indiv_pool)

        crossover = SimulatedBinaryCrossover(rate=0.9, eta=40)

        if model == 'moead':
            ksize = 10
            options = {"ksize":ksize, "normalization":True, 
                        "crossover":crossover}
            optimizer = MOEAD(problem=problem, pool=indiv_pool, **options)
            optimizer.weight_generator(nobj=4, divisions=50)
            popsize = int(popsize)
            epoch = epoch
            
        elif model == 'nsga2':
            optimizer = NSGA2(problem=problem, pool=indiv_pool, normalization=True)
        elif model == 'para':
            optimizer = NSGA2_para(problem=problem, pool=indiv_pool)

        else:
            raise Exception('Unexpected model name')

        # indiv_pool.cls.set_weight([1, -1])
        population = optimizer.init_population(creator, popsize=popsize)
        history = [population]

        for i in range(1,epoch+1):
            if i%50 == 0:
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
    datas2 = np.zeros((len(history), dat_size, len(history[0].data)) )
    genomes = np.zeros((len(history), gene_size, len(history[0].data)) )
    for i, pop in enumerate(history):
        datas[i,0,:] = i
        genomes[i,0,:] = i
        datas2[i,0,:] = i
        datas[i,1:,:] = (np.array([fit.data.value for fit in pop]).T)
        datas2[i,1:,:] = (np.array([fit.data.wvalue for fit in pop]).T)
        genomes[i,1:,:] = (np.array([indv.data.get_variable() for indv in pop]).T)

    return datas,genomes, datas2

def plt_result(out):
    import matplotlib.pyplot as plt
    
    datas, genomes, datas2 = get_gene_data(out)
    datas = datas

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(1,1,1)
    # ax.set_ylim(0,3.0)
    cm = plt.get_cmap("Blues")
    sc = ax.scatter( datas[:,-2], datas[:, -1], c=datas[:,0], cmap=cm)
    plt.colorbar(sc)

    plt.show()

def plt_anim(out):
    import matplotlib.pyplot as plt
    from matplotlib import animation as anim

    env,opt,history = get_model(out)
    datas, genomes, datas2 = get_gene_data(out)
    datas = datas2

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(1,1,1)

    def normalize_line(frame):
        # obj1_max = [[history[frame].max_obj_val[0],0],[0,0]]
        # obj1_min = [[history[frame].min_obj_val[0],0],[0,0]]
        # obj2_max = [[0,history[frame].max_obj_val[1]],[0,0]]
        # obj2_min = [[0,history[frame].min_obj_val[1]],[0,0]]
        obj1_max = [[opt.normalizing.max_obj_val[0],0],[opt.normalizing.max_obj_val[0],100]]
        obj1_min = [[opt.normalizing.min_obj_val[0],0],[opt.normalizing.min_obj_val[0],100]]
        obj2_max = [[0,opt.normalizing.max_obj_val[1]],[100,opt.normalizing.max_obj_val[1]]]
        obj2_min = [[0,opt.normalizing.min_obj_val[1]],[100,opt.normalizing.min_obj_val[1]]]

        ax.plot(obj1_max, c="Blue")
        # ax.plot(obj1_min, c="Blue")
        # ax.plot(obj2_max, c="Red")
        # ax.plot(obj2_min, c="Red")


    def ploting(frame):
        ax.cla()
        # ax.set_xlim(-0.05, 1.2)
        # ax.set_ylim(-0.05, 6.5)
        ax.set_xlim(-0.05, 1.2)
        ax.set_ylim(-0.05, 1.2)
        ax.set_title(f"Generation={frame}")
        # normalize_line(frame)
        # sc = ax.scatter(datas[frame, -2], datas[frame, -1])
        sc = ax.scatter(datas[frame, -2], datas[frame, -1])

        return sc 
    
    frames = range(0, len(history))
    animation = anim.FuncAnimation(fig, ploting, frames=frames, interval=10)
    plt.show()


def __test__(out, model='nsga2'):
    env,opt,history = get_model(out)
    # env = M.Optimize_ENV(model, popsize=len(history[0]), ksize=5).env
    # population = env.optimizer.init_population(env.creator, popsize=5)
    epoch = -1
    population = history[epoch]
    value = []
    wvalue = []

    
    for i,fit in enumerate(population):
        value.append( list(fit.data.value) )
        wvalue.append( list(fit.data.wvalue) )

    dic = {}
    dic["env"] = {"dv_dim":env.__dict__.get("n_dim"), 
                  "opt_weight":env.__dict__.get("opt_weight")}
    dic["optimizer"] = {"name":opt.__class__.__name__}
    dic["epoch"] = {"max":len(history)-1, 
                    "value epoch":len(history)+epoch if epoch<0 else epoch}
    try:
        dic["normalize"] = {"max":list(opt.normalizing.max_obj_val)}
    except:
        pass
    dic["wvalue"] = wvalue
    dic["value"] = value
    # try:
    #     dic["weight_vec"] = list(opt.weight)
    # except:
    #     print("no exist weight vector")

    with open("temp.json", "w") as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)

def rm_result():
    shutil.rmtree("result_test/", ignore_errors=True)
    os.mkdir("result_test/")

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

    out = os.path.join('result_test', model, args.out)

    if args.method == 'r':
        # plt_result(out)
        plt_anim(out)
    elif args.method == "rm":
        rm_result()
    elif args.test:
        __test__(out)
    else:
        print("run test script.")
        main(model ,out)