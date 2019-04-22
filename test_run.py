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
import matplotlib.pyplot as plt

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
from eclib.base import Individual

class Problem():
    def __init__(self):
        pass

    def __call__(self):
        return self.problem()

    def problem(self):
        return zdt1


def main():
    n_dim = 10
    popsize = 100
    epoch = 100

    problem = Problem()

    with Enviroment() as env:
        indiv_pool = env.resister(Individual)
        initializer = UniformInitializer(n_dim)
        creator = Creator(initializer, indiv_pool)

        optimizer = MOEAD(problem=problem, pool=indiv_pool, ksize=3)

        population = optimizer.init_population(creator, popsize=popsize)
        history = [population]

        for i in range(1,epoch+1):
            print("epoch ", i)
            population = optimizer(population)
            history.append(population)

            
    

if __name__ == "__main__":
    pass