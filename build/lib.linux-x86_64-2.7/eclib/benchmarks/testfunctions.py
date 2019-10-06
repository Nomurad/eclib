# -*- coding: utf-8 -*-

#! /usr/bin/env python3

'''
Abstruct
'''

import math

import numpy as np


A = 10
PI2 = 2 * math.pi
PI4 = 4 * math.pi
PI6 = 6 * math.pi
PI10 = 10 * math.pi


################################################################################
# S.O.
################################################################################

def rastrigin(x):
    return A * len(x) + sum(map(lambda v: v ** 2 - A * math.cos(PI2 * v), x))

# rastrigin.range = [-5, 5]


def rosenbrock(x):
    return sum(map(lambda i: 100 * (x[i+1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2,
                   range(len(x) - 1)))

# rosenbrock.range = [-2, 2]


################################################################################
# M.O.
################################################################################

def zdt1(x):
    n = len(x)
    if n == 1:
        return x[0], 1 - math.sqrt(x[0])

    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    return x[0], g * (1 - math.sqrt(x[0] / g))


def zdt2(x):
    if len(x) == 1:
        return x[0], 1 - x[0] ** 2

    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    return x[0], g * (1 - (x[0] / g) ** 2)


def zdt3(x):
    if len(x) == 1:
        return x[0], 1 - math.sqrt(x[0]) - x[0] * math.sin(PI10 * x[0])

    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    v = x[0] / g
    return x[0], g * (1 - math.sqrt(v) - v * math.sin(PI10 * x[0]))


def zdt4(x):
    if len(x) == 1:
        return x[0], 1 - math.sqrt(x[0])

    g = 1 + 10 * (len(x) - 1) \
        + sum(map(lambda v: v ** 2 - 10 * math.cos(PI4 * v), x[1:]))
    return x[0], g * (1 - math.sqrt(x[0] / g))


def zdt6(x):
    f = 1 - math.exp(-4 * x[0]) * math.sin(PI6 * x[0]) ** 6
    if len(x) == 1:
        return f, 1 - f ** 2

    g = 1 + 9 * (np.sum(x[1:]) / (len(x) - 1)) ** 0.25
    return f, g * (1 - (f / g) ** 2)


################################################################################
# M.O. w/ const
################################################################################

def osy(x):
    if len(x) == 1:
        return x[0], 1 - math.sqrt(x[0])

    f1 = sum(map(lambda i, v: (-1 if i else -25) * (x[i] - v) ** 2,
                             enumerate([2, 2, 1, 4, 1])))
    f2 = sum(map(lambda v: v ** 2, x))
    g1 = x[0] + x[1] - 2
    g2 = 6 - x[0] - x[1]
    g3 = 2 - x[1] + x[0]
    g4 = 2 - x[0] + 3 * x[1]
    g5 = 4 - (x[2] - 3) ** 2 - x[3]
    g6 = (x[4] - 3) ** 2 + x[5] - 4
    return (f1, f2), (g1, g2, g3, g4, g5, g6)


################################################################################

def __test__():
    print(rastrigin([0, 0]))


################################################################################


def main():
    '''
    docstring for main.
    '''

    __test__()


if __name__ == '__main__':
    main()
