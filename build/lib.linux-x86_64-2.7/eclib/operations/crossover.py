# -*- coding: utf-8 -*-
import ctypes
import os
import random
import numpy as np


LOADER_PATH = os.path.abspath(os.path.join(__file__, '../../../fortran'))


################################################################################
# double
################################################################################

class BlendCrossover(object):
    def __init__(self, rate=0.9, alpha=0.5, oneout=False):
        self.rate = rate
        self.alpha = alpha
        self.oneout = oneout

    def __call__(self, genes):
        # x1 = origin[0].get_gene()
        # x2 = origin[1].get_gene()
        # x1, x2 = (x.get_gene('d') for x in origin[:2])
        x1, x2 = genes

        if random.random() > self.rate:
            return x1, x2

        gamma = (1 + 2 * self.alpha) * np.random.random(x1.shape) - self.alpha

        if self.oneout:
            y = (1 - gamma) * x1 + gamma * x2
            return y
        else:
            y1 = (1 - gamma) * x1 + gamma * x2
            y2 = gamma * x1 + (1 - gamma) * x2
            return y1, y2


class SimulatedBinaryCrossover(object):
    def __init__(self, rate=0.9, eta=20, oneout=False):
        self.rate = rate
        self.eta = eta
        self.oneout = oneout

    def __call__(self, genes):
        # x1 = origin[0].get_gene()
        # x2 = origin[1].get_gene()

        # y1, y2 = (np.array(x.get_gene('d')) for x in origin[:2])
        y1, y2 = map(np.array, genes)

        if random.random() > self.rate:
            return y1, y2

        size = min(len(y1), len(y2))

        xl, xu = 0.0, 1.0
        eta = self.eta

        for i in range(size):
            if random.random() <= 0.5:
                # This epsilon should probably be changed for 0 since
                # floating point arithmetic in Python is safer
                if abs(y1[i] - y2[i]) > 1e-14:
                    x1 = min(y1[i], y2[i])
                    x2 = max(y1[i], y2[i])
                    rand = random.random()

                    beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                    alpha = 2.0 - beta**-(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha)**(1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))

                    c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                    beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                    alpha = 2.0 - beta**-(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha)**(1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))
                    c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                    c1 = min(max(c1, xl), xu)
                    c2 = min(max(c2, xl), xu)

                    if random.random() <= 0.5:
                        y1[i] = c2
                        y2[i] = c1
                    else:
                        y1[i] = c1
                        y2[i] = c2

        if self.oneout:
            return y1
        else:
            return y1, y2


################################################################################
# int
################################################################################

class OrderCrossover(object):
    def __init__(self, rate=0.9):
        libname = 'libec.dll'
        loader_path = LOADER_PATH
        cdll = np.ctypeslib.load_library(libname, loader_path)

        func_ptr = getattr(cdll, 'order_crossover')
        func_ptr.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.int32),
            ctypes.c_int32,
            np.ctypeslib.ndpointer(dtype=np.int32),
            np.ctypeslib.ndpointer(dtype=np.int32),
        ]
        func_ptr.restype = ctypes.c_void_p

        def f_(x1, x2):
            try:
                n1 = ctypes.c_int32(x1.size)
                y1, y2 = (np.empty_like(x) for x in (x1, x2))
                func_ptr(x1, x2, n1, y1, y2)
                return y1, y2
            except:
                print(x1, x2)
                raise

        self.rate = rate
        self.f_ = f_

    def __call__(self, genes):
        # x1, x2 = (x.get_gene('i') for x in origin[:2])
        x1, x2 = genes
        if random.random() > self.rate:
            return x1, x2
        return self.f_(x1, x2)

    def __reduce_ex__(self, protocol):
        return type(self), (self.rate,)
