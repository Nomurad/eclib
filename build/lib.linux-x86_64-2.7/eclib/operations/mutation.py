# -*- coding: utf-8 -*-
import random
import numpy as np


class PolynomialMutation():
    def __init__(self, rate=0.1, eta=20):
        self.rate = rate
        self.eta = eta

    def __call__(self, gene):
        size = len(gene)
        res = np.array(gene)
        xl, xu = 0.0, 1.0

        for i, x in enumerate(gene):
            if random.random() > self.rate:
                continue
            # x = gene[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (self.eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy**(self.eta + 1)
                delta_q = val**mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy**(self.eta + 1)
                delta_q = 1.0 - val**mut_pow

            y = x + delta_q * (xu - xl)
            y = min(max(y, xl), xu)
            res[i] = y
        return res


class SwapMutation():
    def __init__(self, rate=0.1):
        self.rate = rate

    def __call__(self, gene):
        if random.random() > self.rate:
            return gene

        index = random.sample(range(len(gene)), 2)
        res = np.array(gene)
        res[index] = res[index[::-1]]
        return res


def __test__():
    a = np.arange(10)
    mu = SwapMutation(1.0)
    print(a)
    print(mu(a))


if __name__ == '__main__':
    __test__()
