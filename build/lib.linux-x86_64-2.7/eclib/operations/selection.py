# -*- coding: utf-8 -*-
'''
Abstruct
'''

import random
from operator import itemgetter


def identity(x):
    return x


class RandomSelection(object):
    def __init__(self):
        pass

    def __call__(self, population):
        s = len(population)
        if s == 0:
            return None, []
        pop = list(population)
        index = random.randrange(s)
        return pop.pop(index), pop


class RouletteSelection(object):
    def __init__(self, key=itemgetter(0)):
        # default key: fitst item of touple
        self.key = key

    def __call__(self, population):
        pop = list(population)
        fits = [self.key(x) for x in pop]
        wheel = sum(fits) * random.random() # fitness[0]
        for i, fit in enumerate(fits):
            if fit <= 0:
                continue
            wheel -= fit
            if wheel < 0:
                return pop.pop(i), pop
        raise RuntimeError('Error: in roulette')


class TournamentSelection(object):
    def __init__(self, key=identity, ksize=2):
        self.key = key
        self.ksize = ksize

    def __call__(self, population):
        s = len(population)
        # k = min(self.ksize, s)
        k = self.ksize
        if s < k:
            return None, []
        pop = list(population)
        indices = random.sample(range(s), k)

        # pop = random.sample(population, k)
        index = max(indices, key=pop.__getitem__)
        return pop.pop(index), pop


class TournamentSelectionStrict(object):
    def __init__(self, key=identity, ksize=2):
        self.key = key
        self.ksize = ksize

    def __call__(self, population):
        s = len(population)
        # k = min(self.ksize, s)
        k = self.ksize
        if s < k:
            return None, []
        pop, rest = self.separate_random(population, k)
        return max(pop), rest

    def separate_random(self, pop, k):
        selected = []
        rest = list(pop)
        size = len(pop)
        for i in range(k):
            index = random.randrange(size - i)
            selected.append(rest.pop(index))
        return selected, rest


class TournamentSelectionDCD(object):
    def __init__(self, key=identity):
        self.key = key
        # self.pat = [0, 0, 0, 0, 0]

    def __call__(self, population):
        s = len(population)
        # k = min(self.ksize, s)
        k = 2
        if s < k:
            return None, []
        pop, rest = self.separate_random(population, k)

        # pop = list(population)
        # indices = random.sample(range(s), k)

        # pop = random.sample(population, k)

        # def ret(i):
        #     # return pop.pop(indices[i]), pop
        #     return getpop(i), [x for i, x in enumerate(pop) if i not in indices]
        # def getpop(i):
        #     return pop[indices[i]]

        # 優越関係比較
        if pop[0].dominates(pop[1]):
            # self.pat[0] += 1
            return pop[0], rest
        elif pop[1].dominates(pop[0]):
            # self.pat[1] += 1
            return pop[1], rest

        if len(pop[0]) >= 2 and len(pop[1]) >= 2:
            # 混雑度比較
            if pop[0][1] < pop[1][1]:
                # self.pat[2] += 1
                return pop[1], rest
            elif pop[0][1] > pop[1][1]:
                # self.pat[3] += 1
                return pop[0], rest
        # self.pat[4] += 1

        if random.random() <= 0.5:
            return pop[0], rest
        return pop[1], rest

    def separate_random(self, pop, k):
        selected = []
        rest = list(pop)
        size = len(pop)
        for i in range(k):
            index = random.randrange(size - i)
            selected.append(rest.pop(index))
        return selected, rest
