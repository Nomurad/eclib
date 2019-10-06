# -*- coding: utf-8 -*-

from operator import attrgetter
import numpy as np


SCALE = 1


def key_temp(indiv, i=None):
    if i is None:
        return indiv.wvalue
    return indiv.wvalue[i]


def key_sort(indiv):
    return list(indiv.wvalue)


def key_test(elem, i=None):
    if i is None:
        return np.array(elem)
    return elem[i]


class HyperVolume2DTemp(object):
    def __init__(self, ref):
        self.ref_point = np.array(ref)

    def __call__(self, population, key=key_temp):
        # front = [fit.rank for fit in population if np.all(key(fit.data) <= self.ref_point)]
        # front = [fit.rank for fit in population]
        # print(front)
        # exit()

        if hasattr(population[0], 'rank'):
            # front = [fit.data for fit in population if fit.rank == 1 and np.all(key(fit.data) <= self.ref_point)]
            front = [fit.data for fit in population if np.all(key(fit.data) <= self.ref_point)]
        else:
            front = [x for x in population if np.all(key(x) <= self.ref_point)]

        if not front:
            print('HyperVolume: No data in HV area')
            return 0

        print('HyperVolume:', len(front))

        front.sort(key=key_sort)

        hv = (self.ref_point[0] - key(front[0], 0)) \
           * (self.ref_point[1] - key(front[0], 1))

        for i in range(1, len(front)):
            try:
                v = self.ref_point[0] - key(front[i], 0), 0
                if v[0] < 0:
                    raise
                v = key(front[i - 1], 1) - key(front[i], 1), 1
                if v[0] < 0:
                    raise
            except:
                print(v)
                exit()
            hv += (self.ref_point[0] - key(front[i], 0)) \
                * (key(front[i - 1], 1) - key(front[i], 1))

        return hv * SCALE


def __test__():
    data = [[1, -1], [3, -2], [4, -4]]
    ref = [5, 0]
    key = key_test

    hypervolume = HyperVolume2DTemp(ref)
    print(hypervolume(data, key))


if __name__ == '__main__':
    __test__()
