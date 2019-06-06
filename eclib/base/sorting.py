'''
'''

import numpy as np


def identity(x):
    return x


class NondominatedSort(object):
    def __init__(self, size=100):
        self.size = size

    def __call__(self, population, n=None, return_rank=False):
        popsize = len(population)
        if not popsize:
            Exception('Error: population size is 0')

        is_dominated = np.empty((popsize, popsize), dtype=np.bool)
        num_dominated = np.empty(popsize, dtype=np.int64)
        mask = np.empty(popsize, dtype=np.bool)
        rank = np.zeros(popsize, dtype=np.int64)

        for i in range(popsize):
            for j in range(popsize):
                # iはjに優越されているか
                isdom = i != j and population[j].dominates(population[i])
                is_dominated[i, j] = isdom

        # iを優越している個体を数える
        is_dominated.sum(axis=(1,), out=num_dominated)

        fronts = []
        lim = popsize if n is None else n
        for r in range(popsize):
            # ランク未決定かつ最前線であるかを判定
            front = []
            for i in range(popsize):
                # ランク未決定又は優越される個体がない->ランク決定
                isrankdetermined = not (rank[i] or num_dominated[i])
                mask[i] = isrankdetermined
                if isrankdetermined:
                    rank[i] = r + 1
                    front.append(population[i])

            fronts.append(front)
            lim -= len(front)

            # 終了判定
            if return_rank:
                if rank.all():
                    return rank
            elif lim <= 0:
                return fronts

            # 優越数更新
            num_dominated -= np.sum(mask & is_dominated, axis=(1,))

        raise Exception('Error: reached the end of function')


class NondominatedSortIterator(object):
    def __init__(self, population):
        self._population = population
        self._size = len(population)

        if not self._size:
            Exception('Error: population is empty')

    def __iter__(self):
        pop = self._population
        lim = self._size

        is_dominated = np.empty((self._size, self._size), dtype=np.bool)
        num_dominated = np.empty(self._size, dtype=np.int64)
        mask = np.empty(self._size, dtype=np.bool)
        rank = np.zeros(self._size, dtype=np.int64)

        for i in range(self._size):
            for j in range(self._size):
                # iはjに優越されているか
                is_dominated[i, j] = i != j and pop[j].dominates(pop[i])

        # iを優越している個体を数える
        is_dominated.sum(axis=(1,), out=num_dominated)

        for r in range(self._size):
            # ランク未決定かつ最前線であるかを判定
            front = []
            for i in range(self._size):
                # ランク未決定又は優越される個体がない->ランク決定
                isrankdetermined = not (rank[i] or num_dominated[i])
                mask[i] = isrankdetermined
                if isrankdetermined:
                    rank[i] = r + 1
                    front.append(pop[i])
                    lim -= 1
            yield front

            # 終了判定
            if lim <= 0:
                raise StopIteration()

            # 優越数更新
            num_dominated -= np.sum(mask & is_dominated, axis=(1,))

        raise Exception('Error: reached the end of function')


################################################################################

class CrowdingDistanceCalculator(object):
    ''' key=attrgetter('data')
    '''
    def __init__(self, key=identity):
        self.key = key

    def __call__(self, population, normalization=False):
        popsize = len(population)
        if popsize == 0:
            return

        distances = np.zeros(popsize, dtype=np.float32)

        values = [self.key(x) for x in population]
        index = list(range(popsize))

        nobj = len(values[0])

        for i in range(nobj):
            get_value = lambda idx: values[idx][i]

            index.sort(key=get_value)

            distances[index[0]] = float('inf')
            distances[index[-1]] = float('inf')

            vrange = get_value(-1) - get_value(0)

            if vrange <= 0:
                continue

            norm = nobj * vrange

            for l, c, r in zip(index[:-2], index[1:-1], index[2:]):
                distances[c] += (get_value(r) - get_value(l)) / norm

        return distances


