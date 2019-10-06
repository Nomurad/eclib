# -*- coding: utf-8 -*-
class PartialSelectionIterator(object):
    ''' SelectionIteratorの部分適用オブジェクト
    '''
    def __init__(self, selection, pool):
        self._selection = selection
        self._pool = pool

    def __call__(self, population, reset_cycle=None):
        return SelectionIterator(self._selection, self._pool, population,
                                 reset_cycle)


class SelectionIterator(object):
    ''' 交配の親選択イテレータ
    個体集団から親個体を選択し，選択された親個体を解集団から削除する(削除方式はselection関数に依存)
    reset_cycleが与えられた場合は解をreset_cycle個生成するごとに解集団をpopulationで初期化する
    '''
    def __new__(cls, selection, pool, population=None, reset_cycle=None):
        if population is None:
            return PartialSelectionIterator(selection, pool)
        return super().__new__(cls)

    def __init__(self, selection, pool, population, reset_cycle=None):
        self._selection = selection
        self._pool = pool
        self._population = population
        self._reset_cycle = reset_cycle
        self._stored = []

    def __iter__(self):
        rest = []
        counter = 0
        i = 0
        while True:
            # print('iter:', i)
            if not rest or (self._reset_cycle and counter == self._reset_cycle):
                # print('reset:', i, counter)
                rest = list(self._population)
                counter = 0
            selected, rest = self._selection(rest)
            if selected is None:
                continue
            counter += 1
            self._stored.append(selected)
            i += 1
            yield selected

    def __getnewargs__(self):
        return self._selection, self._pool, self._population, self._reset_cycle

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state['_stored'] = [x.id for x in self._stored]
    #     return state

    # def __setstate__(self, state):
    #     pool = state['_pool']
    #     state['_stored'] = [pool[id] for id in state['_stored']]
    #     self.__dict__.update(state)


################################################################################

class PartialMatingIterator(object):
    ''' MatingIteratorの部分適用オブジェクト
    '''
    def __init__(self, crossover, mutation, pool):
        self._crossover = crossover
        self._mutation = mutation
        self._pool = pool

    def __call__(self, origin):
        return MatingIterator(self._crossover, self._mutation, self._pool,
                              origin)


class MatingIterator(object):
    ''' 親の組からこの組を生成するイテレータ
    crossoverとmutationの直列方式
    '''
    def __new__(cls, crossover, mutation, pool, origin=None):
        if origin is None:
            return PartialMatingIterator(crossover, mutation, pool)
            # return lambda x: cls(crossover, mutation, pool, x)
        return super().__new__(cls)

    def __init__(self, crossover, mutation, pool, origin):
        self._crossover = crossover
        self._mutation = mutation
        self._pool = pool

        if pool.isinstance(origin[0]):
            self._origin = origin
        else:
            self._origin = [fit.get_indiv() for fit in origin] # Indivに変換
        self._stored = []

    def __iter__(self):
        genomes_p = [ind.get_gene() for ind in self._origin] # 親個体の遺伝子の組
        genomes_c = self._crossover(genomes_p) # 子個体の遺伝子の組

        for genome in genomes_c:
            # 1個体の遺伝子
            genome = self._mutation(genome)
            child = self._pool(genome, origin=self)
            self._stored.append(child)
            yield child

    @property
    def origin(self):
        return self._origin

    # def __reduce_ex__(self, protocol):


    def __getnewargs__(self):
        return self._crossover, self._mutation, self._pool, self._origin

'''
    def __getstate__(self):
        state = self.__dict__.copy()
        # pool = state['_pool']
        # state['_origin'] = [x.id for x in self._origin]
        # state['_stored'] = [x.id for x in self._stored]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # pool = self._pool
        # print(len(pool))
        # exit()
        # state['_origin'] = [pool[id] for id in state['_origin']]
        # state['_stored'] = [pool[id] for id in state['_stored']]
'''
