import numpy as np 

class Population(object):
    ''' 解集団
    GAの個体を管理する (デフォルトではシーケンス型)
    島モデルの場合はmigrationを担当する
    '''

    def __init__(self, data=None, capacity=None, origin=None):
        super().__init__()
        if isinstance(data, Population):
            self.data = data.data
            self.capacity = capacity or data.capacity
            self.origin = origin or data.origin
            self.max_obj_val = data.max_obj_val
            self.min_obj_val = data.min_obj_val
        else:
            self.data = data or []
            self.capacity = capacity
            self.origin = origin
            self.max_obj_val = -1
            self.min_obj_val = 1
        
        # self.normalize_para()

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        if isinstance(other, Population):
            return self.data + other.data
        elif iter(other):
            return self.data + list(other)

    def append(self, x):
        self.data.append(x)

    def clear(self):
        self.data.clear()

    def sort(self, *args, **kwargs):
        self.data.sort(*args, **kwargs)

    def filled(self):
        return self.capacity and len(self.data) >= self.capacity

    def max_val(self):
        pop_values = np.array([data.data.value for data in self.data])
        max_values = []
        for i in range(pop_values.shape[-1]):
            max_values.append( max(pop_values[i]) )
        
        return max_values

    def min_val(self):
        pop_values = np.array([data.data.value for data in self.data])
        min_values = []
        for i in range(pop_values.shape[-1]):
            min_values.append( min(pop_values[i]) )
        
        return min_values

    def normalize_para(self):
        fits = self.data
        indivs = [indiv.data for indiv in fits]
        
        if indivs[0].evaluated():
            # n_dim = len(indivs[0].get_variable())
            obj_dim = len(indivs[0].wvalue)
        else:
            raise("Indiv has not been evaluated yet.")
            
        self.max_obj_val = np.full(obj_dim, -np.inf)
        self.min_obj_val = np.full(obj_dim, np.inf)
        # normalized_value = np.zeros( (len(indivs), obj_dim) )

        for indiv in indivs:
            for i in range(obj_dim):
                self.max_obj_val[i] = max(indiv.value[i], self.max_obj_val[i])
                self.min_obj_val[i] = min(indiv.value[i], self.min_obj_val[i])

        return self.max_obj_val, self.min_obj_val

    def normalizing(self, max_paras=None, min_paras=None):
        indivs = [fit.data for fit in self.data]
        # obj_range = max_paras - min_paras
        if max_paras==None:
            max_paras = self.max_obj_val
        if min_paras==None:
            min_paras = self.min_obj_val
        
        obj_range = max_paras - min_paras

        if not indivs[0].evaluated():
            
            raise("Indiv has not been evaluated yet.")

        for indiv in indivs:
            # indiv.wvalue = tuple((indiv.value-min_paras)/obj_range)
            indiv.wvalue = np.array((indiv.wvalue)/obj_range)

###############################################################################

###############################################################################

class Normalization(object):
    '''
        Normalizer
    '''

    def __init__(self, population, max_ref=None, min_ref=None):
        '''
        初期個体を生成した後に呼び出して，解を正規化する
        '''
        indivs = [fit.data for fit in population]
        self.obj_dim = len(indivs[0])
        # self.weights = None
        self.weights = population[0].data.weight
        self.max_obj_val = np.full(self.obj_dim, -np.inf)
        self.min_obj_val = np.full(self.obj_dim, np.inf)
        self.selfset_max = None
        self.selfset_min = None

        # indivs_value = np.array([indiv.data.wvalue for indiv in population])

        if self.weights is not None:
            weight_exist = True
            self.nega_or_posi = np.array([1 if i>0 else (-1) for i in self.weights])
        
            # for i in range(len(self.weights)):
            #     self.max_obj_val[i] = self.nega_or_posi[i]*(self.max_obj_val[i])
            #     self.max_obj_val[i] = self.nega_or_posi[i]*(self.min_obj_val[i])
        else:
            self.nega_or_posi = np.array([1 for i in range(self.obj_dim)])
            
        print("+ or - => ",self.nega_or_posi)

        for fit in population:
            for i in range(self.obj_dim):
                data_value = fit.data.value[i]
                # self.max_obj_val[i] = max((data_value), (self.max_obj_val[i]))
                # self.min_obj_val[i] = min((data_value), (self.min_obj_val[i]))
                self.max_obj_val[i], self.min_obj_val[i] = self.abs_minmax(data_value, i)

        if max_ref is not None:
            self.max_obj_val = max_ref
            self.selfset_max = max_ref
            print("normalizer reference[max]",self.max_obj_val)
        if min_ref is not None:
            self.min_obj_val = min_ref
            self.selfset_min = min_ref
            print("normalizer reference[min]",self.min_obj_val)

        print("normalize para(max,min) ",self.max_obj_val, self.min_obj_val)
        self.obj_range = abs(self.max_obj_val - self.min_obj_val)
        self.ref = self.reference()

    def __call__(self, population, initial=False, **kwargs):
        '''
            Normalizer for population
        '''

        if initial == False:
            self.update_para(population, **kwargs)

        self.ref = self.reference()
        ref = self.ref
        for i in range(len(ref)):
            if self.nega_or_posi[i] < 0:
                ref[i] = self.max_obj_val[i]
        # print("norm ref", ref)

        for i in range(len(population)):
            val = (population[i].data.value)
            wvalue = (abs(val - ref)/self.obj_range)
            population[i].data.wvalue = wvalue

        return population

    def normalize_fit(self, fitness, initial=False, **kwargs):
        '''
            Normalizer for fitness
        '''
        if initial == False:
            pop = Population()
            pop.append(fitness)
            self.update_para(pop, **kwargs)
        
        # ref = self.min_obj_val
        ref = self.reference()
        val = fitness.data.value
        wvalue = (abs(val - ref)/self.obj_range)
        fitness.data.wvalue = wvalue

        return fitness

    def abs_minmax(self, val, index):
        '''
        絶対値が最大最小のものを返す
        '''
        abs_val = abs(val)
        abs_max = abs(self.max_obj_val[index])
        abs_min = abs(self.min_obj_val[index])

        res_max = self.max_obj_val[index]
        res_min = self.min_obj_val[index]

        if abs_val > abs_max:
            res_max = val
        elif np.isinf(abs_max):
            res_max = val
            print("resmax = ",res_max)
        
        if abs_val < abs_min:
            res_min = val
        elif np.isinf(abs_min):
            res_min = val
            print("resmin = ",res_min)

        return res_max, res_min

    def update_para(self, population, max_ref=None, min_ref=None):

        for fit in population:
            for i in range(self.obj_dim):
                data_value = fit.data.value[i]
                # self.max_obj_val[i] = max((data_value), (self.max_obj_val[i]))
                # self.min_obj_val[i] = min((data_value), (self.min_obj_val[i]))
                self.max_obj_val[i], self.min_obj_val[i] = self.abs_minmax(data_value, i)

        if self.selfset_max is not None:
            if max_ref is not None:
                self.max_obj_val = max_ref
            else:
                self.max_obj_val = self.selfset_max
        
        if self.selfset_min is not None:
            if min_ref is not None:
                self.min_obj_val = min_ref
            else:
                self.min_obj_val = self.selfset_min

        self.obj_range = abs(self.max_obj_val - self.min_obj_val)
        # self.ref = self.reference()

        
    def reference(self):
        ref = self.min_obj_val.copy()
        for i in range(len(ref)):
            if self.nega_or_posi[i] < 0:
                ref[i] = self.max_obj_val[i]

        return ref