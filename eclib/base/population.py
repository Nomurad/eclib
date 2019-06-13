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
        else:
            self.data = data or []
            self.capacity = capacity
            self.origin = origin

        self.max_obj_val = None
        self.min_obj_val = None

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
            obj_dim = len(indivs[0].value)
        else:
            raise("Indiv has not been evaluated yet.")
    
        if self.max_obj_val == None and self.min_obj_val == None:
            self.max_obj_val = np.full(obj_dim, -np.inf)
            self.min_obj_val = np.full(obj_dim, np.inf)
        # normalized_value = np.zeros( (len(indivs), obj_dim) )

        for indiv in indivs:
            for i in range(obj_dim):
                self.max_obj_val[i] = max(indiv.value[i], self.max_obj_val[i])
                self.min_obj_val[i] = min(indiv.value[i], self.min_obj_val[i])

        return self.max_obj_val, self.min_obj_val

    def normalizing(self, max_paras, min_paras):
        indivs = [indiv.data for indiv in self.data]
        # obj_range = max_paras - min_paras
        obj_range = max_paras

        if not indivs[0].evaluated():
            
            raise("Indiv has not been evaluated yet.")

        for indiv in indivs:
            # indiv.wvalue = tuple((indiv.value-min_paras)/obj_range)
            indiv.wvalue = tuple((indiv.value)/obj_range)

class Normalization(object):
    # def __init__(self, population):
    #     self.init_paras(population)

    def __init__(self, population, max_ref=None, min_ref=None):
        indivs = [indiv.data for indiv in population]
        self.obj_dim = len(indivs[0])
        self.max_obj_val = np.full(self.obj_dim, -np.inf)
        self.min_obj_val = np.full(self.obj_dim, np.inf)
        self.weights = None
        # self.max_obj_val = -np.inf
        # self.min_obj_val = np.inf
        indivs_value = np.array([indiv.data.value for indiv in population])

        # weight_bool = (np.array(population[0].data.wvalue)*np.array(population[0].data.value) < 0)
        # self.weights = np.zeros(len(weight_bool))
        # for i in range(self.obj_dim):
        #     self.weights[i] = (-1 if weight_bool[i]==True else 1)
        # print((self.weights))

        for fit in population:
            for i in range(self.obj_dim):
                data_value = fit.data.value[i]

                self.max_obj_val[i] = max((data_value), (self.max_obj_val[i]))
                self.min_obj_val[i] = min((data_value), (self.min_obj_val[i]))

        if max_ref is not None:
            self.max_obj_val = max_ref
        if min_ref is not None:
            self.min_obj_val = min_ref

        # print(self.max_obj_val , self.min_obj_val)
        self.obj_range = self.max_obj_val - self.min_obj_val

    def update_para(self, population, max_ref=None, min_ref=None):
        # self.max_obj_val = np.full(self.obj_dim, -np.inf)
        # self.min_obj_val = np.full(self.obj_dim, np.inf)

        indivs_value = np.array([indiv.data.value for indiv in population])

        # print("indivs:", len(indivs_value) )
        for i in range(self.obj_dim):
            self.max_obj_val[i] = max(indivs_value[:,i])
            self.min_obj_val[i] = min(indivs_value[:,i])


        # for indiv in population:
        #     for i in range(self.obj_dim):
        #         self.max_obj_val = max(np.max(indivs_value), self.max_obj_val)
        #         self.min_obj_val = min(np.min(indivs_value), self.min_obj_val)
                

        if max_ref is not None:
            self.max_obj_val = max_ref
        if min_ref is not None:
            self.min_obj_val = min_ref

        # print("max & min:",self.max_obj_val , self.min_obj_val)
        self.obj_range = self.max_obj_val - self.min_obj_val

    def __call__(self, population, initial=False, **kwargs):
        # if initial==False:
        # self.update_para(population, **kwargs)
    
        for i in range(len(population)):
            # population[i].data.wvalue = tuple((population[i].data.value-self.min_obj_val)/self.obj_range)
            population[i].data.wvalue = tuple((population[i].data.value)/self.max_obj_val*1)
            # print(population[i].data.wvalue)
        # print()

        return population