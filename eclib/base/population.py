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

        for indiv in population:
            for i in range(self.obj_dim):
                self.max_obj_val[i] = max(indiv.data.value[i], self.max_obj_val[i])
                self.min_obj_val[i] = min(indiv.data.value[i], self.min_obj_val[i])

        if max_ref is not None:
            self.max_obj_val = max_ref
        if min_ref is not None:
            self.min_obj_val = min_ref

        # print(self.max_obj_val , self.min_obj_val)
        # self.obj_range = self.max_obj_val - self.min_obj_val
        self.obj_range = self.max_obj_val

    def update_para(self, population, max_ref=None, min_ref=None):
        for indiv in population:
            for i in range(self.obj_dim):
                self.max_obj_val[i] = max(indiv.data.value[i], self.max_obj_val[i])
                self.min_obj_val[i] = min(indiv.data.value[i], self.min_obj_val[i])
                
        # indivs_value = np.array([indiv.data.value for indiv in population])

        # # print("indivs:", len(indivs_value) )
        # for j in range(self.obj_dim):
        #     self.max_obj_val[j] = max(indivs_value[:,j])
        #     self.min_obj_val[j] = min(indivs_value[:,j])
        print(self.max_obj_val, self.min_obj_val)

        if max_ref is not None:
            self.max_obj_val = max_ref
        if min_ref is not None:
            self.min_obj_val = min_ref

        # print(self.max_obj_val , self.min_obj_val)
        self.obj_range = self.max_obj_val - self.min_obj_val

    def __call__(self, population, initial=False, **kwargs):
        # if initial==False:
        self.update_para(population, **kwargs)
    
        for i in range(len(population)):
            population[i].data.wvalue = tuple((population[i].data.value-self.min_obj_val)/self.obj_range)
            # population[i].data.wvalue = tuple((population[i].data.value)/self.max_obj_val)
            # print(population[i].data.wvalue)
        # print()

        return population