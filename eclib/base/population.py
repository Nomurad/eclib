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

            #     if indiv.value[i] > self.max_obj_val[i]:
            #         self.max_obj_val[i] = indiv.value[i]
            #     self.max_obj_val[i] = max(indiv.value[i], self.max_obj_val[i])

            #     if indiv.value[i] < self.min_obj_val[i]:
            #         self.min_obj_val[i] = indiv.value[i]
        
        # obj_range = max_obj_val - min_obj_val
        # for i, indiv in enumerate(indivs):
        #     indiv.wvalue = tuple((indiv.value-min_obj_val)/obj_range)
            # indiv.wvalue = tuple(np.zeros(obj_dim))

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
