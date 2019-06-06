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

    def normalizing(self):
        fits = self.data
        indivs = [indiv.data for indiv in fits]
        print(indivs.__doc__)
        
        if indivs[0].evaluated():
            n_dim = len(indivs[0].get_variable())
            obj_dim = len(indivs[0].value)
        else:
            raise("Indiv has not been evaluated yet.")
    
        max_obj_val = np.full(obj_dim, -np.inf)
        min_obj_val = np.full(obj_dim, np.inf)
        normalized_value = np.zeros( (len(indivs), obj_dim) )

        for indiv in indivs:
            for i in range(obj_dim):
                if indiv.value[i] > max_obj_val[i]:
                    max_obj_val[i] = indiv.value[i]
                    print("max:",max_obj_val)
                if indiv.value[i] < min_obj_val[i]:
                    min_obj_val[i] = indiv.value[i]
                    print("min:",min_obj_val)
        
        obj_range = max_obj_val - min_obj_val
        for i, indiv in enumerate(indivs):
            indiv.wvalue = tuple((indiv.value-min_obj_val)/obj_range)

            for j in range(obj_dim):
                print( f'{indiv.value[j]:>7.4f}', end=" ")
            print("| ", end=" ")
            for j in range(obj_dim):
                print( f'{indiv.wvalue[j]:>7.4f}', end=" ")
            print()

        print()
        print(max_obj_val)
        print(min_obj_val) 
