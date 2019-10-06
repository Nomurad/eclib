# -*- coding: utf-8 -*-

class Pool(object):
    def __init__(self, cls):
        self.cls = cls
        self.current_id = 0
        self.data = []
        self.name = cls.__name__

    def __call__(self, *args, **kwargs):
        obj = self.cls(*args, **kwargs)
        obj.id = self.current_id
        self.data.append(obj)
        self.current_id += 1
        return obj

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return self.current_id

    def isinstance(self, obj):
        return isinstance(obj, self.cls)


class Environment(object):
    def __init__(self):
        # self.state = {}
        self.obj_table = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def register(self, cls):
        pool = Pool(cls)
        self.obj_table[cls] = pool
        return pool


class Creator(object):
    def __init__(self, initializer, pool):
        self.initializer = initializer
        self._pool = pool
        # self.pool = env.register(type_)

    def __call__(self):
        data = self.initializer()
        obj = self._pool(data, origin=self)
        return obj

    @property
    def origin(self):
        return ()
