# -*- coding: utf-8 -*-
import numpy as np


class UniformInitializer(object):
    ''' [0, 1)の範囲の一様乱数による実数配列を返す '''

    def __init__(self, size):
        self._size = size

    def __call__(self):
        return np.random.rand(self._size)
