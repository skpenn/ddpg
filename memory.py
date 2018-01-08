import numpy as np


class ReplayBuf(object):
    def __init__(self, max_len: int, shape):
        self.max_len = max_len
        self._data = np.zeros([max_len]+shape)
        self._len = 0
        self._top_pointer = 0

    def __len__(self)->int:
        return self._len

    def __getitem__(self, index):
        if not 0 <= index < self._len:
            raise IndexError()
        return self._data[index]

    def append(self, value):
        self._data[self._top_pointer] = value
        self._top_pointer += 1
        if self._top_pointer >= self.max_len:
            self._top_pointer = self._top_pointer//self.max_len
        if self._len<self.max_len:
            self._len += 1

    def get_by_indexes(self, indexes: list)->np.ndarray:
        return self._data[indexes]

    def empty(self):
        self._len = 0
        self._top_pointer = 0
