import numpy as np

class ReplayBuf(object):
    def __init__(self, maxlen: int):
        self._maxlen = maxlen
        self._data = [None for _ in range(length)]
        self._len = 0
        self._top_pointer = 0

    def __len__(self)->int:
        return self._len

    def __getitem__(self, index):
        if index<0 or index>=self._len:
            raise IndexError()
        return self._data[index]

    def append(self, value):
        self._data[self._top_pointer] = value
        self._top_pointer += 1
        if self._top_pointer>=self._maxlen:
            self._top_pointer = self._top_pointer//self._maxlen
        if self._len<self._maxlen:
            self._len += 1

    def get_by_indexes(self, indexes: list)->list:
        return self._data[indexes]

class State(object):
    def __init__(self, state: Any, action: Any, reward: float, next_state):
        self._state_id = zip_state(state)
        self.state = state
        self.action = action
        self.reward = reward
        self._next_state_id = zip_state(mext_state)

    @property
    def state(self):
        return unzip_state(self._state_id)

    @property
    def next_state(self):
        return unzip_state(self._next_state_id)

    @staticmethod
    def zip_state(state)-> int:
        return 0

    @staticmethod
    def unzip_state(id: int):
        return state

