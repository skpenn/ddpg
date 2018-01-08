class ReplayBuf(object):
    def __init__(self, max_len: int):
        self.max_len = max_len
        self._data = [None for _ in range(max_len)]
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

    def get_by_indexes(self, indexes: list)->map:
        return map(self._data.__getitem__, indexes)

    def empty(self):
        self._len = 0
        self._top_pointer = 0


class Transition(object):
    def __init__(self, state, action, reward: float, next_state):
        self._state_id = self.zip_state(state)
        self.action = action
        self.reward = reward
        self._next_state_id = self.zip_state(next_state)

    @property
    def state(self):
        return self.unzip_state(self._state_id)

    def get_all(self)->tuple:
        return self.state, self.action, self.reward, self.next_state

    @property
    def next_state(self):
        return self.unzip_state(self._next_state_id)

    @staticmethod
    def zip_state(state):
        return state

    @staticmethod
    def unzip_state(state_id: int):
        return state_id

