import tensorflow as tf
import numpy as np
from deep_model import Q_Model, Mu_Model, Inputs
from ddpg import Actor, Critic
from memory import ReplayBuf, Transition

class Model(object):
    def __init__(self, env: function, state_shape: tuple,  network_shape: tuple, buffer_size: int, noise_stddev: float, batch_size: int, epoch: int):
        self._env = env
        self._batch_size = batch_size
        self._state_shape = state_shape
        self._network_shaple = network_shape
