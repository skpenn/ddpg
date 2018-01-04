import tensorflow as tf

class Critic(object):
    def __init__(self, Q_model, Q_model_apo, gamma: float, tau: float, batch_size: int=32, learning_rate: float=0.01 ):
        self._Q_model = Q_model
        self._Q_model_apo = Q_model_apo
        self._gamma = gamma
        self._tau = tau

        self._theta = Q_model.theta
        self._theta_apo = Q_model_apo.theta
        self._state = Q_model.state
        self._action = Q_model.action
        self._action_ = Q_model_apo.action
        self._next_state = Q_model_apo.state

        self._reward = tf.placeholder([batch_size, ], dtype="float32")

    @property
    def s(self):
        return self._state

    @property
    def a(self):
        return self._action

    def a_(self):
        return self._action_

    @property
    def s_iplus1(self):
        return self._next_state

    @property
    def Q(self):
        return Q_model_apo.Q


    @property
    def a_grad(self):
        return tf.grad

    def update_critic(self):
        y_i = self._reward + self._gamma * Q_model_apo.Q
        loss = tf.pow(y_i - Q_model)


class Actor(object):
    def __init__(self, Mu_model):
        pass