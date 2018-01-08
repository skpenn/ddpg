import tensorflow as tf


class Critic(object):
    def __init__(self, Q_model, Q_model_apo, gamma: float, tau: float, batch_size: int=32, learning_rate: float=0.01 ):
        self.Q_model = Q_model
        self.Q_model_apo = Q_model_apo
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate

        self.reward = tf.placeholder(dtype="float32", shape=(batch_size, 1))
        self._theta = Q_model.theta
        self._theta_apo = Q_model_apo.theta

        y_i = self.reward + self.gamma * self.Q_model_apo.Q
        self.loss = tf.pow((y_i - self.Q_model.Q), 2) * 0.5
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    @property
    def s(self):
        return self.Q_model.state

    @property
    def a(self):
        return self.Q_model.action

    @property
    def s_apo(self):
        return self.Q_model_apo.state

    @property
    def a_apo(self):
        return self.Q_model_apo.action

    @property
    def Q(self):
        return self.Q_model.Q

    @property
    def Q_apo(self):
        return self.Q_model_apo.Q

    @property
    def a_grads(self):
        return self.Q_model.a_grads

    def minimize_loss(self):
        return self._optimizer

    def update_target_net(self):
        assignments = (tf.assign(var_apo, var) if "norm" in var.name.lower() else
            tf.assign(var_apo, var * self.tau + var_apo * (1 - self.tau)) for var_apo, var in zip(self._theta_apo, self._theta))
        return tf.group(*assignments)


class Actor(object):
    def __init__(self, Mu_model, Mu_model_apo, gamma: float, tau: float, learning_rate: float=0.01):
        self.Mu_model_apo = Mu_model_apo
        self.Mu_model = Mu_model
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate

        self._theta = Mu_model.theta
        self._theta_apo = Mu_model_apo.theta

        self._optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).apply_gradients(zip(Mu_model.grads, self._theta))

    @property
    def s(self):
        return self.Mu_model.state

    @property
    def a(self):
        return self.Mu_model.a

    @property
    def s_apo(self):
        return self.Mu_model_apo.state

    @property
    def a_apo(self):
        return self.Mu_model_apo.a

    def maximize_action_q(self):
        return self._optimizer

    def update_target_net(self):
        assignments = (tf.assign(var_apo, var) if "norm" in var.name.lower() else
                       tf.assign(var_apo, var * self.tau + var_apo * (1 - self.tau)) for var_apo, var in
                       zip(self._theta_apo, self._theta))
        return tf.group(*assignments)

