import tensorflow as tf
import numpy as np
from deep_model import Q_Model, Mu_Model, Inputs
from ddpg import Actor, Critic
from memory import ReplayBuf, Transition


class Model(object):
    def __init__(self, env: callable, state_shape: list, action_size: int, q_network_shape: tuple,
                 mu_network_shape: tuple, buffer_size: int, gamma: float, tau: float, noise_stddev: float,
                 learning_rate: float, batch_size: int, episode: int, train_epoch: int):
        self.env = env
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        self.action_size = action_size
        self.q_network_shape = q_network_shape
        self.mu_network_shape = mu_network_shape
        self.buffer_size = buffer_size
        self.noise_stddev = noise_stddev
        self.episode = episode
        self.train_epoch = train_epoch

        input_s = Inputs(state_shape, batch_size)
        input_s_plus_1 = Inputs(state_shape, batch_size)
        q = Q_Model("Q", input_s, action_size, q_network_shape, batch_size)
        q_apo = Q_Model("Q_apo", input_s_plus_1, action_size, q_network_shape, batch_size, trainable=False)
        mu = Mu_Model("Mu", input_s, action_size, mu_network_shape, batch_size)
        mu_apo = Mu_Model("Mu", input_s_plus_1, action_size, mu_network_shape, batch_size, trainable=False)
        self._actor = Actor(mu, mu_apo, gamma, tau, batch_size, learning_rate)
        self._critic = Critic(q, q_apo, gamma, tau, batch_size, learning_rate)
        self._replayBuf = ReplayBuf(buffer_size)
        self._sess = tf.Session()

    def train(self):
        # input definition
        s_i = self._actor.s
        s_i_next = self._actor.s_apo
        a_i = self._actor.a
        a_i_next = self._actor.a_apo
        r_i = self._critic.reward

        # data container definition
        data_s_i = np.zeros([self.batch_size] + self.state_shape)
        data_a_i = np.zeros([self.batch_size, self.action_size])
        data_r_i = np.zeros([self.batch_size, 1])
        data_s_i_next = np.zeros([self.batch_size] + self.state_shape)

        self._sess.run(tf.global_variables_initializer())

        for _ in range(self.episode):
            end_state, _, _ = self.env()
            while True:
                start_state = end_state
                data_s_i[0] = start_state
                action = self._sess.run(self._actor.a, {s_i: data_s_i})[0] + np.random.normal(scale=self.noise_stddev, size=self.action_size) # get an action, a = Mu(s)+Noise
                end_state, reward, done = self.env(action)
                if done:   # final state
                    break

                transition = Transition(start_state, action, reward, end_state)
                self._replayBuf.append(transition)

                if len(self._replayBuf) >= self.buffer_size:
                    loss = 0
                    a = 0
                    for _ in range(self.train_epoch):
                        sample = list(range(self.buffer_size))
                        np.random.shuffle(sample)
                        sample_batch = self._replayBuf.get_by_indexes(sample[:self.batch_size])   # get batch
                        for i, data in enumerate(sample_batch):                                   # generate data
                            data_s_i[i] = data.state
                            data_a_i[i] = data.action
                            data_r_i[i] = data.reward
                            data_s_i_next[i] = data.next_state

                        data_a_i_next = self._sess.run(self._actor.a_apo, {s_i_next:data_s_i_next})  # get a_i+1 = Mu(s_i+1)
                        _, loss = self._sess.run([self._critic.minimize_loss(), self._critic.loss],   # minimize critic loss
                                       {s_i: data_s_i,
                                        a_i: data_a_i,
                                        s_i_next: data_s_i_next,
                                        a_i_next: data_a_i_next,
                                        r_i: data_r_i})
                        _, a = self._sess.run([self._actor.maximize_action_q(self._critic.a_grads),self._actor.a],     # maximize actor-critic value
                                       {s_i: data_s_i})
                        q = self._sess.run(self._critic.Q,                  # calculate q value
                                              {s_i: data_s_i, a_i:a})
                        self._sess.run(self._critic.update_target_net())     # update target network
                        self._sess.run(self._actor.update_target_net())
                    print("Average loss: {}".format(loss.mean()))
                    print("Average Q value: {}".format(q.mean()))
