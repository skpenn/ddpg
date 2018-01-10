import tensorflow as tf
import numpy as np
import time
from deep_model import Q_Model, Mu_Model
from ddpg import Actor, Critic
from memory import ReplayBuf


class Model(object):
    def __init__(self, env: callable, state_shape: list, action_size: int, q_network_shape: tuple,
                 mu_network_shape: tuple, buffer_size: int, gamma: float, tau: float, noise_stddev: float,
                 save_dir: str, actor_learning_rate: float, critic_learning_rate: float, batch_size: int, episode: int,
                 train_epoch: int, run_epoch: int=100, action_reshape: callable=None):
        self.env = env
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.action_size = action_size
        self.q_network_shape = q_network_shape
        self.mu_network_shape = mu_network_shape
        self.buffer_size = buffer_size
        self.noise_stddev = noise_stddev
        self.episode = episode
        self.train_epoch = train_epoch
        self.dir = save_dir
        self.action_reshape = action_reshape
        self.run_epoch = run_epoch

        state_i = tf.placeholder("float32", [batch_size]+state_shape)
        state_i_next = tf.placeholder("float32", [batch_size]+state_shape)
        action_i = tf.placeholder("float32", [batch_size, action_size])
        mu_apo = Mu_Model("Mu_apo", state_i_next, action_size, mu_network_shape, batch_size, trainable=False)
        q = Q_Model("Q_0", state_i, action_i, q_network_shape, batch_size)
        mu = Mu_Model("Mu_0", state_i, action_size, mu_network_shape, batch_size, y_grads=q.a_grads)
        q_apo = Q_Model("Q_apo", state_i_next, mu_apo.a, q_network_shape, batch_size, trainable=False)
        self._actor = Actor(mu, mu_apo, gamma, tau, actor_learning_rate)
        self._critic = Critic(q, q_apo, gamma, tau, batch_size, critic_learning_rate)
        self._s_buf = ReplayBuf(buffer_size, self.state_shape)
        self._a_buf = ReplayBuf(buffer_size, [self.action_size])
        self._r_buf = ReplayBuf(buffer_size, [1])
        self._s_next_buf = ReplayBuf(buffer_size, self.state_shape)
        self._sess = None
        self._saver = tf.train.Saver()

    def train(self):
        # input definition
        s_i = self._actor.s
        s_i_next = self._actor.s_apo
        a_i = self._critic.a
        r_i = self._critic.reward

        # data container definition
        data_s_i = np.zeros([self.batch_size] + self.state_shape)

        ck_pt = tf.train.get_checkpoint_state(self.dir)
        if ck_pt is not None:

            self._sess = tf.Session()
            self._saver.restore(self._sess, tf.train.latest_checkpoint(self.dir))
            '''
            except:
                print("Load model error")
                self._sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            '''
        else:
            self._sess = tf.Session()
            self._sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            self._sess.run([self._actor.init_target_net(), self._critic.init_target_net()])

        count = 0
        for _ in range(self.episode):
            end_state = self.env.reset()
            while True:
                try:
                    self.env.render()
                except:
                    pass
                start_state = end_state
                data_s_i[0] = start_state
                action = self._sess.run(self._actor.a, {s_i: data_s_i})[0] + np.random.normal(0, scale=self.noise_stddev, size=self.action_size) # get an action, a = Mu(s)+Noise
                if self.action_reshape is not None:
                    end_state, reward, _done, _ = self.env.step(self.action_reshape(action))
                else:
                    end_state, reward, _done, _ = self.env.step(action)
                #print("Action: {}".format(action))
                #print("Reward: {}".format(reward))


                self._s_buf.append(start_state)
                self._a_buf.append(action)
                self._r_buf.append(np.array([reward]))
                self._s_next_buf.append(end_state)

                count += 1

                if _done:   # final state
                    break

                if len(self._s_buf) >= self.batch_size and count >= self.run_epoch:
                    print("Action: {}".format(action))
                    count = 0
                    loss = np.zeros([0])
                    q = np.zeros([0])

                    for i in range(self.train_epoch):
                        sample = list(range( len(self._s_buf) ))
                        np.random.shuffle(sample)
                        data_s_i = self._s_buf.get_by_indexes(sample[:self.batch_size])   # get batch
                        data_a_i = self._a_buf.get_by_indexes(sample[:self.batch_size])
                        data_r_i = self._r_buf.get_by_indexes(sample[:self.batch_size])
                        data_s_i_next= self._s_next_buf.get_by_indexes(sample[:self.batch_size])

                        _, loss = self._sess.run([self._critic.minimize_loss(), self._critic.loss],   # minimize critic loss
                                       {s_i: data_s_i,
                                        a_i: data_a_i,
                                        s_i_next: data_s_i_next,
                                        r_i: data_r_i})

                        a = self._sess.run(self._actor.a, {s_i: data_s_i})
                        _, a = self._sess.run([self._actor.maximize_action_q(), self._actor.a],     # maximize actor-critic value
                                       {s_i: data_s_i, a_i: a})
                        q = self._sess.run(self._critic.Q,                  # calculate q value
                                              {s_i: data_s_i, a_i: a})
                        self._sess.run(self._critic.update_target_net())     # update target network
                        self._sess.run(self._actor.update_target_net())

                    print("Average loss: {}".format(loss.mean()))
                    print("Average Q value: {}".format(q.mean()))
                    self._saver.save(self._sess, self.dir)
