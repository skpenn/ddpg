import gym
import numpy as np
from model import Model

class Action_reshape(object):
    def __init__(self, action_low, action_high):
        self.action_low = action_low
        self.action_high = action_high

    def reshape(self, action):
        new_actions = np.zeros(action.shape)
        for i, zipped in enumerate(zip(action, self.action_low, self.action_high)):
            act, low, high = zipped
            new_actions[i] = act*(high - low) + low
        return new_actions

if __name__=="__main__":
    env = gym.make('HalfCheetah-v1')
    action_reshape = Action_reshape(env.action_space.low, env.action_space.high)
    model = Model(env=env,
                  state_shape=list(env.observation_space.shape),
                  action_size=len(env.action_space.low),
                  q_network_shape=tuple([[64, 32]]),
                  mu_network_shape=tuple([[48, 48]]),
                  buffer_size=100000,
                  gamma=0.99,
                  tau=0.01,
                  noise_stddev=0.001,
                  save_dir="log/model.ck-pt",
                  actor_learning_rate=0.0001,
                  critic_learning_rate=0.001,
                  batch_size=128,
                  episode=10000,
                  train_epoch=50,
                  run_epoch=100,
                  action_reshape=action_reshape.reshape)
    env.reset()
    model.train()
