import gym
import numpy as np
from model import Model

class Action_reshape(object):
    def __init__(self, action_low, action_high):
        self.action_low = action_low
        self.action_high = action_high

    def reshape(self, action):
        new_actions = np.zeros(action.shape)
        for i, act, low, high in enumerate(zip(action, self.action_low, self.action_high)):
            new_actions[i] = act*(high - low) + low
        return new_actions

if __name__=="__main__":
    env = gym.make('HalfCheetah-v1')
    action_reshape = Action_reshape(env.action_space.low, env.action_space.high)
    model = Model(env=env.step,
                  state_shape=list(env.observation_space.shape),
                  action_size=env.action_size,
                  q_network_shape=tuple([[32, 10]]),
                  mu_network_shape=tuple([[32]]),
                  buffer_size=256,
                  gamma=0.9,
                  tau=0.1,
                  noise_stddev=0,
                  save_dir="log/model.ck-pt",
                  learning_rate=0.01,
                  batch_size=32,
                  episode=10000,
                  train_epoch=50)
    env.reset()
    model.train()
