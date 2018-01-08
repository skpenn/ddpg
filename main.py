from environment import Environment
from model import Model

if __name__=="__main__":
    env = Environment()
    model = Model(env=env,
                  state_shape=env.state_shape,
                  action_size=env.action_size,
                  q_network_shape=((16, 8, 4), (32, 4, 2), (64, 2, 1), [256, 32, 10]),
                  mu_network_shape=((16, 8, 4), (32, 4, 2), (64, 2, 1), [256, 32]),
                  buffer_size=256,
                  gamma=0.1,
                  tau=0.1,
                  noise_stddev=0.1,
                  save_dir="log/model.ck-pt",
                  learning_rate=0.01,
                  batch_size=32,
                  episode=5000,
                  train_epoch=5)
    env.reset()
    model.train()
