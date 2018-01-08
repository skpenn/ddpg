import tensorflow as tf


class Inputs(object):
    def __init__(self, state_shape: list, batch_size):
        self.s = tf.placeholder("float32", shape=[batch_size]+state_shape)


class Q_Model(object):
    def __init__(self, name: str,  inputs: Inputs, action_size: int, network_shape: tuple, batch_size: int= 32, trainable: bool = True):
        self.name = name
        self.state = inputs.s
        self.action = tf.placeholder("float32", shape=[batch_size, action_size])

        with tf.variable_scope(name) as scope:
            out = self.state

            # convolution layer
            for num, kernel_size, stride in network_shape[:-1]:
                out = tf.layers.conv2d(out, num, kernel_size, stride, activation=tf.nn.relu)
                out.trainable = trainable
                out = tf.layers.max_pooling2d(out, 2, 2)
                out.trainable = trainable

            # flatten
            fully_connected = tf.reshape(out, [batch_size, -1])
            fully_connected = tf.concat((fully_connected, self.action), axis=1)

            # fully connected layer
            for num in network_shape[-1]:
                fully_connected = tf.contrib.layers.fully_connected(fully_connected, num_outputs=num, activation_fn=None)
                fully_connected.trainable = trainable
                fully_connected = tf.contrib.layers.layer_norm(fully_connected, center=True, scale=True)
                fully_connected.trainable = trainable


            self._Q = tf.contrib.layers.fully_connected(fully_connected, 1, activation_fn=tf.tanh)
            self._Q.trainable = trainable

    @property
    def Q(self):
        return self._Q

    @property
    def a_grads(self):
        return tf.gradients(-1*self.Q, self.action)

    @property
    def theta(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


class Mu_Model(object):
    def __init__(self, name: str, inputs: Inputs, action_size: int, network_shape: tuple, batch_size: int=32, trainable: bool = True, y_grads=None):
        self.name = name
        self.state = inputs.s
        self.y_grads = y_grads

        with tf.variable_scope(name) as scope:
            out = self.state

            # convolution layer
            for num, kernel_size, stride in network_shape[:-1]:
                out = tf.layers.conv2d(out, num, kernel_size, stride, activation=tf.nn.relu)
                out.trainable = trainable
                out = tf.layers.max_pooling2d(out, 2, 2)
                out.trainable = trainable

            # flatten
            fully_connected = tf.reshape(out, [batch_size, -1])

            # fully connected layer
            for num in network_shape[-1]:
                fully_connected = tf.contrib.layers.fully_connected(fully_connected, num_outputs=num, activation_fn=None)
                fully_connected.trainable = trainable
                fully_connected = tf.contrib.layers.layer_norm(fully_connected, center=True, scale=True)
                fully_connected.trainable = trainable

            self._a = tf.contrib.layers.fully_connected(fully_connected, action_size, activation_fn=tf.sigmoid)
            self._a.trainable = trainable

    @property
    def a(self):
        return self._a

    @property
    def theta(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def grads(self):
        return tf.gradients(self._a, self.theta, self.y_grads)


