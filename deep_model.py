import tensorflow as tf


class Q_Model(object):
    def __init__(self, name: str, state, action, network_shape: tuple, batch_size: int= 32, trainable: bool = True):
        self.name = name
        self.state = state
        self.action = action

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
            fully_connected = tf.contrib.layers.layer_norm(fully_connected, center=True, scale=True)
            fully_connected.trainable = trainable

            # fully connected layer
            for num in network_shape[-1]:
                fully_connected = tf.contrib.layers.fully_connected(fully_connected, num_outputs=num, activation_fn=None)
                fully_connected.trainable = trainable
                fully_connected = tf.contrib.layers.layer_norm(fully_connected, center=True, scale=True)
                fully_connected.trainable = trainable
                fully_connected = tf.nn.relu(fully_connected)

            self._Q = tf.contrib.layers.fully_connected(fully_connected, 1, activation_fn=None)
            self._Q.trainable = trainable

        self.theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def Q(self):
        return self._Q

    @property
    def a_grads(self):
        return tf.gradients(-1*self.Q, self.action)


class Mu_Model(object):
    def __init__(self, name: str, state, action_size: int, network_shape: tuple, batch_size: int=32, trainable: bool = True, y_grads=None):
        self.name = name
        self.state = state
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
            fully_connected = tf.contrib.layers.layer_norm(fully_connected, center=True, scale=True)
            fully_connected.trainable = trainable

            # fully connected layer
            for num in network_shape[-1]:
                fully_connected = tf.contrib.layers.fully_connected(fully_connected, num_outputs=num, activation_fn=None)
                fully_connected.trainable = trainable
                fully_connected = tf.contrib.layers.layer_norm(fully_connected, center=True, scale=True)
                fully_connected.trainable = trainable
                fully_connected = tf.nn.relu(fully_connected)

            self._a = tf.contrib.layers.fully_connected(fully_connected, action_size, activation_fn=tf.sigmoid)
            self._a.trainable = trainable

        self.theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def a(self):
        return self._a

    @property
    def grads(self):
        return tf.gradients(self._a, self.theta, self.y_grads)

