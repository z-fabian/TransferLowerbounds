import tensorflow as tf
from tensorflow.keras import layers


class OneHidden(tf.keras.Model):
    def __init__(self, hidden_units, num_classes, use_bias=False, use_relu=True, softmax=False):
        super(OneHidden, self).__init__()
        self.hidden_layer = Linear(hidden_units, use_bias, name='hidden_layer')
        self.output_layer = Linear(num_classes, use_bias, name='output_layer')
        self.use_relu = use_relu
        self.softmax = softmax

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        if self.use_relu:
            x = tf.nn.relu(x)
        x = self.output_layer(x)
        if self.softmax:
            x = tf.nn.softmax(x)
        return x

    def train_only(self, layer):
        if layer == 'hidden':
            self.output_layer.trainable = False
            self.hidden_layer.reset_weights()
        elif layer == 'output':
            self.hidden_layer.trainable = False
            self.output_layer.reset_weights()

    def get_hidden_layer_as_np(self):
        return self.hidden_layer.get_weights_as_np()

    def get_output_layer_as_np(self):
        return self.output_layer.get_weights_as_np()


class Linear(layers.Layer):
    def __init__(self, hidden_units, use_bias=False, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.use_bias = use_bias

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.hidden_units),
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),
                                 trainable=True,
                                 name='W')

        if self.use_bias:
            self.b = self.add_weight(shape=(self.hidden_units,),
                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),
                                     trainable=True,
                                     name='b')

    def call(self, inputs):
        x = tf.matmul(inputs, self.W)
        if self.use_bias:
            x += self.b
        return x

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'hidden_units': self.hidden_units, 'use_bias': self.use_bias})
        return config

    def reset_weights(self):
        W_init = tf.random.normal(self.W.shape, mean=0.0, stddev=0.05)
        self.W.assign(W_init)
        if self.use_bias:
            b_init = tf.random.normal(self.b.shape, mean=0.0, stddev=0.05)
            self.b.assign(b_init)

    def get_weights_as_np(self):
        if self.use_bias:
            return self.W.numpy(), self.b.numpy()
        else:
            return self.W.numpy()
