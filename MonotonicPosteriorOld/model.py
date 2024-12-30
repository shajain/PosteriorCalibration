import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import numpy as np
from tensorflow.keras.layers import Layer

class MonotonicPosteriorNet(tf.keras.Model):

    def __init__(self, n_units, n_hidden, dropout_rate):
        super(MonotonicPosteriorNet, self).__init__()
        self.n_units = n_units
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate
        self.Dens = list()
        self.BN = list()
        self.Drop = list()
        for i in np.arange(n_hidden):
            if i == 0:
                self.Dens.append(layers.Dense(n_units, activation='relu'))
            else:
                self.Dens.append(layers.Dense(n_units, activation='relu'))
            self.BN.append(layers.BatchNormalization())
            self.Drop.append(layers.Dropout(dropout_rate))
        self.dens_last = layers.Dense(1)
        #self.subtractC = SubtractScalar()

    def call(self, inputs):
        for i in np.arange(len(self.Dens)):
            if i == 0:
                x = self.Dens[i](inputs)
            else:
                x = self.Dens[i](x)
            x = self.BN[i](x)
            x = self.Drop[i](x)
        x = self.dens_last(x)
            # x = self.BN_last(x)
        return activations.softplus(x)

    def copy(self):
        copy = MonotonicPosteriorNet(self.n_units, self.n_hidden, self.dropout_rate)
        input_dim = self.layers[0].weights[0].shape[0]
        copy.build((None, input_dim))
        for l1, l2 in zip(self.layers, copy.layers):
            l2.set_weights(l1.get_weights( ))
        return copy



class SubtractScalar(Layer):

    def __init__(self, **kwargs):
        # self.output_shape = output_shape
        super(SubtractScalar, self).__init__(**kwargs)
        self.C = self.add_weight(name='C',
                                 shape=(1, 1),
                                 initializer='zeros',
                                 trainable=True)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(SubtractScalar, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        out = x -self.C
        # pdb.set_trace()
        return out

    def compute_output_shape(self, input_shape):
        return input_shape