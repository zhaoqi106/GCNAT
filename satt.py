#关于attention的代码
import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow import keras

from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout

class GraphAttentionLayer(keras.layers.Layer):

    def __init__(self,
                 F_ ,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        super(GraphAttentionLayer, self).__init__()
        self.F_ = F_
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        self.kernels = []
        self.biases = []
        self.attn_kernels = []

        if attn_heads_reduction == 'concat':
            self.output_dim = self.F_ * self.attn_heads
        else:
            self.output_dim = self.F_



def build(self, input_shape):
    assert len(input_shape) >= 2
    F = input_shape[0][-1]

    for head in range(self.attn_heads):
        kernel = self.add_weight(shape=(F, self.F_),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 name='kernel_{}'.format(head))
        self.kernels.append(kernel)

        if self.use_bias:
            bias = self.add_weight(shape=(self.F_,),
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint,
                                   name='bias_{}'.format(head))
            self.biases.append(bias)

        attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                           initializer=self.attn_kernel_initializer,
                                           regularizer=self.attn_kernel_regularizer,
                                           constraint=self.attn_kernel_constraint,
                                           name='attn_kernel_self_{}'.format(head), )
        attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                             initializer=self.attn_kernel_initializer,
                                             regularizer=self.attn_kernel_regularizer,
                                             constraint=self.attn_kernel_constraint,
                                             name='attn_kernel_neigh_{}'.format(head))
        self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
    self.built = True


def call(self, inputs):
    X = inputs[0]
    A = inputs[1]

    outputs = []
    for head in range(self.attn_heads):
        kernel = self.kernels[head]
        attention_kernel = self.attn_kernels[head]

        features = K.dot(X, kernel)

        attn_for_self = K.dot(features, attention_kernel[0])
        attn_for_neighs = K.dot(features, attention_kernel[1])

        dense = attn_for_self + K.transpose(attn_for_neighs)

        dense = tf.nn.leaky_relu(alpha=0.2)(dense)

        mask = -10e9 * (1.0 - A)
        dense += mask

        dense = K.softmax(dense)

        dropout_attn = Dropout(self.dropout_rate)(dense)
        dropout_feat = Dropout(self.dropout_rate)(features)

        node_features = K.dot(dropout_attn, dropout_feat)

        if self.use_bias:
            node_features = K.bias_add(node_features, self.biases[head])

        outputs.append(node_features)

    if self.attn_heads_reduction == 'concat':
        output = K.concatenate(outputs)
    else:
        output = K.mean(K.stack(outputs), axis=0)

    output = self.activation(output)
    return output


def compute_output_shape(self, input_shape):
    output_shape = input_shape[0][0], self.output_dim
    return output_shape


