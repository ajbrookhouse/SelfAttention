import numpy as np

import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer

from keras.layers import Dropout
from keras.layers import Softmax
from keras.layers import BatchNormalization


class SelfAttention(Layer):
    """
    My implementation of a self attention layer.

    This class is of type keras.engine.topology.Layer, so it can be added into any keras model easily. You can read more about self attention layers here ().

    Parameters:
    kqLen (int): length of the key and query vectors generated from input
    valLen (int): length of the value vectors generated from the input
    return_sequence (bool): whether to return all timesteps or just the last one, note time steps are not masked so if return_sequence is True, earlier time steps will contain information about future time steps
    dropout (float): value between 0 and 1 to use for dropout in the layer. Dropout is applied to the key, query, and value matricies and also the output of the layer
    bias (bool): whether or not to train an additive bias for the key, query, and value matricies

    Returns:
    Tensor: if return_sequence = True, tensor is of shape (batch size, sequence length, valLen), else it is (batch size, valLen)
    """

    def __init__(self, kqLen, valLen, return_sequence = True, dropout=0, bias = True, cpu=True, **kwargs):
        self.kqLen = kqLen
        self.valLen = valLen
        self.return_sequence = return_sequence
        self.keyDropout = Dropout(dropout)
        self.queryDropout = Dropout(dropout)
        self.valueDropout = Dropout(dropout)
        self.outputDropout = Dropout(dropout)
        self.layerNorm = BatchNormalization()
        self.softmax = Softmax()
        self.bias = bias

        if cpu:
            self.matmul = tf.matmul
        else:
            self.matmul = K.batch_dot

        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.seqLen = input_shape[1]
        self.inputLen = input_shape[2]
        self.scaleNumber = self.inputLen ** .5

        self.keyW=self.add_weight(name="key_weight",shape=(self.inputLen, self.kqLen),initializer="normal")
        self.queryW=self.add_weight(name="query_weight",shape=(self.inputLen, self.kqLen),initializer="normal")
        self.valW=self.add_weight(name="value_weight",shape=(self.inputLen, self.valLen),initializer="normal")

        if self.bias:
            self.keyB = self.add_weight(name="key_bias",shape=(self.seqLen, self.kqLen),initializer="zeros")
            self.queryB = self.add_weight(name="query_bias",shape=(self.seqLen, self.kqLen),initializer="zeros")
            self.valB = self.add_weight(name="value_bias",shape=(self.seqLen, self.valLen),initializer="zeros")
        else:
            self.keyB = K.constant(0)
            self.queryB = K.constant(0)
            self.valB = K.constant(0)

        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        keys = self.matmul(x,self.keyW) 
        keys = keys + self.keyB
        keys = self.keyDropout(keys)

        queries = self.matmul(x, self.queryW) 
        queries = queries + self.queryB
        queries = self.queryDropout(queries)

        values = self.matmul(x, self.valW)
        values = values + self.valB
        values = self.valueDropout(values)

        outputs = self.matmul(queries, K.permute_dimensions(keys, (0,2,1)))
        outputs = outputs / self.scaleNumber
        outputs = self.softmax(outputs)

        w = self.matmul(outputs, values)

        if not self.return_sequence:
            w = w[:,-1,:]

        w = self.layerNorm(w)

        w = self.outputDropout(w)

        return w

    def compute_output_shape(self, input_shape):
        if self.return_sequence:
            return (input_shape[0], input_shape[1], self.valLen)
        else:
            return (input_shape[0], self.valLen)

class AddSinusoidalPositionalEncodings(Layer):
    """
    My implementation of a positional encoding to be used for an attention layer.

    This class is of type keras.engine.topology.Layer, so it can be added into any keras model easily. You can read more about positional encodings here ().
    This layer caches input embeddings in RAM, so if the same shape input is given it does not need to recalculate

    Returns:
    Tensor: will be same shape as the input tensor, positional encodings will be added to the input vector
    """

    def __init__(self, **kwargs):
        super(AddSinusoidalPositionalEncodings, self).__init__(**kwargs)
        self.lookupDic = {}

    def build(self, input_shape):
        super(AddSinusoidalPositionalEncodings, self).build(input_shape)

    def getSinusoidalPositionalEncoding(self, t, d=100):
        a = np.zeros(d)
        for i in range(d):

            if i % 2 == 0:
                sinfun = np.sin
                k = i/2
            else:
                sinfun = np.cos
                k = (i-1)/2

            w = 1/(10000**(2*k/d))
            a[i] = sinfun(w * t)
        return a.astype('float32')

    def getSinusoidalPositionalEncodings(self, batchVector):
        s = tuple(batchVector.shape)
        if s in self.lookupDic:
            return self.lookupDic[s]

        try:
            encodings = np.zeros(batchVector.shape)
        except TypeError: #Will happen when initializing
            return batchVector
        for batchIndex in range(batchVector.shape[0]):
            for timeVectorIndex in range(batchVector[batchIndex].shape[0]):
                encodings[batchIndex][timeVectorIndex] += self.getSinusoidalPositionalEncoding(timeVectorIndex, d=int(batchVector[batchIndex][timeVectorIndex].shape[0]))
        self.lookupDic[s] = tf.constant(encodings.astype('float32'))
        return self.lookupDic[s]

    def call(self, x):
        sinPosEncodings = self.getSinusoidalPositionalEncodings(x)
        x = x + sinPosEncodings
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])

class MultiHeadAttention(Layer):
    """
    My implementation of a Multi-Headed SelfAttention layer.

    This class is of type keras.engine.topology.Layer, so it can be added into any keras model easily. You can read more about self attention layers here ().

    Parameters:
    kqLen (int): length of the key and query vectors generated from input
    valLen (int): length of the value vectors generated from the input
    numHeads (int): number of attention heads for the layer to use
    return_sequence (bool): whether to return all timesteps or just the last one, note time steps are not masked so if return_sequence is True, earlier time steps will contain information about future time steps
    dropout (float): value between 0 and 1 to use for dropout in the layer. Dropout is applied to the key, query, and value matricies and also the output of the layer
    bias (bool): whether or not to train an additive bias for the key, query, and value matricies

    Returns:
    Tensor: if return_sequence = True, tensor is of shape (batch size, sequence length, valLen), else it is (batch size, valLen)
    """
    def __init__(self, kqLen, valLen, numHeads, return_sequence = False, dropout=0, bias=True, cpu=True, **kwargs):
        self.kqLen = kqLen
        self.valLen = valLen
        self.return_sequence = return_sequence
        self.numHeads = numHeads
        self.dropout = dropout
        self.bias = bias
        self.cpu = cpu
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.seqLen = input_shape[1]
        self.inputLen = input_shape[2]

        self.attentionLayers = []
        for i in range(self.numHeads):
            self.attentionLayers.append(SelfAttention(kqLen=self.kqLen, valLen=self.valLen, return_sequence=True, dropout=self.dropout, bias=self.bias, cpu=self.cpu))


        self.resizeWeight=self.add_weight(name="resize_weight",shape=(self.numHeads * self.valLen, self.valLen),initializer="normal")
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, x):
        outputs = self.attentionLayers[0](x)

        for layerIndex in range(1,len(self.attentionLayers)):
            outputs = tf.concat([outputs, self.attentionLayers[layerIndex](x)], 2)
        output = tf.matmul(outputs, self.resizeWeight)

        if not self.return_sequence:
            output = output[:,-1,:]
        return output

    def compute_output_shape(self, input_shape):
        if self.return_sequence:
            return (input_shape[0], input_shape[1], self.valLen)
        else:
            return (input_shape[0], self.valLen)