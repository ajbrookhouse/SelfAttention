import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} | 3 prevents tensorflow from printing a bunch on startup

from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam

from Layers import SelfAttention
from Layers import AddSinusoidalPositionalEncodings
from Layers import MultiHeadAttention


def getModel(seqLen, inputLen, outputDim, learning_rate=.00005):
    """
    A very simple model that can be used to test the attention layer

    The model adds positional encodings to input, runs that through a SelfAttention layer and then runs that output for the last timestep through a single Dense layer for classification.

    Parameters:
    seqLen (int): length of sequences to be fed through model
    inputLen (int): length of the input vectors
    outputDim (bool): how many classes should the model output
    learning_rate (float): value to be used as Adam learning rate

    Returns:
    Keras Model: Can be fed tensors of size (batch_size which can vary, seqLen, inputLen)
    """

    input_ = Input(shape=(seqLen,inputLen))
    pos = AddSinusoidalPositionalEncodings()
    att = SelfAttention(120,120,return_sequence=False, dropout=.2)
    #att = MultiHeadAttention(120, 120, 8, return_sequence=False, dropout=.2) #uncomment to use multihead attention with 8 heads
    l = Dense(outputDim, activation='softmax')

    x = input_
    x = pos(x)
    x = att(x)
    x = l(x)

    model = Model(inputs=input_, outputs=x)
    opt = Adam(lr = learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


seqLen = 100
inputLen = 200
outputDim = 5
batchSize = 20

model = getModel(seqLen, inputLen, outputDim)
#This model could be used on real data to learn with model.fit or predict with model.predict
#For demonstration purposes, random data will be used to show the shape of the model's output



exampleTensor = np.random.random((batchSize, seqLen, inputLen))
output = model.predict(exampleTensor)
print('Output vector shape:',output.shape)