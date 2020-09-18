# SelfAttention
This repository houses my implementation of a Self-Attention layer in Keras to be used for multivariate time series machine learning problems. While it is commonly used for language processing and image processing, I think Self-Attention has great potential to being used in multivariate time-series classification problems. Since inputs can be processed in paralell, a self attention model can be quicker than a LSTM model, and also benifit further from paralellization. The implementations in this repository were developed during my time working at Washington State's Smart Environments summer research program. As the project continues to develop, I will upload improvements to the Layers and upload graphics analyzing their performance.



This repository has two files, Demonstration.py and Layers.py. 

Layers.py contains the code and neccisary imports for my implementations of a SelfAttention layer, a AddSinusoidalPositionalEncodings layer, and a MultiHeadedAttention layer. Demonstration.py has code to create a very simple classification model using these two layers, generates a random batch of data, and calls model.predict() on it. Looking at this example in Demonstration.py should make it easy to include these layers in a Keras model used to classify real data. For more information on the parameters in the constructors of these layers, look at the docstrings in Layers.py

The SelfAttention and AddSinusoidalPositionalEncodings are Keras Layers, so they can be easily imported into Keras machine learning models. 

Currently, the self attention requires all inputs to have the same sequence length. I will work on generalizing this soon so the model can process batches that have different sequence lengths even within the same batch. Also, I am using BatchNormalization at the end of the layer. I want to change this to LayerNormalization eventually, but am not sure how to do it yet without directly importing tensorflow functions. I am trying to keep these layers just in terms of Keras api so that they are backend agnostic.

For more information on the details on how self attention works, I reccomend this great article (https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a). It has great illustrations and explains the concept very well.

The github repo for this code can be found at https://github.com/ajbrookhouse/SelfAttention
