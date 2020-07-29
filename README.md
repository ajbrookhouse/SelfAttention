# SelfAttention
This repository houses my implementation of a Self-Attention layer in Keras to be used for multivariate time series machine learning problems

This repository has two files, Demonstration.py and Layers.py. 

Layers.py contains the code and neccisary imports for my implementations of a SelfAttention layer and a AddSinusoidalPositionalEncodings layer. Demonstration.py has code to create a very simple classification model using these two layers, generates a random batch of data, and calls model.predict() on it. Looking at this example in Demonstration.py should make it easy to include these layers in a Keras model used to classify real data. For more information on the parameters in the constructors of these layers, look at the docstrings in Layers.py

The SelfAttention and AddSinusoidalPositionalEncodings are Keras Layers, so they can be easily imported into Keras machine learning models. 

Currently, the self attention requires all inputs to have the same sequence length. I will work on generalizing this soon so the model can process batches that have different sequence lengths even within the same batch. Also, I am using BatchNormalization at the end of the layer. I want to change this to LayerNormalization.

I am working on a Multiheaded attention class as well. I will upload it when it is finished.

For more information on the details on how self attention works, I reccomend this great article (https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a). It has great illustrations and explains the concept very well.
