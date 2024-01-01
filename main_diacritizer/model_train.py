import numpy as np
import tensorflow as tf
import pandas as pd
import math

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

from constants import *
from encoding_decoding_lookup import *



#should add function train and evaluate

class Diacritizer(Model):
    def __init__(self, embedding_size=DEFAULT_EMBEDDING_SIZE, lstm_size=DEFAULT_LSTM_SIZE, window_size=DEFAULT_WINDOW_SIZE,dropout_rate=DEFAULT_DROPOUT_RATE):
        super(Diacritizer, self).__init__()

        # in the initialization, we define: input, embeddings, and output
        # in lab 5 we had a linear where number of neurons= number of classes but here we have dense

        ######################################## Creating the layers of our model #####################

        # Step 1: define the input
        # input would be the fixed window size after lookup 
        # Input in tf defines the input layer with shape=(window_size,)
        # it represents the input data.
        self.inputs = Input(shape=(window_size,), name='input')


        # Step 2: define the embeddings
        # it converts the integer index to dense vectors with fixed size
        # input dim is dim of the letters (length of valid input letters we have)
        # the output is the dense vector (embeddings size which is 128)
        # the ( ) at the end mean that this layer would be applied to the input layer
        self.embedding = Embedding(input_dim= len(SORTED_VALID_INPUT_LETTERS) + 1, output_dim= embedding_size, name='embedding')(self.inputs)
        

        # Step 3: Define the Bidirectional LSTM layers (we have 4 layers of each class and an initial layer)
        
        # initial layer that would be applied to the embeddings
        self.initial_layer = Bidirectional(LSTM(lstm_size, dropout=dropout_rate, return_sequences=True),
                                      name='initial_layer')(self.embedding)
        
        # first layer is sukun layer that would be applied on initial layer
        self.sukoon_layer = Bidirectional(LSTM(lstm_size, dropout=dropout_rate, return_sequences=True),
                                     name='sukoon_layer')(self.initial_layer)
        
        # sec layer is shadda layer that would be applied to sukun layer
        self.shadda_layer = Bidirectional(LSTM(lstm_size, dropout=dropout_rate, return_sequences=True),
                                     name='shadda_layer')(self.sukoon_layer)
        
        # third layer is sec layer that would be applied to shadda layer
        self.secondary_diacritics_layer = Bidirectional(LSTM(lstm_size, dropout=dropout_rate, return_sequences=True),
                                                   name='secondary_diacritics_layer')(self.shadda_layer)
        
        # forth layer is primary layer that would be applied to sec layer
        self.primary_diacritics_layer = Bidirectional(LSTM(lstm_size, dropout=dropout_rate, return_sequences=True),
                                                 name='primary_diacritics_layer')(self.secondary_diacritics_layer)

        # Step 4: Define the output layers (we have 4 outputs with diff dense)
        # dense functions define the activation functions in tensor flow
        # where dense=1 means sigmoid and dense =2 means softmax

        # Sukun output would have a sigmoid as its binary and would come from the sukun layer
        # dense= 1 means single neuorn we just wanna know if 0 no sukun or 1 yes sukun
        # SIGMOID
        self.sukoon_output = Dense(1, activation='sigmoid',name='sukoon_output')(self.sukoon_layer)

        # shadda output
        # SIGMOID
        self.shadda_output = Dense(1,activation='sigmoid',   name='shadda_output')(self.shadda_layer)

        # sec output
        # dense= 4 as we have multi classification task, where each neuron assigned to one class
        # we have 0,1,2,3 as per the lookup table
        # SOFTMAX
        self.secondary_diacritics_output = Dense(4, activation='softmax' , name='secondary_diacritics_output')(self.secondary_diacritics_layer)

        # prim output
        # SOFTMAX
        self.primary_diacritics_output = Dense(4,  activation='softmax' ,name='primary_diacritics_output')(self.primary_diacritics_layer)
       

    def forward(self, inputs):
        # Step 4: Define the forward pass through the layers
        initial_layer = self.initial_layer(inputs)
        sukoon_layer = self.sukoon_layer(initial_layer)
        shadda_layer = self.shadda_layer(sukoon_layer)
        secondary_diacritics_layer = self.secondary_diacritics_layer(shadda_layer)
        primary_diacritics_layer = self.primary_diacritics_layer(secondary_diacritics_layer)


        # Step 5: Separate outputs for each diacritic
        sukoon_output = self.sukoon_output(sukoon_layer)
        shadda_output = self.shadda_output(shadda_layer)
        secondary_diacritics_output = self.secondary_diacritics_output(secondary_diacritics_layer)
        primary_diacritics_output = self.primary_diacritics_output(primary_diacritics_layer)
       

        return primary_diacritics_output, secondary_diacritics_output, shadda_output, sukoon_output
















# we should have the following functionalities in our class:
# - preprocessing
# - getting list of sentences undiacritized
# - getting list of sentences diacritized
# - prepare for train
# - encoding and decoding  (from the lookup tables)
# - Sliding process (using the sliding window)   - this is for handling variable length input and the overlapping helps to capture contextual meaning
# - train (only used once and then save model for later use)
# - predict
# - load model
# - evaluate

# and any other needed fns along the way

# Steps:
''' 
1. data cleaning 
2. tokenization (sentences with/out diacritics)
3. convert input by the lookup table and concat for all sentences
4. use a sliding window which will be the input to the model  [fixed size]
5. embedding size should be 128   
   #Embedding layers convert the integer-encoded tokens into dense vectors of fixed size (128).
6. use 5 layer lstm with activation tanh to each layer
   while the output of the first 2 layers are sigmoid and the other 2 is softmax
7. convert output by lookup table and concat for all sentences
8. compare golden converted output with output from model and calc loss function
9. get final result of model (save model)
10. load model
11. predict and evaluate model
'''