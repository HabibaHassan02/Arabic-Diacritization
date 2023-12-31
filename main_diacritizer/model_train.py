import numpy as np
import tensorflow as tf

from constants import *
from encoding_decoding_lookup import *


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