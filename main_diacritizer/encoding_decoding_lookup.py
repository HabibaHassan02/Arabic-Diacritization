import re
import tensorflow as tf
from pathlib import Path
import pyarabic.araby as araby
from constants import *


#from documentation:
''' 
tf.lookup.StaticHashTable(
    initializer, default_value, name=None, experimental_is_anonymous=False
)

tf.lookup.KeyValueTensorInitializer(
    keys, values, key_dtype=None, value_dtype=None, name=None
)

Example:
=======
keys_tensor = tf.constant(['a', 'b', 'c'])
vals_tensor = tf.constant([7, 8, 9])
input_tensor = tf.constant(['a', 'f'])
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
    default_value=-1)
table.lookup(input_tensor).numpy()

'''


# 1. chars table  - Encoding & Decoding:

'''
Purpose: It's used to encode characters from the CHARS list to their corresponding indices.
Initialization: It's initialized with a key-value pair where keys are characters from CHARS, and values are their corresponding indices starting from 1.
Default Value: If a key is not found, it returns 0
'''
ENCODE_LETTERS_LOOKUP = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(SORTED_VALID_INPUT_LETTERS), tf.range(1, len(SORTED_VALID_INPUT_LETTERS)+1)), default_value= 0
)
# Note that we should check using regex on the 0-9 digits not only 0
# decode the same way, and default is a (*) which is similar to symbol in paper
DECODE_LETTERS_TABLE = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.range(1, len(SORTED_VALID_INPUT_LETTERS)+1), tf.constant(SORTED_VALID_INPUT_LETTERS)), '*'
)


# 2. short vowels table  - Encoding & Decoding:
'''
Purpose: It's used to encode short vowels and their combinations with 'SHADDA'.
Initialization: It's initialized with a key-value pair where keys are short vowels and their combinations with 'SHADDA', and values are indices starting from 1. The indices are repeated twice.
Default Value: If a key is not found, it returns 0. 
'''
# note en fl paper, kan feh just the 3 main short vowels bs note in momken yb2a m3aha shada, fa we need to concate
# tile: [1, 2, 3, 1, 2, 3] where the range is from 1 to length of short vowels (3)+1 for none, then repeat twice [2]
ENCODE_SHORT_VOWELS_LOOKUP = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(SHORT_VOWELS + [SHADDA + short_vowel for short_vowel in SHORT_VOWELS]),  
                                            tf.tile(tf.range(1, len(SHORT_VOWELS) + 1), [2])),
                                            default_value= 0
)
#for decoding its the same but we get the short vowel withoyt shadda
DECODE_SHORT_VOWELS_LOOKUP = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.range(1, 4), 
                                        tf.constant(SHORT_VOWELS)), 
                                        default_value=''
)

# 3. Double Case Endings table  - Encoding & Decoding:
''' 
Purpose: It's used to encode double case endings and their combinations with 'SHADDA'.
Initialization: It's initialized similar to ENCODE_PRIMARY_TABLE.
Default Value: If a key is not found, it returns 0.
'''
ENCODE_DOUBLE_CASE_ENDINGS_LOOKUP = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(DOUBLE_CASE_ENDINGS + [SHADDA + i for i in DOUBLE_CASE_ENDINGS]),  
                                            tf.tile(tf.range(1, len(DOUBLE_CASE_ENDINGS) + 1), [2])),
                                            default_value= 0
)
#for decoding its the same but we get the dce withoyt shadda
DECODE_DOUBLE_CASE_ENDINGS_LOOKUP = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.range(1, 4), 
                                        tf.constant(DOUBLE_CASE_ENDINGS)), 
                                        default_value=''
)

#Note that we can use one binary table for the encoding as "none"=0 and else (sukun or shadda) 1
# Binary lookup for encoding both shadda and sukun:
ENCODE_BINARY_LOOKUP = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant(['']),
                                        tf.constant([0])), 
                                        default_value= 1
)

# 4. Shadda  table  - Decoding:
DECODE_SHADDA_LOOKUP = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant([0]), 
                                        tf.constant([''])), 
                                        default_value= SHADDA
)

# 5. sukuun table  -  Decoding:
DECODE_SUKOON_LOOKUP = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant([0]), 
                                        tf.constant([''])), 
                                        default_value= SUKOON
)





######################################################################################################

# #look up tables 
# CHAR_MAPPING_ENCODING = {
#     '’ ’': 1,
#     '0-9': 2,
#     araby.HAMZA: 3,
#     araby.ALEF_MADDA: 4,
#     araby.ALEF_HAMZA_ABOVE: 5,
#     araby.WAW_HAMZA: 6,
#     araby.ALEF_HAMZA_BELOW: 7,
#     araby.YEH_HAMZA: 8,
#     araby.ALEF: 9,
#     araby.BEH: 10,
#     araby.TEH_MARBUTA: 11,
#     araby.TEH: 12,
#     araby.THEH: 13,
#     araby.JEEM: 14,
#     araby.HAH: 15,
#     araby.KHAH: 16,
#     araby.DAL: 17,
#     araby.THAL: 18,
#     araby.REH: 19,
#     araby.ZAIN: 20,
#     araby.SEEN: 21,
#     araby.SHEEN: 22,
#     araby.SAD: 23,
#     araby.DAD: 24,
#     araby.TAH: 25,
#     araby.ZAH: 26,
#     araby.AIN: 27,
#     araby.GHAIN: 28,
#     araby.FEH: 29,
#     araby.MOON[12]: 30,
#     araby.KAF: 31,
#     araby.LAM: 32,
#     araby.MEEM: 33,
#     araby.NOON: 34,
#     araby.HEH: 35,
#     araby.WAW: 36,
#     araby.YEHLIKE[2]: 37,
#     araby.YEH: 38,
#     ' ':0
# }

# SHORT_VOWELS_MAPPING_ENCODING={
#     'None':0,
#     araby.FATHA: 1,
#     araby.DAMMA: 2,
#     araby.KASRA: 3
# }

# DOUBLE_CASE_ENDINGS_MAPPING_ENCODING={
#     'None':0,
#     araby.TANWEEN_FATHA: 1,
#     araby.TANWEEN_DAMMA: 2,
#     araby.TANWEEN_KASRA: 3
# }

# SHADDA_MAPPING_ENCODING={
#     'None':0,
#     araby.SHADDA:1
# }
# SUKUN_MAPPING_ENCODING={
#     'None':0,
#     araby.SUKOON:1
# }

# DECODING_CHAR_MAPPING = {value: key for key, value in CHAR_MAPPING_ENCODING.items()}
# DECODING_SHORT_VOWELS_MAPPING = {value: key for key, value in SHORT_VOWELS_MAPPING_ENCODING.items()}
# DECODING_DOUBLE_CASE_ENDINGS_MAPPING = {value: key for key, value in DOUBLE_CASE_ENDINGS_MAPPING_ENCODING.items()}
# DECODING_SHADDA_MAPPING = {value: key for key, value in SHADDA_MAPPING_ENCODING.items()}
# DECODING_SUKUN_MAPPING = {value: key for key, value in SUKUN_MAPPING_ENCODING.items()}
# #I think we need the hash map because the decoding is not accurate here, and the keys of chaprmapping are all read as 0
