import numpy as np
import tensorflow as tf
import pandas as pd
import math

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

from constants import *
from encoding_decoding_lookup import *


# Reading from file
def read_sentences_from_file(file1_path, file2_path):
    sentences_with_diacritics = []
    sentences_without_diacritics = []

    with open(file1_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentences_with_diacritics.append(line)

    with open(file2_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentences_without_diacritics.append(line)

    return sentences_with_diacritics, sentences_without_diacritics

# getting diacritics for a sentence
def getDiacriticsForSentence(listOfSentencesWithDiacritics):
    #now that we have two separated lists we need to get the diacritics list
    short_vowels_list=list()
    double_case_endings_list=list()
    shadda_list=list()
    sukoon_list=list()
    counter=0
    # letters_counter=0
    for word in listOfSentencesWithDiacritics:
        while counter<len(word):
            if word[counter] in LIST_OF_ARABIC_LETTERS: #checking if the character is a letter
                # letters_counter+=1
                if (counter+1)<len(word):
                    #checking if the next character is also a letter, then that means that the diacritics of the current letter is none so add empty string to the list
                    if word[counter +1] in LIST_OF_ARABIC_LETTERS:
                        short_vowels_list.append("")
                        double_case_endings_list.append("")
                        shadda_list.append("")
                        sukoon_list.append("")
                        counter+=2
                        # letters_counter+=1
                        continue
                counter+=1 #if it is the end of the word (no more letters) or the next character is a diacritics -> continue looping
                continue
            else:
                if word[counter] in SHORT_VOWELS:
                    short_vowels_list.append(word[counter])
                    double_case_endings_list.append("")
                    shadda_list.append("")
                    sukoon_list.append("")
                elif word[counter] in DOUBLE_CASE_ENDINGS:
                    double_case_endings_list.append(word[counter])
                    short_vowels_list.append("")
                    shadda_list.append("")
                    sukoon_list.append("")
                elif word[counter] == SHADDA:
                    shadda_list.append(word[counter])
                    short_vowels_list.append("")
                    double_case_endings_list.append("")
                    sukoon_list.append("")
                else:
                    sukoon_list.append(word[counter])
                    short_vowels_list.append("")
                    double_case_endings_list.append("")
                    shadda_list.append("")
                counter+=1
        counter=0
    # print(letters_counter)
    return short_vowels_list,double_case_endings_list,shadda_list,sukoon_list

# getting diacritics for dataset
def getDiacriticsForDataSet(sentences_with_diacritics):
    short_vowels_list=list()
    double_case_endings_list=list()
    shadda_list=list()
    sukoon_list=list()
    for sentence in sentences_with_diacritics:
        list_to_be_sent= sentence.split(" ")
        if ("\n") in list_to_be_sent:
            list_to_be_sent.remove("\n")
        sv_list,dce_list,sh_list,su_list= getDiacriticsForSentence(list_to_be_sent)
        short_vowels_list.append(sv_list)
        double_case_endings_list.append(dce_list)
        shadda_list.append(sh_list)
        sukoon_list.append(su_list)
    return short_vowels_list,double_case_endings_list,shadda_list,sukoon_list

# Encode input sentence
def encode_input_sentence(sentence):
    encoded_sentence=[ENCODE_LETTERS_LOOKUP.lookup(tf.constant(char)).numpy() for char in sentence]
    return encoded_sentence

# Encode all input sentences
def encode_input_sentences(sentences):
        # loop on each sentence and encode it
        encoded_input_sentences=[]
        for sentence in sentences:
              encoded_input_sentences.append(encode_input_sentence(sentence))
        return encoded_input_sentences

# Encode short vowels
def encode_output_sentence_for_short_vowels(sentence):
    encoded_sentence=[ENCODE_SHORT_VOWELS_LOOKUP.lookup(tf.constant(char)).numpy() for char in sentence]
    return encoded_sentence

# Encode double case endings
def encode_output_sentence_for_double_case_endings(sentence):
    encoded_sentence=[ENCODE_DOUBLE_CASE_ENDINGS_LOOKUP.lookup(tf.constant(char)).numpy() for char in sentence]
    return encoded_sentence

# Encode all short vowels
def encode_output_sentences_for_short_vowels(sentences):
    encoded_output_sentences=[]
    for sentence in sentences:
              encoded_output_sentences.append(encode_output_sentence_for_short_vowels(sentence))
    return encoded_output_sentences

# Encode all double case endings
def encode_output_sentences_for_double_case_endings(sentences):
    encoded_output_sentences=[]
    for sentence in sentences:
              encoded_output_sentences.append(encode_output_sentence_for_double_case_endings(sentence))
    return encoded_output_sentences

# Encode Shadda
def encodeShaddaList(shadda_list):
    encoded_shadda_list=list()
    for sentence in shadda_list:
        list_shadda_encoded=[1 if char==SHADDA else 0 for char in sentence]
        encoded_shadda_list.append(list_shadda_encoded)
    return encoded_shadda_list

# Encode Sukuun
def encodeSukoonList(sukoon_list):
    encoded_sukoon_list=list()
    for sentence in sukoon_list:
        list_sukoon_encoded=[1 if char==SUKOON else 0 for char in sentence]
        encoded_sukoon_list.append(list_sukoon_encoded)
    return encoded_sukoon_list

#Read from csv files
def readCSVtoListofLists(filepath):
    major_list = []
    df=pd.read_csv(filepath)
    df=df.fillna(0)
    # Iterate through DataFrame rows and create a list from each row
    for index, row in df.iterrows():
        row_list = row.values.tolist()  # Convert row to a list
        row_list.pop(0)
        # row_list = [x if x != np.nan else 0 for x in row_list]
        row_list = [int(num) for num in row_list]
        major_list.append(row_list)  # Append the list to the major list
    return major_list


# padding

# concat

# window sliding

# train

# evaluate


