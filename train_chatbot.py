import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as numpy
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import gradient_descent_v2 
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add the documents in the corpus
        documents.append((w, intent['tag']))
        print(documents)

        #keep track of the tags in classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
