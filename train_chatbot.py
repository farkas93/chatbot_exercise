import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import gradient_descent_v2 
import random

words_file = "words.pkl"
classes_file = "classes.pkl"
documents_file = "documents.pkl"

ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

def preprocess_intents():
    words = []
    classes = []
    documents = []

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            #tokenize each word
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            #add the documents in the corpus
            documents.append((w, intent['tag']))

            #keep track of the tags in classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])



    # lemmatize, lower each word and remove duplicates
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    # sort alphabethically and remove duplicates
    classes = sorted(list(set(classes)))

    print(len(documents), "documents")

    print(documents)

    print(len(classes), "classes", classes)

    print(len(words), "unique lemmatized words", words)

    pickle.dump(words,open(words_file,'wb'))
    pickle.dump(classes,open(classes_file,'wb'))
    pickle.dump(documents, open(documents_file,'wb'))


def read_words():
    words = []
    with (open(words_file, "rb")) as openfile:        
        words = pickle.load(openfile)
    return words

def read_classes():    
    classes = []
    with (open(classes_file, "rb")) as openfile:
        classes = pickle.load(openfile)
    return classes

    
def read_documents():    
    documents = []
    with (open(documents_file, "rb")) as openfile:
        documents = pickle.load(openfile)
    return documents


def create_train_data():
    documents = read_documents()
    words = read_words()
    classes = read_classes()
    
    
    print(len(classes), "classes", classes)

    print(len(words), "unique lemmatized words", words)

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        # init bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # lemmatize each word - create base word in attempt to represent related words
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        # create bag of words array with 1 if word match was found in current pattern
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    #shuffle the features and turn into a numpy array
    random.shuffle(training)
    training = np.array(training)
    # create train and test lists. X - patterns, Y - intents

    train_x = list(training[:,0])
    train_y = list(training[:,1])
    print("train data was created")
    return train_x, train_y

def get_model(input_size, output_size):
    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
    model = Sequential()
    model.add(Dense(128, input_shape=(input_size,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    optimizer = gradient_descent_v2.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def main():
    preprocess_intents()
    train_x, train_y = create_train_data()
    print(train_x[0])
    model = get_model(len(train_x[0]), len(train_y[0]))

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save('mychatbot_model.h5')

if __name__ == "__main__":
    main()