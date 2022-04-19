
import tkinter
from tkinter import *
import nltk
from nltk.stem import WordNetLemmatizer

import numpy as np
import random

class ChatbotGUI:
    def __init__(self, model, words, classes, intents):
        # set attributes for chatbot
        self.model = model
        self.words = words
        self.classes = classes
        self.intents = intents
        self.lemmatizer = WordNetLemmatizer()

        #Creating GUI with tkinter
        self.base = Tk()
        self.base.title("Hello")
        self.base.geometry("400x500")
        self.base.resizable(width=FALSE, height=FALSE)

        #Create Chat window
        self.ChatLog = Text(self.base, bd=0, bg="white", height="8", width="50", font="Arial",)
        self.ChatLog.config(state=DISABLED)
        #Bind scrollbar to Chat window
        scrollbar = Scrollbar(self.base, command=self.ChatLog.yview, cursor="heart")
        self.ChatLog['yscrollcommand'] = scrollbar.set
        #Create Button to send message
        SendButton = Button(self.base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                            bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                            command= self.send )
        #Create the box to enter message
        self.EntryBox = Text(self.base, bd=0, bg="white",width="29", height="5", font="Arial")
        #EntryBox.bind("<Return>", send)
        #Place all components on the screen
        scrollbar.place(x=376,y=6, height=386)
        self.ChatLog.place(x=6,y=6, height=386, width=370)
        self.EntryBox.place(x=128, y=401, height=90, width=265)
        SendButton.place(x=6, y=401, height=90)



    def clean_up_sentence(self, sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

    def bow(self, sentence, show_details=True):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(self.words) 
        for s in sentence_words:
            for i,w in enumerate(self.words):
                if w == s: 
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
        return(np.array(bag))

    def predict_class(self, sentence):
        # filter out predictions below a threshold
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(self, ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, text):
        ints = self.predict_class(text)
        res = self.getResponse(ints, self.intents)
        return res


    def send(self):
        msg = self.EntryBox.get("1.0",'end-1c').strip()
        self.EntryBox.delete("0.0",END)
        if msg != '':
            self.ChatLog.config(state=NORMAL)
            self.ChatLog.insert(END, "You: " + msg + '\n\n')
            self.ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
            res = self.chatbot_response(msg)
            self.ChatLog.insert(END, "Bot: " + res + '\n\n')
            self.ChatLog.config(state=DISABLED)
            self.ChatLog.yview(END)

    def loop_gui(self):
        self.base.mainloop()

