# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:31:07 2020

@author: Shivam & Shubham
"""

import nltk
import json


df_file = open("intents.json").read()
intents = json.loads(df_file)

words=[]
classes = []
documents = []


for i in intents['intents']:
    for j in i['patterns']:
        w = nltk.word_tokenize(j)
        #w = nltk.sent_tokenize(j)
        words.extend(w)
        documents.append((w, i['tag']))
        if i['tag'] not in classes:
            classes.append(i['tag'])


#print(words)
#print(classes)
#print(document)


from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()


ignore_words = ['?', '!']
words = [lem.lemmatize(w.lower()) for w in words if w not in ignore_words]

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))


#output11111_empty = [0] * len(classes)
#output11111_row = list(output11111_empty)
#len(classes)




# create our training data
training = []
output_empty = [0,0,0,0,0,0,0,0,0]

#bag of words for each sentence
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lem.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
        
        
import random     
import numpy as np
   

training = np.array(training)
# create train and test lists.
train_x = list(training[:,0])
train_y = list(training[:,1])


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
len(train_x[0])

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)


# we will have to convert the input text also i.e lemmetize the input text after lowering it doem
def lem_input_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lem.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array for the input text just like we did it for training data input.
def bow(sentence, words):
    sentence_words = lem_input_sentence(sentence)
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(np.array(bag))
    
    
    
#predict the class after converting it into a list which is input for model
def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]]})
    return return_list


# Now we have defined our classes for the input text, map the class with the json file
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

#Finally when we have class and random response ready, we will return the response
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res




a = (" Hey hi",
     " what you can do",
     " blood level",
     " drug",
     " thanks ",
     " OK bye see you")

for i in range(len(a)):
    print(chatbot_response(a[i]))
    
      

#try asking questions
    
z = (" ",
     " ",
     " ",
     " ",
     " ",
     " ")
    

for i in range(len(z)):
    print(chatbot_response(z[i]))
    
       

#
#
#
#import tkinter as tk
#from tkinter import simpledialog
#
#ROOT = tk.Tk()
#
#ROOT.withdraw()
## the input dialog
#USER_INP = simpledialog.askstring(title="Test",
#                                  prompt="Hello:")
#
## check it out
#print(chatbot_response(USER_INP))
