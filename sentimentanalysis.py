#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import warnings
import csv


# In[2]:


warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


# In[3]:


print(os.listdir("input"))


# In[4]:


# The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and 
# statistical natural language processing for English written in the Python programming language.
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from bs4 import BeautifulSoup
import re
#TQDM is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.
from tqdm import tqdm


# In[5]:


from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import backend
import random
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential


# In[6]:


train= pd.read_csv("input/train.tsv", sep="\t")
review="the movie was superb,fabulous"
#write review string to to be acted as test string to test tsv file
with open('input/test.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['PhraseId', 'SentenceId','Phrase'])
    tsv_writer.writerow(['1', '2',review])

test = pd.read_csv("input/test.tsv", sep="\t")

train.head()


# In[7]:


train.shape


# In[8]:


print(test)
print(type(test))


# In[9]:


#The following function will take each phrase iteratively and it will
#remove html content
#remove non-alphabetic characters
#tokenize the sentences
#lemmatize each word to its lemma
#and then return the result in the list named reviews



def clean_sentences(df):
    reviews = []

    for sent in tqdm(df['Phrase']):
        
        #remove html content
        review_text = BeautifulSoup(sent).get_text()
        
        #remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
    
        #tokenize the sentences
        words = word_tokenize(review_text.lower())
    
        #lemmatize each word to its lemma
        lemma_words = [lemmatizer.lemmatize(i) for i in words]
    
        reviews.append(lemma_words)
        

    return(reviews)


# In[ ]:





# In[10]:


#cleaned reviews for both train and test set retrieved
train_sentences = clean_sentences(train)
test_sentences = clean_sentences(test)
print(len(train_sentences))
print(len(test_sentences))


# In[11]:


test_sentences


# In[12]:


target=train.Sentiment.values
y_target=to_categorical(target)
num_classes=y_target.shape[1]


# In[13]:


target


# In[ ]:





# In[14]:


y_target


# In[15]:


num_classes


# In[16]:


X_train,X_val,y_train,y_val=train_test_split(train_sentences,y_target,test_size=0.2,stratify=y_target)


# In[17]:


#Geting the no of unique words and max length of a review available in the list of cleaned reviews.
#It is needed for initializing tokenizer of keras and subsequent padding

unique_words = set()
len_max = 0

for sent in tqdm(X_train):
    
    unique_words.update(sent)
    
    if(len_max<len(sent)):
        len_max = len(sent)
        
#length of the list of unique_words gives the no of unique words
print(len(list(unique_words)))
print(len_max)


# In[18]:


#Actual tokenizer of keras and convert to sequences

tokenizer = Tokenizer(num_words=len(list(unique_words)))
tokenizer.fit_on_texts(list(X_train))

# Arguments- texts: list of texts to turn to sequences.
# Return: list of sequences (one per text input).
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(test_sentences)


# In[19]:


X_test


# In[20]:


#padding done to equalize the lengths of all input reviews. LSTM networks needs all inputs to be same length.
#Therefore reviews lesser than max length will be made equal using extra zeros at end. This is padding.

X_train = sequence.pad_sequences(X_train, maxlen=len_max)
X_val = sequence.pad_sequences(X_val, maxlen=len_max)
X_test = sequence.pad_sequences(X_test, maxlen=len_max)

print(X_train.shape,X_val.shape,X_test.shape)
print(X_test)


# In[21]:


#Early stopping to prevent overfitting
early_stopping = EarlyStopping(min_delta = 0.001, mode = 'max', monitor='val_acc', patience = 2)
callback = [early_stopping]


# In[22]:


#Model using Keras LSTM

#Multilayer Perceptron (MLP) for multi-class softmax classification:
#Let’s build what’s probably the most popular type of model in NLP at the moment: Long Short Term Memory network. 
#This architecture is specially designed to work on sequence data.
#It fits perfectly for many NLP tasks like tagging and text classification.
#It treats the text as a sequence rather than a bag of words or as ngrams.

#Here’s a possible model definition:

model=Sequential()
model.add(Embedding(len(list(unique_words)),300,input_length=len_max))
model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.005),metrics=['accuracy'])
model.summary()


# In[23]:


mod=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=2, batch_size=256, verbose=1, callbacks=callback)


# In[24]:


#make the predictions with trained model and submit the predictions.
print(X_test)
y_pred=model.predict_classes(X_test)


# In[25]:


sub_file = pd.read_csv('input/sampleSubmission.csv',sep=',')
sub_file.Sentiment=y_pred
sub_file.to_csv('input/sampleSubmission.csv',index=False)


# In[ ]:




