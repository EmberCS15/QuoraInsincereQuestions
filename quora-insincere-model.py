# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# coding: utf-8

# import import_ipynb
# import preprocess_questions as pq
import re
import string
import unicodedata
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import os
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import pandas as pd
# from sklearn.model_selection import train_test_split
# from keras.callbacks import ModelCheckpoint 
# from ipynb.fs.full.nlp import preprocess
# import importlib
import tensorflow as tf
config = tf.ConfigProto(device_count = {'CPU' : 1, 'GPU' : 1})
# config.gpu_options.allow_growth = True 
# config.gpu_options.visible_device_list = "6,7"
session = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(session) 
# pq = importlib.reload(pq)

################### INPUT PATH ##################################################
GLOVE_FILE_PATH = os.path.join('..','input','glove6b100dtxt','glove.6B.100d.txt')
INPUT_DATASET_DIR = os.path.join('..','input','quora-insincere-questions-classification')
FILE_PATH = 'submission.csv'
################### PREPROCESS QUESTION TEXT ####################################
STOP_WORDS = set(stopwords.words('english'))
def preprocess(text):
    """
    preprocess text into clean text for tokenization
    """
    text = normalize_unicode(text)
    text = remove_newline(text)
    text = text.lower()
    text = decontracted(text)
    text = replace_negative(text)
    text = removePunctuations(text)
    text = remove_number(text)
    text = remove_space(text)
    text = removeArticlesAndPronouns(text)
    text = removeNLTKStopWords(text)
    #text = performStemming(text)
    return text

def normalize_unicode(text):
    """
    unicode string normalization
    """
    return unicodedata.normalize('NFKD', text)


def remove_newline(text):
    """
    remove \n and  \t
    """
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('\b', ' ', text)
    text = re.sub('\r', ' ', text)
    return text


def decontracted(text):
    """
    de-contract the contraction
    """
    # specific
    text = re.sub(r"(W|w)on(\'|\’)t", "will not", text)
    text = re.sub(r"(C|c)an(\'|\’)t", "can not", text)
    text = re.sub(r"(Y|y)(\'|\’)all", "you all", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll", "you all", text)
    # general
    text = re.sub(r"(I|i)(\'|\’)m", "i am", text)
    text = re.sub(r"(A|a)in(\'|\’)t", "is not", text)
    text = re.sub(r"n(\'|\’)t", " not", text)
    text = re.sub(r"(\'|\’)re", " are", text)
    text = re.sub(r"(\'|\’)s", " is", text)
    text = re.sub(r"(\'|\’)d", " would", text)
    text = re.sub(r"(\'|\’)ll", " will", text)
    text = re.sub(r"(\'|\’)t", " not", text)
    text = re.sub(r"(\'|\’)ve", " have", text)
    return text

def remove_number(text):
    """
    numbers are not toxic
    """
    return re.sub('\d+', ' ', text)


def remove_space(text):
    """
    remove extra spaces and ending space if any
    """
    text = re.sub('\s+', ' ', text)
    text = re.sub('\s+$', '', text)
    return text

# remove articles and pronouns
def removeArticlesAndPronouns(text):
    filteredTokens = []
    for word, pos in nltk.pos_tag(nltk.word_tokenize(text)):
        # check for articles or pronoun 'I'
        if word == "a" or word == "an" or word == "the" or word == "i":
            continue
        # check for remaining pronouns
        if not (pos == 'PRP' or pos == 'PRP$' or pos == 'WP' or pos == 'WP$'):
            filteredTokens.append(word)
    return " ".join(filteredTokens)


# removes all nltk stop words
def removeNLTKStopWords(text):
    words = word_tokenize(text)
    wordsFiltered = []
    for w in words:
        if w not in STOP_WORDS:
            wordsFiltered.append(w)
    return " ".join(wordsFiltered)


def performStemming(text):
    porter_stemmer = PorterStemmer()
    nltk_tokens = nltk.word_tokenize(text)
    filteredTokens = []
    for w in nltk_tokens:
        # print ("Actual: %s Stem: %s" % (w, porter_stemmer.stem(w)))
        filteredTokens.append(str(porter_stemmer.stem(w)))
    return " ".join(filteredTokens)

def removePunctuations(text):
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    text = ' '.join(words)
    return text


def replace_negative(text):
    text = re.sub('\S+n\'\S+', 'not', text)
    return text

#################### UTILITY FUNCTION FOR MP #############################
def preprocess_question(questions_array, index):
    questions_array[index] = preprocess(questions_array[index]) 

################### INPUT DATA AND CREATE MODEL #########################
# models_dir = "./models"
# model_save_path = os.path.join(models_dir,"final-lstm-model.model") 
train_path = os.path.join(INPUT_DATASET_DIR, "train.csv")
test_path = os.path.join(INPUT_DATASET_DIR, "test.csv")
df_train = pd.read_csv(train_path, engine='python')
df_test = pd.read_csv(test_path, engine='python')
print("train data shape : ", df_train.shape)
print("test data shape : ", df_test.shape)
# Class count
count_class_0, count_class_1 = df_train.target.value_counts()
print("Class 0 count : ",count_class_0)
print("Class 1 count : ",count_class_1)
# Divide by class
df_class_0 = df_train[df_train['target'] == 0]
df_class_1 = df_train[df_train['target'] == 1]
# df_class_0_under = df_class_0.sample(count_class_1)
# df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
# print('Random under-sampling :')
# print(df_train_under.target.value_counts())
# questions_train = df_train_under['question_text'].values
questions_train = df_train['question_text'].values
questions_test = df_test['question_text'].values
print('Shape of questions_train = {}'.format(questions_train.shape))
print(questions_train[0:2])
target_train = df_train['target'].values
print('Shape of target_train = {}'.format(target_train.shape))

for i,ques in enumerate(questions_train):
    preprocess_question(questions_train, i)
    
for i,ques in enumerate(questions_test):
    preprocess_question(questions_test, i)

print('First two Questions of Preprocessed Train Set')
print(questions_train[0])
print(questions_train[1])

max_len = sum(len(x.split(' ')) for x in questions_train)/len(questions_train)
print('Average Question Length = {}'.format(max_len))

# load the whole embedding into memory
embeddings_index = dict()
glove_file = open(GLOVE_FILE_PATH)
for line in glove_file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
glove_file.close()
print('Loaded %s word vectors.' % len(embeddings_index))

questions = np.concatenate((questions_train, questions_test), axis=0)
t = Tokenizer()
max_length = 40

####################### FITTING ON ENTIRE QUESTION SET ################################
t.fit_on_texts(questions)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_ques_train = t.texts_to_sequences(questions_train)
print(encoded_ques[0])
# pad documents to a max length of 4 word
padded_ques_train = pad_sequences(encoded_ques_train, maxlen=max_length, padding='post')
print('Padded question Train = ' + padded_ques_train[0])
# (trainX, testX, trainY, testY) = train_test_split(padded_ques, target_train, test_size=0.2, random_state=42)
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


################## CREATING PADDED QUESTION FOR TEST ########################
# integer encode the documents
encoded_ques_test = t.texts_to_sequences(questions_test)
print(encoded_ques[0])
padded_ques_test = pad_sequences(encoded_ques_test, maxlen=max_length, padding='post')
print('Padded question Test = ' + padded_ques_test[0])

print('Model Definition and Trainig......')
# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# filepath=os.path.join(models_dir,"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only = False, mode = 'max', period = 10)
# callbacks_list = [checkpoint]

model.fit(padded_ques_train, target_train, epochs=100, batch_size=64, verbose = 2)
# Final evaluation of the model
# scores = model.evaluate(x_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
# model.save(model_save_path)
y_pred = model.predict(padded_ques_test, batch_size = 64, verbose = 2)
df_pred = pd.DataFrame({'qid': df_test.qid, 'prediction': y_pred})
df_pred.to_csv(FILE_PATH, index=False)
print('Save submission file to {}'.format(FILE_PATH))