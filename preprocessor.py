import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob, Word
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from sklearn.feature_extraction.text import CountVectorizer
import spacy
nlp = spacy.load('en_core_web_sm')
stopwords_en = set(stopwords.words('english'))


## STEMMING AND CLEANING TEXT ##
def stem_clean_text(data):
    cleaned_text = []
    for sent in data['tweet']:
        sent = sent.lower()
        sent = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', 
                    '', sent, flags=re.MULTILINE)
        sent = re.sub('[^a-zA-Z]', ' ', sent)
        sent = re.sub(r"won\'t", "will not", sent)
        sent = re.sub(r"can\'t", "can not", sent)
        sent = re.sub(r"n\'t", " not", sent)
        sent = re.sub(r"\'re", " are", sent)
        sent = re.sub(r"\'s", " is", sent)
        sent = re.sub(r"\'d", " would", sent)
        sent = re.sub(r"\'ll", " will", sent)
        sent = re.sub(r"\'t", " not", sent)
        sent = re.sub(r"\'ve", " have", sent)
        sent = re.sub(r"\'m", " am", sent)
        ps = PorterStemmer()
        sent = " ".join([ps.stem(word) for word in sent.split() if word not in stopwords_en and not word.isdigit()])
        cleaned_text.append(sent)
    return cleaned_text

## LEMMATIZING AND CLEANING TEXT ##

def lemmatize_clean_text(data):
    cleaned_text = []
    for i in data['tweet']:
        i = i.lower()
        i = TextBlob(i)
        i = " ".join([word.lemmatize() for word in i.words if word not in stopwords_en and not word.isdigit()])
        i = re.sub('[^a-zA-Z]', ' ', i)
        i = re.sub(r"won\'t", "will not", i)
        i = re.sub(r"can\'t", "can not", i)
        i = re.sub(r"n\'t", " not", i)
        i = re.sub(r"\'re", " are", i)
        i = re.sub(r"\'s", " is", i)
        i = re.sub(r"\'d", " would", i)
        i = re.sub(r"\'ll", " will", i)
        i = re.sub(r"\'t", " not", i)
        i = re.sub(r"\'ve", " have", i)
        i = re.sub(r"\'m", " am", i)
        i = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', 
                        '', i, flags=re.MULTILINE)
        cleaned_text.append(i)
    return cleaned_text

## SENTIMENT POLARITY, SUBJECTIVITY ##

def sentiment_polarity(data,clean_col):
    subjectivity = []
    polarity = []
    sense = []
    for tweet in data[clean_col]:
        subjectivity.append(TextBlob(tweet).sentiment.subjectivity)
        pol = TextBlob(tweet).sentiment.polarity
        polarity.append(pol)
        if pol>0:
            sense.append(1)
        elif pol<0:
            sense.append(-1)
        else:
            sense.append(0)
    return subjectivity, polarity, sense

## FREQUENCY DISTRIBUTION OF TOKENS ##

def freqDist(clean_col):
    wordlist = nltk.FreqDist(clean_col)
    features = wordlist.keys()
    return wordlist, features

## COUNT VECTORIZER MATRIX OF DATA ##

def Count_Vectorize(X_train,X_test):
    cv = CountVectorizer(max_features=1500)
    cv.fit(X_train)
    xtrain_cv_matrix = cv.transform(X_train)
    xtest_cv_matrix = cv.transform(X_test)
    return xtrain_cv_matrix, xtest_cv_matrix

## TFIDF NGRAMS LEVEL OF DATA ##

def tfidfMatrix(X_train,X_test):
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=6000)
    tfidf_vect_ngram.fit(X_train)
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)
    return xtrain_tfidf_ngram, xvalid_tfidf_ngram

## TEXT TO SEQUENCE OF VECTORS ##

def tokenized_seq_vectors(X_train,X_test):
    tokenizer = text.Tokenizer(num_words = 1000)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    maxlength = len(max(X_train, key = len))
    X_train = sequence.pad_sequences(X_train,maxlen=maxlength)
    X_test = sequence.pad_sequences(X_test,maxlen=maxlength)
    index = tokenizer.word_index
    return X_train, X_test, index


## ENTITY TAGGER ##
def tagger(data,col):
    tags = []
    for t in data[col]:
        e = []
        doc = nlp(t)
        for token in doc:
            e.append((token.text, token.tag_))
        tags.append(e)
    return tags

## ENTITY RECOGNIZER ##

def get_ner(data,col):
    ent_labels = []
    for t in data[col]:
        e = []
        doc = nlp(t)
        for ent in doc.ents:
            e.append((ent.text,ent.label_))
        ent_labels.append(e)
    return ent_labels
