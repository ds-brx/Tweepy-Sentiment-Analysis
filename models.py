import pandas as pd
import numpy as np

import xgboost
from xgboost import XGBClassifier


from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


## SVM MODEL ##
def model_svm(X_train,y_train, X_test, y_test):
    model = svm.LinearSVC()
    model.fit(X_train, y_train)
    predicted= model.predict(X_test)
    score=accuracy_score(y_test,predicted)
    cm = confusion_matrix(y_test,predicted)
    return cm,score

## XGBOOST MODEL ##
def model_xgbClassifier(X_train,y_train,X_test,y_test):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test,y_pred)
    return cm,score

## NAIVE BAYES ##
def nbClassifier(X_train,y_train,X_test,y_test):
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    score = accuracy_score(y_test,y_pred)
    return cm,score

## LSTM ##
def model_LSTM(num_words,emd_dim,input_length,lstm_dim):
    model = Sequential()
    model.add(Embedding(num_words,emd_dim,input_length=input_length))
    model.add(LSTM(lstm_dim))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['acc'])
    return model.summary()

def prediction_lstm(model,X_test,y_test):
    predictions = model.predict(X_test)
    predictions = np.round(predictions)

    score = accuracy_score(y_test,predictions)
    cm = confusion_matrix(y_test,predictions)
    return cm,score

