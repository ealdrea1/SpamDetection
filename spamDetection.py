#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:50:05 2018

@author: esraa_aldreabi
"""

from sklearn.naive_bayes import MultinomialNB
import pandas as pd #import data
import numpy as np

data = pd.read_csv('spambase.data').as_matrix()# convert to matrix
np.random.shuffle(data)#to make the data different everytime after splitting to train and test data
# the data 1-48 col input (word frequency measure: number of times word appears in a document divided by the num of words in document then multibly that by 100)
#last column is lable (1 spam 0 not spam)
# term-document matrix terms go along colums documents(emails) rows
X = data[:, :48]#all rows first 48 col
Y = data[:, -1]#all rows last col
#already shuffled so dosen't matter which order to choose train and test
Xtrain = X[:-100,]# first 100 rows (-100 means: 0...N-100; N size of the array)
Ytrain = Y[:-100,]
Xtest = X[-100:,]#last 100 rows for test set
Ytest = Y[-100:,]

model = MultinomialNB()#object
model.fit(Xtrain, Ytrain)
print ("Classification rate for NB:")
print (model.score(Xtest,Ytest))

