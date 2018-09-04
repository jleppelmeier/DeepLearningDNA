#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnn_classify_crossval.py:
    Cross-validates classifier performance on the UCI Splice dataset using
    a 1D CNN. Requires the splice dataset:

https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/

    A keras implementation of the paper, "DNA Sequence Classification
    by Convolutional Neural Network" by Nguyen, et al. (2016)
    UCI's Splice is 1 of 12 datasets tested in the original paper.

    Target variable is categorical: EI, IE, or N (exon-intron junction,
    intron-exon junction, or neither). Predictors are Primate DNA sequences
    of 60 nucleotides, 30 preceding the junction, and 30 after the junction.

    Performs 10-fold cross-validation of the architecture in the script cnn_classify.py


JLeppelmeier 3-Sep-2018
"""

import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.optimizers import Adam


print("Loading Dataset...")

dataset = pd.read_csv('splice.data', header=None)

#Prepare the labels:
labels = dataset.iloc[:,0]
le = LabelEncoder()
y = le.fit_transform(labels)


print("Processing Sequences...")

#Process nucleotide sequences:
sequences = dataset.iloc[:,2]
sequences = sequences.str.strip() #remove whitespace

# pass a window (size=3) over each sequence
# and collect the resulting "words"
windowed_seqs = []
for x in sequences:
    windowed_seqs.append([x[i-1]+x[i]+x[i+1] for i in range(1,len(x)-1)])

# join each row's windows with spaces
# so we can process later with text Tokenizers:
X = [" ".join(row) for row in windowed_seqs]

#Tokenize the nucleotide sequence features:
tkn = Tokenizer(oov_token=None)
tkn.fit_on_texts(X)
X = np.array(tkn.texts_to_sequences(X))

# one-hot encode the labels:
y_enc = np_utils.to_categorical(y)

#vocab_size = 150 #...better way to get this?

print("Generating Models...")

val_accuracies = []

cv_folds=10
skf = StratifiedKFold(n_splits=cv_folds)
fold_count = 0

for trn_ix,val_ix in skf.split(X,y):

    #init model:
    model = Sequential()

    #define CNN layers/architecture:
    model.add(Embedding(150, 200, input_length=58))
    model.add(Conv1D(200,3,strides=1,activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(4))
    model.add(Conv1D(200,3,strides=1,activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(6))
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation = 'softmax'))

    adam = Adam(lr=0.0005)
    model.compile(optimizer='adam', loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

    #train model:
    model.fit(X[trn_ix],y_enc[trn_ix],batch_size=128, epochs=10,verbose=0)

    #collect predictions:
    y_pred = model.predict(X[val_ix])
    y_pred = np.argmax(y_pred, axis=1)

    val_acc = sum([x1==x2 for x1,x2 in zip(y[val_ix],y_pred)])/len(y_pred)
    val_f1 = f1_score(y[val_ix],y_pred,average='weighted')

    val_accuracies.append(val_acc)

    fold_count += 1
    print("Fold {0} - Validation Accuracy: {1:.4f}".format(fold_count, val_acc))

output = np.array(val_accuracies)
print("\nmean accuracy (10 folds): {0:.4f}".format(np.mean(output)))
np.savetxt('cnn_classify_crossval_output.csv',output)
