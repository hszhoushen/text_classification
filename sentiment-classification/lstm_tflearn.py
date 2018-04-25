"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""
from __future__ import division, print_function, absolute_import
import pandas as pd
import numpy as np
import data_process as dp
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='./data/imdb.pkl', n_words=10000,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
# model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
# batch_size=32)
# model.save('imdb_tflearn.h5')


MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000

# depending on how large the wording embedding file you choose such as 100d, 200d, 300d
EMBEDDING_DIM = 100
epochs = 10
batch_size = 16

# read the dataset from file
traindata_path = './data/binary_sst/train.csv'
valdata_path = './data/binary_sst/val.csv'
testdata_path = './data/binary_sst/test.csv'
data_train = pd.read_csv(traindata_path, sep=',')
data_val = pd.read_csv(valdata_path, sep=',')
data_test = pd.read_csv(testdata_path, sep=',')

print('validation set shape:', data_val.shape[0])

data = pd.concat([data_train, data_val], axis=0)

print (data.shape)
texts, labels = dp.sst_data_preprocess(data_train)
test_texts, test_labels = dp.sst_data_preprocess(data_test)

labels = to_categorical(np.asarray(labels), nb_classes=2)
test_labels = to_categorical(np.asarray(test_labels), nb_classes=2)

print('Shape of data tensor:', len(texts))
print('Shape of label tensor:', len(labels))
print('Shape of data tensor:', len(test_texts))
print('Shape of label tensor:', len(test_labels))


# tokenizer the data
data, word_index = dp.tokenizer_data(texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
data_test, _ = dp.tokenizer_data(test_texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

# shuffle the data
# train_data,  train_labels = dp.shuffle_data(data, labels)

# split the data into training and validating set
nb_validation_set = data_val.shape[0]
# get the data from 0 to end-nb_validation_set set as training dataset
x_train = data[:-nb_validation_set]
y_train = labels[:-nb_validation_set]
# get the data from end to the end-nb_validation_set as validation dataset
x_val = data[-nb_validation_set:]
y_val = labels[-nb_validation_set:]

x_test = data_test
y_test = test_labels

# Training
# print('load model')
# model.load('imdb_tflearn.h5')
# print('success')
model.fit(x_train, y_train, validation_set=(x_val, y_val), show_metric=True, batch_size=16)
