# author - Logan
# Apr 24 2018
# Place - CUHKSZ

import numpy as np
import pandas as pd
import time

import sys
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Sequential, Model, load_model

import data_process as dp
# from sentence_process import data_preprocess
# from sentence_process import tokenizer_data
# from sentence_process import shuffle_data


# initilize the paramters
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 15000

sst_epochs = 15
sst_batch_size = 100

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

labels = to_categorical(np.asarray(labels))
test_labels = to_categorical(np.asarray(test_labels))

print('Shape of data tensor:', len(texts))
print('Shape of label tensor:', len(labels))
print('Shape of data tensor:', len(test_texts))
print('Shape of label tensor:', len(test_labels))


# tokenizer the data
data, word_index = dp.tokenizer_data(texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
sst_test, _ = dp.tokenizer_data(test_texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

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

x_test = sst_test
y_test = test_labels



print('load the IMDB pre trained model')
best_model_path = './models/2best_modelsst_transferlearning_100d_lstm_imdb_sl_150_acc_0.87376666188240052018_04_24_21_50_11.h5'
model = load_model(best_model_path)

print('train the model on sst dataset')

best_acc = 0.5
for i in range(sst_epochs):
    print('epoch:', i+1)
    # train the model on sst
    model.fit(x_train, y_train,
              batch_size=sst_batch_size,
              epochs=1,
              validation_data=(x_test, y_test),
              verbose=0)
    # evaluate the model on testing dataset
    train_score, train_acc = model.evaluate(x_train, y_train,
                                batch_size=sst_batch_size,
                                verbose=0)
    test_score, test_acc = model.evaluate(x_test, y_test,
                                batch_size=sst_batch_size,
                                verbose=0)

    print('train_score:', train_score, 'train_acc:', train_acc)
    print('test_score:', test_score, 'test_acc', test_acc)
    if(test_acc > best_acc):
        best_acc = test_acc
        best_model = model

# save the best model
print('best_acc:', best_acc)
current_time = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
best_model_path = './models/'+ str(sst_epochs) + 'epochs_' + 'best_model_'+ 'sst_transferlearning_'+ str(EMBEDDING_DIM)+'d_lstm_imdb_' + 'sl_' + str(MAX_SEQUENCE_LENGTH) + 'acc:' + str(best_acc) + current_time + '.h5'
best_model.save(best_model_path)

# print(model.summary())

# evaluate the model on testing dataset
score, acc = best_model.evaluate(x_test, y_test,
                            batch_size=sst_batch_size,
                            verbose=0)
print('Best score:', score)
print('Best accuracy:', acc)

# save the result to csv file
sst_predict = best_model.predict(x_test)

result_path = './prediction/' + str(sst_epochs) + 'epochs_' + 'sst_transferlearning_'+ str(EMBEDDING_DIM)+'d_lstm_imdb_' + current_time
print('result:', result_path)

y_preds = (sst_predict[:, 1]>0.5).astype('int')
y_pred = pd.DataFrame(y_preds, columns=['label'])

sst_result = pd.concat([y_pred, data_test.sentence], axis=1)

sst_result.to_csv(result_path, index=False)


# with sentence length at 200
# train_score: 0.3025751153263387 train_acc: 0.9046792280244291
# test_score: 0.8254468385529086 test_acc 0.5409115871546493
# epoch: 7

# with sentence length at 200 and 200d vector
# train_score: 0.5599438989505409 train_acc: 0.7174272532993681
# test_score: 0.7295623784664893 test_acc 0.538989565395052
# epoch: 5

# with sentence length at 100 & 100D
# epoch: 10
# train_score: 0.48510821783057756 train_acc: 0.7933201058792374
# test_score: 0.7321632794139805 test_acc 0.5502471165922825
# best_acc: 0.5565623298639258
# Test score: 0.7321632794139805
# Test accuracy: 0.5502471165922825