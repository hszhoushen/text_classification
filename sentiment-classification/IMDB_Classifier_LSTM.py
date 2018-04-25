# author - Logan
# Time - April 16 2017
# Place - CUHKSZ

import numpy as np
import pandas as pd
import time
import data_process as dp


from keras.utils.np_utils import to_categorical
from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Sequential, Model



# define parameters for training
SENTENCE_NUM = 25000
MAX_SEQUENCE_LENGTH = 500   # cut texts after this number of words (among top max_features most common words)
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
epochs = 5
batch_size = 50


# load the dataset
data_path = './data/imdb_train.csv'

print('reading...')
data_train = pd.read_csv(data_path)
print('reading success')
print (data_train.head())
print (data_train.shape)

# process the sentence
texts, labels = dp.data_preprocess(data_train)

# load the glove for word embedding
embeddings_index = dp.create_embedding_index()

# from [1, 1, 0 ...] to [[0, 1], [0, 1], [1, 0], ...]
labels = to_categorical(np.asarray(labels))


print('Total %s word vectors.' % len(embeddings_index))
print('Shape of data tensor:', len(texts))
print('Shape of label tensor:', len(labels))

# tokenizer the data
data, word_index = dp.tokenizer_data(texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

# create the embedding matrix
embedding_matrix = dp.create_embedding_matrix(word_index, embeddings_index, EMBEDDING_DIM)

# shuffle the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# split the data into training and validation set
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


print('Traing and validation set number of positive and negative reviews')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))



print('Build model...')
model = Sequential()
model.add(Embedding(len(word_index)+1,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    mask_zero=False,
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))
model.add(Dropout(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(100, activation='tanh'))
model.add(Dense(2, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
# load the pretrained weights from IMDB dataset before training
# model.load_weights('./models/20_epochs_300dlstm_imdb.h5')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val))


score, acc = model.evaluate(x_val, y_val,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


# print(model.summary())

current_time = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
model_path = './models/'+ str(epochs) + 'epochs_IMDB_300dlstm_' + current_time + '.h5'
model.save(model_path)
