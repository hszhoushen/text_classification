# author - Logan
# Apr 24 2018
# Place - CUHKSZ


import time
from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, MaxPooling1D
from keras.models import Sequential, Model
from keras import optimizers
import data_process as dp


# initilize the paramters
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 15000

# model parameter
imdb_epochs = 2
imdb_batch_size = 50
dropout = 0.25
lstm_dropout = 0.4
LR = 0.003

# LSTM
lstm_output_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4
VALIDATION_SPLIT = 0.6




# load the imdb dataset
print('start load the imdb')
x_train, y_train, x_val, y_val, embedding_matrix, len_train_word_index = dp.load_imdb(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, VALIDATION_SPLIT)

print('Traing and validation set number of positive and negative reviews')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))


print('Build model...')
model = Sequential()
model.add(Embedding(len_train_word_index+1,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    mask_zero=False,
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))
# model.add(Embedding(len_train_word_index+1,
#                     EMBEDDING_DIM,
#                     weights=[embedding_matrix],
#                     mask_zero=False,
#                     input_length=MAX_SEQUENCE_LENGTH,
#                     trainable=False))
# model.add(Conv1D(filters,
#                  kernel_size,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
#model.add(MaxPooling1D(pool_size=pool_size))
model.add(Dropout(dropout))
model.add(Bidirectional(LSTM(lstm_output_size, dropout=lstm_dropout)))
# model.add(Dense(lstm_output_size, activation='tanh'))
model.add(Dropout(dropout))
model.add(Dense(2, activation='softmax'))
adams = optimizers.Adam(lr=LR, beta_1=0.8, beta_2=0.95, epsilon=None, decay=0.0, amsgrad=False)
# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer=adams,
              metrics=['accuracy'])
# binary_crossentropy

# training
print('Train...')
best_acc = 0.5
for i in range(imdb_epochs):
    print('epoch: ', i + 1)

    model.fit(x_train, y_train,
              batch_size=imdb_batch_size,
              epochs=1,
              validation_data=(x_val, y_val),
              verbose=0)

    # evaluate the model on training dataset
    test_score, test_acc = model.evaluate(x_val, y_val,
                                batch_size=imdb_batch_size,
                                verbose=0)
    # evaluate the model on testing dataset
    train_score, train_acc = model.evaluate(x_train, y_train,
                                batch_size=imdb_batch_size,
                                verbose = 0)

    print('Train score:', train_score)
    print('Train accuracy:', train_acc)
    print('Test score:', test_score)
    print('Test accuracy:', test_acc)

    if(test_acc > best_acc):
        best_acc = test_acc
        best_model = model
    else:
        best_model = model

# save the best_model
print('best accuracy=', best_acc)
current_time = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
best_model_path = './models/'+ str(imdb_epochs) + 'best_model'+ 'sst_transferlearning_' + str(EMBEDDING_DIM) + 'd_lstm_imdb_' + 'sl_' + str(MAX_SEQUENCE_LENGTH) + '_acc_' + str(best_acc) + current_time + '.h5'
print('best_model_path:', best_model_path)
best_model.save(best_model_path)


# with sentence length at 500
# epooch 20 loss: 0.0984 - acc: 0.9606 - val_loss: 0.3102 - val_acc: 0.9026
# print(model.summary())
#
#Train score: 0.030577751987613738
#Train accuracy: 0.9914600032567978


# with sentence length at 300
# epooch 20 loss: 0.0984 - acc: 0.9606 - val_loss: 0.3102 - val_acc: 0.9026
# print(model.summary())

# Train score: 0.03135960669349879
# Train accuracy: 0.9926800043582916
# Test Acc: 0.8959199948310852
# Test_Score: 0.3197254096418619

# with sentence length at 150


# with sentence length at 200
# Train score: 0.14852381101250647
# Train accuracy: 0.945499999165535
# Test score: 0.27344141171127556
# Test accuracy: 0.8905799957513809
# best accuracy= 0.8942999963760376

# with 200D word embedding and sentence length at 200
# Train score: 0.2528523686975241
# Train accuracy: 0.8923799946308136
# Test score: 0.2995964899659157
# Test accuracy: 0.8696999943256378
# best accuracy= 0.8696999943256378

# with 100D word embedding and sentence length at 200
