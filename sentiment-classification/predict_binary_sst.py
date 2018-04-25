# author - Logan
# Mar 26 2017


import numpy as np
import pandas as pd


import sys
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Sequential, Model
from keras.models import load_model
from pandas import Series, DataFrame

SENTENCE_NUM = 25000
MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = string.decode('utf-8')         # for python3
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

# read the testing data to dataframe
data_path = './data/binary_sst/test.csv'
data_test = pd.read_csv(data_path, sep=',')
y_true = data_test['label']

print (data_test.shape)
print (data_test.head())

texts = []

for idx in range(data_test.shape[0]):
    text = BeautifulSoup(data_test.sentence[idx], "lxml")
    texts.append(clean_str(text.get_text().encode('ascii','ignore')))


# load the glove model for word embedding
GLOVE_DIR = "./glove/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),  'r', encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))
print('Shape of data tensor:', len(texts))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
# pad the sentence length to be MAX_SEQUENCE_LENGTH
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


# get the input data for testing
x_test = data

# load the model
model_path = './models/5binary_sst_300dlstm_imdb2018_04_16_19_01_22.h5'
model = load_model(model_path)
print('load model')

# predict the data according to the model we trained
predict = model.predict(x_test, batch_size=100)
print('after prediction')

# save the final result to the csv file
result = DataFrame(predict[:, 1])
result.to_csv('result_bianry_sst.csv')


# calculate the accuracy
y_pred = result[0] > 0.5    # or < 0.5 ?
accuracy = (y_true == y_pred)
accuracy = sum(accuracy) / len(accuracy)
print(accuracy)