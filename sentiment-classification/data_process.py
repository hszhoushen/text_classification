# author - Logan
# Time - April 16 2017
# Place - CUHKSZ


from bs4 import BeautifulSoup
import re
import os
import numpy as np
import pandas  as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical




# process the sentence
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

# data process
def data_preprocess(input_data):
    texts = []
    labels = []
    for idx in range(input_data.sentence.shape[0]):
        text = BeautifulSoup(input_data.sentence[idx], "lxml")
        texts.append(clean_str(text.get_text().encode('ascii','ignore')))
        labels.append(input_data.polarity[idx])
    print('end data preprocess')
    return texts, labels

# sst data process
def sst_data_preprocess(data):
    texts = []
    labels = []
    for idx in range(data.shape[0]):
        # pandas??
        # print(idx, data.sentence[idx])
        text = BeautifulSoup(data.sentence[idx], "lxml")
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data.label[idx])

    print('end sst data preprocess')
    return texts, labels


# load glove for word embeddings
def create_embedding_index(word_dimension):
    GLOVE_DIR = "./glove/"
    embeddings_index = {}
    glove_path = 'glove.6B.' + str(word_dimension) + 'd'+ '.txt'
    f = open(os.path.join(GLOVE_DIR, glove_path),  'r', encoding='UTF-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

# tokenizer sentence
def tokenizer_data(texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
    # set the max length and pad the sentences
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    # print('tokenizer:', tokenizer)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    # print('sequences', sequences)
    word_index = tokenizer.word_index
    # print('word index:', word_index)
    # pad the sentence length to be MAX_SEQUENCE_LENGTH
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    # print('data:', data)

    return data, word_index

def create_embedding_matrix(word_index, embeddings_index, EMBEDDING_DIM):
    # create the embedding matrix
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# shuffle data
def shuffle_data(data, labels):
    # create the indices
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    return data, labels

def split_data(data, labels, VALIDATION_SPLIT):
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    return x_train, y_train, x_val, y_val

def load_mr_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    MR_data_pos_path = './data/MR/rt-polarity.pos'
    MR_data_neg_path = './data/MR/rt-polarity.neg'

    # Load data from files
    # with open(MR_data_pos_path) as file:
    #     data_pos = np.loadtxt(MR_data_pos_path, encoding='gbk')
    # with open(MR_data_neg_path) as file:
    #     data_neg = np.loadtxt(MR_data_neg_path, encoding='utf-8')
    # # Split by words
    # print(len(data_pos))
    # print(len(data_neg))
    # print(data_pos.shape)
    # print(data_neg.shape)
    # # Generate labels
    #
    # return [x_text, y]

def load_imdb(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, VALIDATION_SPLIT):

    # load the dataset
    train_data_path = './data/imdb/imdb_train.csv'
    test_data_path = './data/imdb/imdb_test.csv'

    data_train = pd.read_csv(train_data_path)
    data_test = pd.read_csv(test_data_path)

    # concat data vertically
    data = pd.concat([data_train, data_test], axis=0, ignore_index=True)
    print(data.shape)
    # process the sentence
    train_texts, train_labels = data_preprocess(data)

    # load the glove for word embedding
    embeddings_index = create_embedding_index(EMBEDDING_DIM)

    # from [1, 1, 0 ...] to [[0, 1], [0, 1], [1, 0], ...]
    train_labels = to_categorical(np.asarray(train_labels))

    print('Total %s word vectors.' % len(embeddings_index))
    print('Shape of data tensor:', len(train_texts))
    print('Shape of label tensor:', len(train_labels))

    # tokenizer the data
    train_data, train_word_index = tokenizer_data(train_texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

    # create the embedding matrix
    embedding_matrix = create_embedding_matrix(train_word_index, embeddings_index, EMBEDDING_DIM)

    # shuffle the data
    # train_data, train_labels = shuffle_data(train_data, train_labels)

    # split the data into training and validation set (1:1)
    x_train, y_train, x_val, y_val = split_data(train_data, train_labels, VALIDATION_SPLIT)

    len_train_word_index = len(train_word_index)
    return x_train, y_train, x_val, y_val, embedding_matrix, len_train_word_index

