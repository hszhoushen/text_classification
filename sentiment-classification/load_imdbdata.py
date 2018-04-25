# author - Logan
# Time - April 16 2017
# Place - CUHKSZ

import os
import pandas as pd
import tensorflow as tf
import re

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):

  train_path = os.path.join("./data","aclImdb", "train")

  test_path = os.path.join("./data","aclImdb", "test")

  train_df = load_dataset(train_path)
  print('finish train_df')
  test_df = load_dataset(test_path)
  print('finish test_df')

  return train_df, test_df


train_df, test_df = download_and_load_datasets()

train_df = pd.concat([train_df.sentence, train_df.polarity], axis=1)
test_df = pd.concat([test_df.sentence, test_df.polarity], axis=1)

print(train_df.head())
print(test_df.head())
train_df.to_csv('./data/imdb_train.csv')
test_df.to_csv('./data/imdb_test.csv')
print('end')
