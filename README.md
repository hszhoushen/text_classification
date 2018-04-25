# text_classification
This is a project for text classification based on IMDB and SST2 dataset.

We first pretrain the model using the LSTM based on the IMDB dataset.

Then, the pretrain model is used to train the SST2 dataset and evaluate the model on test dataset on SST2 which consists of 1820 short movie reviews.

First we run the pretrain_imdb.py to get the pretrain model on imdb.
Then, we get the best pretrain model path and write the path in sst_training.py file and train model on SST2 dataset.

data_process.py is used for data loading and preprocessing.
