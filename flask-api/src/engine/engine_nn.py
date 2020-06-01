import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Embedding, Input, Dropout, LSTM, TimeDistributed, Bidirectional

from src import config, utils
from src.processing import emb_processing




class EmbeddingLayer(keras.layers.Layer):

    def __init__(self, embedding_matrix):

        super(EmbeddingLayer, self).__init__()

        num_words = embedding_matrix.shape[0]
        # embedding_matrix = emb_processing.get_embedding_matrix(embedding_dict, tokenizer)
        embedding_dim = embedding_matrix.shape[1]

        self.embedding_layer = Embedding(num_words,
                                    embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=config.MAX_LEN,
                                    trainable=False)

        
    def call(self, inputs):
        return self.embedding_layer(inputs)


    
class FeedForwardNn(keras.Model):

    def __init__(self, embedding_matrix, n_classes):

        super(FeedForwardNn, self).__init__()
        

        self.embedding_layer = EmbeddingLayer(embedding_matrix)

        self.first_conv1d = Conv1D(64, 3, activation=tf.nn.relu)
        self.first_maxpool = MaxPooling1D(3)
        self.second_conv1d = Conv1D(64, 3, activation=tf.nn.relu)
        self.second_maxpool = MaxPooling1D(3)
        #model.add(SpatialDropout2D(0.5))
        self.flatten = Flatten()
        self.dense_layer = Dense(128, activation=tf.nn.relu)
        self.dropout = Dropout(0.2)
        self.out_layer = Dense(n_classes, activation=tf.nn.softmax)

    def call(self, inputs, training=None):

        x = self.embedding_layer(inputs)
        x = self.first_conv1d(x)  
        x = self.first_maxpool(x)
        x = self.second_conv1d(x) 
        x = self.second_maxpool(x)  
        x = self.flatten(x)
        x = self.dense_layer(x)
        if training:
            x = self.dropout(x)
        return(self.out_layer(x))





class LstmNetwork(keras.Model):

    def __init__(self, embedding_matrix, n_classes):

        super(LstmNetwork, self).__init__()
        
        self.embedding_layer = EmbeddingLayer(embedding_matrix)
        self.lstm_layer = Bidirectional(LSTM(128, return_sequences=True))
        self.time_distributed_layer = TimeDistributed(Dense(5))
        self.flatten_layer = Flatten()
        self.dense_layer = Dense(128, activation=tf.nn.relu)
        self.dropout = Dropout(0.2)
        self.out_layer = Dense(n_classes, activation=tf.nn.softmax)

        
    def call(self, inputs, training=None):
        
        x = self.embedding_layer(inputs)
        #x = self.flatten_layer(x)
        x = self.lstm_layer(x)
        #x = self.time_distributed_layer(x)
        x = self.flatten_layer(x)
        
        x = self.dense_layer(x) 

        if training:
            x = self.dropout(x)
        
        return(self.out_layer(x))

class MixNetwork(keras.Model):

    def __init__(self, embedding_matrix, n_classes):

        super(MixNetwork, self).__init__()
        
        self.embedding_layer = EmbeddingLayer(embedding_matrix)
        self.lstm_layer = Bidirectional(LSTM(64, return_sequences=True))
        self.time_distributed_layer = TimeDistributed(Dense(3))
        self.flatten_layer = Flatten()
        self.first_conv1d = Conv1D(128, 2, activation=tf.nn.relu)
        self.first_maxpool = MaxPooling1D(3)
        self.second_conv1d = Conv1D(128, 2, activation=tf.nn.relu)
        self.second_maxpool = MaxPooling1D(3)
        self.dropout = Dropout(0.5)
        self.dense_layer = Dense(128, activation=tf.nn.relu)

        self.out_layer = Dense(n_classes, activation=tf.nn.softmax)

        
    def call(self, inputs, training=None):
        
        x = self.embedding_layer(inputs)
        x = self.lstm_layer(x)
        x = self.time_distributed_layer(x)
        #x = self.flatten_layer(x)
        x = self.first_conv1d(x)
        x = self.first_maxpool(x)
        x = self.second_conv1d(x)
        x = self.second_maxpool(x)
        #x = self.third_conv1d(x)
        #x = self.third_maxpool(x)
        x = self.flatten_layer(x)
        x = self.dense_layer(x) 

        if training:
            x = self.dropout(x)
        
        return(self.out_layer(x))



# def train_fn(X_data, y_data, keras_model):


#     keras_model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=[mcc_metric, "acc"])
#     keras_model.fit(x=X_data, y=y_data, verbose=2, epochs=epochs, class_weight=class_weight)

#     return keras_model






