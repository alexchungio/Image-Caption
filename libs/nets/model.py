#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : model.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/22 下午4:42
# @ Software   : PyCharm
#-------------------------------------------------------


import tensorflow as tf


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units=units)
        self.W2 = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)


    def __call__(self, feature, hidden):
        """

        :param feature: (batch_size, 64, embedding_dim)
        :param hidden: (batch_size, hidden_size)
        :return:
        """
        # (batch_size, 1,  hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, axis=1)

        # attention_hidden_layer => (batch_size, 64, num_units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(feature) +
                                             self.W2(hidden_with_time_axis)))

        # scores => (batch_size, 64, 1)
        scores = self.V(attention_hidden_layer)

        # attention_weights => (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(scores, axis=1)

        # context_vector => (batch_size, embedding_dim)
        context_vector = attention_weights * feature  #  (batch_size, 64, embedding_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1) #  (batch_size, embedding_dim)

        return context_vector, attention_weights



class CNNEencoder(tf.keras.Model):
    """

    """
    def __init__(self, embedding_dim):
        super(CNNEencoder, self).__init__()
        self.fc = tf.keras.layers.Dense(units=embedding_dim)

    def __call__(self, x):
        """

        :param x: (batch_size, 64, 2048)
        :return: (batch_size, 64, embedding_dim)
        """
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNNDecoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):

        super(RNNDecoder, self).__init__()
        self.units = units
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(self.vocab_size)

        self.attention = BahdanauAttention(units=self.units)

    def __call__(self, x, feature, hidden):
        """

        :param x: (batch_size, 1)
        :param feature: (batch_size, 64, embedding_dim)
        :param hidden:  (batch_size, hidden_size)
        :return:
        """
        # context_vector => (batch_size, embedding_dim(hidden_size))
        # attention_weights => (batch_size, 64, 1)
        context_vector, attention_weights = self.attention(feature, hidden)

        # embedding layer x => (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x after concat => (batch_size, 1,  embedding_dim + embedding_dim(hidden_size))
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=-1)

        # passing the concatenated vector to gru
        # outputs => (batch_size, 1, num_units)
        # states => (batch_size, num_units)
        outputs, states = self.gru(x)

        # x => (batch_size, 1, units)
        x = self.fc1(x)

        # x => (batch_size, num_units)
        x = tf.reshape(x, (-1, x.shape[-1]))

        # x => (batch_size, vocab_size)
        x = self.fc2(x)

        return x, states, attention_weights


    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
