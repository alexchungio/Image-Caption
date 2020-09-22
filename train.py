#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/21 上午10:18
# @ Software   : PyCharm
#-------------------------------------------------------

# ------------------------import package-----------------------------
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import tensorflow as tf

from libs.configs import cfgs
from data.dataset_pipeline import dataset_batch, read_from_pickle, tokenize, split_dataset, load_dataset
from libs.nets.model import CNNEencoder, RNNDecoder, BahdanauAttention



if __name__ == "__main__":
    train_image_path = os.path.join(cfgs.DATASET_PATH, 'train2017')
    train_annotation_path = os.path.join(cfgs.DATASET_PATH, 'annotations', 'captions_train2017.json')

    train_images, train_captions = load_dataset(train_image_path, train_annotation_path, num_examples=50000)
    print(len(train_images), len(train_captions))

    # initialize inception_v3 and construct model
    train_sequence = tokenize(train_captions)
    img_name_train, img_name_val, cap_train, cap_val = split_dataset(train_images, train_sequence,
                                                                     split_ratio=cfgs.SPLIT_RATIO)
    print(len(img_name_train), len(img_name_val), len(cap_train), len(cap_val))

    # get word_index and index word
    word_index = read_from_pickle(cfgs.WORD_INDEX)
    index_word = {index: word for word, index in word_index.items()}

    vocab_size = len(word_index)
    #
    train_dataset = dataset_batch(img_name_train, cap_train, batch_size=cfgs.BATCH_SIZE)

    example_image_batch, example_cap_batch = next(iter(train_dataset))



    # show shape
    embedding_dim = 256
    units = 512

    encoder = CNNEencoder(embedding_dim=embedding_dim)
    decoder = RNNDecoder(embedding_dim, units, vocab_size)

    feature = encoder(example_image_batch)
    print('Encoder output shape: (batch size, 64, embedding_dim) {}'.format(feature.shape)) # (32, 64, 256)


    hidden = decoder.reset_state(batch_size = cfgs.BATCH_SIZE)
    attention_layer = BahdanauAttention(units)
    context_vector, attention_weights = attention_layer(feature=feature, hidden=hidden)

    print("Context_vector shape: (batch size, units) {}".format(context_vector.shape)) # (32, 256)
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape)) # (32, 64, 1)
    #

    sample_decoder_output, _, _ = decoder(tf.random.uniform((cfgs.BATCH_SIZE, 1)), feature=feature, hidden=hidden)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape)) # (32, 10057)

    print('Done')
