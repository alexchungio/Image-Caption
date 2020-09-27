#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/23 下午6:26
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


from data.dataset_pipeline import  load_image, read_from_pickle, feature_extract_model
from libs.nets.model import CNNEencoder, RNNDecoder
from libs.configs import cfgs


# get word_index and index word
word_index = read_from_pickle(cfgs.WORD_INDEX)
index_word = {index: word for word, index in word_index.items()}

max_length = read_from_pickle(cfgs.SEQ_MAX_LENGTH)['max_length']



def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


def evaluate(image):
    image_features_extract_model = feature_extract_model()
    attention_plot = np.zeros((max_length, cfgs.ATTENTION_FEATURE_SHAPE))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(index_word[predicted_id])

        if index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


if __name__ == "__main__":


    encoder = CNNEencoder(cfgs.EMBEDDING_DIM)
    decoder = RNNDecoder(cfgs.EMBEDDING_DIM, cfgs.NUM_UNITS, cfgs.TOP_WORDS +1)

    optimizer = tf.keras.optimizers.Adam()
    checkpoint_prefix = os.path.join(cfgs.TRAINED_CKPT, "ckpt_{epoch}")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    # restoring the latest checkpoint in checkpoint_dir
    ckpt_path = tf.train.latest_checkpoint(cfgs.TRAINED_CKPT)
    print(ckpt_path)

    checkpoint.restore(ckpt_path)

    image_path = os.path.abspath(os.path.join('../tools/demo', 'surf.jpg'))

    result, attention_plot = evaluate(image_path)
    print('Prediction Caption:', ' '.join(result))
    plot_attention(image_path, result, attention_plot)
    # opening the image
    Image.open(image_path)