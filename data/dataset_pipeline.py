#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dataset_pipeline.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/21 上午11:31
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import os
import json
import pickle
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

from libs.configs import cfgs
from utils.tools import makedir


def dataset_batch(img_name, img_caption, batch_size, buffer_size=256):
    """

    :param img_name:
    :param img_caption:
    :param batch_size:
    :param buffer_size:
    :param epoch:
    :return:
    """

    dataset = tf.data.Dataset.from_tensor_slices((img_name, img_caption))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_feature,
                                                                 inp=[item1, item2],
                                                                 Tout=[tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def load_feature(img_name, img_caption):
    feature_path = os.path.join(cfgs.IMAGE_FEATURE_PATH, os.path.basename(img_name.decode('utf-8')) + '.npy')
    img_feature = np.load(feature_path)

    return img_feature, img_caption

def tokenize(texts):
    """

    :param texts:
    :return:
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=cfgs.NUM_WORDS,
                                                      char_level=False,
                                                      oov_token='<unk>',
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(texts)
    tokenizer.texts_to_sequences(texts)

    # update word index
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # create tokenized vector
    sequence = tokenizer.texts_to_sequences(texts)
    # pad each vector to the max_length of the caption
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post')

    word_index = tokenizer.word_index
    save_to_pickle(cfgs.WORD_INDEX, word_index)

    seq_max_length = {
        'max_length': sequence.shape[-1],
    }
    save_to_pickle(cfgs.SEQ_MAX_LENGTH, seq_max_length)

    return sequence

def load_dataset(image_path, annotation_path, num_examples=50000):
    """
    load image and caption
    :param image_path:
    :param annotation_path:
    :param num_examples:
    :return:
    """
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    # store caption and image_id in vectors
    all_captions = []
    all_images = []
    images_id = []
    for annotation in annotations['annotations']:
        caption = '<start> ' + annotation['caption'] + ' <end>'
        image_id = annotation['image_id']
        img_path = os.path.join(image_path, '{:012d}.jpg'.format(image_id))

        all_captions.append(caption)
        all_images.append(img_path)

    # note one image have
    print(len(set(all_images)), len(set(all_captions))) # 118287 570281
    # shuffle captions and image_path
    train_captions, train_images = shuffle(all_captions, all_images, random_state=0)

    captions = train_captions[:num_examples]
    images = train_images[:num_examples]

    return images, captions


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def extract_feature(image_path):
    """

    :param image_path:
    :return:
    """
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    print('input: name {} -- shape {}'.format(new_input.op.name, new_input.shape))
    print('output: name {} -- shape {}'.format(hidden_layer.op.name, hidden_layer.shape))
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    sorted_image = sorted(set(image_path))
    image_dataset = tf.data.Dataset.from_tensor_slices(sorted_image)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    makedir(cfgs.IMAGE_FEATURE_PATH)
    for batch_image, batch_path in tqdm(image_dataset, desc="extract feature"):
        batch_features = image_features_extract_model(batch_image)  # (batch_size, 8, 8, 2048)
        batch_features = tf.reshape(batch_features,
                                    shape=(
                                    batch_features.shape[0], -1, batch_features.shape[-1]))  # (batch_size, 64, 2048)
        # save feature
        for feature, path in zip(batch_features, batch_path):
            path = path.numpy().decode('utf-8')
            np.save(os.path.join(cfgs.IMAGE_FEATURE_PATH, os.path.basename(path)), feature)

    return True


def get_max_length(tensor):

    return max(len(t) for t in tensor)


def save_to_pickle(filename, vocab):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)


def read_from_pickle(filename):
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
        return vocab


def index_to_word(tensor, index_word):
    text = ' '.join([index_word[t] for t in tensor if t != 0])
    return text


def split_dataset(image_name, sequence, split_ratio):
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(image_name,
                                                                                  sequence,
                                                                                  test_size=split_ratio)
    return img_name_train, img_name_val, cap_train, cap_val


# def word_to_index(text, word_index):
#     text = preprocess_sentence(text)
#     tensor = [word_index[t] for t in text]
#
#     return tensor

if __name__ == "__main__":

    train_image_path = os.path.join(cfgs.DATASET_PATH, 'train2017')
    train_annotation_path = os.path.join(cfgs.DATASET_PATH, 'annotations', 'captions_train2017.json')

    train_images, train_captions = load_dataset(train_image_path, train_annotation_path, num_examples=50000)
    print(len(train_images), len(train_captions))

    # initialize inception_v3 and construct model

    # image_features_extract_model.summary()
    # catching the feature and extract from inception V3
    # extract_feature(train_images)
    train_sequence = tokenize(train_captions)
    img_name_train, img_name_val, cap_train, cap_val = split_dataset(train_images, train_sequence,
                                                                     split_ratio=cfgs.SPLIT_RATIO)

    print(len(img_name_train), len(img_name_val), len(cap_train), len(cap_val))

    train_dataset = dataset_batch(img_name_train, cap_train, batch_size=cfgs.BATCH_SIZE)

    word_index = read_from_pickle(cfgs.WORD_INDEX)
    index_word = {index:word for word, index in word_index.items()}
    # for _ in range(10):
    #     feature_batch, cap_batch = next(iter(train_dataset))
    #     print(feature_batch.shape, cap_batch.shape)  # (batch_size, 8*8, 2048), (batch_size, max_length)
    #     print(index_to_word(cap_batch[0].numpy(), index_word))
    for (batch, (feature_batch, cap_batch)) in enumerate(train_dataset.take(2)):
        print(feature_batch.shape, cap_batch.shape)  # (batch_size, 8*8, 2048), (batch_size, max_length)
        print(index_to_word(cap_batch[0].numpy(), index_word))

    print('Done ')

