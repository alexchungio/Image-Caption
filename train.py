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

    # feature = encoder(example_image_batch)
    # print('Encoder output shape: (batch size, 64, embedding_dim) {}'.format(feature.shape)) # (32, 64, 256)
    #
    #
    # hidden = decoder.reset_state(batch_size = cfgs.BATCH_SIZE)
    # attention_layer = BahdanauAttention(units)
    # context_vector, attention_weights = attention_layer(feature=feature, hidden=hidden)
    #
    # print("Context_vector shape: (batch size, units) {}".format(context_vector.shape)) # (32, 256)
    # print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape)) # (32, 64, 1)
    # #
    #
    # sample_decoder_output, _, _ = decoder(tf.random.uniform((cfgs.BATCH_SIZE, 1)), feature=feature, hidden=hidden)
    #
    # print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape)) # (32, 10057)
    #
    # print('Done')

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfgs.LEARNING_RATE)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


    def loss_function(target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = loss_object(target, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    # checkpoint
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=cfgs.TRAINED_CKPT, max_to_keep=5)

    # --------------------------train start with latest checkpoint----------------------------
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)

    # --------------------------------- train_step---------------------------------------------
    @tf.function
    def train_step(image_feature, target):
        """

        :param image_feature:
        :param target:
        :return:
        """

        loss = 0
        # decoder hidden state
        hidden_states = decoder.reset_state(batch_size=target.shape[0])
        # decoder input per step
        decoder_input = tf.expand_dims([word_index['<start>']] * target.shape[0], axis=1)

        with tf.GradientTape() as tape:
            # get encoder feature
            feature = encoder(image_feature)

            for i in range(1, target.shape[1]):
                predictions, hidden_states, _ = decoder(x=decoder_input, feature=feature, hidden=hidden_states)

                loss += loss_function(target[:, i], predictions)

                # teacher forcing the target word is passed as the next input to the decoder
                decoder_input = tf.expand_dims(target[:, i], axis=1)

        total_loss = (loss / int(target.shape[1]))
        trainable_variables = encoder.trainable_variables + encoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss


    summary_writer = tf.summary.create_file_writer(cfgs.SUMMARY_PATH)
    loss_plot = []
    for epoch in range(start_epoch, cfgs.NUM_EPOCH):

        start_time = time.time()
        num_steps = 0
        total_loss = []

        for (batch, (image_feature, image_caption)) in enumerate(train_dataset):

            batch_loss, t_loss = train_step(image_feature, image_caption)
            total_loss += t_loss

            num_steps += 1

            if batch % cfgs.SHOW_TRAIN_INFO_INTE == 0:
                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, batch_loss.numpy() / int(image_caption.shape[1])))


        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss[0] / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss[0] / num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))

        with summary_writer.as_default():
            tf.summary.scalar('loss', total_loss.numpy(), step=epoch)


    #