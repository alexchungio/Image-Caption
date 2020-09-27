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
from data.dataset_pipeline import dataset_batch, read_from_pickle, tokenize, split_dataset, load_dataset, generate_train_samples
from libs.nets.model import CNNEencoder, RNNDecoder, BahdanauAttention


def main():

    # ----------------------------------generate dataset pipeline---------------------------------------
    train_image_path = os.path.join(cfgs.DATASET_PATH, 'train2017')
    train_annotation_path = os.path.join(cfgs.DATASET_PATH, 'annotations', 'captions_train2017.json')

    train_images_captions = load_dataset(train_image_path, train_annotation_path)

    train_images, train_captions = generate_train_samples(train_images_captions)

    print(len(train_images), len(train_captions))

    # get step per epoch
    # step_per_epoch = int(len(train_images) / cfgs.BATCH_SIZE)

    # initialize inception_v3 and construct model
    train_sequence = tokenize(train_captions)
    img_name_train, img_name_val, cap_train, cap_val = split_dataset(train_images, train_sequence,
                                                                     split_ratio=cfgs.SPLIT_RATIO)
    print(len(img_name_train), len(img_name_val), len(cap_train), len(cap_val))

    # get word_index and index word
    word_index = read_from_pickle(cfgs.WORD_INDEX)

    train_dataset = dataset_batch(img_name_train, cap_train, batch_size=cfgs.BATCH_SIZE)
    # example_image_batch, example_cap_batch = next(iter(train_dataset))

    # show shape
    encoder = CNNEencoder(embedding_dim=cfgs.EMBEDDING_DIM)
    decoder = RNNDecoder(embedding_dim=cfgs.EMBEDDING_DIM,
                         units=cfgs.NUM_UNITS,
                         vocab_size=cfgs.TOP_WORDS + 1)  # due to add '<pad>' word to corpus

    # --------------------------------- test model class--------------------------------------------

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

    # -------------------------------------initial optimizer--------------------------------------
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
    def train_step(img_feature, target):
        """

        :param img_feature:
        :param target:
        :return:
        """
        loss = 0
        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])

        # initial decode input with <start>
        dec_input = tf.expand_dims([word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            # get encoder feature
            features = encoder(img_feature)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        # get all trainable variables
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    # ------------------------------ execute train----------------------------------------
    summary_writer = tf.summary.create_file_writer(cfgs.SUMMARY_PATH)
    for epoch in range(start_epoch, cfgs.NUM_EPOCH):

        start_time = time.time()
        num_steps = 0
        total_loss = 0

        for (batch, (image_feature, image_caption)) in enumerate(train_dataset):

            batch_loss, t_loss = train_step(image_feature, image_caption)
            total_loss += t_loss
            num_steps += 1

            if batch % cfgs.SHOW_TRAIN_INFO_INTE == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss / int(image_caption.shape[1])))
        if epoch % 5 == 0:
            ckpt_manager.save()

        with summary_writer.as_default():
            tf.summary.scalar('loss', (total_loss / num_steps), step=epoch)

        print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss / num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))


if __name__ == "__main__":
    main()