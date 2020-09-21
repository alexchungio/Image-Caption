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
# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle





if __name__ == "__main__":
    train_image_path = os.path.join('/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/COCO_2017', 'train2017')
    train_annotation_path = os.path.join('/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/COCO_2017', 'annotations',
                                         'captions_train2017.json')

    with open(train_annotation_path, 'r') as f:
        annotations = json.load(f)


    # set number examples
    num_examples = 50000

    # process annotation
    with open(train_annotation_path, 'r') as f:
        annotations = json.load(f)

    # store caption and image_id in vectors
    all_captions = []
    all_images = []

    for annotation in annotations['annotations']:
        caption = '<start> ' + annotation['caption'] + ' <end>'
        image_id = annotation['image_id']
        img_path = os.path.join(train_image_path, '{:012d}.jpg'.format(image_id))

        all_captions.append(caption)
        all_images.append(img_path)

    # shuffle captions and image_path
    train_captions, train_images = shuffle(all_captions, all_images, random_state=0)

    train_captions = train_captions[:num_examples]
    train_images = train_images[:num_examples]

    print('Done ')