#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

python3 -u bson2rec_simple.py \
    --bson          ${ROOT}/data/train_split_train.bson \
    --out-rec       ${ROOT}/data/train_split_train.rec \
    --func          read_images

python3 -u bson2rec_simple.py \
    --bson          ${ROOT}/data/train_split_val.bson \
    --out-rec       ${ROOT}/data/train_split_val.rec \
    --func          read_images
