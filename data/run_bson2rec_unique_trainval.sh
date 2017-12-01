#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

python3 -u ${ROOT}/data/bson2rec_simple.py \
    --bson          ${ROOT}/data/train_split_train.bson \
    --out-rec       ${ROOT}/data/train_split_train.rec \
    --md5-dict-pkl  ${ROOT}/data/train_split_train_md5.pkl \
    --unique-md5

python3 -u ${ROOT}/data/bson2rec_simple.py \
    --bson          ${ROOT}/data/train_split_val.bson \
    --out-rec       ${ROOT}/data/train_split_val.rec \
    --unique-md5

python3 -u ${ROOT}/data/bson2rec_simple.py \
    --bson          ${ROOT}/data/train_split_train.bson \
    --out-rec       ${ROOT}/data/train_split_train_single_label.rec \
    --md5-dict-pkl  ${ROOT}/data/train_split_train_md5.pkl
