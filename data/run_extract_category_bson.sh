#!/usr/bin/env bash

python3 -u extract_category_bson.py \
    --bson train_split_train.bson \
    --include 27 35

python3 -u extract_category_bson.py \
    --bson train_split_val.bson \
    --include 27 35

python3 -u bson2rec_simple.py \
    --cate-type     1 \
    --bson          ./category/train_split_train_27.bson \
    --out-rec       ./category/train_split_train_27.rec

python3 -u bson2rec_simple.py \
    --cate-type     1 \
    --bson          ./category/train_split_train_35.bson \
    --out-rec       ./category/train_split_train_35.rec

python3 -u bson2rec_simple.py \
    --cate-type     1 \
    --bson          ./category/train_split_val_27.bson \
    --out-rec       ./category/train_split_val_27.rec

python3 -u bson2rec_simple.py \
    --cate-type     1 \
    --bson          ./category/train_split_val_35.bson \
    --out-rec       ./category/train_split_val_35.rec

