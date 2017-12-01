#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

python3 -u split_train_bson.py \
    --input-bson    /home/deploy/dylan/dataset/cdiscount/train.bson \
    --val-ratio     0.05 \
    --save-train-bson   ${ROOT}/data/train_split_train.bson \
    --save-val-bson     ${ROOT}/data/train_split_val.bson
