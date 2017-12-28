

#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

python3 -u bson2rec_simple.py \
    --bson          ${ROOT}/data/train_split_A_train.bson \
    --out-rec       ${ROOT}/data/train_split_C_train.rec \
    --unique-md5

python3 -u bson2rec_simple.py \
    --bson          ${ROOT}/data/train_split_A_val.bson \
    --out-rec       ${ROOT}/data/train_split_C_val.rec \
    --unique-md5

