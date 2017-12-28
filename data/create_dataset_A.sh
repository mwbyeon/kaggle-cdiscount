

#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

python3 -u split_train_bson.py \
    --input-bson        ${ROOT}/data/train.bson \
    --val-ratio         0.05 \
    --save-train-bson   ${ROOT}/data/train_split_A_train.bson \
    --save-val-bson     ${ROOT}/data/train_split_A_val.bson \
    --random-seed       12648430

python3 -u bson2rec_simple.py \
    --bson          ${ROOT}/data/train_split_A_train.bson \
    --out-rec       ${ROOT}/data/train_split_A_train.rec

python3 -u bson2rec_simple.py \
    --bson          ${ROOT}/data/train_split_A_val.bson \
    --out-rec       ${ROOT}/data/train_split_A_val.rec

