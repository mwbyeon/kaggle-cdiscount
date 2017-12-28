

#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

python3 -u split_train_bson.py \
    --input-bson        ${ROOT}/data/train.bson \
    --val-ratio         0.05 \
    --save-train-bson   ${ROOT}/data/train_split_B_train.bson \
    --save-val-bson     ${ROOT}/data/train_split_B_val.bson \
    --random-seed       1

python3 -u bson2rec_simple.py \
    --bson          ${ROOT}/data/train_split_B_train.bson \
    --out-rec       ${ROOT}/data/train_split_B_train.rec

python3 -u bson2rec_simple.py \
    --bson          ${ROOT}/data/train_split_B_val.bson \
    --out-rec       ${ROOT}/data/train_split_B_val.rec

