#!/usr/bin/env bash

python3 -u split_train_bson.py \
    --input-bson    /home/deploy/dylan/dataset/cdiscount/train.bson \
    --val-ratio     0.05 \
    --save-train-bson   /home/deploy/dylan/projects/kaggle-cdiscount/data/refined_train.bson \
    --save-val-bson     /home/deploy/dylan/projects/kaggle-cdiscount/data/refined_val.bson
