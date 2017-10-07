#!/usr/bin/env bash

python3 -u bson2rec_simple.py \
    --bson          /home/deploy/dylan/projects/kaggle-cdiscount/data/refined_train.bson \
    --category-csv  /home/deploy/dylan/dataset/cdiscount/category_names.csv \
    --out-rec       /home/deploy/dylan/projects/kaggle-cdiscount/data/refined_train.rec \
    --md5-dict-pkl  /home/deploy/dylan/projects/kaggle-cdiscount/data/md5_dict.pkl \
    --unique-md5
