#!/usr/bin/env bash

python bson2rec.py \
    --prefix /home/deploy/dylan/projects/kaggle-cdiscount/data/cdiscount \
    --bson /home/deploy/dylan/dataset/cdiscount/train.bson \
    --csv /home/deploy/dylan/dataset/cdiscount/category_names.csv \
    --pass-through \
    --num-thread 1 \
    --shuffle \
    --val-ratio 0.05

