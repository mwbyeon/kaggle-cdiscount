#!/usr/bin/env bash

python3 -u predict.py \
    --zmq-port      18300 \
    --bson          /home/deploy/dylan/projects/kaggle-cdiscount/data/refined_val.bson \
    --csv           /home/deploy/dylan/dataset/cdiscount/category_names.csv \
    --params        /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-newset/imagenet1k-resnext-101-0006.params \
    --symbol        /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-newset/imagenet1k-resnext-101-symbol.json \
    --batch-size    1024 \
    --data-shape    3,180,180 \
    --gpus          0,1,2,3,4,5,6,7 \
    --num-procs     24 \
    --multi-view    0
