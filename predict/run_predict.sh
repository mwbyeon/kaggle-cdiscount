#!/usr/bin/env bash

python3 -u predict.py \
    --zmq-port      18400 \
    --bson          /home/deploy/dylan/dataset/cdiscount/test.bson \
    --csv           /home/deploy/dylan/dataset/cdiscount/category_names.csv \
    --params        /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d-0015.params \
    --symbol        /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d-symbol.json \
    --batch-size    1024 \
    --data-shape    3,180,180 \
    --gpus          0,1,2,3,4,5,6,7 \
    --num-procs     24 \
    --md5-dict-pkl  "" \
    --multi-view    1 \
    --output        output_resnext101_64x4d_e15_mv1.csv

python3 -u predict.py \
    --zmq-port      18400 \
    --bson          /home/deploy/dylan/dataset/cdiscount/test.bson \
    --csv           /home/deploy/dylan/dataset/cdiscount/category_names.csv \
    --params        /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d-0015.params \
    --symbol        /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d-symbol.json \
    --batch-size    1024 \
    --data-shape    3,180,180 \
    --gpus          0,1,2,3,4,5,6,7 \
    --num-procs     24 \
    --md5-dict-pkl  /home/deploy/dylan/projects/kaggle-cdiscount/data/md5_dict.pkl \
    --multi-view    1 \
    --output        output_resnext101_64x4d_e15_mv1_md5.csv

