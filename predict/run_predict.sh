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
    --md5-dict-pkl  /home/deploy/dylan/projects/kaggle-cdiscount/data/train_md5_dict.pkl \
    --md5-dict-type majority \
    --multi-view    1 \
    --output        20171101/output_resnext101_64x4d_e15_mv1_md5_major.csv


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
    --md5-dict-pkl  /home/deploy/dylan/projects/kaggle-cdiscount/data/train_md5_dict.pkl \
    --md5-dict-type l1 \
    --multi-view    1 \
    --output        20171101/output_resnext101_64x4d_e15_mv1_md5_l1.csv


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
    --md5-dict-pkl  /home/deploy/dylan/projects/kaggle-cdiscount/data/train_md5_dict.pkl \
    --md5-dict-type l2 \
    --multi-view    1 \
    --output        20171101/output_resnext101_64x4d_e15_mv1_md5_l2.csv


python3 -u predict.py \
    --zmq-port      18400 \
    --bson          /home/deploy/dylan/dataset/cdiscount/test.bson \
    --csv           /home/deploy/dylan/dataset/cdiscount/category_names.csv \
    --params        /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-64x4d-seed1/resnext-101-64x4d-0015.params \
                    /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d-0015.params \
                    /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/dpn98-02/dpn98-0015.params \
    --symbol        /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-64x4d-seed1/resnext-101-64x4d-symbol.json \
                    /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d-symbol.json \
                    /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/dpn98-02/dpn98-symbol.json \
    --batch-size    1024 \
    --data-shape    3,180,180 \
    --gpus          0,1,2,3,4,5,6,7 \
    --num-procs     24 \
    --md5-dict-pkl  /home/deploy/dylan/projects/kaggle-cdiscount/data/train_md5_dict.pkl \
    --md5-dict-type l1 \
    --multi-view    1 \
    --output        20171101/output_dpn98_C0FFEE_e15__resnext101_COFFEE_e15__resnext101_seed1_e15__mv1__md5__l1_geomean.csv
