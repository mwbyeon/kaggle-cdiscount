#!/usr/bin/env bash


python predict.py \
    --bson          /home/deploy/dylan/dataset/cdiscount/test.bson \
    --csv           /home/deploy/dylan/dataset/cdiscount/category_names.csv \
    --params        /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101/imagenet1k-resnext-101-0008.params \
    --symbol        /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101/imagenet1k-resnext-101-symbol.json \
    --batch-size    1024 \
    --data-shape    3,180,180 \
    --gpus          0,1,2,3,4,5,6,7 \
    --num-procs     24 \
    --output        output_resnext101_e8_rgb.csv

