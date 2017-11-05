#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

python3 analyze.py \
    --predict-csv   ${ROOT}/predict/output_val_resnext101_64x4d_e15_mv1.csv \
    --bson-path     ${ROOT}/data/train_split_val.bson \
    --save-incorrect ./incorrects

