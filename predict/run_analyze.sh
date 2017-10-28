#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

cd ${ROOT}
python3 -m predict.analyze \
    --predict-csv   ${ROOT}/predict/output_val_resnext101_64x4d_e15_mv1.csv \
    --bson-path     ${ROOT}/data/train_split_val.bson \
    --category-csv  ${ROOT}/data/category_names.csv

