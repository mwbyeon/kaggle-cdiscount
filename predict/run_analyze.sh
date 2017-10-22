#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

cd ${ROOT}
python3 -m predict.analyze \
    --predict-csv   ${ROOT}/predict/val_debug.csv \
    --bson-path     ${ROOT}/data/refined_val.bson \
    --category-csv  /home/deploy/dylan/dataset/cdiscount/category_names.csv
