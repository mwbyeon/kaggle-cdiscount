#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

python3 -u ${ROOT}/predict/predict.py \
    --zmq-port          18300 \
    --bson              ${ROOT}/data/train_split_val.bson \
    --csv               ${ROOT}/data/category_names.csv \
    --params            ${ROOT}/predict/models/resnext-101-64x4d-model1-0015.params \
                        ${ROOT}/predict/models/resnext-101-64x4d-model2-0015.params \
                        ${ROOT}/predict/models/dpn98-0015.params \
    --symbol            ${ROOT}/predict/models/resnext-101-64x4d-model1-symbol.json \
                        ${ROOT}/predict/models/resnext-101-64x4d-model2-symbol.json \
                        ${ROOT}/predict/models/dpn98-symbol.json \
    --batch-size        1024 \
    --resize            0 \
    --data-shape        3,180,180 \
    --gpus              0,1,2,3,4,5,6,7 \
    --num-procs         24 \
    --cate-level        3 \
    --md5-dict-pkl      ${ROOT}/data/train_split_train_md5.pkl \
    --md5-dict-type     unique \
    --multi-view        1 \
    --multi-view-mean   geometric \
    --mean-max-pooling  0 \
    --output            ""
