#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 -u ${ROOT}/train/train_model.py \

for md5_type in unique majority l1 l2 softmax none
do
    for mean in arithmetic geometric
    do
        python3 -u ${ROOT}/predict/predict.py \
            --zmq-port          18300 \
            --bson              ${ROOT}/data/train_split_val.bson \
            --csv               ${ROOT}/category_names.csv \
            --params            ${ROOT}/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d-0015.params \
            --symbol            ${ROOT}/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d-symbol.json \
            --batch-size        1024 \
            --data-shape        3,180,180 \
            --gpus              0,1,2,3,4,5,6,7 \
            --num-procs         24 \
            --md5-dict-pkl      ${ROOT}/data/train_split_train_md5.pkl \
            --md5-dict-type     ${md5_type} \
            --multi-view        1 \
            --multi-view-mean   ${mean} \
            --output            ""
    done
done
