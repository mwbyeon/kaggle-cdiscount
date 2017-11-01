#!/usr/bin/env bash

for md5_type in unique majority l1 l2 softmax
do
    for mean in arithmetic geometric
    do
        python3 -u predict.py \
            --zmq-port          18300 \
            --bson              /home/deploy/dylan/projects/kaggle-cdiscount/data/train_split_val.bson \
            --csv               /home/deploy/dylan/dataset/cdiscount/category_names.csv \
            --params            /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d-0015.params \
            --symbol            /home/deploy/dylan/projects/kaggle-cdiscount/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d-symbol.json \
            --batch-size        1024 \
            --data-shape        3,180,180 \
            --gpus              0,1,2,3,4,5,6,7 \
            --num-procs         24 \
            --md5-dict-pkl      /home/deploy/dylan/projects/kaggle-cdiscount/data/train_split_train_md5.pkl \
            --md5-dict-type     ${md5_type} \
            --multi-view        1 \
            --multi-view-mean   ${mean} \
            --output            ""
    done
done
