
#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

python3 -u bson_md5_dict.py \
    --bson        ${ROOT}/data/train.bson \
    --md5-dict-pkl      ${ROOT}/data/train_md5_dict_new.pkl

