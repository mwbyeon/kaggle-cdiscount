#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 -u ${ROOT}/train/train_model.py \
    --gpus              0,1,2,3,4,5,6,7 \
    --kv-store          device \
    --symbol            resnext \
    --num-layers        101 \
    --use-squeeze-excitation \
    --num-conv-groups   64 \
    --dropout-ratio     0.0 \
    --params            ./se-resnext-101-64x4d/se-resnext-101-64x4d-0015.params \
    --model-prefix      ./se-resnext-101-64x4d/se-resnext-101-64x4d \
    --data-train        ${ROOT}/data/train_split_train.rec \
    --data-val          ${ROOT}/data/train_split_val.rec \
    --image-shape       3,180,180 \
    --data-nthread      6 \
    --optimizer         nadam \
    --lr                0.0001 \
    --lr-factor         0.2 \
    --lr-step-epochs    10,15,18,20 \
    --disp-batches      100 \
    --num-epoch         20 \
    --load-epoch        15 \
    --mom               0.9 \
    --wd                0.00004 \
    --top-k             5 \
    --batch-size        512 \
    --num-classes       5270 \
    --smooth-alpha      0.0 \
    --num-examples      11754490 \
    --rgb-mean          0,0,0 \
    --rgb-scale         1.0 \
    --random-crop       0 \
    --random-mirror     0 \
    --max-random-h      0 \
    --max-random-s      0 \
    --max-random-l      0 \
    --min-random-scale  1.0 \
    --max-random-scale  1.0 \
    --max-random-rotate-angle   0 \
    --max-random-shear-ratio    0 \
    --max-random-aspect-ratio   0

