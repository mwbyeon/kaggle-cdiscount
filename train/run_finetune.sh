#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 -u fine-tune.py \
    --gpus              0,1,2,3,4,5,6,7 \
    --kv-store          device \
    --pretrained-model  ${ROOT}/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d \
    --model-prefix      ${ROOT}/train/checkpoints/resnext-101-64x4d/resnext-101-64x4d \
    --data-train        ${ROOT}/data/train_split_train.rec \
    --data-val          ${ROOT}/data/train_split_val.rec \
    --layer-before-fullc flatten0 \
    --image-shape       3,180,180 \
    --data-nthread      6 \
    --optimizer         adam \
    --lr                0.0001 \
    --lr-factor         0.2 \
    --lr-step-epochs    15,20,25 \
    --disp-batches      100 \
    --num-epoch         25 \
    --load-epoch        15 \
    --mom               0.9 \
    --wd                0.00004 \
    --top-k             5 \
    --batch-size        512 \
    --num-classes       5270 \
    --num-examples      11754490 \
    --rgb-mean          0,0,0 \
    --rgb-scale         1.0 \
    --random-crop       1 \
    --random-mirror     1 \
    --max-random-h      20 \
    --max-random-s      30 \
    --max-random-l      50 \
    --min-random-scale  1.0 \
    --max-random-scale  1.1 \
    --max-random-rotate-angle   0 \
    --max-random-shear-ratio    0 \
    --max-random-aspect-ratio   0 \
    --max-random-scale  1.0
