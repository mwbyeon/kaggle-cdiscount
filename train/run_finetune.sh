#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

C5270_NUM_EXAMPLES=11754490
REFINE_NUM_EXAMPLES=7310313
MODEL=resnext-101

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 -u fine-tune.py \
    --gpus              0,1,2,3,4,5,6,7 \
    --kv-store          device \
    --pretrained-model  ${ROOT}/train/model/DPNs/dpn68-5k \
    --model-prefix      ${ROOT}/train/checkpoints/dpn68-5k/dpn68-newset \
    --fix-last-layer \
    --layer-before-fullc flatten \
    --data-train        ${ROOT}/data/train_split_train.rec \
    --data-val          ${ROOT}/data/train_split_val.rec \
    --image-shape       3,180,180 \
    --data-nthread      6 \
    --optimizer         adam \
    --lr                0.0001 \
    --lr-factor         0.2 \
    --lr-step-epochs    6,10,13,15 \
    --disp-batches      100 \
    --num-epoch         15 \
    --load-epoch        0 \
    --mom               0.9 \
    --wd                0.00004 \
    --top-k             5 \
    --batch-size        1024 \
    --num-classes       5270 \
    --num-examples      ${C5270_NUM_EXAMPLES} \
    --rgb-mean          0,0,0 \
    --rgb-scale         1.0 \
    --random-crop       1 \
    --random-mirror     1 \
    --max-random-h      20 \
    --max-random-s      20 \
    --max-random-l      20 \
    --max-random-rotate-angle   0 \
    --max-random-shear-ratio    0 \
    --max-random-aspect-ratio   0 \
    --max-random-scale  1.0
