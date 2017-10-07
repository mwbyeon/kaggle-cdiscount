#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

C5270_NUM_EXAMPLES=11752908
REFINE_NUM_EXAMPLES=7310313
MODEL=resnext-101

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 -u fine-tune.py \
    --gpus              0,1,2,3,4,5,6,7 \
    --kv-store          device \
    --pretrained-model  ${ROOT}/train/checkpoints/${MODEL}/imagenet1k-${MODEL} \
    --model-prefix      ${ROOT}/train/checkpoints/${MODEL}/imagenet1k-${MODEL} \
    --fix-last-layer \
    --data-train        ${ROOT}/data/refined_train.rec \
    --data-val          ${ROOT}/data/refined_val.rec \
    --image-shape       3,180,180 \
    --data-nthread      6 \
    --optimizer         adam \
    --lr                0.0005 \
    --lr-factor         0.2 \
    --lr-step-epochs    8,18,20 \
    --num-epoch         20 \
    --load-epoch        16 \
    --mom               0.9 \
    --wd                0.00004 \
    --top-k             5 \
    --batch-size        512 \
    --num-classes       5270 \
    --num-examples      ${REFINE_NUM_EXAMPLES} \
    --rgb-mean          0,0,0 \
    --rgb-scale         1.0 \
    --random-crop       1 \
    --random-mirror     1 \
    --max-random-h      0 \
    --max-random-s      0 \
    --max-random-l      0 \
    --max-random-rotate-angle   0 \
    --max-random-shear-ratio    0 \
    --max-random-aspect-ratio   0 \
    --max-random-scale  1.0
