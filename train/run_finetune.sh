#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount/

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 fine-tune.py \
    --gpus              0,1,2,3,4,5,6,7 \
    --kv-store          device \
    --pretrained-model  $ROOT/train/checkpoints/imagenet1k-resnext-50 \
    --model-prefix      $ROOT/train/checkpoints/resnext-50/imagenet1k-resnext-50 \
    --fix-last-layer \
    --data-train        $ROOT/data/cdiscount_train.rec \
    --data-val          $ROOT/data/cdiscount_val.rec  \
    --image-shape       3,180,180 \
    --data-nthread      6 \
    --lr                0.05 \
    --lr-factor         0.1 \
    --lr-step-epochs    6,14,17 \
    --num-epoch         15 \
    --load-epoch        11 \
    --top-k             5 \
    --batch-size        512 \
    --num-classes       5270 \
    --optimizer         nag \
    --rgb-mean          0,0,0 \
    --max-random-h      0 \
    --max-random-s      0 \
    --max-random-l      0 \
    --max-random-rotate-angle   0 \
    --max-random-shear-ratio    0 \
    --max-random-aspect-ratio   0 \
    --max-random-scale  1.0 \
    --num-examples      11752908
