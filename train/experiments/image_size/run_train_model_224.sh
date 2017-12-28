#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 -u ${ROOT}/train/train_model.py \
    --gpus              6,7 \
    --kv-store          device \
    --symbol            ${ROOT}/train/model/imagenet1k-resnet-34-symbol.json \
    --params            ${ROOT}/train/model/imagenet1k-resnet-34-0000.params \
    --model-prefix      ${ROOT}/train/experiments/augmentation/checkpoints/resnet-34-224 \
    --data-train        ${ROOT}/data/train_split_A_train.rec \
    --data-val          ${ROOT}/data/train_split_A_val.rec \
    --feature-layer     flatten0 \
    --resize            224 \
    --inter-method      1 \
    --image-shape       3,224,224 \
    --data-nthread      6 \
    --optimizer         sgd \
    --lr                0.005 \
    --lr-factor         0.2 \
    --lr-step-epochs    8,12,15,17 \
    --disp-batches      100 \
    --num-epoch         17 \
    --load-epoch        0 \
    --mom               0.9 \
    --wd                0.00004 \
    --top-k             5 \
    --batch-size        512 \
    --num-classes       5270 \
    --dropout-ratio     0.0 \
    --smooth-alpha      0.0 \
    --num-examples      11754490 \
    --rgb-mean          0,0,0 \
    --rgb-scale         1.0 \
    --random-crop       1 \
    --random-mirror     1 \
    --max-random-h      0 \
    --max-random-s      0 \
    --max-random-l      0 \
    --min-random-scale  1.0 \
    --max-random-scale  1.0 \
    --max-random-rotate-angle   0 \
    --max-random-shear-ratio    0 \
    --max-random-aspect-ratio   0

