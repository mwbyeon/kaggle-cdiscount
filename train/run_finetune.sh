#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/kaggle-cdiscount

C2048_NUM_EXAMPLES=11095745
C5270_NUM_EXAMPLES=11752908
UNDER10K_NUM_EXAMPLES=6859873
MODEL=dpn92-5k

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 -u fine-tune.py \
    --gpus              0,1,2,3,4,5,6,7 \
    --kv-store          device \
    --pretrained-model  ${ROOT}/train/model/DPNs/dpn92-5k \
    --model-prefix      ${ROOT}/train/checkpoints/dpn92-5k/dpn92-5k \
    --layer-before-fullc flatten \
    --data-train        ${ROOT}/data/cdiscount_c5270_train.rec \
    --data-val          ${ROOT}/data/cdiscount_c5270_val.rec \
    --image-shape       3,180,180 \
    --data-nthread      6 \
    --optimizer         adam \
    --lr                0.0001 \
    --lr-factor         0.2 \
    --lr-step-epochs    8,16,20 \
    --mom               0.9 \
    --wd                0.00004 \
    --num-epoch         20 \
    --load-epoch        0 \
    --top-k             5 \
    --batch-size        512 \
    --num-classes       5270 \
    --num-examples      ${C5270_NUM_EXAMPLES} \
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
