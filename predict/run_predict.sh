#!/usr/bin/env bash

ROOT=/home/deploy/dylan/projects/temp/kaggle-cdiscount

# create prediction output
python3 -u ${ROOT}/predict/predict.py \
    --zmq-port      18400 \
    --bson          ${ROOT}/data/test.bson \
    --csv           ${ROOT}/data/category_names.csv \
    --params        ${ROOT}/train/M14/dpn107-0020.params \
                    ${ROOT}/train/M13/se-resnext-101-64x4d-seed1-0013.params \
                    ${ROOT}/train/M12/dpn131-seed1-0017.params \
                    ${ROOT}/train/M11/se-resnext-101-64x4d-0015.params \
                    ${ROOT}/train/M10/resnext-101-0019.params \
                    ${ROOT}/train/M09/resnext-101-64x4d-seed1-0015.params \
                    ${ROOT}/train/M08/dpn131-0018.params \
                    ${ROOT}/train/M07/resnext-101-64x4d-0015.params \
                    ${ROOT}/train/M06/dpn92-0011.params \
                    ${ROOT}/train/M05/dpn98-0020.params \
                    ${ROOT}/train/M04/densenet-161-0020.params \
                    ${ROOT}/train/M03/dpn92-unique-02-0013.params \
                    ${ROOT}/train/M02/dpn92-unique-01-0013.params \
                    ${ROOT}/train/M01/resnext-101-unique-0023.params \
    --symbol        ${ROOT}/train/M14/dpn107-symbol.json \
                    ${ROOT}/train/M13/se-resnext-101-64x4d-seed1-symbol.json \
                    ${ROOT}/train/M12/dpn131-seed1-symbol.json \
                    ${ROOT}/train/M11/se-resnext-101-64x4d-symbol.json \
                    ${ROOT}/train/M10/resnext-101-symbol.json \
                    ${ROOT}/train/M09/resnext-101-64x4d-seed1-symbol.json \
                    ${ROOT}/train/M08/dpn131-symbol.json \
                    ${ROOT}/train/M07/resnext-101-64x4d-symbol.json \
                    ${ROOT}/train/M06/dpn92-symbol.json \
                    ${ROOT}/train/M05/dpn98-symbol.json \
                    ${ROOT}/train/M04/densenet-161-symbol.json \
                    ${ROOT}/train/M03/dpn92-unique-02-symbol.json \
                    ${ROOT}/train/M02/dpn92-unique-01-symbol.json \
                    ${ROOT}/train/M01/resnext-101-unique-symbol.json \
    --batch-size    512 \
    --data-shape    3,180,180 \
    --gpus          0,1,2,3,4,5,6,7 \
    --num-procs     24 \
    --md5-dict-pkl  ${ROOT}/data/train_md5_dict.pkl \
    --md5-dict-type unique \
    --md5-mode      0 \
    --multi-view    1 \
    --ensembles     8 11 14 \
    --output        output_ensemble14_20171211.csv

# create output using md5 dictionary
python3 -u ${ROOT}/predict/md5_predict.py \
    --train-bson    ${ROOT}/data/train.bson \
    --test-bson     ${ROOT}/data/test.bson \
    --md5-dict-pkl  ${ROOT}/data/train_md5_dict.pkl \
    --output        ${ROOT}/predict/md5_output.txt


# overwrite
python3 -u ${ROOT}/predict/postprocess_product.py \
    --src           output_ensemble14_20171211.csv.e14.m1 \
    --dst           output_ensemble14_20171211.csv.e14.m1.md5 \
    --md5           ${ROOT}/predict/md5_output.txt
