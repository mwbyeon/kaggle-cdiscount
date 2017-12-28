
*I'm not fluent in English. Please let me know if I need to fix it.*

# Kaggle Cdiscount’s Image Classification Challenge
  * 3rd place solution for Cdiscount’s Image Classification Challenge.
  * https://www.kaggle.com/c/cdiscount-image-classification-challenge

## Prerequisite
  * Python 3.6.2
  * MXNet 0.12.0 (GPU enabled)
  * and need GPU Machines as many as possible...
  * run `$ pip install -r requirements` to install required packages.

## Prepare data files for training

#### Dataset Preview
  * train.bson (58.2 GB)
    * 5270 Categories
    * 7,069,896 products, 12,371,293 images (each product contains between 1-4 images).
    * imbalanced dataset
    ![Cateogry Distribution](assets/category_distribution.png)
  * test.bson (14.5GB)
  * see https://www.kaggle.com/c/cdiscount-image-classification-challenge/data

#### Split the BSON file to Training and Validation
  * split products in the `train.bson` to `train_train.bson` and `train_valid.bson`
    * randomly selected with seed
    * Training(0.95) : Validation(0.05)
  * see `data/split_train_bson.py` and `data/run_split_train.sh`

#### Convert BSON files to `.rec` file 
  * MXNet supports efficient data loaders(`.rec` format) for fast training
    (see https://mxnet.incubator.apache.org/architecture/note_data_loading.html)
  * create `.rec` files for training CNN models
    - assign unique `class_id`(0-based) to each category.
    - images in the same product are assigned same `class_id`.
  * see [data/bson2rec_simple.py](data/bson2rec_simple.py) and [data/run_bson2rec_trainval.sh](data/run_bson2rec_trainval.sh)

#### Create different datasets
  * `DATASET_A`
    - split products to 0.95(training) : 0.05(validation)
    - random seed: 12648430 (`0xC0FFEE`)
  * `DATASET_B`
    - split products to 0.95(training) : 0.05(validation)
    - random seed: 1 `0x1`
  * `DATASET_C`
    - split products to 0.95(training) : 0.05(validation)
    - random seed: 12648430 (`0xC0FFEE`)
    - remove duplicated images from `DATASET_A` (it can reduce training time)


## Train CNN Models
  * trained 14 CNN models
  
    | No. | network              | transfer from        | dataset   | epochs | single val-acc |
    |-----|----------------------|----------------------|-----------|--------|----------------|
    | M01 | ResNext-101          | ImageNet-1k          | DATASET_C | 23     | 0.660329       |
    | M02 | DPNs-92              | ImageNet-1k          | DATASET_C | 13     | 0.662091       |
    | M03 | DPNs-92              | ImageNet-1k          | DATASET_C | 13     | 0.663739       |
    | M04 | DenseNet-161         | ImageNet-1k          | DATASET_A | 20     | 0.726930       |
    | M05 | DPNs-98              | ImageNet-1k          | DATASET_A | 20     | 0.732641       |
    | M06 | DPNs-92              | ImageNet-1k          | DATASET_A | 11     | 0.734462       |
    | M07 | ResNext-101-64x4d    | ImageNet-1k          | DATASET_A | 15     | 0.735808       |
    | M08 | DPNs-131             | ImageNet-1k          | DATASET_A | 18     | 0.736697       |
    | M09 | ResNext-101-64x4d    | ImggeNet-1k          | DATASET_B | 15     | 0.737427       |
    | M10 | ResNext-101          | ImageNet-1k          | DATASET_A | 19     | 0.738542       |
    | M11 | SE-ResNext-101-64x4d | M10                  | DATASET_A | 15     | 0.739272       |
    | M12 | DPNs-131             | ImageNet-1k          | DATASET_B | 17     | 0.742388       |
    | M13 | SE-ResNext-101-64x4d | M09                  | DATASET_B | 13     | 0.743221       |
    | M14 | DPNs-107             | ImageNet-1k (Link)   | DATASET_A | 20     | 0.743781       |

    * ResNext: https://arxiv.org/abs/1611.05431
    * SE-ResNext: https://arxiv.org/abs/1709.01507
    * DPNs: https://arxiv.org/abs/1707.01629
    * DenseNet: https://arxiv.org/abs/1608.06993

#### Hyper-parameters
  * Batch Size: 512 (using 8-GPUs)
  * Optimizer (in most cases)
    * [NADAM](https://mxnet.incubator.apache.org/api/python/optimization.html#mxnet.optimizer.Nadam) (it's better than SGD, NAG and ADAM optimizer for this dataset)
    * learning rate
      * 0.000100: 1~10 epochs
      * 0.000020: 11~13 epochs
      * 0.000004: 13~ epochs
  * label smoothing
    * smooth alpha: 0.1

#### Data augmentation
  * input: 180x180x3
  * did not use random crop
  * use only random flip (it's enough for training on 15~20 epochs)

## Experiments
#### dropout
  * did not use dropout after GAP layer
  
  |  Dropout Ratio  | Local Train (Top-1) | Local Validation (Top-1) |
  |-----------------|---------------------|--------------------------|
  | **0.0**         | **0.788686**        | **0.693604**             |
  | 0.2             | 0.743764            | 0.689954                 |

  * see [train/train_model.py](train/train_model.py#L48-L49) and [scripts/logs](train/experiments/dropout) for more details.

#### input size of image
  * The larger the image size, the higher the accuracy.
  * However, I used 180x180 because the larger the image size, the slower the learning.
  
  | image size | train-acc | valid-acc | training speed |
  |------------|-----------|-----------|----------------|
  | 160 x 160  | 0.753975  | 0.678730  | 1130/sec       |
  | **180 x 180**  | **0.750308**  | **0.680361**  | **872/sec**        |
  | 192 x 192  | 0.751713  | 0.680706  | 884/sec        |
  | 224 x 224  | 0.748252  | 0.681723  | 647/sec        |
  
  * see [scripts/logs](train/experiments/image_size) for more details.

#### label smoothing
  * Label smoothing(`0.1`) increases accuracy.

  | Smooth alpha | train-acc | val-acc  |
  |--------------|-----------|----------|
  | 0.0          | 0.785019  | 0.691917 |
  | **0.1**          | **0.779468**  | **0.694035** |
  | 0.2          | 0.770593  | 0.692766 |

  * see [code](train/train_model.py#L52) and [scripts/logs](train/experiments/label_smoothing) for more details.

#### augmentation
  * Just flip is enough.

  | Input Size      | Option     | train-acc  | val-acc  |
  |-----------------|------------|------------|----------|
  | 180 x 180       | No Aug.    | 0.862802   | 0.688090 |
  | **180 x 180**       | **Flip**       | **0.787349**   | **0.692491** |
  | 180 x 180       | Flip + HSL | 0.774842   | 0.690422 |
  | 224 x 224 (NN)  | Flip       | 0.789405   | 0.695116 |

  * see [train/common/data.py](train/common/data.py#L34-L57) and [scripts/logs](train/experiments/augmentation) for more details.


## Predict
#### Ensemble of images in a product
  * Arithmetic Mean (sum of probabilities of each images)
  * see [predict/predict.py]() and [script](predict/run_predict.sh)

#### Ensemble of models
  * Arithmetic Mean (sum of probabilities of each models)

#### Post-processing to improve accuracy using MD5 hash of products
  * calculate the hash of each products in the training set.
  * the hash of the product is a set of md5 hash of each images.
  * create a dictionary of (hash, category)
  * for testing time, if exists same hash in the dictionary,
    the category of this product is over-written by the dictionary.
  * see [predict/md5_predict.py](predict/md5_predict.py)
  * also you can implement this logic in the `predict.py`. 

#### Accelerate prediction speed
  * needs multi-processing for faster prediction. but `multiprocessing.Queue` module is very slow.
  * I use **ZeroMQ**(https://github.com/zeromq/pyzmq) instead of `multiprocessing.Queue` for process communication.

## Results (Kaggle Leaderboard)
  * https://www.kaggle.com/c/cdiscount-image-classification-challenge/leaderboard

## Appendix
  * see [this slide](assets/Kaggle_Cdiscount_Image_Classification.pdf) for more details.
