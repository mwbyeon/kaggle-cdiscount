# -*- coding: utf-8 -*-

import os
import random
import logging
import coloredlogs
coloredlogs.install(level=logging.INFO)

import bson
from tqdm import tqdm


def products_iter(input_bson_path, product_ids):
    with open(input_bson_path, 'rb') as reader:
        data = bson.decode_file_iter(reader)
        for i, prod in tqdm(enumerate(data), unit='products', total=args.num_products, disable=True):
            prod_id = prod.get('_id')
            if prod_id in product_ids:
                yield prod


def encode_dict_list(products, output_bson_path, total=None):
    with open(output_bson_path, 'wb') as writer:
        for i, prod in tqdm(enumerate(products), unit='products', total=total):
            obj = bson._dict_to_bson(prod, False, bson.DEFAULT_CODEC_OPTIONS)
            writer.write(obj)


def main(args):
    if os.path.exists(args.save_train_bson):
        raise FileExistsError(args.save_train_bson)
    if os.path.exists(args.save_val_bson):
        raise FileExistsError(args.save_val_bson)

    logging.info('aggregating id of products...')
    product_ids = list()
    with open(args.input_bson, 'rb') as reader:
        data = bson.decode_file_iter(reader)
        for x in tqdm(data, unit='products', total=args.num_products):
            product_ids.append(x.get('_id'))

    logging.info('shuffle train and val ids...')
    num_val = int(len(product_ids) * args.val_ratio)
    random.seed(args.random_seed)
    random.shuffle(product_ids)
    val_product_ids = set(random.sample(product_ids, num_val))
    train_product_ids = set(product_ids) - val_product_ids

    logging.info('writing {} products for validation: {}'.format(len(val_product_ids), args.save_val_bson))
    encode_dict_list(products=products_iter(args.input_bson, val_product_ids),
                     output_bson_path=args.save_val_bson,
                     total=len(val_product_ids))

    logging.info('writing {} products for training: {}'.format(len(train_product_ids), args.save_train_bson))
    encode_dict_list(products=products_iter(args.input_bson, train_product_ids),
                     output_bson_path=args.save_train_bson,
                     total=len(train_product_ids))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-bson', type=str, required=True)
    parser.add_argument('--num-products', type=int, default=7069896)
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument('--random-seed', type=int, default=0xC0FFEE)
    parser.add_argument('--save-train-bson', type=str, required=True)
    parser.add_argument('--save-val-bson', type=str, required=True)
    args = parser.parse_args()

    main(args)
