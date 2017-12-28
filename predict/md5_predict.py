
import sys
import os
import logging
import pickle
import hashlib
from collections import defaultdict, Counter

from tqdm import tqdm
import bson
import coloredlogs
coloredlogs.install(level=logging.INFO)


def main(args):
    # logging.info('loading md5 dict...')
    # md5_dict = pickle.load(open(args.md5_dict_pkl, 'rb')) if args.md5_dict_pkl else dict()
    # logging.info('loaded {} images'.format(len(md5_dict)))

    train_data = bson.decode_file_iter(open(args.train_bson, 'rb'))

    product_dict = defaultdict(Counter)
    image_dict = defaultdict(Counter)
    for i, d in tqdm(enumerate(train_data), unit='products'):
        product_id = d.get('_id')
        category_id = d.get('category_id', None)  # This won't be in Test data

        hash_list = []
        for img in d['imgs']:
            h = hashlib.md5(img['picture']).hexdigest()
            hash_list.append(h)
            image_dict[h][category_id] += 1
        product_hash = ','.join(sorted(set(hash_list)))
        product_dict[product_hash][category_id] += 1

    test_data = bson.decode_file_iter(open(args.test_bson, 'rb'))
    for i, d in tqdm(enumerate(test_data), unit='products'):
        product_id = d.get('_id')
        category_id = d.get('category_id', None)  # This won't be in Test data

        hash_list = []
        for img in d['imgs']:
            h = hashlib.md5(img['picture']).hexdigest()
            hash_list.append(h)

        product_hash = ','.join(sorted(set(hash_list)))
        if product_hash in product_dict:
            if len(product_dict[product_hash]) == 1:
                most_id, most_cnt = product_dict[product_hash].most_common(1)[0]
                pred = most_id
                args.output.write('{},{}\n'.format(product_id, pred))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-bson', type=str, required=True)
    parser.add_argument('--test-bson', type=str, required=True)
    parser.add_argument('--md5-dict-pkl', type=str, required=True)
    parser.add_argument('--output', type=argparse.FileType('w'), default='-')
    parser.add_argument('--cut', type=int, default=0)
    args = parser.parse_args()

    main(args)

