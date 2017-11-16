# -*- coding: utf-8 -*-

import sys
import os
import hashlib
import pickle
import random
import logging
import coloredlogs
coloredlogs.install(level=logging.INFO)

import mxnet as mx
from collections import Counter
import bson
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.category import get_category_dict
from data import utils


def read_images(args):
    cate1_dict, cate2_dict, cate3_dict = get_category_dict()

    logging.info('read bson file: {}'.format(args.bson))
    total_count = utils.get_bson_count(args.bson)
    data = bson.decode_file_iter(open(args.bson, 'rb'))

    idx = 0
    category_counter = Counter()
    md5_dict = pickle.load(open(args.md5_dict_pkl, 'rb')) if args.md5_dict_pkl else None
    if md5_dict is None:
        logging.info('md5_dict is not provided')
    else:
        logging.info('md5_dict has {} keys'.format(len(md5_dict)))

    used_md5_set = set()
    for i, prod in tqdm(enumerate(data), unit='products', total=total_count):
        product_id = prod.get('_id')
        category_id = prod.get('category_id', None)  # This won't be in Test data
        images = prod.get('imgs')

        for img in images:
            img_bytes = img['picture']
            h = hashlib.md5(img_bytes).hexdigest()
            if md5_dict is not None and len(md5_dict.get(h, [])) != 1:  # save only single label
                continue
            if args.unique_md5:
                if h in used_md5_set:
                    continue
                used_md5_set.add(h)

            item = None
            if category_id is None:
                item = (idx, img_bytes, -1)
            elif category_counter[category_id] < args.under_sampling:
                if args.cate_type == 1:
                    class_id = cate1_dict[(cate3_dict[category_id]['names'][0],)]['child_cate3'][category_id]
                elif args.cate_type == 3:
                    class_id = cate3_dict[category_id]['cate3_class_id']
                else:
                    raise ValueError('invalid cate type: {}'.format(args.cate_type))
                item = (idx, img_bytes, class_id)
                category_counter[category_id] += 1

            if item is not None:
                idx += 1
                yield item  # id, img_bytes, label


def main(args):
    if os.path.exists(args.out_rec):
        raise FileExistsError(args.out_rec)

    logging.info('write rec file to {}'.format(args.out_rec))
    rec_writer = mx.recordio.MXRecordIO(args.out_rec, 'w')
    count = 0
    random.seed(args.random_seed)
    images_buf = []
    for item in tqdm(read_images(args), unit='images'):
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)
        rec = mx.recordio.pack(header, item[1])
        images_buf.append(rec)
        if len(images_buf) >= args.shuffle_size:
            logging.info('shuffle {} images'.format(len(images_buf)))
            item_perm = [i for i in range(len(images_buf))]
            random.shuffle(item_perm)
            logging.info('write {} images'.format(len(images_buf)))
            for i in tqdm(item_perm, total=len(item_perm), unit='images', desc='write to rec file'):
                rec_writer.write(images_buf[i])
            images_buf = []
        count += 1

    logging.info('shuffle {} images'.format(len(images_buf)))
    item_perm = [i for i in range(len(images_buf))]
    random.shuffle(item_perm)
    logging.info('write {} images'.format(len(images_buf)))
    for i in tqdm(item_perm, total=len(item_perm), unit='images', desc='write to rec file'):
        rec_writer.write(images_buf[i])

    rec_writer.close()
    logging.info('complete. {} images'.format(count))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bson', type=str, required=True)
    parser.add_argument('--out-rec', type=str, required=True)
    parser.add_argument('--md5-dict-pkl', type=str, default=None)
    parser.add_argument('--cate-type', type=int, default=3)
    parser.add_argument('--shuffle-size', type=int, default=99999999)
    parser.add_argument('--random-seed', type=int, default=0xC0FFEE)
    parser.add_argument('--unique-md5', action='store_true')
    parser.add_argument('--under-sampling', type=int, default=99999999)
    args = parser.parse_args()

    main(args)
