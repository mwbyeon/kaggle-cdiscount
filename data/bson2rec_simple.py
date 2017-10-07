# -*- coding: utf-8 -*-

import csv
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


def category_csv_to_dict(category_csv):
    cate2cid, cid2cate = dict(), dict()
    with open(category_csv, 'r') as reader:
        csvreader = csv.reader(reader, delimiter=',', quotechar='"')
        for i, row in enumerate(csvreader):
            if i == 0:
                continue
            try:
                cateid, cate1, cate2, cate3 = row
                cid = len(cate2cid)
                cate2cid[int(cateid)] = cid
                cid2cate[cid] = int(cateid)
            except Exception as e:
                logging.error('cannot parse line: {}, {}'.format(row, e))
    logging.info('{} categories in {}'.format(len(cate2cid), category_csv))
    return cate2cid, cid2cate


def get_bson_count(bson_path):
    logging.info('counting bson items...')
    count = 0
    with open(bson_path, 'rb') as reader:
        data = bson.decode_file_iter(reader)
        for _ in tqdm(data, unit='products'):
            count += 1
    return count


def read_images(args):
    cate2cid, cid2cate = category_csv_to_dict(args.category_csv)

    logging.info('read bson file: {}'.format(args.bson))
    total_count = get_bson_count(args.bson)
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
            if md5_dict is not None and h not in md5_dict:
                continue
            if args.unique_md5:
                if h in used_md5_set:
                    continue
                used_md5_set.add(h)

            item = None
            if category_id is None:
                item = (idx, img_bytes, -1)
            elif category_counter[category_id] < args.under_sampling:
                item = (idx, img_bytes, cate2cid[category_id])
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
    parser.add_argument('--category-csv', type=str, required=True)
    parser.add_argument('--out-rec', type=str, required=True)
    parser.add_argument('--md5-dict-pkl', type=str, default=None)
    parser.add_argument('--shuffle-size', type=int, default=99999999)
    parser.add_argument('--random-seed', type=int, default=0xC0FFEE)
    parser.add_argument('--unique-md5', action='store_true')
    parser.add_argument('--under-sampling', type=int, default=99999999)
    args = parser.parse_args()

    main(args)
