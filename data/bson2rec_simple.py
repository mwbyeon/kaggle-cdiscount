# -*- coding: utf-8 -*-

import sys
import os
import hashlib
import pickle
import random
import logging
from multiprocessing import Process
import coloredlogs
coloredlogs.install(level=logging.INFO)

from joblib import Parallel, delayed
import cv2
import numpy as np
import mxnet as mx
from collections import Counter
import bson
from tqdm import tqdm
import zmq


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


def stitch_product(prod):
    product_id = prod.get('_id')
    category_id = prod.get('category_id', None)  # This won't be in Test data
    images = prod.get('imgs')
    decoded = [cv2.imdecode(np.fromstring(images[x % len(images)]['picture'], np.uint8), cv2.IMREAD_COLOR) for x in
               range(4)]
    stitched = np.zeros((360, 360, 3), dtype=decoded[0].dtype)
    stitched[:180, :180, :] = decoded[0]
    stitched[180:, :180, :] = decoded[1]
    stitched[:180, 180:, :] = decoded[2]
    stitched[180:, 180:, :] = decoded[3]
    img_bytes = cv2.imencode('.jpg', stitched)[1].tostring()
    return category_id, img_bytes


def _func_reader(args):
    logging.info('read bson file: {}'.format(args.bson))
    data = bson.decode_file_iter(open(args.bson, 'rb'))

    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.set_hwm(0)
    zmq_socket.bind('tcp://0.0.0.0:{port}'.format(port=args.zmq_port))
    logging.info('_func_reader started')

    product_count = 0
    for i, prod in enumerate(data):
        zmq_socket.send_pyobj(prod)
        product_count += 1

    for _ in range(args.num_procs):
        zmq_socket.send_pyobj(None)

    logging.info('reader finished (product: {})'.format(product_count))


def _func_proc(args):
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PULL)
    zmq_socket.set_hwm(0)
    zmq_socket.connect('tcp://0.0.0.0:{port}'.format(port=args.zmq_port))
    logging.info('processor started')

    ext_socket = context.socket(zmq.PUSH)
    ext_socket.set_hwm(0)
    ext_socket.connect('tcp://0.0.0.0:{port}'.format(port=args.zmq_port+1))

    while True:
        prod = zmq_socket.recv_pyobj()
        if prod is None:
            ext_socket.send_pyobj(None)
            return
        ext_socket.send_pyobj(stitch_product(prod))


def read_products(args):
    cate1_dict, cate2_dict, cate3_dict = get_category_dict()

    context = zmq.Context()
    ext_socket = context.socket(zmq.PULL)
    ext_socket.set_hwm(0)
    ext_socket.bind('tcp://0.0.0.0:{port}'.format(port=args.zmq_port+1))
    logging.info('read_products started (port: {port})'.format(port=args.zmq_port+1))

    total_count = utils.get_bson_count(args.bson)

    proc_reader = Process(target=_func_reader, args=(args,))
    proc_processors = [Process(target=_func_proc, args=(args,)) for _ in range(args.num_procs)]

    proc_reader.start()
    [x.start() for x in proc_processors]

    bar = tqdm(total=total_count, unit='products')
    idx = 0
    term_count = 0
    finished = False
    while not finished:
        item = ext_socket.recv_pyobj()
        if item is None:
            term_count += 1
            if term_count == args.num_procs:
                finished = True
            continue

        category_id, img_bytes = item
        class_id = cate3_dict[category_id]['cate3_class_id']
        item = (idx, img_bytes, class_id)

        bar.update(n=1)
        idx += 1
        yield item  # id, img_bytes, label

    proc_reader.join()
    [x.join() for x in proc_processors]


def main(args):
    if os.path.exists(args.out_rec):
        raise FileExistsError(args.out_rec)

    logging.info('write rec file to {}'.format(args.out_rec))
    rec_writer = mx.recordio.MXRecordIO(args.out_rec, 'w')
    count = 0
    random.seed(args.random_seed)
    images_buf = []
    for item in globals()[args.func](args):
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
            del images_buf[:]
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
    parser.add_argument('--func', type=str, default='read_images')

    parser.add_argument('--num-procs', type=int, default=1)
    parser.add_argument('--zmq-port', type=int, default=18300)
    args = parser.parse_args()

    main(args)
