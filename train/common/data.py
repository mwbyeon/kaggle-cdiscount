# -*- coding: utf-8 -*-


import os
import sys
import time
import logging
from multiprocessing import Process
import random

import cv2
import mxnet as mx
from mxnet.io import DataBatch, DataIter, DataDesc
import numpy as np
import zmq
import bson
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.category import get_category_dict
from data import utils


def add_data_args(parser):
    data = parser.add_argument_group('Data', 'the input images')
    data.add_argument('--label-width', type=int, default=1)
    data.add_argument('--data-train', type=str, help='the training data')
    data.add_argument('--data-val', type=str, help='the validation data')
    data.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',
                      help='a tuple of size 3 for the mean rgb')
    data.add_argument('--rgb-scale', type=float, default=1.0)
    data.add_argument('--pad-size', type=int, default=0,
                      help='padding the input image')
    data.add_argument('--resize', type=int, default=-1,
                      help='Down scale the shorter edge to a new size before applying other augmentations')
    data.add_argument('--inter-method', type=int, default=9,
                      help='0-NN, 1-bilinear, 2-cubic, 3-area, 4-lanczos4, 9-auto, 10-rand.')
    data.add_argument('--image-shape', type=str,
                      help='the image shape feed into the network, e.g. (3,224,224)')
    data.add_argument('--num-classes', type=int, help='the number of classes')
    data.add_argument('--num-examples', type=int, help='the number of training examples')
    data.add_argument('--data-name', type=str, default='data')
    data.add_argument('--label-name', type=str, default='softmax_label')
    data.add_argument('--data-nthreads', type=int, default=4,
                      help='number of threads for data decoding')
    data.add_argument('--benchmark', type=int, default=0,
                      help='if 1, then feed the network with synthetic data')
    return data


def add_data_aug_args(parser):
    aug = parser.add_argument_group(
        'Image augmentations', 'implemented in src/io/image_aug_default.cc')
    aug.add_argument('--random-crop', type=int, default=1,
                     help='if or not randomly crop the image')
    aug.add_argument('--random-mirror', type=int, default=1,
                     help='if or not randomly flip horizontally')
    aug.add_argument('--max-random-h', type=int, default=0,
                     help='max change of hue, whose range is [0, 180]')
    aug.add_argument('--max-random-s', type=int, default=0,
                     help='max change of saturation, whose range is [0, 255]')
    aug.add_argument('--max-random-l', type=int, default=0,
                     help='max change of intensity, whose range is [0, 255]')
    aug.add_argument('--max-random-aspect-ratio', type=float, default=0,
                     help='max change of aspect ratio, whose range is [0, 1]')
    aug.add_argument('--max-random-rotate-angle', type=int, default=0,
                     help='max angle to rotate, whose range is [0, 360]')
    aug.add_argument('--max-random-shear-ratio', type=float, default=0,
                     help='max ratio to shear, whose range is [0, 1]')
    aug.add_argument('--max-random-scale', type=float, default=1,
                     help='max ratio to scale')
    aug.add_argument('--min-random-scale', type=float, default=1,
                     help='min ratio to scale, should >= img_size/input_shape. otherwise use --pad-size')
    return aug


def set_data_aug_level(aug, level):
    if level >= 1:
        aug.set_defaults(random_crop=1, random_mirror=1)
    if level >= 2:
        aug.set_defaults(max_random_h=36, max_random_s=50, max_random_l=50)
    if level >= 3:
        aug.set_defaults(max_random_rotate_angle=10, max_random_shear_ratio=0.1, max_random_aspect_ratio=0.25)


class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, dtype):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size, ])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))
        self.label = mx.nd.array(label, dtype=self.dtype, ctx=mx.Context('cpu_pinned', 0))

    def __iter__(self):
        return self

    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]

    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label', (self.batch_size,), self.dtype)]

    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_iter = 0


def get_rec_iter(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    if 'benchmark' in args and args.benchmark:
        data_shape = (args.batch_size,) + image_shape
        train = SyntheticDataIter(args.num_classes, data_shape, 500, np.float32)
        return train, None
    if kv:
        rank, nworker = (kv.rank, kv.num_workers)
    else:
        rank, nworker = (0, 1)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    train = mx.io.ImageRecordIter(
        path_imgrec=args.data_train,
        label_width=args.label_width,
        mean_r=rgb_mean[0],
        mean_g=rgb_mean[1],
        mean_b=rgb_mean[2],
        scale=args.rgb_scale,
        data_name=args.data_name,
        label_name=args.label_name,
        resize=args.resize,
        inter_method=args.inter_method,
        data_shape=image_shape,
        batch_size=args.batch_size,
        rand_crop=args.random_crop,
        pad=args.pad_size,
        fill_value=127,
        min_random_scale=args.min_random_scale,
        max_random_scale=args.max_random_scale,
        max_aspect_ratio=args.max_random_aspect_ratio,
        random_h=args.max_random_h,
        random_s=args.max_random_s,
        random_l=args.max_random_l,
        max_rotate_angle=args.max_random_rotate_angle,
        max_shear_ratio=args.max_random_shear_ratio,
        rand_mirror=args.random_mirror,
        preprocess_threads=args.data_nthreads,
        shuffle=True,
        num_parts=nworker,
        part_index=rank)
    if args.data_val is None:
        return train, None

    val = mx.io.ImageRecordIter(
        path_imgrec=args.data_val,
        label_width=args.label_width,
        mean_r=rgb_mean[0],
        mean_g=rgb_mean[1],
        mean_b=rgb_mean[2],
        scale=args.rgb_scale,
        data_name=args.data_name,
        label_name=args.label_name,
        batch_size=args.batch_size,
        resize=args.resize,
        inter_method=args.inter_method,
        data_shape=image_shape,
        preprocess_threads=args.data_nthreads,
        rand_crop=False,
        rand_mirror=False,
        num_parts=nworker,
        part_index=rank)
    return train, val


class CategoricalImageRecordIter(mx.io.DataIter):
    def __init__(self, rec_iter):
        super(CategoricalImageRecordIter, self).__init__()
        self._rec_iter = rec_iter
        cate1_dict, cate2_dict, cate3_dict = get_category_dict()
        self._cate3_dict = cate3_dict

    def next(self):
        if self._rec_iter.iter_next():
            batch = self._rec_iter.next()
            cate3_label = batch.label[0].asnumpy()
            cate1_label = np.zeros(cate3_label.shape, dtype=np.float32)
            cate2_label = np.zeros(cate3_label.shape, dtype=np.float32)
            for i, x in enumerate(cate3_label):
                cate1_label[i] = self._cate3_dict[int(x)]['cate1_class_id']
                cate2_label[i] = self._cate3_dict[int(x)]['cate2_class_id']
            batch.label = [mx.nd.array(cate1_label, ctx=mx.Context('cpu_pinned', 0)),
                           mx.nd.array(cate2_label, ctx=mx.Context('cpu_pinned', 0)),
                           mx.nd.array(cate3_label, ctx=mx.Context('cpu_pinned', 0))
                           ]
            batch.provide_label = self.provide_label
            return batch
        else:
            raise StopIteration

    def reset(self):
        self._rec_iter.reset()

    @property
    def provide_data(self):
        return self._rec_iter.provide_data

    @property
    def provide_label(self):
        # (batch_size,)
        label_shape = self._rec_iter.provide_label[0].shape
        return [DataDesc(name='cate1_softmax_label', shape=label_shape),
                DataDesc(name='cate2_softmax_label', shape=label_shape),
                DataDesc(name='cate3_softmax_label', shape=label_shape)]

    def getdata(self):
        return self._rec_iter.data

    def getlabel(self):
        return self._rec_iter.label

    def getindex(self):
        return self._rec_iter.index

    def getpad(self):
        return self._rec_iter.pad


def get_categorical_rec_iter(args, kv=None):
    train, val = get_rec_iter(args, kv)
    return CategoricalImageRecordIter(train), CategoricalImageRecordIter(val)


def _hwc_to_chw(img):
    img_chw = np.swapaxes(img, 0, 2)
    img_chw = np.swapaxes(img_chw, 1, 2)
    return img_chw


def _func_product(product_socket_port, data_socket_port, random_flip=False, image_shuffle=False):
    context = zmq.Context()
    product_socket = context.socket(zmq.PULL)
    product_socket.set_hwm(0)
    product_socket.connect(f'tcp://0.0.0.0:{product_socket_port}')
    logging.info(f'connect product socket (port: {product_socket_port})')

    data_socket = context.socket(zmq.PUSH)
    data_socket.set_hwm(0)
    data_socket.connect(f'tcp://0.0.0.0:{data_socket_port}')
    logging.info(f'connect data socket (port: {data_socket_port})')

    while True:
        '''
        # opencv version
        idx, images, class_id = product_socket.recv_pyobj()
        image_array = [cv2.imdecode(np.fromstring(x, np.uint8), cv2.IMREAD_COLOR) for x in images]
        data = np.concatenate([_hwc_to_chw(image_array[i % len(image_array)]) for i in range(4)])
        data_socket.send_pyobj((data, class_id))
        '''

        idx, images, class_id = product_socket.recv_pyobj()
        image_array = [mx.img.imdecode(x, to_rgb=1) for x in images]
        data_array = [mx.nd.transpose(mx.nd.flip(image_array[i % len(image_array)], axis=1) if random_flip and random.random() < 0.5 else image_array[i % len(image_array)],
                                      axes=(2, 0, 1)) for i in range(4)]
        if image_shuffle:
            random.shuffle(data_array)
        data = mx.nd.concat(*data_array, dim=0)
        data_socket.send_pyobj((data, class_id))


class ProductDataIter(mx.io.DataIter):
    def __init__(self, bson_path, batch_size, data_shape,
                 data_name='data', label_name='softmax_label',
                 rand_crop=False, rand_mirror=False,
                 num_procs=1, shuffle_product=False, shuffle_image=False):
        super(ProductDataIter, self).__init__()
        cate1_dict, cate2_dict, cate3_dict = get_category_dict()
        self._cate3_dict = cate3_dict

        self._shuffle_product = shuffle_product
        self._shuffle_image = shuffle_image
        self._rand_crop = rand_crop
        self._rand_mirror = rand_mirror
        self._batch_size = batch_size
        self._data_shape = data_shape
        self._num_procs = num_procs

        self._products = []
        self._num_products = 0
        total_count = None  # utils.get_bson_count(bson_path)
        with open(bson_path, 'rb') as reader:
            logging.info(f'load bson: {bson_path} ({total_count})')
            data = bson.decode_file_iter(reader)
            for idx, prod in tqdm(enumerate(data), unit='products', total=total_count):
                product_id = prod.get('_id')
                category_id = prod.get('category_id', None)  # This won't be in Test data
                images = prod.get('imgs')
                class_id = cate3_dict[category_id]['cate3_class_id']
                self._products.append((idx, [x['picture'] for x in images], class_id))
                self._num_products += 1
            logging.info(f'ready {self._num_products} products')

        self._perm = np.arange(self._num_products)
        self._curr = 0

        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.provide_label = [(label_name, (batch_size,))]
        self.reset()

        context = zmq.Context()
        self._product_socket = context.socket(zmq.PUSH)
        self._product_socket.set_hwm(0)
        self._product_socket_port = self._product_socket.bind_to_random_port(addr='tcp://0.0.0.0')
        logging.info('start product socket(port: {port})'.format(port=self._product_socket_port))

        self._data_socket = context.socket(zmq.PULL)
        self._data_socket.set_hwm(0)
        self._data_socket_port = self._data_socket.bind_to_random_port(addr='tcp://0.0.0.0')
        logging.info('start data socket (port: {port})'.format(port=self._data_socket_port))

        proc_processors = [Process(target=_func_product, args=(self._product_socket_port,
                                                               self._data_socket_port,
                                                               self._rand_mirror,
                                                               self._shuffle_image))
                           for _ in range(num_procs)]
        [x.start() for x in proc_processors]

    def reset(self):
        self._curr = 0
        if self._shuffle_product:
            np.random.shuffle(self._perm)

    def next(self):
        if self._curr < self._num_products:
            batch_data = mx.nd.empty(self.provide_data[0][1])
            batch_label = mx.nd.empty(self.provide_label[0][1])
            i = 0
            _t1 = time.time()
            while i < self._batch_size:
                idx = self._perm[self._curr % self._num_products]
                prod = self._products[idx]
                self._product_socket.send_pyobj(prod)
                self._curr += 1
                i += 1

            for j in range(i):
                data, label = self._data_socket.recv_pyobj()
                batch_data[j] = data
                batch_label[j] = label
            _t2 = time.time()
            # logging.info('batch {:d}: {:.3f}sec'.format(i, _t2 - _t1))
            # TODO: pad is not working while training
            return DataBatch(data=[batch_data], label=[batch_label], pad=self._batch_size-i)
        else:
            raise StopIteration


def get_product_iter(args, kv=None):
    data_shape = tuple([int(l) for l in args.image_shape.split(',')])

    train = ProductDataIter(bson_path=args.data_train,
                            batch_size=args.batch_size,
                            data_shape=data_shape,
                            data_name=args.data_name,
                            label_name=args.label_name,
                            rand_crop=args.random_crop,
                            rand_mirror=args.random_mirror,
                            num_procs=args.data_nthreads,
                            shuffle_product=True,
                            shuffle_image=False,
                            )
    val = ProductDataIter(bson_path=args.data_val,
                          batch_size=args.batch_size,
                          data_shape=data_shape,
                          data_name=args.data_name,
                          label_name=args.label_name,
                          rand_crop=False,
                          rand_mirror=False,
                          num_procs=args.data_nthreads,
                          shuffle_product=False,
                          shuffle_image=False,
                          )
    return train, val
