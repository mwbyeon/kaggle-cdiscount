# -*- coding: utf-8 -*-

import time
import csv
import hashlib
import logging
import coloredlogs
import pickle
import sys
import os
from operator import itemgetter
from multiprocessing import Process
coloredlogs.install(level=logging.DEBUG, milliseconds=True)

from collections import namedtuple, Counter, defaultdict

import mxnet as mx
import numpy as np
import cv2
import bson
import zmq
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.category import get_category_dict
from data import utils


Batch = namedtuple('Batch', ['data'])


class Tester(object):
    def __init__(self, symbol_path, params_path, data_shape, device_type='gpu', gpus='0'):
        self._data_shape = data_shape
        self._arg_params, self._aux_params = {}, {}
        save_dict = mx.nd.load(params_path)
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                self._arg_params[name] = v
            if tp == 'aux':
                self._aux_params[name] = v

        self._symbol = mx.symbol.load(symbol_path)

        ctx = [mx.gpu(int(x)) for x in gpus.split(',')] if device_type == 'gpu' else mx.cpu()

        self._module = mx.mod.Module(symbol=self._symbol, label_names=None, context=ctx)
        self._module.bind(data_shapes=[('data', self._data_shape)],
                          label_shapes=None,
                          for_training=False)
        self._module.set_params(self._arg_params, self._aux_params, allow_missing=True)

    def get_output(self, batch_data):
        self._module.forward(Batch([mx.nd.array(batch_data)]), is_train=False)
        output = self._module.get_outputs()
        return output


def category_csv_to_dict(category_csv):
    cate2cid, cid2cate = dict(), dict()
    with open(category_csv, 'r', encoding='utf-8') as reader:
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


def read_images(bson_path, cut=None, product_unique_md5=False):
    with open(bson_path, 'rb') as reader:
        data = bson.decode_file_iter(reader)

        product_count, image_count = 0, 0
        for c, d in enumerate(data):
            if cut and c == cut:
                break
            product_id = d.get('_id')
            category_id = d.get('category_id', None)  # This won't be in Test data
            items = []
            prod_md5_set = set()
            for i, pic in enumerate(d['imgs']):
                img_bytes = pic['picture']
                h = hashlib.md5(img_bytes).hexdigest()
                if product_unique_md5 and h in prod_md5_set:
                    continue
                prod_md5_set.add(h)
                item = (product_id, i, img_bytes)
                items.append(item)
                image_count += 1
            product_count += 1
            yield items  # list of [id, picture, label, [label,]]
    logging.info('read finished (product:{}, image:{})'.format(product_count, image_count))


def _func_reader(args):
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.set_hwm(1)
    zmq_socket.bind('tcp://0.0.0.0:{port}'.format(port=args.zmq_port))
    logging.info('reader started (port: {port})'.format(port=args.zmq_port))

    product_count = 0
    for items in read_images(args.bson, args.cut, args.product_unique_md5):
        zmq_socket.send_pyobj(items)
        product_count += 1

    for _ in range(args.num_procs):
        zmq_socket.send_pyobj(None)

    logging.info('reader finished (product: {})'.format(product_count))


def _do_forward(models, batch_data, batch_ids, batch_raw, cate3_dict, md5_dict=None, md5_type=None, cate_level=3):
    probs_dict = defaultdict(lambda: defaultdict(list))
    for model_id, model in enumerate(models):
        output = model.get_output(batch_data)
        probs = output[0].asnumpy()
        for i, (product_id, image_id) in enumerate(batch_ids):
            if product_id is not None:
                prob = probs[i]  # softmax
                if md5_dict:
                    h = hashlib.md5(batch_raw[i]).hexdigest()
                    if h in md5_dict:
                        if md5_type == 'unique' and len(md5_dict[h]) == 1:  # BEST!
                            prob = np.full(probs.shape[1:], 0.0)
                            most_label, most_count = md5_dict[h].most_common(1)[0]
                            class_id = cate3_dict[most_label]['cate1_sub_class_id'] if cate_level == 1 else cate3_dict[most_label]['cate3_class_id']
                            prob[class_id] = 1.0  # NOTE: 10.0?
                        elif md5_type == 'majority':
                            prob = np.full(probs.shape[1:], 0.0)
                            most_label, most_count = md5_dict[h].most_common(1)[0]
                            class_id = cate3_dict[most_label]['cate1_sub_class_id'] if cate_level == 1 else cate3_dict[most_label]['cate3_class_id']
                            prob[class_id] = 1.0
                        elif md5_type == 'l1':
                            prob = np.full(probs.shape[1:], 0.0)
                            for cate, cnt in md5_dict[h].items():
                                class_id = cate3_dict[cate]['cate1_sub_class_id'] if cate_level == 1 else cate3_dict[cate]['cate3_class_id']
                                prob[class_id] = cnt
                            prob /= sum(list(md5_dict[h].values()))
                        elif md5_type == 'l2':
                            prob = np.full(probs.shape[1:], 0.0)
                            for cate, cnt in md5_dict[h].items():
                                class_id = cate3_dict[cate]['cate1_sub_class_id'] if cate_level == 1 else cate3_dict[cate]['cate3_class_id']
                                prob[class_id] = cnt
                            prob /= np.linalg.norm(list(md5_dict[h].values()))
                        elif md5_type == 'softmax':
                            prob = np.full(probs.shape[1:], 0.0)
                            for cate, cnt in md5_dict[h].items():
                                class_id = cate3_dict[cate]['cate1_sub_class_id'] if cate_level == 1 else cate3_dict[cate]['cate3_class_id']
                                prob[class_id] = cnt
                            e_p = np.exp(prob - np.max(prob))
                            prob = e_p / e_p.sum()
                probs_dict[product_id][image_id].append((model_id, prob))
    return probs_dict


def _predict(probs_dict, max_ensemble, mode=0):
    assert 0 < max_ensemble <= 22
    result = dict()
    for product_id, prod in probs_dict.items():
        product_prob = None
        images_prob = []
        for image_id, image in prod.items():
            image_prob = None
            n = 0
            for model_id, prob in image:
                if model_id < max_ensemble:
                    if image_prob is None:
                        image_prob = np.copy(prob)
                    else:
                        image_prob = image_prob + prob
                    n += 1
            image_prob /= n
            images_prob.append(image_prob)

        for prob in images_prob:
            if product_prob is None:
                if mode == 0:
                    product_prob = prob
                else:
                    product_prob = prob ** mode
            else:
                if mode == 0:
                    product_prob = product_prob * prob
                else:
                    product_prob = product_prob + prob ** mode
        result[product_id] = int(np.argmax(product_prob))
    return result


def _md5_predict(images, cnt, cate3_counter, mode=0):
    if mode >= 1:
        perfect_matches = []
        for _k, _v in cnt.items():
            if _v >= len(images):
                perfect_matches.append(_k)

        if perfect_matches:
            return max(perfect_matches, key=lambda x: cate3_counter[x])

    if mode >= 2:
        if len(cnt) == 1:
            return list(cnt.keys())[0]

    # bad result
    # if len(cnt) > 0:
    #     return max(cnt.keys(), key=lambda x: cate3_counter[x])
    #     return cnt.most_common(1)[0][0]

    return None


def _write(writer, product_id, pred, cate3_dict):
    if writer:
        cate_id = cate3_dict[pred]['cate_id'] if args.cate_level == 3 else pred
        writer.write('{0:d},{1:d}\n'.format(product_id, cate_id))
        writer.flush()


def _func_predict(args):
    # cate2cid, cid2cate = category_csv_to_dict(args.csv)
    cate1_dict, cate2_dict, cate3_dict = get_category_dict()

    md5_dict = pickle.load(open(args.md5_dict_pkl, 'rb')) if args.md5_dict_pkl else None
    ground_truths = dict()
    cate3_counter = Counter()
    with open(args.bson, 'rb') as reader:
        for d in bson.decode_file_iter(reader):
            prod_id = d.get('_id')
            cate_id = d.get('category_id')
            if cate_id is None:
                continue
            cate1, cate2, cate3 = cate3_dict[cate_id]['names']
            if args.cate_level == 1:
                ground_truths[prod_id] = cate1_dict[(cate1,)]['child_cate3'][cate_id]
            elif args.cate_level == 3:
                cate3_class_id = cate3_dict[cate_id]['cate3_class_id']
                ground_truths[prod_id] = cate3_class_id
                cate3_counter[cate3_class_id] += 1

    logging.info('ground_truths: {}'.format(len(ground_truths)))

    data_shape = [int(x) for x in args.data_shape.split(',')]
    batch_shape = [args.batch_size] + data_shape
    logging.info('batch_shape: {}'.format(batch_shape))

    testers = []
    for symbol, params in zip(args.symbol, args.params):
        logging.info('load: {}'.format(symbol))
        testers.append(Tester(symbol, params, batch_shape, gpus=args.gpus))

    context = zmq.Context()
    ext_socket = context.socket(zmq.PULL)
    ext_socket.set_hwm(args.batch_size)
    ext_socket.bind('tcp://0.0.0.0:{port}'.format(port=args.zmq_port+1))
    logging.info('tester started (port: {port})'.format(port=args.zmq_port+1))

    writer = open(args.output, 'w') if args.output else None
    if writer:
        writer.write('_id,category_id\n')  # csv header

    ensemble_writer = dict()
    for _k in args.ensembles:
        for _m in range(2):
            w = open(args.output + f'.e{_k}.m{_m}', 'w') if args.output else None
            if w:
                w.write('_id,category_id\n')  # csv header
                ensemble_writer[(_k, _m)] = w

    __t0 = time.time()
    batch_data = np.zeros(batch_shape)
    batch_ids, batch_raw = [], []
    term_count = 0
    product_count = 0
    correct_count = 0
    catetory_count_dict, correct_count_dict = Counter(), Counter()
    incorrect_count_dict = defaultdict(Counter)

    total_count = utils.get_bson_count(args.bson)
    bar = tqdm(total=total_count, unit='products')
    finished = False
    while not finished:
        images = ext_socket.recv_pyobj()
        pad_forward = False
        if images is None:
            images = []
            term_count += 1
            if term_count == args.num_procs:
                finished = True
                pad_forward = True
        else:
            if len(images) + len(batch_ids) <= args.batch_size:
                for img, product_id, image_id, image_raw in images:
                    batch_data[len(batch_ids)] = img
                    batch_ids.append((product_id, image_id))
                    batch_raw.append(image_raw)
                product_count += 1
                bar.update(n=1)
            else:
                pad_forward = True

        if pad_forward or len(batch_ids) == args.batch_size:
            __t1 = time.time()
            probs_dict = _do_forward(testers, batch_data, batch_ids, batch_raw, cate3_dict, md5_dict, args.md5_dict_type, args.cate_level)
            __t2 = time.time()
            if args.output:
                for _k in args.ensembles:
                    for _m in range(2):
                        preds_dict = _predict(probs_dict, _k, _m)
                        for product_id, pred in preds_dict.items():
                            _write(ensemble_writer[(_k, _m)], product_id, pred, cate3_dict)

            preds_dict = _predict(probs_dict, len(testers))
            for product_id, pred in preds_dict.items():
                if product_id in ground_truths:
                    label = ground_truths.get(product_id)
                    catetory_count_dict[label] += 1
                    if label == pred:  # correct
                        correct_count += 1
                        correct_count_dict[label] += 1
                    else:
                        incorrect_count_dict[label][pred] += 1
                _write(writer, product_id, pred, cate3_dict)
            __t3 = time.time()
            bar.write('[{0:8d}] acc={1:.6f} batch:{2:.3f}, forward:{3:.3f}, write:{4:.3f} ({5:.1f}images/s)'.format(
                product_count, correct_count / product_count,
                __t1-__t0, __t2-__t1, __t3-__t2, len(batch_ids) / (__t3-__t0)))

            batch_ids[:] = []
            batch_raw[:] = []

            if pad_forward and images:
                for img, product_id, image_id, image_raw in images:
                    batch_data[len(batch_ids)] = img
                    batch_ids.append((product_id, image_id))
                    batch_raw.append(image_raw)
                product_count += 1
                bar.update(n=1)
            __t0 = time.time()

    logging.info('tester finished (product_count:{0}, accuracy={1:.6f})'.format(
        product_count, correct_count / product_count))
    if writer:
        writer.close()

    if args.print_summary:
        category_accuracy = [(cate, count, correct_count_dict[cate], correct_count_dict[cate] / count)
                             for cate, count in catetory_count_dict.items()]
        category_accuracy = sorted(category_accuracy, key=itemgetter(3, 1), reverse=True)
        for i, (cate, count, correct, accuracy) in enumerate(category_accuracy):
            line = '{0:4d}\t{1:10d}\t{2:8d}\t{3:8d}\t{4:.6f}'.format(i, cate, count, correct, accuracy)
            if incorrect_count_dict[cate]:
                second = incorrect_count_dict[cate].most_common(1)[0]
                line += '\t{0:10d}\t{1:.6f}'.format(second[0], second[1] / count)
            else:
                line += '\n'
            logging.info(line)


def _hwc_to_chw(img):
    img_chw = np.swapaxes(img, 0, 2)
    img_chw = np.swapaxes(img_chw, 1, 2)
    return img_chw


def _func_processor(args):
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PULL)
    zmq_socket.set_hwm(0)
    zmq_socket.connect('tcp://0.0.0.0:{port}'.format(port=args.zmq_port))
    # logging.info('processor started')

    ext_socket = context.socket(zmq.PUSH)
    ext_socket.set_hwm(args.batch_size)
    ext_socket.connect('tcp://0.0.0.0:{port}'.format(port=args.zmq_port+1))

    data_shape = [int(x) for x in args.data_shape.split(',')]

    while True:
        items = zmq_socket.recv_pyobj()
        # logging.info('items: %s' % (None if items is None else len(items),))
        if items is None:
            logging.info('processor finished')
            ext_socket.send_pyobj(None)
            return

        images = []
        for product_id, image_id, img_bytes in items:
            img = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if args.resize > 0:
                img = cv2.resize(img, (args.resize, args.resize), interpolation=cv2.INTER_CUBIC)
            if args.multi_view >= 0:
                images.append((_hwc_to_chw(img), product_id, image_id, img_bytes))
            if args.multi_view >= 1:
                img_flip = cv2.flip(img, flipCode=1)
                images.append((_hwc_to_chw(img_flip), product_id, image_id, img_bytes))
            if args.multi_view >= 2:
                img_flip = cv2.flip(img, flipCode=0)
                images.append((_hwc_to_chw(img_flip), product_id, image_id, img_bytes))
            if args.multi_view >= 3:
                img_crop = cv2.resize(img[5:-5, 5:-5, :], tuple(data_shape[1:]))
                images.append((_hwc_to_chw(img_crop), product_id, image_id, img_bytes))
        ext_socket.send_pyobj(images)


def main(args):
    assert len(args.params) == len(args.symbol)

    logging.info('Arguments')
    for _k, _v in vars(args).items():
        logging.info('  {}: {}'.format(_k, _v))

    proc_predict = Process(target=_func_predict, args=(args,))
    proc_reader = Process(target=_func_reader, args=(args,))
    proc_processors = [Process(target=_func_processor, args=(args,)) for _ in range(args.num_procs)]

    try:
        proc_predict.start()
        proc_reader.start()
        [x.start() for x in proc_processors]

        [x.join() for x in proc_processors]
        proc_predict.join()
        proc_reader.join()
    except KeyboardInterrupt:
        logging.warning('Keyboard Interrupted. Terminate all processes.')
        proc_reader.terminate()
        proc_predict.terminate()
        [x.terminate() for x in proc_processors]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bson', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--symbol', type=str, nargs='+', required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--data-shape', type=str, default='3,180,180')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--num-procs', type=int, default=1)
    parser.add_argument('--zmq-port', type=int, default=18300)
    parser.add_argument('--cut', type=int, default=0)

    parser.add_argument('--cate-level', type=int, default=3)
    parser.add_argument('--md5-dict-pkl', type=str, default='')
    parser.add_argument('--md5-dict-type', type=str, choices=['none', 'unique', 'majority', 'l1', 'l2', 'softmax'])
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--multi-view', type=int, default=0)
    parser.add_argument('--product-unique-md5', action='store_true')

    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--ensembles', type=int, nargs='*', default=[])
    parser.add_argument('--print-summary', action='store_true')

    parser.add_argument('--md5-mode', type=int, default=0)
    args = parser.parse_args()

    main(args)

