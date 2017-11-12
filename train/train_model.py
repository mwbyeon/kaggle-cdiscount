# -*- coding: utf-8 -*-

import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import logging
import coloredlogs
coloredlogs.install(level=logging.INFO, milliseconds=True)

import mxnet as mx

from train.common import data, fit, modelzoo


def load_symbol(symbol_path):
    if not os.path.exists(symbol_path):
        raise FileNotFoundError('not exists: {}'.format(symbol_path))
    symbol = mx.symbol.load(symbol_path)
    return symbol


def load_params(params_path):
    if not os.path.exists(params_path):
        raise FileNotFoundError('not exists: {}'.format(params_path))

    arg_params, aux_params = {}, {}
    save_dict = mx.nd.load(params_path)
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def load_symbol_params(symbol_path, params_path):
    return load_symbol(symbol_path) + load_params(params_path)


def get_finetune_model(symbol, arg_params, aux_params, num_classes, feature_layer_name):
    logging.info('fine-tune to {} classes from {} layer'.format(num_classes, feature_layer_name))
    all_layers = symbol.get_internals()
    net = all_layers[feature_layer_name + '_output']  # embedding
    label = mx.sym.Variable('softmax_label')
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    new_symbol = mx.symbol.SoftmaxOutput(data=net, label=label, name='softmax')

    new_arg_params = dict({k: arg_params[k] for k in arg_params if 'fc' not in k})
    return new_symbol, new_arg_params, aux_params


def train(args):
    symbol, arg_params, aux_params = None, {}, {}
    if os.path.exists(args.symbol):
        symbol = load_symbol(args.symbol)
    else:
        from importlib import import_module
        symbol = import_module('symbols.' + args.symbol).get_symbol(**vars(args))

    if os.path.exists(args.params):
        arg_params, aux_params = load_params(args.params)

    if args.feature_layer:
        symbol, arg_params, aux_params = get_finetune_model(symbol, arg_params, aux_params, args.num_classes, args.feature_layer)

    fit.fit(args=args,
            network=symbol,
            data_loader=data.get_rec_iter,
            arg_params=arg_params,
            aux_params=aux_params)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # add args
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)

    parser.add_argument('--symbol', type=str, default='', help='symbol name or .json path')
    parser.add_argument('--params', type=str, default='', help='')
    parser.add_argument('--feature-layer', type=str, default='', help='')
    args = parser.parse_args()

    train(args)

