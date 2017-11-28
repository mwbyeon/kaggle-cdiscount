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


def get_finetune_model(base_symbol, arg_params, aux_params, num_classes, feature_layer, dropout_ratio, smooth_alpha, **kwargs):
    logging.info('fine-tune to {} classes from {} layer'.format(num_classes, feature_layer))
    all_layers = base_symbol.get_internals()
    label = mx.sym.Variable('softmax_label')
    net = all_layers[feature_layer + '_output']  # embedding
    if dropout_ratio is not None and dropout_ratio > 0.0:
        net = mx.sym.Dropout(net, p=dropout_ratio)

    net = mx.sym.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    new_symbol = mx.sym.SoftmaxOutput(data=net, label=label, name='softmax', smooth_alpha=smooth_alpha)

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
        symbol, arg_params, aux_params = get_finetune_model(symbol, arg_params, aux_params, **vars(args))

    fit.fit(args=args,
            network=symbol,
            data_loader=data.get_categorical_rec_iter if args.categorical else data.get_rec_iter,
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
    parser.add_argument('--params', type=str, default='', help='.params path for fine-tuning')
    parser.add_argument('--feature-layer', type=str, default='', help='for fine-tuning')
    parser.add_argument('--smooth-alpha', type=float, default=0.0, help='label smoothing')
    parser.add_argument('--dropout-ratio', type=float, default=None, help='use dropout')

    # arguments for ResNext
    parser.add_argument('--num-conv-groups', type=int, default=32)

    # arguments for SENets
    parser.add_argument('--use-squeeze-excitation', action='store_true')
    parser.add_argument('--excitation-ratio', type=float, default=1/16)

    # for squeeze-and-excitation
    args = parser.parse_args()

    # categorical
    parser.add_argument('--categorical', type=int, default=0)

    train(args)

