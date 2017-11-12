import os
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet
from common import data, fit, modelzoo
import mxnet as mx


def get_fine_tune_model(symbol, arg_params, num_classes, layer_name, use_lsoftmax=False):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']  # embedding
    label = mx.sym.Variable('softmax_label')
    if use_lsoftmax:
        # Large-Margin Softmax Loss (https://arxiv.org/abs/1612.02295)
        # https://github.com/luoyetx/mx-lsoftmax
        beta_min, beta = 0.0, 100
        scale = 0.99
        margin = 2
        net = mx.sym.LSoftmax(data=net, label=label, num_hidden=num_classes,
                              beta=beta, margin=margin, scale=scale,
                              beta_min=beta_min, verbose=True)
    else:
        net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.SoftmaxOutput(data=net, label=label, name='softmax')

    if not args.fix_last_layer:
        new_args = dict({k: arg_params[k] for k in arg_params if 'fc' not in k})
    else:
        new_args = arg_params
    return net, new_args


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    parser.add_argument('--pretrained-model', type=str,
                        help='the pre-trained model')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    parser.add_argument('--fix-last-layer', action='store_true')
    # use less augmentations for fine-tune
    data.set_data_aug_level(parser, 1)
    # use a small learning rate and less regularizations
    parser.set_defaults(image_shape='3,224,224', num_epochs=30,
                        lr=.01, lr_step_epochs='20', wd=0, mom=0)

    args = parser.parse_args()

    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    prefix, epoch = (args.pretrained_model, args.load_epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args.num_classes, args.layer_before_fullc)

    # train
    fit.fit(args=args,
            network=new_sym,
            data_loader=data.get_rec_iter,
            arg_params=new_args,
            aux_params=aux_params)
