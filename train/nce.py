import mxnet as mx
import mxnet.metric
import numpy as np
from operator import itemgetter


def nce_loss(data, label, label_weight, embed_weight, vocab_size, num_hidden, num_label):
    label_embed = mx.sym.Embedding(data=label, input_dim=vocab_size,
                                   weight=embed_weight,
                                   output_dim=num_hidden, name='label_embed')
    data = mx.sym.Reshape(data=data, shape=(-1, 1, num_hidden))
    pred = mx.sym.broadcast_mul(data, label_embed)
    pred = mx.sym.sum(data=pred, axis=2)
    return mx.sym.SoftmaxOutput(data=pred, label=label_weight)
    # return mx.sym.LogisticRegressionOutput(data=pred, label=label_weight)


class NceAccuracy(mx.metric.Accuracy):
    def __init__(self):
        super(NceAccuracy, self).__init__('nce-accuracy')

    def update(self, labels, preds):
        label_weight = labels[1].asnumpy()
        preds = preds[0].asnumpy()

        if preds.shape != label_weight.shape:
            preds = mx.nd.argmax(preds, axis=self.axis)
        preds = preds.asnumpy().astype('int32')
        label_weight = label_weight.asnumpy().astype('int32')

        mx.metric.check_label_shapes(label_weight, preds)

        self.sum_metric += (preds.flat == label_weight.flat).sum()
        self.num_inst += len(preds.flat)
