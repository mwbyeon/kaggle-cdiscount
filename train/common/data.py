# -*- coding: utf-8 -*-


import mxnet as mx


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


def get_rec_iter(args, kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])

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
