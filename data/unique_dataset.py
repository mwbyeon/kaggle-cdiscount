# -*- coding: utf-8 -*-

"""
total image count: 12371293
unique image count: 7772910
label conflict count: 114209
refined image count: 7658701
"""

import hashlib
import io
import os
from operator import itemgetter
import pickle

import bson
from tqdm import tqdm
from collections import defaultdict, Counter


def save_image(path, imgid, data):
    if not os.path.exists(path):
        os.makedirs(path)
    img_path = os.path.join(path, imgid + '.jpg')
    with open(img_path, 'wb') as writer:
        writer.write(data)
    return img_path


def main(args):
    data = bson.decode_file_iter(open(args.bson, 'rb'))

    md5_dict = defaultdict(Counter)
    category_counter = Counter()
    for i, d in tqdm(enumerate(data)):
        product_id = d.get('_id')
        category_id = d.get('category_id', None)  # This won't be in Test data

        for e, pic in enumerate(d['imgs']):
            picture = pic['picture']
            h = hashlib.md5(picture).hexdigest()
            md5_dict[h][category_id] += 1
            category_counter[category_id] += 1

    image_count, label_conflict = 0, 0
    relabel_dict = dict()
    relabel_counter = Counter()
    refined = dict()
    for h, v in md5_dict.items():
        s = sum(v.values())
        image_count += s
        most_label, most_count = v.most_common(1)[0]
        if len(v) > 1:
            for l, c in v.items():
                if c == most_count and category_counter[l] > category_counter[most_label]:
                    most_label = l
            label_conflict += 1
            relabel_dict[h] = most_label
        else:
            relabel_dict[h] = most_label
            refined[h] = most_label
        relabel_counter[most_label] += 1

    relabel_counter = sorted(relabel_counter.items(), key=itemgetter(1), reverse=True)
    for i, (k, v) in enumerate(relabel_counter):
        print(i, k, v)

    with open(args.md5_dict_pkl, 'wb') as writer:
        pickle.dump(refined, writer)

    print('total image count: {}'.format(image_count))
    print('unique image count: {}'.format(len(md5_dict)))
    print('label conflict count: {}'.format(label_conflict))
    print('refined image count: {}'.format(len(refined)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bson', type=str, required=True)
    parser.add_argument('--md5-dict-pkl', type=str, required=True)
    args = parser.parse_args()

    main(args)
