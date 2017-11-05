# -*- coding: utf-8 -*-

import sys
import os
import csv
from collections import defaultdict, Counter

import bson
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.category import get_category_dict


def main(args):
    cate1_dict, cate2_dict, cate3_dict = get_category_dict()

    products = dict()
    with open(args.bson_path, 'rb') as reader:
        data = bson.decode_file_iter(reader)
        for c, d in tqdm(enumerate(data), unit='products'):
            product_id = d.get('_id')
            category_id = d.get('category_id', None)  # This won't be in Test data
            products[product_id] = {
                'product_id': product_id,
                'category_id': category_id,
                'images': [pic.get('picture') for e, pic in enumerate(d['imgs'])],
            }

    cate1_total_counter, cate1_correct_counter = Counter(), Counter()
    with open(args.predict_csv, 'r') as reader:
        csvreader = csv.reader(reader, delimiter=',', quotechar='"')
        save_count = 0
        for i, row in enumerate(csvreader):
            if i == 0:  # ignore header line
                continue
            prod_id, cate_id = (int(x) for x in row)
            cate_answer = products[prod_id]['category_id']
            cate1, cate2, cate3 = cate3_dict[cate_answer]['names']
            cate1_total_counter[cate1] += 1
            if cate_answer == cate_id:  # correct
                cate1_correct_counter[cate1] += 1
            else:  # incorrect
                if args.save_incorrect and save_count < args.save_cut:
                    save_count += 1
                    save_dir = os.path.join(args.save_incorrect, '%03d' % cate1_dict[(cate1,)]['cate1_class_id'])
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, '%d.png' % prod_id)

                    with open(save_path, 'wb') as writer:
                        writer.write(products[prod_id]['images'][0])

    print('Accuracy: {:.6f}'.format(sum(cate1_correct_counter.values())/sum(cate1_total_counter.values())))
    for cate1 in cate1_total_counter:
        cate1_accuracy = cate1_correct_counter[cate1] / cate1_total_counter[cate1]
        print('{:8d}\t{:8d}\t{:8d}\t{:.6f}\t[{:02d}] {}'.format(cate1_total_counter[cate1],
                                                                cate1_correct_counter[cate1],
                                                                cate1_total_counter[cate1] - cate1_correct_counter[cate1],
                                                                cate1_accuracy,
                                                                cate1_dict[(cate1,)]['cate1_class_id'], cate1))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict-csv', type=str, required=True)
    parser.add_argument('--bson-path', type=str, required=True)
    parser.add_argument('--save-incorrect', type=str, default='')
    parser.add_argument('--save-cut', type=int, default=1000)
    args = parser.parse_args()

    main(args)
