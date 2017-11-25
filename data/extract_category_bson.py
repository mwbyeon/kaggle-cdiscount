
import sys
import os
from collections import defaultdict
import logging

import coloredlogs
coloredlogs.install(level=logging.INFO)

import bson

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.category import get_category_dict
from data.utils import get_bson_count, encode_dict_list


def main(args):
    args.save_to = os.path.abspath(args.save_to)
    if not os.path.exists(args.save_to) or not os.path.isdir(args.save_to):
        raise NotADirectoryError(args.save_to)

    cate1_dict, cate2_dict, cate3_dict = get_category_dict()

    total_count = get_bson_count(args.bson)
    logging.info('load bson ({})'.format(args.bson))
    logging.info('read {} products'.format(total_count))
    data = bson.decode_file_iter(open(args.bson, 'rb'))

    category_products = defaultdict(list)
    for i, prod in tqdm(enumerate(data), unit='products', total=total_count, ascii=True):
        category_id = prod.get('category_id', None)  # This won't be in Test data
        cate1, cate2, cate3 = cate3_dict[category_id]['names']
        if cate1_dict[(cate1,)]['cate1_sub_class_id'] in args.include:
            category_products[cate1].append(prod)

    logging.info('{} categories'.format(len(category_products)))
    bson_filename, _ = os.path.splitext(os.path.basename(args.bson))
    for cate1 in category_products:
        prods = category_products[cate1]
        cate1_sub_class_id = cate1_dict[(cate1,)]['cate1_sub_class_id']
        logging.info(' [{:02d}] {}: {}'.format(cate1_sub_class_id, cate1, len(prods)))
        save_path = os.path.join(args.save_to, '{}_{:02d}.bson'.format(bson_filename, cate1_sub_class_id))
        encode_dict_list(prods, save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bson', type=str, required=True)
    parser.add_argument('--include', type=int, nargs='*')
    parser.add_argument('--save-to', type=str, default='./category')

    args = parser.parse_args()

    main(args)

