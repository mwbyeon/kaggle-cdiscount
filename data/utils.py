
import os
import logging

from tqdm import tqdm
import bson


def get_bson_count(bson_path):
    logging.info('counting bson items...')
    count = 0
    with open(bson_path, 'rb') as reader:
        data = bson.decode_file_iter(reader)
        for _ in tqdm(data, unit='products'):
            count += 1
    return count


def encode_dict_list(products, output_bson_path, total=None, overwrite=False):
    if os.path.exists(output_bson_path) and overwrite is False:
        logging.info('already exists ({})'.format(output_bson_path))
        return

    logging.info('write {} products to {}'.format(len(products), output_bson_path))
    with open(output_bson_path, 'wb') as writer:
        for i, prod in tqdm(enumerate(products), unit='products', total=total or len(products), ascii=True):
            obj = bson._dict_to_bson(prod, False, bson.DEFAULT_CODEC_OPTIONS)
            writer.write(obj)

