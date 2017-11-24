# -*- coding: utf-8 -*-

import os
import csv
import logging
import coloredlogs
coloredlogs.install(level=logging.INFO)


def get_category_dict():
    category_names_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'category_names.csv')
    logging.info('load category csv: {}'.format(category_names_path))

    cate1_dict, cate2_dict, cate3_dict = dict(), dict(), dict()
    cate1_counter, cate2_counter, cate3_counter = 0, 0, 0
    with open(category_names_path, 'r', encoding='utf-8') as reader:
        csv_reader = csv.reader(reader, delimiter=',', quotechar='"')
        for i, row in enumerate(csv_reader):
            if i == 0:  # header line
                continue
            try:
                cate_id, cate1, cate2, cate3 = row
                cate_id = int(cate_id)

                cate1_names = (cate1,)
                cate2_names = (cate1, cate2)
                cate3_names = (cate1, cate2, cate3)

                if cate1_names not in cate1_dict:
                    v = {
                        'names': cate1_names,
                        'cate1_class_id': cate1_counter,
                        'child_cate3': dict(),
                    }
                    cate1_dict[cate1_names] = v
                    cate1_dict[v['cate1_class_id']] = v
                    cate1_counter += 1

                if cate_id not in cate1_dict[cate1_names]['child_cate3']:
                    cate1_dict[cate1_names]['child_cate3'][cate_id] = len(cate1_dict[cate1_names]['child_cate3'])

                if cate2_names not in cate2_dict:
                    v = {
                        'names': cate2_names,
                    }
                    cate2_dict[cate2_names] = v

                if cate3_names not in cate3_dict:
                    v = {
                        'names': cate3_names,
                        'cate_id': cate_id,
                        'cate1_class_id': cate1_dict[cate1_names]['child_cate3'][cate_id],
                        'cate3_class_id': cate3_counter,
                    }
                    cate3_dict[cate3_names] = v
                    cate3_dict[cate_id] = v
                    cate3_dict[cate3_counter] = v
                    cate3_counter += 1

            except Exception as e:
                logging.error('cannot parse a line: {}, {}'.format(row, e))
    logging.info('cate1: {} categories'.format(len(cate1_dict)))
    logging.info('cate3: {} categories'.format(len(cate3_dict) // 3))
    return cate1_dict, cate2_dict, cate3_dict


def __category_csv_to_dict(category_csv):
    logging.error('deprecated')
    cate2cid, cid2cate, cate2name = dict(), dict(), dict()
    with open(category_csv, 'r', encoding='utf-8') as reader:
        csv_reader = csv.reader(reader, delimiter=',', quotechar='"')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            try:
                cateid, cate1, cate2, cate3 = row
                cateid = int(cateid)
                cid = len(cate2cid)
                cate2cid[cateid] = cid
                cid2cate[cid] = cateid
                cate2name[cateid] = (cate1, cate2, cate3)
            except Exception as e:
                logging.error('cannot parse line: {}, {}'.format(row, e))
    logging.info('{} categories in {}'.format(len(cate2cid), category_csv))
    return cate2cid, cid2cate, cate2name


if __name__ == '__main__':
    cate1_dict, cate2_dict, cate3_dict = get_category_dict()

    for cate1, value in cate1_dict.items():
        print('[{:02d}] {}'.format(value['cate1_class_id'], cate1))
        print(' - child_cate3: {}'.format(len(value['child_cate3'])))
        print()
