# -*- coding: utf-8 -*-

import csv
import logging


def category_csv_to_dict(category_csv):
    cate2cid, cid2cate, cate2name = dict(), dict(), dict()
    with open(category_csv, 'r') as reader:
        csvreader = csv.reader(reader, delimiter=',', quotechar='"')
        for i, row in enumerate(csvreader):
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
