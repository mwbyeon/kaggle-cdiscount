

def main(args):
    gt = dict(x.strip().split(',') for x in args.md5)

    for line in args.src:
        prod_id, pred = line.strip().split(',')
        args.dst.write('%s,%s\n' % (prod_id, gt.get(prod_id, pred)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=argparse.FileType('r'), default='output_ensemble14_20171211.csv.e14.m1')
    parser.add_argument('--dst', type=argparse.FileType('w'), default='output_ensemble14_20171211.csv.e14.m1.md5')
    parser.add_argument('--md5', type=argparse.FileType('r'), default='md5_output.txt')
    args = parser.parse_args()

    main(args)
