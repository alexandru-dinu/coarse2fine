import argparse
import logging
import os

import coloredlogs
import torch

import options
import table
import table.IO
from table.Utils import set_seed

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

arg_parser = argparse.ArgumentParser(description='preprocess.py')

arg_parser.add_argument('-config', help="Read options from this file")

# TODO
# arg_parser.add_argument('-src_vocab', help="Path to an existing source vocabulary")
# arg_parser.add_argument('-tgt_vocab', help="Path to an existing target vocabulary")
# arg_parser.add_argument('-report_every', type=int, default=100000, help="Report status every this many sentences")
#  ---

options.set_common_options(arg_parser)
options.set_preprocess_options(arg_parser)

args = arg_parser.parse_args()

args.train_anno = os.path.join(args.root_dir, args.dataset, 'train.json')
args.valid_anno = os.path.join(args.root_dir, args.dataset, 'dev.json')
args.test_anno = os.path.join(args.root_dir, args.dataset, 'test.json')
args.save_data = os.path.join(args.root_dir, args.dataset)

if args.cuda and args.seed is not None:
    set_seed(args.seed)


def main():
    fields = table.IO.TableDataset.get_fields()

    logger.info(" * building training")
    train = table.IO.TableDataset(args.train_anno, fields, args.permute_order, args, True)

    if os.path.isfile(args.valid_anno):
        logger.info(" * building valid")
        valid = table.IO.TableDataset(args.valid_anno, fields, permute_order=0, args=args, filter_ex=True)
    else:
        valid = None

    if os.path.isfile(args.test_anno):
        logger.info(" * building test")
        test = table.IO.TableDataset(args.test_anno, fields, permute_order=0, args=args, filter_ex=False)
    else:
        test = None

    logger.info(" * building vocab")
    table.IO.TableDataset.build_vocab(train, valid, test, args)

    logger.info(" * saving vocab.pt")
    torch.save(table.IO.TableDataset.save_vocab(fields), open(os.path.join(args.save_data, 'vocab.pt'), 'wb'))

    # can't save fields, so remove/reconstruct at training time.

    train.fields = []
    logger.info(" * saving train.pt")
    torch.save(train, open(os.path.join(args.save_data, 'train.pt'), 'wb'))

    if os.path.isfile(args.valid_anno):
        valid.fields = []
        logger.info(" * saving valid.pt")
        torch.save(valid, open(os.path.join(args.save_data, 'valid.pt'), 'wb'))

    if os.path.isfile(args.test_anno):
        test.fields = []
        logger.info(" * saving test.pt")
        torch.save(test, open(os.path.join(args.save_data, 'test.pt'), 'wb'))


if __name__ == "__main__":
    main()
