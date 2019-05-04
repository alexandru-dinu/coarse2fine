import argparse
import os

import torch
from path import Path

import cli_logger
import options
import table
import table.IO
from table.Utils import set_seed

arg_parser = argparse.ArgumentParser(description='preprocess.py')

arg_parser.add_argument('-config', help="Read options from this file")
arg_parser.add_argument('-src_vocab', help="Path to an existing source vocabulary")
arg_parser.add_argument('-tgt_vocab', help="Path to an existing target vocabulary")
arg_parser.add_argument('-seed', type=int, default=123, help="Random seed")
arg_parser.add_argument('-report_every', type=int, default=100000, help="Report status every this many sentences")

options.set_preprocess_options(arg_parser)

args = arg_parser.parse_args()
set_seed(args.seed)

args.train_anno = os.path.join(args.root_dir, args.dataset, 'train.json')
args.valid_anno = os.path.join(args.root_dir, args.dataset, 'dev.json')
args.test_anno = os.path.join(args.root_dir, args.dataset, 'test.json')
args.save_data = os.path.join(args.root_dir, args.dataset)

assert torch.cuda.is_available()


def main():
    cli_logger.info('Preparing training ...')
    fields = table.IO.TableDataset.get_fields()

    cli_logger.info("Building Training...")
    train = table.IO.TableDataset(args.train_anno, fields, args.permute_order, args, True)

    if Path(args.valid_anno).exists():
        cli_logger.info("Building Valid...")
        valid = table.IO.TableDataset(
            args.valid_anno, fields, 0, args, True)
    else:
        valid = None

    if Path(args.test_anno).exists():
        cli_logger.info("Building Test...")
        test = table.IO.TableDataset(
            args.test_anno, fields, 0, args, False)
    else:
        test = None

    cli_logger.info("Building Vocab...")
    table.IO.TableDataset.build_vocab(train, valid, test, args)

    # Can't save fields, so remove/reconstruct at training time.
    cli_logger.info("Saving vocab.pt")
    torch.save(table.IO.TableDataset.save_vocab(fields), open(os.path.join(args.save_data, 'vocab.pt'), 'wb'))
    train.fields = []

    torch.save(train, open(os.path.join(args.save_data, 'train.pt'), 'wb'))
    cli_logger.info("Saving train.pt")

    if Path(args.valid_anno).exists():
        valid.fields = []
        torch.save(valid, open(os.path.join(args.save_data, 'valid.pt'), 'wb'))
        cli_logger.info("Saving valid.pt")

    if Path(args.test_anno).exists():
        test.fields = []
        torch.save(test, open(os.path.join(args.save_data, 'test.pt'), 'wb'))
        cli_logger.info("Saving test.pt")


if __name__ == "__main__":
    main()
