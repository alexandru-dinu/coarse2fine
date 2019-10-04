from __future__ import division

import argparse
import logging
from pprint import pprint

import coloredlogs
import random
import string
import torch

import options
import table
import table.IO

arg_parser = argparse.ArgumentParser()
options.set_common_options(arg_parser)
options.set_model_options(arg_parser)
options.set_translation_options(arg_parser)
args = arg_parser.parse_args()

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

if args.cuda:
    torch.cuda.set_device(args.gpu_id[0])

if args.beam_size > 0:
    args.batch_size = 1


def do_test(example_list):
    metric_name_list = ['lay-token', 'lay', 'tgt-token', 'tgt']

    args.model = args.model_path  # TODO??
    translator = table.Translator(args)
    data = table.IO.TableDataset(example_list, translator.fields, 0, None, False)

    test_data = table.IO.OrderedIterator(
        dataset=data, device=args.gpu_id[0] if args.cuda else -1,
        batch_size=args.batch_size, train=False, sort=True, sort_within_batch=False
    )

    out_list = []
    for i, batch in enumerate(test_data):
        r = translator.translate(batch)
        logger.info(r[0])
        out_list += r

    out_list.sort(key=lambda x: x.idx)
    assert len(out_list) == len(example_list), 'len(out_list) != len(js_list): {} != {}'.format(len(out_list), len(example_list))

    # evaluation
    for pred, gold in zip(out_list, example_list):
        pred.eval(gold)

    for metric_name in metric_name_list:
        if metric_name.endswith("-token"):
            c_correct = sum([len(set(x.get_by_name(metric_name)) - set(y[metric_name.split("-")[0]])) == 0 for x, y in zip(out_list, example_list)])
            acc = c_correct / len(out_list)

            out_str = '{}: {} / {} = {:.2%}'.format(metric_name.upper(), c_correct, len(out_list), acc)
            logger.info(out_str)

        else:
            c_correct = sum((x.correct[metric_name] for x in out_list))
            acc = c_correct / len(out_list)

            out_str = '{}: {} / {} = {:.2%}'.format(metric_name.upper(), c_correct, len(out_list), acc)
            logger.info(out_str)

            for x in out_list:
                for prd, tgt in x.incorrect[metric_name]:
                    logger.warning("\nprd: %s\ntgt: %s" % (" ".join(prd), " ".join(tgt)))


def main():
    ex = {
        "token": ["fp", "=", "kwargs", ".", "pop", "(", "\" _STR:0_ \"", ",", "sys", ".", "stdout", ")"],
        "src"  : ["remove", "_STR:0_", "key", "from", "the", "kwargs", "dictionary", ",", "if", "it", "exists", "substitute", "it", "for", "fp", ",", "if", "not", "substitute",
                  "sys.stdout", "[", "sys", ".", "stdout", "]", "for", "fp", "."],
        "type" : "NAME = NAME . FUNC#2 ( STRING , NAME . NAME )".split()
    }

    # ex = {
    #     "token": ["self", ".", "error", "(", "self", ".", "cmd", ".", "missing_args_message", ")"],
    #     "src"  : ["call", "the", "method", "self.error", "[", "self", ".", "error", "]", "with", "an", "argument", "self.cmd.missing_args_message", "[", "self", ".", "cmd", ".",
    #               "missing_args_message", "]"],
    #     "type" : ["self", ".", "FUNC#1", "(", "self", ".", "NAME", ".", "NAME", ")"]
    # }

    # ex = {
    #     'token': "dir = os . path . join ( app_config . path , self . base )".split(),
    #     'src'  : "join app_config.path [ app_config . path ] and self.base [ self . base ] into a file path , substitute it for dir".split(),
    #     'type' : "NAME = NAME . NAME . FUNC#2 ( NAME . NAME , self . NAME )".split()
    # }

    js_list = [
        table.IO.preprocess_json(ex)
    ]

    js_list = []

    for _ in range(10):
        n = random.randint(1, 5)
        s = ''.join(random.sample(string.ascii_lowercase, n))
        ex = {
            "token": ('%s = ' % s).split() + [" _STR:0_ "],
            "type" : "NAME = STRING".split(),
            "src"  : ("%s is an empty string" % s).split()
        }
        js_list.append(table.IO.preprocess_json(ex))

    do_test(js_list)


if __name__ == "__main__":
    main()
