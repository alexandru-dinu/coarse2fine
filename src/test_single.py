from __future__ import division

import argparse
import logging
from pprint import pprint

import coloredlogs
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


def main():
    # ex = {
    #     "token": ["self", ".", "error", "(", "self", ".", "cmd", ".", "missing_args_message", ")"],
    #     "src"  : ["call", "the", "method", "self.error", "[", "self", ".", "error", "]", "with", "an", "argument", "self.cmd.missing_args_message", "[", "self", ".", "cmd", ".",
    #               "missing_args_message", "]"],
    #     "type" : ["self", ".", "FUNC#1", "(", "self", ".", "NAME", ".", "NAME", ")"]
    # }

    ex = {
        'token': "ajkk = dict ( )".split(),
        'src'  : "ajkk is an empty dict".split(),
        'type' : "NAME = dict ( )".split()
    }

    js_list = [
        table.IO.preprocess_json(ex)
    ]

    metric_name_list = ['lay-token', 'lay', 'tgt-token', 'tgt']

    args.model = args.model_path  # TODO??
    translator = table.Translator(args)
    data = table.IO.TableDataset(js_list, translator.fields, 0, None, False)

    test_data = table.IO.OrderedIterator(
        dataset=data, device=args.gpu_id[0] if args.cuda else -1,
        batch_size=args.batch_size, train=False, sort=True, sort_within_batch=False
    )

    r_list = []
    for i, batch in enumerate(test_data):
        r = translator.translate(batch)
        logger.info(r[0])
        r_list += r

    r_list.sort(key=lambda x: x.idx)
    assert len(r_list) == len(js_list), 'len(r_list) != len(js_list): {} != {}'.format(len(r_list), len(js_list))

    # evaluation
    for pred, gold in zip(r_list, js_list):
        pred.eval(gold)

    for metric_name in metric_name_list:
        if metric_name.endswith("-token"):
            c_correct = sum([len(set(x.get_by_name(metric_name)) - set(y[metric_name.split("-")[0]])) == 0 for x, y in zip(r_list, js_list)])
            acc = c_correct / len(r_list)

            out_str = '{}: {} / {} = {:.2%}'.format(metric_name.upper(), c_correct, len(r_list), acc)
            logger.info(out_str)

        else:
            c_correct = sum((x.correct[metric_name] for x in r_list))
            acc = c_correct / len(r_list)

            out_str = '{}: {} / {} = {:.2%}'.format(metric_name.upper(), c_correct, len(r_list), acc)
            logger.info(out_str)

            for x in r_list:
                for prd, tgt in x.incorrect[metric_name]:
                    logger.warning("\nprd: %s\ntgt: %s" % (" ".join(prd), " ".join(tgt)))


if __name__ == "__main__":
    main()
