from __future__ import division

import argparse

import torch

import options
import table
import table.IO
from pprint import pprint

arg_parser = argparse.ArgumentParser()
options.set_common_options(arg_parser)
options.set_model_options(arg_parser)
options.set_translation_options(arg_parser)
args = arg_parser.parse_args()

if args.cuda:
    torch.cuda.set_device(args.gpu_id[0])

if args.beam_size > 0:
    args.batch_size = 1


def main():
    js_list = [
        table.IO.preprocess_json({
            "token": ["val", "=", "1", "+", "2"],
            "src"  : "val is the sum of 1 and 2".split(),
            "type" : ["NAME", "OP", "NUMBER", "OP", "NUMBER"]
        })
    ]

    metric_name_list = ['tgt']

    args.model = args.model_path  # TODO??
    translator = table.Translator(args)
    data = table.IO.TableDataset(js_list, translator.fields, 0, None, False)

    test_data = table.IO.OrderedIterator(
        dataset=data, device=args.gpu_id[0] if args.cuda else -1,
        batch_size=args.batch_size, train=False, sort=True, sort_within_batch=False
    )

    print("inference")
    r_list = []
    for i, batch in enumerate(test_data):
        r = translator.translate(batch)
        pprint(r[0])
        r_list += r

    r_list.sort(key=lambda x: x.idx)
    assert len(r_list) == len(js_list), 'len(r_list) != len(js_list): {} != {}'.format(len(r_list), len(js_list))

    # evaluation
    for pred, gold in zip(r_list, js_list):
        pred.eval(gold)

    print('Results:')
    for metric_name in metric_name_list:
        c_correct = sum((x.correct[metric_name] for x in r_list))
        acc = c_correct / len(r_list)
        print('{}: {} / {} = {:.2%}'.format(metric_name, c_correct, len(r_list), acc))


if __name__ == "__main__":
    main()
