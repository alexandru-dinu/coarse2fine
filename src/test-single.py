from __future__ import division

import argparse
import glob
import os

import torch

import options
import table
import table.IO

arg_parser = argparse.ArgumentParser(description='evaluate.py')
options.set_translation_options(arg_parser)
args = arg_parser.parse_args()

torch.cuda.set_device(args.gpu)

# TODO: don't hardcode, add arg
args.pre_word_vecs = os.path.join(args.root_dir, args.dataset, 'embedding')
# args.pre_word_vecs = args.root_dir

if args.beam_size > 0:
    args.batch_size = 1


def main():
    dummy_parser = argparse.ArgumentParser()
    options.set_model_options(dummy_parser)
    options.set_train_options(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    js_list = [
        table.IO.preprocess_json({
            "token": ["SomeMethod", "(", "min", ",", "max", ")"],
            "src"  : "call the method SomeMethod with arguments min and max".split(),
            "succ" : True,
            "type" : ["NAME", "OP", "NAME", "OP", "NAME", "OP"]  # TODO
        })
    ]

    metric_name_list = ['tgt']

    args.model = args.model_path
    translator = table.Translator(args, dummy_opt.__dict__)
    data = table.IO.TableDataset(js_list, translator.fields, 0, None, False)
    test_data = table.IO.OrderedIterator(dataset=data, device=args.gpu, batch_size=args.batch_size, train=False, sort=True, sort_within_batch=False)

    # inference
    r_list = []
    for i, batch in enumerate(test_data):
        r = translator.translate(batch)
        print(i, r)
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
