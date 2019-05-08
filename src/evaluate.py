from __future__ import division

import argparse
import codecs
import glob
import os

import torch
from tqdm.auto import tqdm

import options
import table
import table.IO

arg_parser = argparse.ArgumentParser()
options.set_common_options(arg_parser)
options.set_model_options(arg_parser)
options.set_translation_options(arg_parser)
args = arg_parser.parse_args()

args.anno = os.path.join(args.root_dir, args.dataset, '{}.json'.format(args.split))

if args.cuda:
    torch.cuda.set_device(args.gpu_id[0])

if args.beam_size > 0:
    args.batch_size = 1


def main():
    js_list = table.IO.read_anno_json(args.anno)

    metric_name_list = ['tgt']
    prev_best = (None, None)

    for cur_model in glob.glob(args.model_path):
        args.model = cur_model

        # translator model
        translator = table.Translator(args)
        test_data = table.IO.OrderedIterator(
            dataset=table.IO.TableDataset(js_list, translator.fields, 0, None, False),
            device=args.gpu_id[0] if args.cuda else -1,  # -1 is CPU
            batch_size=args.batch_size,
            train=False, sort=True, sort_within_batch=False
        )

        r_list = []
        for batch in tqdm(test_data, desc="Inference"):
            r = translator.translate(batch)
            r_list += r

        r_list.sort(key=lambda x: x.idx)
        assert len(r_list) == len(js_list), 'len(r_list) != len(js_list): {} != {}'.format(len(r_list), len(js_list))

        for pred, gold in tqdm(zip(r_list, js_list), total=len(r_list), desc="Evaluation"):
            pred.eval(gold)

        print('Results:')
        for metric_name in metric_name_list:
            c_correct = sum((x.correct[metric_name] for x in r_list))
            acc = c_correct / len(r_list)
            print('{}: {} / {} = {:.2%}'.format(metric_name, c_correct, len(r_list), acc))
            if metric_name == 'tgt' and (prev_best[0] is None or acc > prev_best[1]):
                prev_best = (cur_model, acc)

    if (args.split == 'dev') and (prev_best[0] is not None):
        with codecs.open(os.path.join(args.root_dir, args.dataset, 'dev_best.txt'), 'w', encoding='utf-8') as f_out:
            f_out.write('{}\n'.format(prev_best[0]))


if __name__ == "__main__":
    main()
