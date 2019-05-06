from __future__ import division
import os
import argparse
import torch
import codecs
import glob

import table
import table.IO
import options

arg_parser = argparse.ArgumentParser(description='evaluate.py')
options.set_translation_options(arg_parser)
args = arg_parser.parse_args()

torch.cuda.set_device(args.gpu)
args.anno = os.path.join(args.root_dir, args.dataset, '{}.json'.format(args.split))
args.pre_word_vecs = os.path.join(args.root_dir, args.dataset, 'embedding')

if args.beam_size > 0:
    args.batch_size = 1


def main():
    dummy_parser = argparse.ArgumentParser()
    options.set_model_options(dummy_parser)
    options.set_train_options(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    js_list = table.IO.read_anno_json(args.anno, args)

    metric_name_list = ['tgt']
    prev_best = (None, None)

    for fn_model in glob.glob(args.model_path):
        args.model = fn_model
        print(fn_model)
        print(args.anno)

        translator = table.Translator(args, dummy_opt.__dict__)
        data = table.IO.TableDataset(js_list, translator.fields, 0, None, False)
        test_data = table.IO.OrderedIterator(dataset=data, device=args.gpu, batch_size=args.batch_size, train=False, sort=True, sort_within_batch=False)

        # inference
        r_list = []
        for batch in test_data:
            r = translator.translate(batch)
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
            if metric_name == 'tgt' and (prev_best[0] is None or acc > prev_best[1]):
                prev_best = (fn_model, acc)

    if (args.split == 'dev') and (prev_best[0] is not None):
        with codecs.open(os.path.join(args.root_dir, args.dataset, 'dev_best.txt'), 'w', encoding='utf-8') as f_out:
            f_out.write('{}\n'.format(prev_best[0]))


if __name__ == "__main__":
    main()