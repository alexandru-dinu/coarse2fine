from __future__ import division

import argparse
import codecs
import glob
import logging
import os

import coloredlogs
import torch
from tqdm.auto import tqdm

import options
import table
import table.IO

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

arg_parser = argparse.ArgumentParser()
options.set_common_options(arg_parser)
options.set_model_options(arg_parser)
options.set_translation_options(arg_parser)
args = arg_parser.parse_args()

logger.warning('* evaluating on %s split' % args.split)
args.anno = os.path.join(args.root_dir, args.dataset, '{}.json'.format(args.split))

if args.cuda:
    torch.cuda.set_device(args.gpu_id[0])

if args.beam_size > 0:
    args.batch_size = 1


def dump_cfg(fp, cfg: dict) -> None:
    cfg = sorted(cfg.items(), key=lambda x: x[0])
    for k, v in cfg:
        fp.write("%32s: %s\n" % (k, v))


def dict_update(src: dict, new_data: dict):
    for k, v in new_data.items():
        src[k] = v
    return src


def _apply_twice(f, x):
    return f(f(x))

def get_exp_name(cur_model):
    dd = os.path.dirname(os.path.dirname(cur_model))
    return dd.split('/')[-1]


def main():
    js_list = table.IO.read_anno_json(args.anno)

    metric_name_list = ['tgt-token', 'lay-token', 'tgt', 'lay']

    prev_best = (None, None)

    model_range = range(10, 101, 5)

    if os.path.isfile(args.model_path):
        model_list = [args.model_path]
    elif os.path.isdir(args.model_path):
        model_list = sorted(glob.glob('%s/**/*.pt' % args.model_path, recursive=True))
    else:
        raise RuntimeError('Incorrect model path')

    for i, cur_model in enumerate(model_list):
        assert cur_model.endswith(".pt")

        # TODO: make better
        # if int(os.path.basename(cur_model)[2:4]) not in model_range:
        #     continue

        exp_name = get_exp_name(cur_model)

        args.model = cur_model
        logger.info(" * evaluating model [%s]" % cur_model)

        checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
        model_args = checkpoint['opt']

        fp = open("./experiments/%s/%s-%s-eval.txt" % (exp_name, args.model.split("/")[-1], args.split), "wt")

        # translator model
        translator = table.Translator(args, checkpoint)
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

        for metric_name in tqdm(metric_name_list, desc="Dump results by metric"):

            if metric_name.endswith("-token"):
                c_correct = sum([len(set(x.get_by_name(metric_name)) - set(y[metric_name.split("-")[0]])) == 0 for x, y in zip(r_list, js_list)])
                acc = c_correct / len(r_list)

                out_str = 'result: {}: {} / {} = {:.2%}'.format(metric_name, c_correct, len(r_list), acc)
                fp.write(out_str + "\n")
                print(out_str)

            else:
                c_correct = sum((x.correct[metric_name] for x in r_list))
                acc = c_correct / len(r_list)

                out_str = 'result: {}: {} / {} = {:.2%}'.format(metric_name, c_correct, len(r_list), acc)
                fp.write(out_str + "\n")
                print(out_str)

                # dump incorrect examples
                for x in r_list:
                    for prd, tgt in x.incorrect[metric_name]:
                        fp.write("\tprd: %s\n\ttgt: %s\n\n" % (" ".join(prd), " ".join(tgt)))

            if metric_name == 'tgt' and (prev_best[0] is None or acc > prev_best[1]):
                prev_best = (cur_model, acc)
        # ---

        # save model args
        fp.write("\n\n")
        dump_cfg(fp, cfg=dict_update(args.__dict__, model_args.__dict__))
        fp.close()

    # if (args.split == 'dev') and (prev_best[0] is not None):
    #     with codecs.open(os.path.join(args.root_dir, args.dataset, 'dev_best.txt'), 'w', encoding='utf-8') as f_out:
    #         f_out.write('{}\n'.format(prev_best[0]))


if __name__ == "__main__":
    main()
