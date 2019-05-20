from __future__ import division

import argparse
import logging
import os
from argparse import Namespace

import coloredlogs
import torch
from tensorboardX import SummaryWriter

import options
import table
import table.ModelConstructor
import table.Models
import table.modules
from table.Utils import set_seed

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

arg_parser = argparse.ArgumentParser()

options.set_common_options(arg_parser)
options.set_model_options(arg_parser)
options.set_train_options(arg_parser)

args = arg_parser.parse_args()

# experiment
args.dataset_dir = os.path.join(args.root_dir, args.dataset)

EXP_BASE_DIR = "./experiments/%s" % args.exp_name

os.makedirs(EXP_BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(EXP_BASE_DIR, "tb-logs"), exist_ok=True)
os.makedirs(os.path.join(EXP_BASE_DIR, "checkpoints"), exist_ok=True)

args.checkpoint_save_path = os.path.join(EXP_BASE_DIR, "checkpoints")


# --


def dump_cfg(file: str, cfg: Namespace) -> None:
    cfg = sorted(vars(cfg).items(), key=lambda x: x[0])
    fp = open(file, "wt")
    for k, v in cfg:
        fp.write("%32s: %s\n" % (k, v))
    fp.close()


dump_cfg(os.path.join(EXP_BASE_DIR, "train-cfg.txt"), cfg=args)
# --


if args.layers != -1:
    args.enc_layers = args.layers
    args.dec_layers = args.layers

args.brnn = (args.encoder_type == "brnn")

if args.seed is not None:
    logger.info(" * using seed: %d" % args.seed)
    set_seed(args.seed)
else:
    logger.warning(" * not using custom seed")


def report_func(epoch: int, batch: int, num_batches: int, start_time: float, lr: float, report_stats: table.Statistics):
    """
    This is the user-defined batch-level traing progress report function.

    Args:
        epoch: current epoch count.
        batch: current batch count.
        num_batches: total number of batches.
        start_time: last report time.
        lr: current learning rate.
        report_stats: old Statistics instance.

    Returns:
        report_stats updated Statistics instance.
    """

    is_new_report = batch % args.batch_report_every == -1 % args.batch_report_every

    if is_new_report:
        report_stats.print_output(epoch, batch + 1, num_batches, start_time)
        report_stats = table.Statistics(loss=0, eval_result={})

    return report_stats, is_new_report, args.batch_report_every


def load_fields(train_data, valid_data, checkpoint):
    fields = table.IO.TableDataset.load_fields(torch.load(os.path.join(args.dataset_dir, 'vocab.pt')))
    fields = dict([(k, f) for (k, f) in fields.items() if k in train_data.examples[0].__dict__])

    train_data.fields = fields
    valid_data.fields = fields

    if args.train_from:
        logger.info(' * loading vocab from checkpoint at %s' % args.train_from)
        fields = table.IO.TableDataset.load_fields(checkpoint['vocab'])

    return fields


def build_model(model_args, fields, checkpoint):
    model = table.ModelConstructor.make_base_model(model_args, fields, checkpoint)
    # print(model)

    with open(os.path.join(EXP_BASE_DIR, 'model.txt'), 'wt') as fp:
        fp.write(str(model) + "\n")

    return model


def build_optimizer(model, checkpoint=None):
    if args.train_from:
        assert checkpoint is not None

        logger.info(' * loading optimizer from checkpoint')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())

    else:
        optim = table.Optim(
            method=args.optim,
            lr=args.learning_rate,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm,
            lr_decay=args.learning_rate_decay,
            start_decay_at=args.start_decay_at,
            opt=args
        )

    optim.set_parameters(model.parameters())

    return optim


def train(model, train_data, valid_data, fields, optim):
    # tensorboard
    summary_writer = SummaryWriter(os.path.join(EXP_BASE_DIR, "tb-logs"))

    train_iter = table.IO.OrderedIterator(
        dataset=train_data, batch_size=args.batch_size, device=args.gpu_id[0], repeat=False
    )

    valid_iter = table.IO.OrderedIterator(
        dataset=valid_data, batch_size=args.batch_size, device=args.gpu_id[0], train=False, sort=True, sort_within_batch=False
    )

    train_loss = table.Loss.LossCompute(smooth_eps=model.args.smooth_eps).cuda()
    valid_loss = table.Loss.LossCompute(smooth_eps=model.args.smooth_eps).cuda()

    trainer = table.Trainer(model, train_iter, valid_iter, train_loss, valid_loss, optim, summary_writer)

    logger.debug("Training from epoch %d, total: %d" % (args.start_epoch, args.epochs))

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.fix_word_vecs:
            model.q_encoder.embeddings.set_update(epoch >= args.update_word_vecs_after)

        train_stats = trainer.train(epoch, fields, report_func)
        logger.info('Train accuracy: %s' % train_stats.accuracy(return_str=True))

        for k, v in train_stats.accuracy(return_str=False).items():
            summary_writer.add_scalar("train/accuracy/%s" % k, v / 100.0, trainer.global_timestep)

        valid_stats = trainer.validate(epoch, fields)
        logger.info('Validation accuracy: %s' % valid_stats.accuracy(return_str=True))

        for k, v in valid_stats.accuracy(return_str=False).items():
            summary_writer.add_scalar("valid/accuracy/%s" % k, v / 100.0, trainer.global_timestep)

        # Update the learning rate
        trainer.epoch_step(eval_metric=None, epoch=epoch)

        if epoch >= args.start_checkpoint_at:
            trainer.drop_checkpoint(args, epoch, fields, valid_stats)

    logger.info('Training done')
    summary_writer.close()


def main():
    logger.info(" * loading train and valid data from %s" % args.dataset_dir)

    train_data = torch.load(os.path.join(args.dataset_dir, 'train.pt'))
    valid_data = torch.load(os.path.join(args.dataset_dir, 'valid.pt'))

    logger.info(' * number of training examples: %d' % len(train_data))
    logger.info(' * number of validation examples: %d' % len(valid_data))
    logger.info(' * maximum batch size: %d' % args.batch_size)

    if args.train_from:
        logger.info(' * loading checkpoint from %s' % args.train_from)

        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
        model_args = checkpoint['opt']

        args.start_epoch = checkpoint['epoch'] + 1
    else:
        logger.info(' * training from scratch')
        checkpoint = None
        model_args = args

    logger.info(" * loading fields generated from preprocessing phase")
    fields = load_fields(train_data, valid_data, checkpoint)

    logger.info(" * building model")
    model = build_model(model_args, fields, checkpoint)

    logger.info(" * building optimizer")
    optim = build_optimizer(model, checkpoint)

    logger.info(" * start training")
    train(model, train_data, valid_data, fields, optim)


if __name__ == "__main__":
    main()
