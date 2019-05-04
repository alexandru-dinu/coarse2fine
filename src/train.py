from __future__ import division

import argparse
import os

import torch

import cli_logger
import options
import table
import table.ModelConstructor
import table.Models
import table.modules
from table.Utils import set_seed

arg_parser = argparse.ArgumentParser()

options.set_model_options(arg_parser)
options.set_train_options(arg_parser)

args = arg_parser.parse_args()

args.data = os.path.join(args.root_dir, args.dataset)

# experiment
os.makedirs("./experiments/%s" % args.exp, exist_ok=True)

# args.save_dir = os.path.join(args.root_dir, args.dataset)
args.save_path = "./experiments/%s" % args.exp

if args.layers != -1:
    args.enc_layers = args.layers
    args.dec_layers = args.layers

args.brnn = (args.encoder_type == "brnn")
args.pre_word_vecs = os.path.join(args.data, 'embedding')

set_seed(args.seed)


def report_func(epoch: int, batch: int, num_batches: int, start_time: float, lr: float, report_stats: table.Statistics) -> table.Statistics:
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

    if batch % args.report_every == -1 % args.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = table.Statistics(0, {})

    return report_stats


def load_fields(train_data, valid_data, checkpoint):
    fields = table.IO.TableDataset.load_fields(torch.load(os.path.join(args.data, 'vocab.pt')))
    fields = dict([(k, f) for (k, f) in fields.items() if k in train_data.examples[0].__dict__])

    train_data.fields = fields
    valid_data.fields = fields

    if args.train_from:
        cli_logger.info('Loading vocab from checkpoint at %s' % args.train_from)
        fields = table.IO.TableDataset.load_fields(checkpoint['vocab'])

    return fields


def build_model(model_opt, fields, checkpoint):
    # defaults on cuda
    model = table.ModelConstructor.make_base_model(model_opt, fields, checkpoint)
    # print(model)

    return model


def build_optimizer(model, checkpoint):
    if args.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        # what members of opt does Optim need?
        optim = table.Optim(
            args.optim, args.learning_rate, args.alpha, args.max_grad_norm,
            lr_decay=args.learning_rate_decay,
            start_decay_at=args.start_decay_at,
            opt=args
        )

    optim.set_parameters(model.parameters())

    return optim


def train(model, train_data, valid_data, fields, optim):
    train_iter = table.IO.OrderedIterator(
        dataset=train_data, batch_size=args.batch_size, device=args.gpuid[0], repeat=False
    )

    valid_iter = table.IO.OrderedIterator(
        dataset=valid_data, batch_size=args.batch_size, device=args.gpuid[0], train=False, sort=True, sort_within_batch=False
    )

    train_loss = table.Loss.LossCompute(smooth_eps=model.opt.smooth_eps).cuda()
    valid_loss = table.Loss.LossCompute(smooth_eps=model.opt.smooth_eps).cuda()

    trainer = table.Trainer(model, train_iter, valid_iter, train_loss, valid_loss, optim)

    cli_logger.debug("Training from epoch %d, total: %d" % (args.start_epoch, args.epochs))

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.fix_word_vecs:
            model.q_encoder.embeddings.set_update(epoch >= args.update_word_vecs_after)

        train_stats = trainer.train(epoch, fields, report_func)
        cli_logger.info('Train accuracy: %s' % train_stats.accuracy(return_str=True))

        valid_stats = trainer.validate(epoch, fields)
        cli_logger.info('Validation accuracy: %s' % valid_stats.accuracy(return_str=True))

        # Update the learning rate
        trainer.epoch_step(eval_metric=None, epoch=epoch)

        if epoch >= args.start_checkpoint_at:
            trainer.drop_checkpoint(args, epoch, fields, valid_stats)


def main():
    cli_logger.info("Loading train and valid data from %s" % args.data)

    train_data = torch.load(os.path.join(args.data, 'train.pt'))
    valid_data = torch.load(os.path.join(args.data, 'valid.pt'))

    cli_logger.info(' * number of training sentences: %d' % len(train_data))
    cli_logger.info(' * maximum batch size: %d' % args.batch_size)

    if args.train_from:
        cli_logger.info('Loading checkpoint from %s' % args.train_from)

        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']

        args.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = args

    cli_logger.info("Loading fields generated from preprocessing phase")
    fields = load_fields(train_data, valid_data, checkpoint)

    cli_logger.info("Building model")
    model = build_model(model_opt, fields, checkpoint)

    cli_logger.info("Building optimizer")
    optim = build_optimizer(model, checkpoint)

    cli_logger.info("Start training")
    train(model, train_data, valid_data, fields, optim)


if __name__ == "__main__":
    main()
