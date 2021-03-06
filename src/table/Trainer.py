"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
"""
from comet_ml import Experiment

import logging
import os
import time
from copy import deepcopy

import coloredlogs
import torch

import table
import table.modules
from table.Utils import argmax

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')


class Statistics(object):
    def __init__(self, loss, eval_result):
        self.loss = loss
        self.eval_result = eval_result
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss

        for k, v in stat.eval_result.items():
            if k in self.eval_result:
                v0 = self.eval_result[k][0] + v[0]
                v1 = self.eval_result[k][1] + v[1]
                self.eval_result[k] = (v0, v1)
            else:
                self.eval_result[k] = (v[0], v[1])

    def accuracy(self, return_str=False):
        d = sorted([(k, v) for k, v in self.eval_result.items()], key=lambda x: x[0])

        if return_str:
            return '; '.join((('{}: {:.2%}'.format(k, v[0] / v[1], )) for k, v in d))
        else:
            return dict([(k, 100.0 * v[0] / v[1]) for k, v in d])

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_output(self, epoch, batch, n_batches, start):
        logger.info(
            "Epoch %2d, Batch %5d/%5d, Loss %f; Accuracy %s; Time %.0f s elapsed" %
            (epoch, batch, n_batches, self.loss, self.accuracy(return_str=True), time.time() - start)
        )

    def log(self, split, logger, lr, step):
        pass


def count_accuracy(scores, target, mask=None, row=False):
    pred = argmax(scores)

    if mask is None:
        m_correct = pred.eq(target)
        num_all = m_correct.numel()

    elif row:
        m_correct = pred.eq(target).masked_fill_(mask, 1).prod(0, keepdim=False)
        num_all = m_correct.numel()

    else:
        non_mask = mask.ne(1)
        m_correct = pred.eq(target).masked_select(non_mask)
        num_all = non_mask.sum()

    return m_correct, num_all


def count_token_prune_accuracy(scores, target, _mask, row=False):
    # 0 -> 0.5 by sigmoid

    pred = scores.gt(0).long()
    target = target.long()
    mask = torch.ByteTensor(_mask).cuda().unsqueeze(1).expand_as(target)

    if row:
        m_correct = pred.eq(target).masked_fill_(
            mask, 1).prod(0, keepdim=False)
        num_all = m_correct.numel()

    else:
        non_mask = mask.ne(1)
        m_correct = pred.eq(target).masked_select(non_mask)
        num_all = non_mask.sum()

    return m_correct, num_all


def aggregate_accuracy(r_dict, metric_name_list):
    m_list = []

    for metric_name in metric_name_list:
        m_list.append(r_dict[metric_name][0])

    agg = torch.stack(m_list, 0).prod(0, keepdim=False)

    return agg.sum(), agg.numel()


def _debug_batch_content(vocab, ts_batch):
    seq_len = ts_batch.size(0)
    batch_size = ts_batch.size(1)
    for b in range(batch_size):
        tk_list = []
        for i in range(seq_len):
            tk = vocab.itos[ts_batch[i, b]]
            tk_list.append(tk)
        print(tk_list)


class Trainer(object):
    def __init__(self, model, train_iter, valid_iter, train_loss, valid_loss, optim, summary_writer=None, experiment: Experiment = None):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
        """
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim

        # tensorboard writer
        self.summary_writer = summary_writer

        # comet-ml experiment
        self.experiment = experiment

        # global reporting timestep
        self.global_timestep = 0

        if self.model.args.moving_avg > 0:
            self.moving_avg = deepcopy(
                list(p.data for p in model.parameters()))
        else:
            self.moving_avg = None

        # Set model in training mode.
        self.model.train()

    def forward(self, epoch, batch, criterion, fields):
        # 1. F-prop.
        q, q_len = batch.src
        lay, lay_len = batch.lay

        lay_out, tgt_out, token_out, loss_coverage = self.model(
            q, q_len, None, lay, batch.lay_e, lay_len, batch.lay_index, batch.tgt_mask,
            batch.tgt, batch.lay_parent_index, batch.tgt_parent_index, batch.copy_to_ext,
            batch.copy_to_tgt
        )

        # 2. Compute loss.
        pred = {'lay': lay_out, 'tgt': tgt_out, 'token': token_out}
        gold = {}
        mask_loss = {}
        gold['lay'] = lay[1:]
        tgt_copy_mask = batch.tgt_copy_ext.ne(fields['tgt_copy_ext'].vocab.stoi[table.IO.UNK_WORD]).long()[1:]
        tgt_org_mask = batch.tgt_copy_ext.eq(fields['tgt_copy_ext'].vocab.stoi[table.IO.UNK_WORD]).long()[1:]
        gold['tgt'] = torch.mul(tgt_copy_mask, batch.tgt_copy_ext[1:] + len(fields['tgt_loss'].vocab)) + \
                      torch.mul(tgt_org_mask, batch.tgt_loss[1:])

        if self.model.args.coverage_loss > 0 and epoch > 10:
            gold['cover'] = loss_coverage * self.model.args.coverage_loss

        loss = criterion.compute_loss(pred, gold, mask_loss)

        # 3. Get the batch statistics.
        r_dict = {}

        for metric_name in ('lay', 'tgt'):
            p = pred[metric_name].data
            g = gold[metric_name].data
            r_dict[metric_name + '-token'] = count_accuracy(p, g, mask=g.eq(table.IO.PAD), row=False)
            r_dict[metric_name] = count_accuracy(p, g, mask=g.eq(table.IO.PAD), row=True)

        st = dict([(k, (v[0].sum(), v[1])) for k, v in r_dict.items()])
        st['all'] = aggregate_accuracy(r_dict, ('lay', 'tgt'))

        if self.model.args.coverage_loss > 0 and epoch > 10:
            st['attn_impor_loss'] = (gold['cover'].data[0], 1)

        batch_stats = Statistics(loss.data[0], st)

        return loss, batch_stats

    def train(self, epoch, fields, report_func=None):
        """ Called for each epoch """

        total_stats = Statistics(loss=0, eval_result={})
        report_stats = Statistics(loss=0, eval_result={})

        avg_loss = 0.0

        for i, batch in enumerate(self.train_iter):
            self.model.zero_grad()

            loss, batch_stats = self.forward(epoch, batch, self.train_loss, fields)

            avg_loss += loss.data[0]

            # Update the parameters and statistics.
            loss.backward()
            self.optim.step()

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            if report_func is not None:
                report_stats, is_new_report, interval_len = report_func(
                    epoch, i, len(self.train_iter), total_stats.start_time, self.optim.lr, report_stats
                )

                if is_new_report:
                    self.summary_writer.add_scalar('train/avg_loss', avg_loss / interval_len, self.global_timestep)
                    self.experiment.log_metric('avg_loss', avg_loss / interval_len, step=self.global_timestep)

                    for name, param in self.model.named_parameters():
                        self.summary_writer.add_histogram(name, param, self.global_timestep)

                    avg_loss = 0.0
                    self.global_timestep += 1
            # --

            if self.model.args.moving_avg > 0:
                decay_rate = min(self.model.args.moving_avg, (1 + epoch) / (1.5 + epoch))
                for p, avg_p in zip(self.model.parameters(), self.moving_avg):
                    avg_p.mul_(decay_rate).add_(1.0 - decay_rate, p.data)

        return total_stats

    def validate(self, epoch, fields):
        """ Called for each epoch to validate. """
        self.model.eval()

        stats = Statistics(0, {})

        for batch in self.valid_iter:
            loss, batch_stats = self.forward(epoch, batch, self.valid_loss, fields)
            stats.update(batch_stats)

        self.model.train()

        return stats

    def epoch_step(self, eval_metric, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim.update_learning_rate(eval_metric, epoch)

    def drop_checkpoint(self, args, epoch, fields, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """

        model_state_dict = self.model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}

        checkpoint = {
            'model'     : model_state_dict,
            'vocab'     : table.IO.TableDataset.save_vocab(fields),
            'opt'       : args,
            'epoch'     : epoch,
            'optim'     : self.optim,
            'moving_avg': self.moving_avg
        }

        eval_result = valid_stats.accuracy()
        torch.save(checkpoint, os.path.join(args.checkpoint_save_path, 'm_%d.pt' % epoch))
