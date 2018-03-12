from __future__ import absolute_import

import torch

import laia.plugins.logging as log
from laia.engine.engine import Engine
from laia.engine.feeders import (ImageFeeder, ItemFeeder, PHOCFeeder,
                                 VariableFeeder)
from laia.meters import (AllPairsMetricAveragePrecisionMeter,
                         RunningAverageMeter, TimeMeter)


class PHOCEngineWrapper(object):
    r"""Engine wrapper to perform KWS experiments with PHOC networks."""

    ON_BATCH_START = Engine.ON_BATCH_START
    ON_EPOCH_START = Engine.ON_EPOCH_START
    ON_BATCH_END = Engine.ON_BATCH_END
    ON_EPOCH_END = Engine.ON_EPOCH_END

    _logger = log.get_logger(__name__)

    def __init__(self, symbols_table, phoc_levels, train_engine,
                 valid_engine=None, gpu=0):
        self._tr_engine = train_engine
        self._va_engine = valid_engine

        # If the trainer was created without any criterion, set it properly.
        if not self._tr_engine.criterion:
            self._tr_engine.set_criterion(torch.nn.BCEWithLogitsLoss())

        # Set trainer's batch_input_fn and batch_target_fn if not already set.
        if not self._tr_engine.batch_input_fn:
            self._tr_engine.set_batch_input_fn(
                ImageFeeder(device=gpu,
                            keep_padded_tensors=False,
                            parent_feeder=ItemFeeder('img')))
        if not self._tr_engine.batch_target_fn:
            self._tr_engine.set_batch_target_fn(
                VariableFeeder(device=gpu,
                               parent_feeder=PHOCFeeder(
                                   syms=symbols_table,
                                   levels=phoc_levels,
                                   parent_feeder=ItemFeeder('txt'))))

        self._train_timer = TimeMeter()
        self._train_loss_meter = RunningAverageMeter()

        self._tr_engine.add_hook(self.ON_EPOCH_START, self._train_reset_meters)
        self._tr_engine.add_hook(self.ON_BATCH_END, self._train_accumulate_loss)

        if valid_engine:
            # Set batch_input_fn and batch_target_fn if not already set.
            if not self._va_engine.batch_input_fn:
                self._va_engine.set_batch_input_fn(
                    self._tr_engine.batch_input_fn)
            if not self._va_engine.batch_target_fn:
                self._va_engine.set_batch_target_fn(
                    self._tr_engine.batch_target_fn)

            self._valid_timer = TimeMeter()
            self._valid_loss_meter = RunningAverageMeter()
            self._valid_ap_meter = AllPairsMetricAveragePrecisionMeter(
                metric='braycurtis', ignore_singleton=True)

            self._va_engine.add_hook(
                self.ON_EPOCH_START, self._valid_reset_meters)
            self._va_engine.add_hook(
                self.ON_BATCH_END, self._valid_accumulate_loss)
            self._va_engine.add_hook(
                self.ON_EPOCH_END, self._report_epoch_train_and_valid)

            # Add evaluator to the trainer engine
            self._tr_engine.add_evaluator(self._va_engine)
        else:
            self._valid_timer = None
            self._valid_loss_meter = None
            self._valid_ap_meter = None
            self._tr_engine.add_hook(
                self.ON_EPOCH_END, self._report_epoch_train_only)

    @property
    def train_timer(self):
        return self._train_timer

    @property
    def valid_timer(self):
        return self._valid_timer

    @property
    def train_loss(self):
        return self._train_loss_meter

    @property
    def valid_loss(self):
        return self._valid_loss_meter

    @property
    def valid_ap(self):
        return self._valid_ap_meter

    def run(self):
        self._tr_engine.run()
        return self

    @property
    def logger(self):
        return self._logger

    def _train_reset_meters(self, **_):
        self._train_timer.reset()
        self._train_loss_meter.reset()

    def _valid_reset_meters(self, **_):
        self._valid_timer.reset()
        self._valid_loss_meter.reset()
        self._valid_ap_meter.reset()

    def _train_accumulate_loss(self, batch_loss, **_):
        self._train_loss_meter.add(batch_loss)
        # Note: Stop training timer to avoid including extra costs
        # (e.g. the validation epoch)
        self._train_timer.stop()

    def _valid_accumulate_loss(self, batch, batch_output, batch_target, **_):
        batch_loss = self._tr_engine.criterion(batch_output, batch_target)
        self._valid_loss_meter.add(batch_loss)

        batch_output_phoc = torch.nn.functional.sigmoid(batch_output.data)
        self._valid_ap_meter.add(batch_output_phoc.cpu().numpy(),
                                 [''.join(w) for w in batch['txt']])
        self._valid_timer.stop()

    def _report_epoch_train_only(self, **_):
        self._logger.info('Epoch {_tr_engine.epochs:4d}, '
                          'TR Loss = {train_loss.value[0]:.3e}, '
                          'TR Time = {train_timer.value:.2f}s',
                          **vars(self))

    def _report_epoch_train_and_valid(self, **_):
        self._logger.info('Epoch {_tr_engine.epochs:4d}, '
                          'TR Loss = {train_loss.value[0]:.3e}, '
                          'VA Loss = {valid_loss.value[0]:.3e}, '
                          'VA gAP = {valid_ap.value[0]:5.1%}, '
                          'VA mAP = {valid_ap.value[1]:5.1%}, '
                          'TR Time = {train_timer.value:.2f}s, '
                          'VA Time = {valid_timer.value:.2f}s',
                          **vars(self))
