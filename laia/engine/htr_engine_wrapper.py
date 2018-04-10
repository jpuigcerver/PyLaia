from __future__ import absolute_import

from laia.decoders import CTCDecoder
from laia.engine.engine import ON_EPOCH_START, ON_BATCH_END, ON_EPOCH_END
from laia.hooks import action
from laia.hooks.meters import RunningAverageMeter, SequenceErrorMeter, TimeMeter
from laia.logging import get_logger
from laia.losses import CTCLoss

_logger = get_logger(__name__)


class HtrEngineWrapper(object):
    """Engine wrapper to perform HTR experiments."""

    def __init__(self, train_engine, valid_engine=None):
        self._tr_engine = train_engine
        self._va_engine = valid_engine

        # If the trainer was created without any criterion, or it is not
        # the CTCLoss, set it properly.
        if not self._tr_engine.criterion:
            self._tr_engine.criterion = CTCLoss()
        elif not isinstance(self._tr_engine.criterion, CTCLoss):
            self.logger.warn('Overriding the criterion of the trainer to CTC.')
            self._tr_engine.criterion = CTCLoss()

        self._ctc_decoder = CTCDecoder()
        self._train_timer = TimeMeter()
        self._train_loss_meter = RunningAverageMeter()
        self._train_cer_meter = SequenceErrorMeter()

        self._tr_engine.add_hook(ON_EPOCH_START, self._train_reset_meters)
        self._tr_engine.add_hook(ON_BATCH_END, self._train_accumulate_loss)
        self._tr_engine.add_hook(ON_BATCH_END, self._train_compute_cer)

        if valid_engine:
            self._valid_timer = TimeMeter()
            self._valid_loss_meter = RunningAverageMeter()
            self._valid_cer_meter = SequenceErrorMeter()

            self._va_engine.add_hook(ON_EPOCH_START, self._valid_reset_meters)
            self._va_engine.add_hook(ON_BATCH_END, self._valid_accumulate_loss)
            self._va_engine.add_hook(ON_BATCH_END, self._valid_compute_cer)
            self._va_engine.add_hook(ON_EPOCH_END, self._report_epoch_train_and_valid)

            # Add evaluator to the trainer engine
            self._tr_engine.add_evaluator(self._va_engine)
        else:
            self._valid_timer = None
            self._valid_loss_meter = None
            self._valid_cer_meter = None
            self._tr_engine.add_hook(ON_EPOCH_END, self._report_epoch_train_only)

    @property
    def logger(self):
        return _logger

    def train_timer(self):
        return self._train_timer

    def valid_timer(self):
        return self._valid_timer

    def train_loss(self):
        return self._train_loss_meter

    def valid_loss(self):
        return self._valid_loss_meter

    def train_cer(self):
        return self._train_cer_meter

    def valid_cer(self):
        return self._valid_cer_meter

    def run(self):
        self._tr_engine.run()
        return self

    @action
    def _train_reset_meters(self):
        self._train_timer.reset()
        self._train_loss_meter.reset()
        self._train_cer_meter.reset()

    @action
    def _valid_reset_meters(self):
        self._valid_timer.reset()
        self._valid_loss_meter.reset()
        self._valid_cer_meter.reset()

    @action
    def _train_accumulate_loss(self, batch_loss):
        self._train_loss_meter.add(batch_loss)
        # Note: Stop training timer to avoid including extra costs
        # (e.g. the validation epoch)
        self._train_timer.stop()

    @action
    def _valid_accumulate_loss(self, batch_output, batch_target):
        batch_loss = self._tr_engine.criterion(batch_output, batch_target)
        self._valid_loss_meter.add(batch_loss)
        # Note: Stop timer to avoid including extra costs
        self._valid_timer.stop()

    @action
    def _train_compute_cer(self, batch_output, batch_target):
        batch_decode = self._ctc_decoder(batch_output)
        self._train_cer_meter.add(batch_target, batch_decode)

    @action
    def _valid_compute_cer(self, batch_output, batch_target):
        batch_decode = self._ctc_decoder(batch_output)
        self._valid_cer_meter.add(batch_target, batch_decode)

    @action
    def _report_epoch_train_only(self):
        self.logger.info('Epoch {epochs:4d}, '
                         'TR Loss = {train_loss.value[0]:.3e}, '
                         'TR CER = {train_cer.value:5.1%}, '
                         'TR Time = {train_timer.value:.2f}s',
                         epochs=self._tr_engine.epochs(),
                         train_loss=self.train_loss(),
                         train_cer=self.train_cer(),
                         train_timer=self.train_timer())

    @action
    def _report_epoch_train_and_valid(self):
        self.logger.info('Epoch {epochs:4d}, '
                         'TR Loss = {train_loss.value[0]:.3e}, '
                         'VA Loss = {valid_loss.value[0]:.3e}, '
                         'TR CER = {train_cer.value:5.1%}, '
                         'VA CER = {valid_cer.value:5.1%}, '
                         'TR Time = {train_timer.value:.2f}s, '
                         'VA Time = {valid_timer.value:.2f}s',
                         epochs=self._tr_engine.epochs(),
                         train_loss=self.train_loss(),
                         valid_loss=self.valid_loss(),
                         train_cer=self.train_cer(),
                         valid_cer=self.valid_cer(),
                         train_timer=self.train_timer(),
                         valid_timer=self.valid_timer())
