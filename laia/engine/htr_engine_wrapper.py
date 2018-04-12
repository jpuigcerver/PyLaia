from __future__ import absolute_import

from laia.decoders import CTCGreedyDecoder
from laia.engine.engine import ON_EPOCH_START, ON_BATCH_END, ON_EPOCH_END
from laia.hooks import action
from laia.hooks.meters import RunningAverageMeter, SequenceErrorMeter, TimeMeter
from laia.logging import get_logger
from laia.losses import CTCLoss
from laia.utils.char_to_word_seq import char_to_word_seq

_logger = get_logger(__name__)


def batch_char_to_word_seq(batch_sequence_of_characters, delimiters):
    return ['_'.join(map(str, char_to_word_seq(seq_chars, delimiters)))
            for seq_chars in batch_sequence_of_characters]


class HtrEngineWrapper(object):
    """Engine wrapper to perform HTR experiments."""

    def __init__(self, train_engine, valid_engine=None, word_delimiters=None):
        self._tr_engine = train_engine
        self._va_engine = valid_engine
        self._word_delimiters = word_delimiters

        # If the trainer was created without any criterion, or it is not
        # the CTCLoss, set it properly.
        if not self._tr_engine.criterion:
            self._tr_engine.criterion = CTCLoss()
        elif not isinstance(self._tr_engine.criterion, CTCLoss):
            self.logger.warn('Overriding the criterion of the trainer to CTC.')
            self._tr_engine.criterion = CTCLoss()

        self._ctc_decoder = CTCGreedyDecoder()
        self._train_timer = TimeMeter()
        self._train_loss_meter = RunningAverageMeter()
        self._train_cer_meter = SequenceErrorMeter()
        self._train_wer_meter = SequenceErrorMeter()

        self._tr_engine.add_hook(ON_EPOCH_START, self._train_reset_meters)
        self._tr_engine.add_hook(ON_BATCH_END, self._train_update_meters)

        if valid_engine:
            self._valid_timer = TimeMeter()
            self._valid_loss_meter = RunningAverageMeter()
            self._valid_cer_meter = SequenceErrorMeter()
            self._valid_wer_meter = SequenceErrorMeter()

            self._va_engine.add_hook(ON_EPOCH_START, self._valid_reset_meters)
            self._va_engine.add_hook(ON_BATCH_END, self._valid_update_meters)

            # Add evaluator to the trainer engine
            self._tr_engine.add_evaluator(self._va_engine)
        else:
            self._valid_timer = None
            self._valid_loss_meter = None
            self._valid_cer_meter = None

        self._summary_format = None
        self._summary_params = None
        self._tr_engine.add_hook(ON_EPOCH_END, self._epoch_summary)

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

    def train_wer(self):
        return self._train_wer_meter

    def valid_wer(self):
        return self._valid_cer_meter

    def run(self):
        self._summary_format, self._summary_params = \
            self._prepare_epoch_summary()
        self._tr_engine.run()
        return self

    def set_word_delimiters(self, delimiters):
        self._word_delimiters = delimiters

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
    def _train_update_meters(self, batch_loss, batch_output, batch_target):
        self._train_loss_meter.add(batch_loss)
        # Compute character error rate
        batch_decode = self._ctc_decoder(batch_output)
        self._train_cer_meter.add(batch_target, batch_decode)
        # Compute word error rate, if word delimiters are given
        if self._word_delimiters is not None:
            decode_words = batch_char_to_word_seq(batch_decode,
                                                  self._word_delimiters)
            target_words = batch_char_to_word_seq(batch_target,
                                                  self._word_delimiters)
            self._train_wer_meter.add(target_words, decode_words)
        # Note: Stop training timer to avoid including extra costs
        # (e.g. the validation epoch)
        self._train_timer.stop()

    @action
    def _valid_update_meters(self, batch_output, batch_target):
        batch_loss = self._tr_engine.criterion(batch_output, batch_target)
        self._valid_loss_meter.add(batch_loss)
        # Compute character error rate
        batch_decode = self._ctc_decoder(batch_output)
        self._valid_cer_meter.add(batch_target, batch_decode)
        # Compute word error rate, if word delimiters are given
        if self._word_delimiters is not None:
            decode_words = batch_char_to_word_seq(batch_decode,
                                                  self._word_delimiters)
            target_words = batch_char_to_word_seq(batch_target,
                                                  self._word_delimiters)
            self._valid_wer_meter.add(target_words, decode_words)
        # Note: Stop timer to avoid including extra costs
        self._valid_timer.stop()

    def _prepare_epoch_summary(self):
        fmt = 'Epoch {epoch:4d}'
        # Note: We cannot add the epochs here, sine trainer.epochs is a method.
        params = {'epoch': None}

        fmt += ', TR Loss = {train_loss.value[0]:.3e}'
        params['train_loss'] = self.train_loss()
        if self._va_engine:
            fmt += ', VA Loss = {valid_loss.value[0]:.3e}'
            params['valid_loss'] = self.valid_loss()

            fmt += ', TR CER = {train_cer.value:5.1%}'
        params['train_cer'] = self.train_cer()
        if self._va_engine:
            fmt += ', VA CER = {valid_cer.value:5.1%}'
            params['valid_cer'] = self.valid_cer()

        if self._word_delimiters is not None:
            fmt += ', TR WER = {train_wer.value:5.1%}'
            params['train_wer'] = self.train_wer()
            if self._va_engine:
                fmt += ', VA CER = {valid_wer.value:5.1%}'
                params['valid_wer'] = self.valid_cer()

        fmt += ', TR Time = {train_timer.value:.2f}s'
        params['train_timer'] = self.train_timer()
        if self._va_engine:
            fmt += ', VA Time = {valid_timer.value:.2f}s'
            params['valid_timer'] = self.valid_timer()

        return fmt, params

    @action
    def _epoch_summary(self, epoch):
        self._summary_params['epoch'] = epoch
        self.logger.info(self._summary_format, **self._summary_params)
