from __future__ import absolute_import
from __future__ import print_function

from laia.decoders import CTCDecoder
from laia.meters import RunningAverageMeter, SequenceErrorMeter, TimeMeter


class HtrEngineWrapper(object):
    def __init__(self, train_engine, valid_engine=None):
        self._tr_engine = train_engine
        self._va_engine = valid_engine

        self._ctc_decoder = CTCDecoder()
        self._train_timer = TimeMeter()
        self._train_loss_meter = RunningAverageMeter()
        self._train_cer_meter = SequenceErrorMeter()

        self._tr_engine.add_hook('on_start_epoch', self.__train_reset_meters)
        self._tr_engine.add_hook('on_end_batch', self.__train_accumulate_loss)
        self._tr_engine.add_hook('on_end_batch', self.__train_compute_cer)

        if valid_engine:
            self._valid_timer = TimeMeter()
            self._valid_loss_meter = RunningAverageMeter()
            self._valid_cer_meter = SequenceErrorMeter()

            self._va_engine.add_hook('on_start_epoch', self.__valid_reset_meters)
            self._va_engine.add_hook('on_end_batch', self.__valid_accumulate_loss)
            self._va_engine.add_hook('on_end_batch', self.__valid_compute_cer)

            self._tr_engine.add_evaluator(self._va_engine)
        else:
            self._valid_timer = None
            self._valid_loss_meter = None
            self._valid_cer_meter = None

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
    def train_cer(self):
        return self._train_cer_meter

    @property
    def valid_cer(self):
        return self._valid_cer_meter

    def __train_reset_meters(self, **kwargs):
        self._train_timer.reset()
        self._train_loss_meter.reset()
        self._train_cer_meter.reset()

    def __valid_reset_meters(self, **kwargs):
        self._valid_timer.reset()
        self._valid_loss_meter.reset()
        self._valid_cer_meter.reset()

    def __train_accumulate_loss(self, **kwargs):
        self._train_loss_meter.add(kwargs['batch_loss'])

    def __valid_accumulate_loss(self, **kwargs):
        loss_fn = self._tr_engine.criterion
        batch_loss = loss_fn(kwargs['batch_output'], kwargs['batch_target'])
        self._valid_loss_meter.add(batch_loss)

    def __train_compute_cer(self, **kwargs):
        batch_decode = self._ctc_decoder(kwargs['batch_output'])
        self._train_cer_meter.add(kwargs['batch_target'], batch_decode)

    def __valid_compute_cer(self, **kwargs):
        batch_decode = self._ctc_decoder(kwargs['batch_output'])
        self._valid_cer_meter.add(kwargs['batch_target'], batch_decode)

    def run(self):
        self._tr_engine.run()

    def report_epoch_info(self, **kwargs):
        # Average training loss in the last EPOCH
        tr_loss = self.train_loss.value
        # Average training CER in the last EPOCH
        tr_cer = self.train_cer.value * 100
        # Average validation CER in the last EPOCH
        va_cer = self.valid_cer.value * 100
        print('Epoch {:4d}, '
              'TR Loss = {:.3e}, '
              'TR CER = {:3.2f}%, '
              'TR Elapsed Time = {:.2f}s, '
              'VA CER = {:3.2f}%, '
              'VA Elapsed Time = {:.2f}s'.format(
            kwargs['epoch'],
            tr_loss,
            tr_cer,
            self.train_timer,
            va_cer,
            self.valid_timer))
