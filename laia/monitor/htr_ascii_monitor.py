from laia.hooks.meters import RunningAverageMeter, SequenceErrorMeter, TimeMeter
from .ascii_monitor import AsciiMonitor
from ..decoders import CTCDecoder
from ..losses import CTCLoss


class HtrAsciiMonitor(AsciiMonitor):
    def __init__(self, filename, train_engine, valid_engine=None):
        super(HtrAsciiMonitor, self).__init__(filename)
        self._tr_engine = train_engine
        self._va_engine = valid_engine

        self._ctc_decoder = CTCDecoder()
        self._train_timer = TimeMeter()
        self._valid_timer = TimeMeter()
        self._train_loss_meter = RunningAverageMeter()
        self._valid_loss_meter = RunningAverageMeter()
        self._train_cer_meter = SequenceErrorMeter()
        self._valid_cer_meter = SequenceErrorMeter()

        self._va_engine.add_hook('on_start_epoch', self.__valid_reset_meters)
        self._va_engine.add_hook('on_end_batch', self.__valid_accumulate_loss)
        self._va_engine.add_hook('on_end_batch', self.__valid_compute_cer)

        self._tr_engine.add_hook('on_start_epoch', self.__train_reset_meters)
        self._tr_engine.add_hook('on_end_batch', self.__train_accumulate_loss)
        self._tr_engine.add_hook('on_end_batch', self.__train_compute_cer)
        self._tr_engine.add_evaluator(self._va_engine)

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
        batch_loss = CTCLoss()(kwargs['batch_output'], kwargs['batch_target'])
        self._valid_loss_meter.add(batch_loss)

    def __train_compute_cer(self, **kwargs):
        batch_decode = self._ctc_decoder(kwargs['batch_output'])
        self._train_cer_meter.add(kwargs['batch_target'], batch_decode)

    def __valid_compute_cer(self, **kwargs):
        batch_decode = self._ctc_decoder(kwargs['batch_output'])
        self._valid_cer_meter.add(kwargs['batch_target'], batch_decode)
