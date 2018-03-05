#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import logging
import torch

import laia.data
import laia.engine
import laia.nn
import laia.utils
from laia.engine.feeders import ImageFeeder, ItemFeeder, PHOCFeeder, VariableFeeder
from laia.engine.triggers import Any, MaxEpochs
from laia.meters import TimeMeter, RunningAverageMeter, AllPairsMetricAveragePrecisionMeter
from laia.utils.arguments import add_argument, add_defaults, args
from laia.savers import SaverTrigger, SaverTriggerCollection


import torch.nn.functional as F
class MyBCELoss(laia.losses.loss.Loss):
    def __call__(self, output, target):
        loss = F.binary_cross_entropy(output, target, weight=None,
                                      size_average=False)
        return loss / output.size(0)

def load_caffe_model_txt(txtfile):
    if isinstance(txtfile, str):
        txtfile = open(txtfile, 'r')
    weights = []
    player = None
    for line in txtfile:
        line = line.split()
        l = '_'.join(line[0].split('_')[:-1])
        i = int(line[0].split('_')[-1])
        d = int(line[1])
        shape = [int(x) for x in line[2:(d+2)]]
        numel = reduce(lambda y, x: y * x, shape, 1)
        w = np.zeros(numel)
        for k, x in enumerate(line[(d+2):]):
            w[k] = float(x)
        w = w.reshape(shape)
        if l != player:
            weights.append(('%s.weight' % l, torch.from_numpy(w)))
            player = l
        else:
            weights.append(('%s.bias' % l, torch.from_numpy(w)))
    return OrderedDict(weights)

def set_model_weights_from_caffe(pytorch_model, caffe_weights):
    pytorch_model.load_state_dict(caffe_weights)
    return pytorch_model


def Model(phoc_size, levels=5):
    return torch.nn.Sequential(OrderedDict([
        # conv1_1
        ('conv1_1', torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)),
        ('relu1_1', torch.nn.ReLU(inplace=True)),
        # conv1_2
        ('conv1_2', torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)),
        ('relu1_2', torch.nn.ReLU(inplace=True)),
        ('maxpool1', torch.nn.MaxPool2d(2)),
        # conv2_1
        ('conv2_1', torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)),
        ('relu2_1', torch.nn.ReLU(inplace=True)),
        # conv2_2
        ('conv2_2', torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)),
        ('relu2_2', torch.nn.ReLU(inplace=True)),
        ('maxpool2', torch.nn.MaxPool2d(2)),
        # conv3_1
        ('conv3_1', torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)),
        ('relu3_1', torch.nn.ReLU(inplace=True)),
        # conv3_2
        ('conv3_2', torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ('relu3_2', torch.nn.ReLU(inplace=True)),
        # conv3_3
        ('conv3_3', torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ('relu3_3', torch.nn.ReLU(inplace=True)),
        # conv3_4
        ('conv3_4', torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ('relu3_4', torch.nn.ReLU(inplace=True)),
        # conv3_5
        ('conv3_5', torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ('relu3_5', torch.nn.ReLU(inplace=True)),
        # conv3_6
        ('conv3_6', torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ('relu3_6', torch.nn.ReLU(inplace=True)),
        # conv4_1
        ('conv4_1', torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)),
        ('relu4_1', torch.nn.ReLU(inplace=True)),
        # conv4_2
        ('conv4_2', torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)),
        ('relu4_2', torch.nn.ReLU(inplace=True)),
        # conv4_3
        ('conv4_3', torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)),
        ('relu4_3', torch.nn.ReLU(inplace=True)),
        # SPP layer
        ('tpp5', laia.nn.PyramidMaxPool2d(levels=levels)),
        # Linear layers
        ('fc6', torch.nn.Linear(512 * sum(range(1, levels + 1)), 4096)),
        ('relu6', torch.nn.ReLU(inplace=True)),
        ('drop6', torch.nn.Dropout()),
        ('fc7', torch.nn.Linear(4096, 4096)),
        ('relu7', torch.nn.ReLU(inplace=True)),
        ('drop7', torch.nn.Dropout()),
        ('fc8', torch.nn.Linear(4096, phoc_size)),
        # Predicted PHOC
        ('sigmoid', torch.nn.Sigmoid())
    ]))


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    add_defaults('batch_size', 'learning_rate', 'momentum', 'gpu', 'seed', 'max_epochs')
    add_argument('--phoc_levels', type=int, default=[1, 2, 3, 4, 5], nargs='+',
                 help='PHOC levels used to encode the transcript')
    add_argument('--show_progress_bar', type=bool, default=True,
                 help='If true, show progress bar for each epoch')
    add_argument('--num_samples_per_epoch', type=int, default=None,
                 help='Use this number of samples randomly sampled '
                 'from the dataset in each epoch')
    add_argument('--num_iterations_to_update', type=int, default=10,
                 help='Update parameters every n iterations')
    add_argument('syms')
    add_argument('tr_img_dir')
    add_argument('tr_txt_table')
    add_argument('va_txt_table')
    args = args()

    laia.manual_seed(args.seed)

    syms = laia.utils.SymbolsTable(args.syms)

    phoc_size = sum(args.phoc_levels) * len(syms)
    model = Model(phoc_size)
    if args.gpu > 0:
        model = model.cuda(args.gpu - 1)
    else:
        model = model.cpu()

    logging.info('Model has {} parameters'.format(
        sum(param.data.numel() for param in model.parameters())))
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    tr_ds = laia.data.TextImageFromTextTableDataset(
        args.tr_txt_table, args.tr_img_dir,
        img_transform=laia.utils.ImageToTensor())
    if args.num_samples_per_epoch is None:
        tr_ds_loader = torch.utils.data.DataLoader(
            tr_ds, batch_size=args.batch_size, num_workers=8, shuffle=True,
            collate_fn=laia.data.PaddingCollater({
                'img': [1, None, None],
            }, sort_key=lambda x: -x['img'].size(2)))
    else:
        tr_ds_loader = torch.utils.data.DataLoader(
            tr_ds, batch_size=args.batch_size, num_workers=8,
            sampler=FixedSizeSampler(tr_ds, args.num_samples_per_epoch),
            collate_fn=laia.data.PaddingCollater({
                'img': [1, None, None],
            }, sort_key=lambda x: -x['img'].size(2)))

    va_ds = laia.data.TextImageFromTextTableDataset(
        args.va_txt_table, args.tr_img_dir,
        img_transform=laia.utils.ImageToTensor())
    va_ds_loader = torch.utils.data.DataLoader(
        va_ds, args.batch_size, num_workers=8,
        collate_fn=laia.data.PaddingCollater({
            'img': [1, None, None],
        }, sort_key=lambda x: -x['img'].size(2)))

    # List of early stop triggers.
    # If any of these returns True, training will stop.
    early_stop_triggers = []

    # Configure MaxEpochs trigger
    if args.max_epochs and args.max_epochs > 0:
        early_stop_triggers.append(
            MaxEpochs(trainer=trainer, max_epochs=args.max_epochs))

    batch_input_fn = ImageFeeder(device=args.gpu,
                                 keep_padded_tensors=False,
                                 requires_grad=True,
                                 parent_feeder=ItemFeeder('img'))
    batch_target_fn = VariableFeeder(device=args.gpu,
                                     parent_feeder=PHOCFeeder(
                                         syms=syms,
                                         levels=args.phoc_levels,
                                         parent_feeder=ItemFeeder('txt')))

    trainer = laia.engine.Trainer(
        model=model,
        criterion=MyBCELoss(),
        optimizer=optimizer,
        data_loader=tr_ds_loader,
        batch_input_fn=batch_input_fn,
        batch_target_fn=batch_target_fn,
        progress_bar='Train' if args.show_progress_bar else False)

    evaluator = laia.engine.Evaluator(
        model=model,
        data_loader=va_ds_loader,
        batch_input_fn=batch_input_fn,
        batch_target_fn=batch_target_fn,
        progress_bar='Valid' if args.show_progress_bar else False)

    trainer.set_early_stop_trigger(Any(*early_stop_triggers)).add_evaluator(evaluator)

    train_timer = TimeMeter()
    train_loss_meter = RunningAverageMeter()
    valid_timer = TimeMeter()
    valid_loss_meter = RunningAverageMeter()
    ap_meter = AllPairsMetricAveragePrecisionMeter(
        metric='braycurtis',
        ignore_singleton=True)


    def train_reset_meters(**kwargs):
        train_timer.reset()
        train_loss_meter.reset()


    def valid_reset_meters(**kwargs):
        valid_timer.reset()
        valid_loss_meter.reset()
        ap_meter.reset()


    def train_accumulate_loss(batch_loss, **kwargs):
        train_loss_meter.add(batch_loss)
        train_timer.stop()

    def valid_accumulate_loss(batch, batch_output, batch_target, **kwargs):
        batch_loss = trainer.criterion(batch_output, batch_target)
        valid_loss_meter.add(batch_loss)
        valid_timer.stop()
        ap_meter.add(batch_output.data.cpu().numpy(),
                     [''.join(w) for w in batch['txt']])

    def valid_report_epoch(epoch, **kwargs):
        # Average loss in the last EPOCH
        tr_loss, _ = train_loss_meter.value
        va_loss, _ = valid_loss_meter.value
        # Timers
        tr_time = train_timer.value
        va_time = valid_timer.value
        # Global and Mean AP for validation
        g_ap, m_ap = ap_meter.value
        logging.info('Epoch {:4d}, '
                     'TR Loss = {:.3e}, '
                     'VA Loss = {:.3e}, '
                     'VA gAP  = {:.2%}, '
                     'VA mAP  = {:.2%}, '
                     'TR Time = {:.2f}s, '
                     'VA Time = {:.2f}s'.format(
                         epoch,
                         tr_loss,
                         va_loss,
                         g_ap,
                         m_ap,
                         tr_time,
                         va_time))


    trainer.add_hook(trainer.ON_EPOCH_START, train_reset_meters)
    evaluator.add_hook(evaluator.ON_EPOCH_START, valid_reset_meters)
    trainer.add_hook(trainer.ON_BATCH_END, train_accumulate_loss)
    evaluator.add_hook(evaluator.ON_BATCH_END, valid_accumulate_loss)
    evaluator.add_hook(evaluator.ON_EPOCH_END, valid_report_epoch)

    """
    class LastParametersSaver(object):
        def __init__(self, base_path, keep_checkpoints=5):
            self._base_path = base_path
            self._last_checkpoints = []
            self._keep_checkpoints = keep_checkpoints
            self._nckpt = 0

        def __call__(self, trainer):
            path = '{}-{}'.format(self._base_path, trainer.epochs)
            print('Saving model parameters to {!r}'.format(path))
            try:
                torch.save(trainer.model.state_dict(), path)
            except:
                # TODO(jpuigcerver): Log error saving new checkpoint
                return False

            if len(self._last_checkpoints) < self._keep_checkpoints:
                self._last_checkpoints.append(path)
            else:
                print('Removing old parameters remove: {!r}'.format(
                    self._last_checkpoints[self._nckpt]))
                try:
                    os.remove(self._last_checkpoints[self._nckpt])
                except:
                    # TODO(jpuigcerver): Log error deleting old checkpoint
                    pass
                self._last_checkpoints[self._nckpt] = path
                self._nckpt = (self._nckpt + 1) % self._keep_checkpoints

            return True

    class ParametersSaver(object):
        def __init__(self, path):
            self._path = path

        def __call__(self, trainer):
            print('Saving model parameters to {!r}'.format(self._path))
            try:
                torch.save(trainer.model.state_dict(), self._path)
                return True
            except:
                # TODO(jpuigcerver): Log error saving checkpoint
                return False


    trainer.set_epoch_saver_trigger(
        SaverTriggerCollection(
            SaverTrigger(EveryEpoch(trainer, 10),
                         LastParametersSaver('./checkpoint')),
            SaverTrigger(MeterDecrease(engine_wrapper.valid_cer),
                         ParametersSaver('./checkpoint-best-valid-cer')),
            SaverTrigger(MeterDecrease(engine_wrapper.train_cer),
                         ParametersSaver('./checkpoint-best-train-cer'))))

    # Start training
    engine_wrapper.run()
    """
    trainer.run()
