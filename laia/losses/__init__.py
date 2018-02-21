from __future__ import absolute_import

try:
    from laia.losses.ctc_loss import CTCLoss
except:
    # TODO(jpuigcerver): Show some kind of warning, since CTC is probably a key component!
    pass
