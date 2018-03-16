from __future__ import absolute_import

import laia.plugins.logging as log

try:
    from laia.losses.ctc_loss import CTCLoss
except:
    log.warning('Missing CTC loss function library')
