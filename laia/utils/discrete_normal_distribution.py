import math

from numpy import logaddexp

import laia.plugins.logging as log


class DiscreteNormalDistribution(object):
    def __init__(self, mean, var, eps=1e-9, debug_compute_constant=False):
        assert mean >= 0, 'Mean must be a real value greater than or equal to 0'
        assert var > 0, 'Variance must be real value greater than 0'
        self._mean = mean
        self._var = var
        self._eps = eps
        self._debug_compute_constant = debug_compute_constant
        self._log_z = self.__compute_constant(eps)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

    def __unorm_log_pdf(self, x):
        return -0.5 * (x - self._mean) ** 2 / self._var

    def __compute_constant(self, eps):
        p_acc = self.__unorm_log_pdf(0)
        acc = logaddexp(p_acc, self.__unorm_log_pdf(1))
        i = 2
        while True:
            acc = logaddexp(acc, self.__unorm_log_pdf(i))
            i += 1
            if math.fabs(acc - p_acc) / math.fabs(p_acc) < eps:
                break
            p_acc = acc
        if self._debug_compute_constant:
            log.debug('Computing Discrete Normal Distribution Constant. '
                      'Mean = {:.6e}, var = {:.6e}, log_z = {.6e}, iters = {}',
                      self._mean, self._var, acc, i)
            return acc

    def pdf(self, x):
        return math.exp(self.log_pdf(x))

    def log_pdf(self, x):
        if x < 0.0:
            return -float('inf')
        else:
            return self.__unorm_log_pdf(x) - self._log_z
