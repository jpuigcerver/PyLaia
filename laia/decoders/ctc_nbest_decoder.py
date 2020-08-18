import torch

from laia.losses.ctc_loss import transform_output


class CTCNBestDecoder:
    """N-best decoder based on CTC output."""

    def __init__(self, nbest):
        assert isinstance(nbest, int) and nbest > 0
        self._nbest = nbest
        self._output = None

    def __call__(self, x):
        x, _ = transform_output(x)
        x = x.permute(1, 0, 2)  # batch first
        best = [CTCNBestDecoder.get_nbest(self._nbest, x_n) for x_n in x]
        best = [tuple(zip(v, p)) for v, p in best]
        self._output = [[(v.item(), p.tolist()) for v, p in b] for b in best]
        return self._output

    @staticmethod
    def get_nbest(n, x):
        x = x.detach()
        k = min(n, x.size(1))
        val, idx = x.topk(k, dim=1, sorted=True)
        # v: current bests, p: current paths
        v, p = val[0, :], idx[0, :].unsqueeze(1)
        for i in range(1, val.size(0)):
            # calculate all next values
            next_v = torch.cartesian_prod(v, val[i, :]).sum(dim=1)
            # keep the k best ones
            v, indices = next_v.topk(k, sorted=True)
            # form paths associated to best values
            next_p = torch.cartesian_prod(p[:, -1], idx[i, :])
            next_p = torch.cat((p[:, :-1].repeat_interleave(k, 0), next_p), dim=1)
            p = next_p[indices]
        return v, p

    @property
    def output(self):
        return self._output
