import numpy as np


def ctc_alignment(logpost_matrix, seq, ctc_sym=0):
    """Perform CTC forced alignment of the given sequence in the log-posteriors
    matrix.

    This obtains the most likely sequence of symbols (incl. CTC-blank symbols)
    that generate the given sequence of symbols, according to the input matrix.

    Args:
        logpost_matrix (Union[np.ndarray, torch.Tensor]): Input log-posteriors
            matrix used for alignment.
        seq (Iterable[int]): Sequence of symbols to align.
        ctc_sym (int): Symbol used as the CTC blank.

    Returns:
        Tuple[float, List[int]]: Log-likelihood and best alignment for the
        input sequence.
    """
    assert logpost_matrix.ndim == 2
    T, S = logpost_matrix.shape
    L = len(seq)

    # Add CTC symbols to the reference to form a canonical sequence:
    # <ctc> s_1 <ctc> s_2 <ctc> ... <ctc> s_n <ctc>
    # len(canonical) = 2 * len(reference) + 1
    canonical = [ctc_sym]
    for i, sym in enumerate(seq):
        canonical.append(sym)
        canonical.append(ctc_sym)
        assert sym < S, (
            f"Reference symbol ({sym}) at position {i} "
            f"is too large for the given matrix {T}x{S}"
        )
        assert sym != ctc_sym, (
            "Reference includes the CTC symbol " f"({ctc_sym}) at position {i}"
        )

    best_logp = np.ndarray((T, len(canonical)))
    best_logp.fill(np.NINF)
    best_alig = np.zeros((T, len(canonical)), dtype=np.int64)

    # t = 0
    best_logp[0, 0] = logpost_matrix[0, ctc_sym]  # emit CTC symbol
    best_alig[0, 0] = 0
    if L > 0:
        best_logp[0, 1] = logpost_matrix[0, seq[0]]  # emit first ref sym
        best_alig[0, 1] = 0

    for t in range(1, T):
        # k = 0 (no symbol emitted yet)
        best_logp[t, 0] = logpost_matrix[t, ctc_sym] + best_logp[t - 1, 0]
        best_alig[t, 0] = 0

        # k = 1..(2 * L + 1)
        # TODO(jpuigcerver): Some of these iterations can be avoided.
        for k, s in enumerate(canonical[1:], 1):
            if s == ctc_sym or k == 1 or s == canonical[k - 2]:
                prev_logp, prev_k = max(
                    (best_logp[t - 1, k - 1], k - 1), (best_logp[t - 1, k], k)
                )
            else:
                prev_logp, prev_k = max(
                    (best_logp[t - 1, k - 2], k - 2),
                    (best_logp[t - 1, k - 1], k - 1),
                    (best_logp[t - 1, k], k),
                )
            best_logp[t, k] = logpost_matrix[t, s] + prev_logp
            best_alig[t, k] = prev_k

    if L > 0 and best_logp[-1, -2] > best_logp[-1, -1]:
        k = len(canonical) - 2
        best_logp = best_logp[-1, -2]
    else:
        k = len(canonical) - 1
        best_logp = best_logp[-1, -1]

    best_alignment = [0] * T
    for t in range(T - 1, -1, -1):
        best_alignment[t] = canonical[k]
        k = best_alig[t, k]

    return best_logp, best_alignment
