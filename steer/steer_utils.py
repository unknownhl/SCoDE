import random
import torch
import torch.nn as nn
import numpy as np


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Hack_no_grad(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        with torch.no_grad():
            return self.module(*inputs, **kwargs)


class RunningMean:
    def __init__(self, gamma):
        self.gamma = gamma
        self.count = 0
        self._value = None

    def update(self, value):
        value = value.detach().cpu()
        if value.ndim == 0:
            self._update(value)
        else:
            for _v in value:
                self._update(_v)

    def _update(self, value):
        self.count += 1
        if self._value is None:
            self._value = value
        else:
            w1 = self.gamma * (1 - self.gamma ** (self.count - 1))
            w2 = (1 - self.gamma)
            wt = w1 + w2
            w1 = w1 / wt
            w2 = w2 / wt
            self._value = w1 * self._value + w2 * value

    @property
    def value(self):
        if self._value is None:
            return 0
        return self._value * 1


def find_max_subspans(sequence, n_spans, max_length):
    length = len(sequence)
    inner_scores = np.zeros((length, n_spans + 1, max_length + 1, 2))
    trace = np.zeros((length, n_spans + 1, max_length + 1, 2, 3), dtype=int)
    # trace[:, n_spans, max_length, 0] = (n_spans, max_length, 0)
    inner_scores[-1, :, :, 1] = -1e5
    for _i in range(length):
        for _j in range(n_spans+1):
            for _k in range(max_length+1):
                trace[_i, _j, _k, 0] = (_j, max_length, 0)

    for _i in range(length):
        for _j in range(n_spans):
            for _k in range(max_length+1):
                inner_scores[_i, _j, _k, 0], trace[_i, _j, _k, 0] = (
                    inner_scores[_i-1, _j, max_length, 0],
                    (_j, max_length, 0)
                )
                max_taken = inner_scores[_i-1, _j, :, 1].max()
                if max_taken > inner_scores[_i, _j, _k, 0]:
                    inner_scores[_i, _j, _k, 0] = max_taken
                    trace[_i, _j, _k, 0] = (
                        _j, inner_scores[_i-1, _j, :, 1].argmax(), 1)

                if _k < max_length:
                    inner_scores[_i, _j, _k, 1], trace[_i, _j, _k, 1] = (
                        (
                            inner_scores[_i-1, _j, _k+1, 1] + sequence[_i],
                            (_j, _k+1, 1)
                        )
                        if (inner_scores[_i-1, _j, _k+1, 1] >
                            inner_scores[_i-1, _j+1, max_length, 0])
                        else (
                            inner_scores[_i-1, _j+1, max_length, 0] +
                            sequence[_i],
                            (_j+1, max_length, 0)
                        )
                    )

    max_score = 0
    argmax = (0, 0, 0)
    for _j in reversed(range(n_spans + 1)):
        for _k in reversed(range(max_length)):
            if inner_scores[-1, _j, _k, 0] > max_score:
                max_score = inner_scores[-1, _j, _k, 0]
                argmax = (_j, _k, 0)
            if inner_scores[-1, _j, _k, 1] > max_score:
                max_score = inner_scores[-1, _j, _k, 1]
                argmax = (_j, _k, 1)

    trace_back = argmax
    tags = []
    for _i in reversed(range(length)):
        tags.append(trace_back[2])
        trace_back = trace[_i, trace_back[0], trace_back[1], trace_back[2]]

    tags.reverse()
    segments = []
    start = None
    for _i in range(length + 1):
        if _i < length and tags[_i] == 1 and start is None:
            start = _i
        elif (_i == length or tags[_i] == 0) and start is not None:
            segments.append((start, _i))
            start = None
    return segments, max_score, tags  # , inner_scores, trace