"""Small numerical primitives that more than one module needs, written for stability.

Textbook formulas are written for a blackboard, where numbers have infinite range. The
translations here are the ones that survive float64: they compute the same quantity as the
formula they are named after, and they do it without overflowing on inputs that occur in
practice.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def sigmoid(z: Array) -> Array:
    """Logistic function, evaluated so that large-magnitude inputs do not overflow.

    The textbook form ``1 / (1 + exp(-z))`` evaluates ``exp(709)`` and beyond once ``z``
    drops past about -709, which is the largest power of e a float64 holds. Numpy warns and
    returns ``inf``, and the quotient then collapses to exactly ``0``.

    Notice what that does *not* do: it does not produce a visibly wrong number. The true
    value there is around ``1e-309``, so zero is nearly right, and the damage only surfaces
    later — as a ``RuntimeWarning`` raised from inside a training loop, and as a probability
    of exactly zero whose logarithm is ``-inf`` where the honest answer is ``-800``. Errors
    that look harmless at the point they are made are the ones worth engineering away.

    Branching on the sign keeps the argument of ``exp`` at or below zero on both halves, so
    nothing overflows and even subnormal probabilities survive.
    """
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exponential = np.exp(z[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def log_sigmoid(z: Array) -> Array:
    """``log(sigmoid(z))``, computed without forming ``sigmoid(z)`` first.

    Going through the probability loses the answer twice over: it underflows to zero for
    very negative ``z``, and ``log(0)`` is then ``-inf`` where the true value is a perfectly
    ordinary ``-800``. Log-loss on a confidently wrong prediction is exactly that case.
    """
    return -np.logaddexp(0.0, -z)
