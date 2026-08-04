"""Machine learning fundamentals, implemented from scratch and checked against references.

Every estimator in this package is plain numpy. The point is not to be fast or to be a
library you would depend on — it is to be *readable*, and to be *verified*: each one is
pinned by a test against an independent reference, a numerical gradient check, or a
recovery test on data whose true parameters are known by construction.

See ``tests/`` for which of the three applies to what, and ``lessons/`` for the prose.
"""

__version__ = "0.1.0"
