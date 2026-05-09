"""
Evaluation scope and limitations (walk-forward).

This repository focuses on **single-path backtests** on a fixed historical CSV: one train of
bars, one set of execution assumptions, and reported cash curves or returns from that path.

**Walk-forward validation** (rolling or expanding windows, with parameters fit only on past
segments and tested on strictly future segments) is **not implemented** in code. Without it,
any parameter search or model fitting on the same timeline that you then backtest is vulnerable
to **in-sample optimism**. The UI’s optional “optimization” loop is explicitly a research
convenience, not an out-of-sample validation protocol.

For production-style evaluation you would: (1) split time into disjoint train / validation /
hold-out segments, (2) fit only on train, (3) simulate forward on the next segment with frozen
parameters, (4) repeat and aggregate metrics across folds. That workflow is left as a documented
gap so users do not mistake a single backtest for a full validation design.
"""

WALK_FORWARD_NOT_IMPLEMENTED = True
