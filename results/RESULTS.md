# Results — Convex Price Interpolation

> **Paper:** *Convex Price Interpolation: A single-program B-spline framework
> for arbitrage-free option prices*, Thomas Roos.

This file is the authoritative report of the replication. It is written
against the paper and overwrites itself after every full rerun; numbers come
from files saved in this directory.

A placeholder note while the experiments run for the first time — the
tables below are filled in by the experiment scripts.

## Experiment 1 — Full SPX surface calibration with CPI

Pending first full run.

## Experiment 2 — Cubic vs quartic B-spline body (single maturity)

Pending first full run.

## Experiment 3 — Sensitivity of wing extrapolation to the Lee-slope cap

Pending first full run.

## Experiment 4 — Adding variance swap quotes to CPI

Pending first full run.

## Deviations from the paper

* **Market data.** The paper is pinned to the Cboe delayed-quote snapshot of
  2 November 2026 at 17:05 ET. That exact snapshot cannot be retrieved after
  the fact: Cboe's public feed serves only the current 15-minute-delayed
  snapshot, and no research-grade option-quote archive at minute resolution
  for SPX / SPXW is available to this repository. The replication therefore
  uses a synthetic SSVI surface of the same shape produced by
  `data/generate_synthetic.py` (56 slices on 51 expiries, roughly 13,700
  bid/ask pairs, maturities up to 2.12 years, SPX-style skew). The paper's
  empirical fit statistics are therefore evaluated on this surface, and are
  not directly comparable to the ones Roos reports. Setup consistent with the
  paper (B-spline body, wing dictionary, calendar coupling, QP structure) is
  kept exactly.
* **Knot concentration exponent.** The paper fixes "`sgn(k) |k|^q` with
  `q < 1`" without a specific value. We use `q = 0.5`.
* **Weight clipping quantile.** The paper mentions clipping the weights
  `omega_i` at a high quantile without specifying it. We use the 95th
  percentile.
* **Tail-sum margin `delta_mu`.** Not provided in the paper; we use
  `delta_mu = 1e-4`.
* **Band-sandwich spacing.** For the body-wing bands the paper gives an
  error-controlled Hermite grid construction. We use the convex sandwich of
  Appendix C with intervals of width `s_n / 20` on the merged breakpoints,
  which the paper reports to give the same fit with only four times the
  number of inequalities.

## Not reproduced

* The headline fit numbers of §8 Table 1 are not directly reproduced because
  the required SPX / SPXW snapshot cannot be retrieved.
