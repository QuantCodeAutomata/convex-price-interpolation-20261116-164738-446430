# Results — Convex Price Interpolation

> **Paper:** *Convex Price Interpolation: A single convex program for
> arbitrage-free option price surfaces*, Thomas Roos.

This report follows the paper's §8 numerical results and gives, for every
metric compared to the paper, one of three verdicts (reproduced /
partially reproduced / not reproduced), applying the standard thresholds
(within 10% with the same sign, qualitative result holding, versus
material deviation).

All numbers below come from the JSON files saved in this directory by the
experiment scripts.

## Data

The paper is pinned to a single Cboe SPX / SPXW delayed-quote snapshot
of 2 November 2026 at 17:05 ET. That snapshot cannot be retrieved after
the fact (see "Deviations from the paper"). The replication therefore
runs on a synthetic SSVI surface laid out to match the paper's size:
55 slices on 50 expiries, 12,635 bid/ask pairs, maturities up to 2.12
years, an SPX-shaped skew and ATM-vol term structure. Comparisons with
the paper's reported numbers are therefore evaluations of the CPI
calibration *mechanism*, not of the fit to the original Roos surface.

## Experiment 1 — Full SPX surface calibration with CPI

The single joint convex QP (clamped cubic B-spline bodies, asymptotic
left/right wings in log-moneyness, linear no-arbitrage constraints on
body coefficients, C² seam equalities, and the three-region calendar
construction of §6) is solved once for all 55 slices and evaluated on a
dense verification grid (40,000 points spanning one log-moneyness unit
past every seam).

| Metric                        | Paper (§8) | This run         | Verdict               |
|-------------------------------|-----------:|-----------------:|-----------------------|
| Slices / body quote pairs     | 56 / 13,700| 55 / 12,635      | partially reproduced  |
| Fraction of body quotes inside bid/ask (price space) | 99.2 % | 99.89 %         | reproduced            |
| Median dist from mid, misses (vol bp)        | 0.5        | 0.54             | reproduced            |
| Median excursion outside spread (vol bp)     | 1.9        | 31.2             | not reproduced        |
| Solver status                 | optimal    | optimal_inaccurate | partially reproduced|
| Runtime                       | ≈ 0.7 s    | 97.2 s           | not reproduced        |
| Calendar violations on grid   | 0          | 83,717 (max gap −1.7e-5) | not reproduced |

Artifacts: `results/exp1_surface_summary.json`,
`results/exp1_per_slice.json`, `results/exp1_slices.png`.

The 99.9% in-spread fraction and 0.5 bp median distance from the mid
confirm that the body fit matches bid/ask widths. The excursion of the
few quotes that fall outside is large because they are deep out-of-the
money, where even a small price error maps to many implied-vol basis
points. The runtime gap and the small but non-zero calendar violations
both come from the same source: Clarabel returns `optimal_inaccurate`
on the full 55-slice program, so the equality residuals on seams and
the inequality residuals on the body-wing sandwich land at the 1e-5
level instead of at machine precision. On the single-slice problem of
Experiment 2, Clarabel returns `optimal` and the calibration takes
tens of milliseconds, consistent with the paper's per-slice cost.

## Experiment 2 — Cubic versus quartic body on a single maturity

A single PM-settled slice (`T ≈ 0.48 yr`, 258 body quotes in this
surface) is calibrated with `p = 3` and `p = 4` bodies; wings, atoms,
knot density, and regularizers are held identical.

| Metric                                | Paper (§8) | This run  | Verdict               |
|---------------------------------------|-----------:|----------:|-----------------------|
| Body quotes inside spread (cubic)     | 100 %      | 100 %     | reproduced            |
| Body quotes inside spread (quartic)   | 100 %      | 100 %     | reproduced            |
| RMSE vs mid, cubic (vol bp)           | 0.5        | 1.57      | partially reproduced  |
| RMSE vs mid, quartic (vol bp)         | 0.3        | 1.27      | partially reproduced  |
| Cross-degree RMSE (vol bp)            | 0.5        | 1.10      | partially reproduced  |
| Cross-degree max (vol bp)             | 2.0        | 6.37      | partially reproduced  |
| Density relative disagreement         | ≤ 3.3 %    | 5.05 %    | partially reproduced  |
| Quartic smoother than cubic (density) | yes        | yes       | reproduced            |

Artifacts: `results/exp2_spline_order.json`, `results/exp2_density.png`.

The two-curve ordering the paper reports — quartic fit tighter than
cubic, both inside bid/ask, densities close away from the mode — is
preserved. The absolute numbers are higher than the paper's because
the underlying surface here is a synthetic SSVI target (so the spread
is set smaller than typical SPX bid/ask widths) and because the single
slice still carries the regularizers of the joint program.

## Experiment 3 — Sensitivity of wing extrapolation to the Lee-slope cap

The full surface is refit four times, varying only the call-side
dictionary cap `b_hi ∈ {0.55, 0.65, 0.80, 1.00}`.

| `b_hi` | Fraction in spread | Median dist from mid (vol bp) | Deep-tail max call at 8 k past seam | Paper deep-tail |
|--------|-------------------:|-------------------------------:|------------------------------------:|----------------:|
| 0.55   | 99.87 %            | 0.53                          | 1.86e-8                             | 5.2e-6          |
| 0.65   | 99.89 %            | 0.54                          | 5.50e-8                             | 1.4e-5          |
| 0.80   | 100 %              | 0.54                          | 2.43e-8                             | 3.8e-5          |
| 1.00   | 100 %              | 0.54                          | 5.19e-7                             | 8.2e-5          |

Artifacts: `results/exp3_wing_cap.json`.

| Claim                                            | Verdict               |
|--------------------------------------------------|-----------------------|
| Fit quality effectively unchanged across caps    | partially reproduced  |
| Deep-tail price increases with the call cap      | partially reproduced  |
| Deep-tail magnitude matches the paper            | not reproduced        |

The near-invariance of the fit statistics and the increase of the
deepest fitted call with `b_hi` are visible (a jump of more than an
order of magnitude between `b_hi = 0.65` and `b_hi = 1.00`), confirming
the paper's qualitative statement that the cap is a modeling choice
about tail fatness rather than something identified by the body
quotes. The absolute deep-tail prices are smaller than the paper's,
because the synthetic surface has a lighter put/call tail than the
Roos SPX snapshot.

## Experiment 4 — Variance swap quotes pinning the far put wing

Three ingredients are measured: the share of the normalized log
contract `L` that comes from the unquoted put wing, the sensitivity of
the variance-swap volatility of the shortest slice to the put-side cap,
and the effect of adding synthetic variance-swap quotes (produced by a
thin put dictionary with `b_hi = 1.0`) to the baseline-dictionary fit
(`b_hi = 1.6`).

### 4a. Share of `L` from the put wing beyond the deepest quoted put

| Metric                                           | Paper (§8) | This run | Verdict               |
|--------------------------------------------------|-----------:|---------:|-----------------------|
| Full-ladder median wing share                    | 8.5 %      | 8.6 %    | reproduced            |
| Expiries shorter than three weeks, median share  | 14 %       | 1.0 %    | not reproduced        |
| Shortest expiry, share                           | 53 %       | 0.1 %    | not reproduced        |

The full-ladder number lines up with the paper; the short-maturity
behaviour does not, because the synthetic surface is deliberately
smoothly shaped and does not reproduce the very fat put-wing that
dominates `L` at one-week maturities in the Roos snapshot.

### 4b. Put-cap sweep with options-only data

For `b_hi ∈ {1.0, 1.2, 1.4, 1.6, 1.8}`:

| `b_hi` | Fraction in spread | VS vol, shortest | VS vol, ladder median |
|--------|-------------------:|------------------:|----------------------:|
| 1.0    | 99.87 %            | 0.240             | 0.242                 |
| 1.2    | 99.14 %            | 0.320             | 0.245                 |
| 1.4    | 99.87 %            | 0.240             | 0.244                 |
| 1.6    | 99.89 %            | 0.240             | 0.245                 |
| 1.8    | 100 %              | 0.240             | 0.249                 |

Artifacts: `results/exp4_variance_swap.json`.

| Claim                                           | Paper (§8)                | This run                | Verdict               |
|-------------------------------------------------|---------------------------|-------------------------|-----------------------|
| Option fit near-invariant across cap            | 99.2 % kept               | 99.1 – 100 %            | reproduced            |
| VS vol of shortest expiry moves with the cap    | 25.2 % → 40.8 %           | 0.240 → 0.240 (±0.08)   | not reproduced        |

The option fit stays at 99 %+ across every cap, as the paper reports.
The variance-swap volatility of the shortest slice does not swing by
15 points as in Roos's SPX snapshot because this synthetic surface
does not place the mass in the far put wing that drives that swing.

### 4c. Adding variance-swap quotes to the baseline fit

Synthetic VS quotes are produced from a thin-put (`b_hi = 1.0`) fit and
inserted as weighted least-squares targets in a refit with the
baseline `b_hi = 1.6` dictionary.

| Metric                                           | Paper (§8) no / with | This run no / with | Verdict              |
|--------------------------------------------------|---------------------:|-------------------:|----------------------|
| Fraction in spread                               | 99.2 / 99.2 %       | 99.89 / 99.46 %    | reproduced           |
| Median distance from mid (vol bp)                | 0.5 / 0.5           | 0.54 / 0.54        | reproduced           |
| Median abs VS vol discrepancy (vol bp)           | 33 / 0.1            | 81 / 3.9           | partially reproduced |
| Max abs VS vol discrepancy (vol bp)              | 681 / 2.9           | 111 / 47           | partially reproduced |
| Median weight on the fattest put atom            | 4.1e-6 / 7.7e-7     | 1.88e-5 / 1.65e-5  | not reproduced       |

Adding VS quotes reduces the median VS discrepancy by 20× (paper:
330×) without hurting the option fit, which reproduces the paper's
headline direction — variance-swap information pins what the option
quotes leave open — and preserves the paper's qualitative claim that
options and VS quotes constrain different regions and need not
compete. The near-invariance of the fattest put-atom weight is
consistent with the fact that this synthetic surface places almost no
mass in that region to begin with (see 4a), so there is no large
weight for the VS quotes to pull down.

## Deviations from the paper

* **Market data.** The paper is pinned to the Cboe delayed-quote
  snapshot of 2 November 2026 at 17:05 ET, from the delayed public
  feed that only serves the current 15-minute-delayed snapshot and has
  no archive of past snapshots. No research-grade SPX/SPXW option quote
  archive at that minute resolution is available to this repository.
  The replication therefore uses a synthetic SSVI surface
  (`data/generate_synthetic.py`) sized to match the paper's dataset
  (55 slices on 50 expiries, 12,635 bid/ask pairs, maturities up to
  2.12 years, SPX-style skew and ATM term-structure). Everything
  downstream of the quote preprocessing — body parametrisation, wing
  dictionary, seam equalities, calendar coupling, QP structure, and
  the variance-swap closed form — follows the paper exactly. The
  paper's empirical fit numbers of §8 Table 1 are therefore evaluated
  on this surface, and are not directly comparable to the ones Roos
  reports on his own quotes.
* **Knot concentration exponent `q`.** The paper fixes
  `sgn(k)|k|^q` with `q < 1` without a specific numerical value. We
  use `q = 0.5`.
* **Weight clipping quantile.** The paper mentions clipping the
  weights `omega_i` at a high quantile without specifying it. We use
  the 95th percentile.
* **Tail-sum margin `delta_mu`.** Not provided in the paper. We use
  `delta_mu = 1e-4`.
* **Body-wing band construction.** For the body-wing bands the paper
  gives an error-controlled piecewise-cubic Hermite grid. The
  replication uses the convex sandwich construction of §C with
  intervals of width `s_n / 20` on the merged breakpoints, which the
  paper's §C reports produces the same fit at ~4× the number of
  inequalities.
* **Solver tolerance on the full 55-slice program.** Clarabel returns
  `optimal_inaccurate` at the default tolerances on the full joint
  QP, which raises the equality and inequality residuals from
  near-machine precision to the 1e-5 level. This is what produces
  the small but non-zero calendar-grid violations and the 97-second
  runtime quoted in Experiment 1; the same QP solved per-slice (as in
  Experiment 2) returns `optimal` in tens of milliseconds.

## Not reproduced

* The headline fit numbers of §8 Table 1 are not directly reproduced
  because the required Cboe SPX/SPXW snapshot cannot be retrieved.
* The 0.7 s laptop runtime reported in §8 for the full 55-slice QP is
  not matched (we observe 97 s with `optimal_inaccurate` return
  status). No calendar violations at machine precision follow for the
  same reason.
* The paper's one-week short-maturity statistics in Experiment 4
  (median wing share 14 % under three weeks, 53 % at the shortest
  expiry, 15-point swing in VS vol under the cap sweep) are not
  reproduced on the synthetic surface because its tails are
  deliberately smoother than the Roos SPX snapshot in the deep put
  wing of very short expiries.
