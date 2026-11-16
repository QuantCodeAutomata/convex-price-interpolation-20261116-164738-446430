# Convex Price Interpolation

Reproduction of

> **Convex Price Interpolation: A single-program B-spline framework for
> arbitrage-free option prices.** Thomas Roos.

The paper calibrates the full SPX / SPXW option price surface in one convex
quadratic program (QP). Each maturity is represented in normalised call-price
space as a clamped B-spline body on the traded strike range and two
non-negative asymptotic wings in log-moneyness. Butterfly, slope, monotonicity
and calendar no-arbitrage are enforced as linear inequalities on the
coefficients. The repository reproduces the four experiments of §8.

## Experiments

| Script | Paper reference | Purpose |
| --- | --- | --- |
| `exp.exp1_surface` | §7, §8 | Full 56-slice SPX surface calibration with CPI, fit statistics, solver runtime, calendar verification on a dense grid. |
| `exp.exp2_spline_order` | §8 single-maturity figure | Cubic (p=3) vs quartic (p=4) B-spline body on the 30 April 2027 PM-settled slice. |
| `exp.exp3_wing_cap` | §5.4, §8 | Sensitivity of deep-tail extrapolation to the call-wing Lee-slope cap `b_hi`. |
| `exp.exp4_variance_swap` | §7, §8, Appendix B | Augment CPI with the affine variance-swap functional and compare option-only versus option+variance-swap fits. |

## Layout

```
src/        core CPI implementation (B-splines, wings, calendar constraints, QP)
exp/        experiment scripts
data/       data_loader.py and generate_synthetic.py
tests/      pytest suite
results/    saved tables, plots and RESULTS.md
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add DATALOADER_API_KEY / MASSIVE_API_KEY if available
```

The paper is pinned to Cboe's delayed-quote snapshot of 2 November 2026 at
17:05 ET. That exact snapshot is not retrievable from any research-grade
market-data service after the fact: Cboe's public site serves only the current
15-minute-delayed snapshot and neither the QuantX Dataloader nor Massive REST
expose a historical SPX / SPXW option quote archive at minute resolution for
general users. When `data/cache/spx_raw.csv` is absent, the loader falls back
to a synthetic SSVI surface of the same shape (56 slices, 51 expiries,
~13,700 bid/ask pairs, maturities up to 2.12 years, negative SPX-style skew).
All deviations are documented in `results/RESULTS.md`.

## Running the experiments

```bash
python -m exp.exp1_surface              # full paper configuration
python -m exp.exp2_spline_order
python -m exp.exp3_wing_cap
python -m exp.exp4_variance_swap

./reproduce.sh                          # runs tests then every experiment
```

Each run overwrites its artefacts in `results/`. The numbers and verdicts in
`results/RESULTS.md` are checked against the newly saved files and updated by
hand.

## Tests

```bash
python -m pytest                        # all tests
python -m pytest -m "not market_data"   # skip anything that needs API access
```

## Conventions

* All experiment parameters live in `src/config.py` as module-level upper-case
  variables, each with a comment giving its meaning, unit and paper reference.
* Market data is loaded only from `data/data_loader.py`; synthetic surfaces
  only from `data/generate_synthetic.py`.
* API keys are read from an uncommitted `.env`; `.env.example` has empty
  placeholders only.
