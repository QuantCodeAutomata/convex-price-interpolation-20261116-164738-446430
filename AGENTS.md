# AGENTS.md

A short guide for coding agents and contributors working on this
reproduction of **Convex Price Interpolation** (Thomas Roos).

## What this repository implements

A single joint convex QP (CVXPY + Clarabel) that calibrates the whole SPX
option price surface in normalised call-price space with clamped B-spline
bodies, non-negative asymptotic wings and exact linear no-arbitrage
constraints across strike and maturity. The four experiments of §8 are
reproduced in `exp/`.

## Layout

```
README.md                    paper metadata, run commands
AGENTS.md                    this file
requirements.txt
reproduce.sh                 runs tests then every experiment
.env.example                 empty API-key placeholders
src/                         library modules (flat, no subdirectories)
exp/                         experiment entry points (python -m exp.<name>)
tests/                       pytest suite
results/                     generated tables, plots, RESULTS.md
data/                        data_loader.py, generate_synthetic.py
```

`src/`, `exp/` and `data/` each have an empty `__init__.py`.

## Commands

Set up:

```bash
pip install -r requirements.txt
cp .env.example .env         # add DATALOADER_API_KEY / MASSIVE_API_KEY if available
```

Tests:

```bash
python -m pytest
python -m pytest -m "not market_data"
```

Experiments (from the repository root):

```bash
python -m exp.exp1_surface
python -m exp.exp2_spline_order
python -m exp.exp3_wing_cap
python -m exp.exp4_variance_swap
```

The full paper configuration is selected by default.

## Conventions

* Experiment parameters live in `src/config.py` as module-level upper-case
  variables. Each has a comment with meaning, unit and paper reference.
* Market data is loaded only through `data/data_loader.py`; synthetic data
  only through `data/generate_synthetic.py`.
* API keys are read from an uncommitted `.env`; `.env.example` holds only
  empty placeholders.
* Tests that need a live market-data API are marked `@pytest.mark.market_data`
  and skip when `DATALOADER_API_KEY` is not set.

## After a rerun

Experiment commands overwrite the files in `results/`. The numbers and
verdicts in `results/RESULTS.md` are checked against the new files and
updated by hand.
