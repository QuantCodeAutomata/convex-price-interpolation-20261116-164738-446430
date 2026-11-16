#!/usr/bin/env bash
# Run the test suite, then every experiment at its paper configuration.
set -euo pipefail
cd "$(dirname "$0")"

python -m pytest -m "not market_data"

python -m exp.exp1_surface
python -m exp.exp2_spline_order
python -m exp.exp3_wing_cap
python -m exp.exp4_variance_swap
