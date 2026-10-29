#!/usr/bin/env bash
# Rebuild every figure and table number of the paper from results/. Output: figures/
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../figures
python build_results.py         # Figures 1-3, numbers in Section 5, Table 6 -> figures/results_facts.json
python build_fig_arms.py        # Figure 4
python build_appendix_ledger.py # Figures 5-6
python build_tables.py         # Tables 4-5 -> figures/t_app_*.tex
python robust.py                # Table 7 -> figures/robust.json
