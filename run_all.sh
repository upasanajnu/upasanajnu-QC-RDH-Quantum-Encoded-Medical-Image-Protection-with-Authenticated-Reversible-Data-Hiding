#!/usr/bin/env bash
# Regenerates every result of the article. Run from the repository root after `pip install -r requirements.txt`.
# The DICOM steps need pydicom-data (see README). Full-size circuit executions and the noise study take the longest.
set -e
python exp.py                 # main results, BB84 key, attack, Theorem 3 check, sensitivity, ablation
python checks.py              # numerical audit (Table 10)
python fig_example.py
python exp_medical.py
python exp_medical_ext.py
python exp_dicom.py
python exp_fresh.py
python exp_compact.py
python exp_rr.py
python exp_rr_fill.py
python exp_baseline.py
python exp_benchmark.py
python exp_malformed.py
python mcx_cost.py
python resources512.py
python ft_estimate.py
python depth.py
python mcx_verify.py
python rr_statevector.py
python circsim.py
python circsim_param.py
python circsim_rr.py
python exp_noise.py
python bb84fk/bb84_fk.py
python fp/compare.py
python figs.py
python fig_layout.py
python fig_medical.py
python fig_capacity.py
python recheck.py             # final consistency audit against the stored results
