# QC-RDH: code and data

Code, data and results for the article

> U. Singh and V. B. Singh, **"Fixed-Key Unitary Ciphers Cannot Diffuse: A Limit Theorem for Quantum Image Security and a Provably Secure Reversible Data Hiding Scheme"**, submitted to *Journal of Information Security and Applications*, 2026.

Every number, table and figure in the article is produced by the scripts in this repository from public data and fixed random seeds. The stored outputs are in `results/`, so the article's numbers can be checked without rerunning the long experiments (`recheck.py`).

## Contents

| Path | What it is |
|---|---|
| `core.py` | QC-RDH reference implementation (classical twin of the circuits): key schedule, synthetic nonce, embed/extract, room-reserving configuration, freshness counter |
| `qcirc.py` | Gate-level Qiskit circuits of QC-RDH |
| `novelty.py` | Chosen-cover attack (Theorem 4) and gate-level check of the diffusion bound (Theorem 3) |
| `exp*.py`, `circsim*.py`, ... | One script per experiment (table below) |
| `bb84fk/` | Finite-key BB84 between two processes over TCP, with Wegman–Carter authentication, Cascade and Toeplitz privacy amplification |
| `fp/` | Chaotic-digest trajectory in C, JavaScript, Java, Go and Rust, plus the stored outputs (floating-point reproducibility test) |
| `results/` | All stored results (JSON, CSV, NPZ) |
| `data_med/` | Four public CT/MRI test volumes from pyradiomics (BSD licence) |
| `figs/` | Output folder for the figure scripts |

## Installation

Python 3.13 was used.

```bash
pip install -r requirements.txt
```

The DICOM experiments also need the pydicom test collection:

```bash
git clone https://github.com/pydicom/pydicom-data data_dicom
# or point to an existing copy:
export PYDICOM_DATA=/path/to/pydicom-data/data_store/data
```

## Quick check (minutes)

```bash
python recheck.py        # recomputes the quoted numbers from the stored results
python checks.py         # numerical audit of every lemma and formula (Table 10)
python ft_estimate.py    # surface-code estimate (Section 5.7)
python exp_malformed.py  # malformed-input handling (Section 7.3)
```

## Script for each result

Run the scripts from the repository root. `exp.py` must run first because it creates the BB84 key in `results/results.json` that the other experiments use.

| Article | Script | Output in `results/` |
|---|---|---|
| Table 4, Figures 4–5, Table 8, Table 9 (main results, key material, sensitivity, ablation) | `exp.py`, `fig_layout.py`, `figs.py` | `results.json`, `qcrdh_results.csv`, `ablation.csv`, `qber_runs.json` |
| Figure 2 (worked example) | `fig_example.py` | `example.npz` |
| Table 3, Section 5.7 (resources) | `resources512.py`, `mcx_cost.py`, `ft_estimate.py` | `resources512.json`, `resources_s1_k8.json`, `resources_rr.json`, `mcx_cost.json`, `ft_estimate.json` |
| Figure 3, Section 6.3 (chosen-cover attack) | `novelty.py`, `exp.py` | `attack_imgs.npz` |
| Figure 8 (check of Theorem 3) | `novelty.py`, `exp.py` | `theorem1_trials.csv` |
| Section 7.2 (freshness extension) | `exp_fresh.py` | `fresh.json` |
| Section 7.3 (malformed inputs) | `exp_malformed.py` | `malformed.json` |
| Section 7.4 (floating-point reproducibility) | `fp/compare.py` (sources and build commands in `fp/`) | `fp.json` |
| Section 8.1 (finite-key BB84) | `bb84fk/bb84_fk.py` | `bb84_finite_key.json` |
| Section 8.3, Table 5, Figure 6 (clinical CT/MRI) | `exp_medical.py`, `exp_medical_ext.py`, `fig_medical.py` | `medical.json`, `medical_ext.json`, `medical_imgs.npz` |
| Section 8.3 (29 DICOM images, five modalities) | `exp_dicom.py` | `dicom_modalities.json` |
| Section 8.4 (compact configurations) | `exp_compact.py`, `circsim_param.py` | `compact.json`, `circsim_s1_k8.json` |
| Section 8.4, Table 6 (room-reserving configuration) | `exp_rr.py`, `exp_rr_fill.py`, `circsim_rr.py`, `rr_statevector.py` | `rr.json`, `circsim_rr.json`, `rr_statevector.json` |
| Section 8.5, Table 7 (benchmark against AES-GCM/AES-SIV) | `exp_benchmark.py` | `benchmark.json` |
| Section 8.8, Figure 7 (circuit-level verification) | `circsim.py`, `depth.py`, `mcx_verify.py`, `fig_layout.py` | `circsim.json`, `depth.json`, `mcx_verify.json` |
| Section 8.9 (noise study) | `exp_noise.py` (shots and output file set by `SH`, `PS`, `OUT`) | `noise.json`, `noise_b.json` |
| Table 10 (numerical audit) | `checks.py`, `recheck.py` | `checks.json`, `recheck.json` |
| Section 9.2, Figure 9 (same-data JPEG-LS reference, capacity) | `exp_baseline.py`, `fig_capacity.py` | `baseline_jpegls.json` |

`run_all.sh` runs everything in this order. The full-size circuit executions (`circsim*.py`) and the noise study take the longest.

## Data

- **Natural images:** scikit-image test images (installed with scikit-image).
- **CT/MRI volumes:** pyradiomics test data (Harvard Medical School, BSD licence), included in `data_med/`.
- **DICOM images:** pydicom test files (MIT licence), from pydicom and pydicom-data.

No private patient data were used.

## Licence

The code is released under the MIT licence (`LICENSE`). The data in `data_med/` keep their original BSD licence.
