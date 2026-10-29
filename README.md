# When Does Jacobian Fine-Tuning Stabilize Neural PDE Emulators? — code and results

Code, starting checkpoints and per-run results for the paper. Everything in the paper's
figures and tables can be rebuilt from `results/` in about half a minute on a CPU:

```bash
pip install numpy scipy matplotlib jax equinox
bash analysis/make_all.sh          # writes figures/
```

Rerunning the experiments themselves needs a GPU (we used single NVIDIA T4s on Kaggle).

## Layout

```
code/
  small/          C1_small_{ks,gks,cgle}.ipynb   main study (8-10k parameters): data, pretraining of
                                                 five architectures x 3 seeds, all four fine-tuning
                                                 methods, analysis. One notebook per PDE.
  33k/pretrain/   jacobian_study_JOB{0_fno,1_resnet}.py   pretraining of the 33k APEBench presets
  33k/finetune/   fine-tuning of the 33k models (see the table below)
  33k/noise/      A5_noise_ks_resnet.ipynb       input-noise baseline, KS ResNet, 3 noise levels
checkpoints_33k/  the 40 pretrained 33k starting models + their training logs (see its README)
results/          one .npz per (PDE, architecture, seed, method, checkpoint): test rollout error,
                  counterfactual error, one-step error, ...; plus training histories
  small/{ks,gks,cgle}/
  33k_round1/  33k/  33k_noise/
analysis/         scripts that turn results/ into every figure and number in the paper
```

Each notebook is self-contained: it writes the shared modules it needs (`pdecore.py`: ETDRK4
solver and architectures; `ccore.py`: Jacobian-vector products, counterfactual error,
Lyapunov exponents; `oagg.py`: aggregation; `worker.py` for the small study) and runs a set of
built-in checks (solver tangent map against finite differences, penalty gradients against
finite differences) before training.

## Running the experiments

All notebooks read the starting models and the Lyapunov exponents from `checkpoints_33k/`.
On Kaggle, attach that folder as a dataset; elsewhere set

```bash
export PDE_DATA=/path/to/checkpoints_33k
export OUT=/path/to/output            # default ./own_out
```

and run the notebook top to bottom. Data are regenerated from fixed seeds
(train 1000, validation 4242, test 773), so nothing needs to be downloaded.
Output goes to `$OUT/own/{res,hist,ckpt}`, the same layout as `results/`.

| results folder | notebook | runs |
|---|---|---|
| `small/ks`, `small/gks`, `small/cgle` | `code/small/C1_small_*.ipynb` | 5 architectures x 3 seeds; Cont. training, RJM, Noise, EDJM |
| `33k/A0_screen_resnet`, `33k/A0_screen_fno` | `A0_screen_*.ipynb` | analysis of the starting models (no training) |
| `33k_round1/b_ks_resnet_own`, `d_gks_resnet_own` | `own_error_finetune.ipynb` | KS and gKS ResNet, seeds 0-1 |
| `33k_round1/a_gks_resnet_randcos`, `c_ks_resnet_randcos` | `randcos_finetune.ipynb` | KS and gKS ResNet, seeds 0-1 |
| `33k/A1_ks_resnet_s234` | `A1_ks_resnet_s234.ipynb` | KS ResNet, seeds 2-4 |
| `33k/A1_ks_fno`, `A1_nik_resnet`, `A1_cgle_resnet` | `A1_*.ipynb` | 5 seeds each |
| `33k/A2_gks_fno`, `A2_cgle_fno` | `A2_*.ipynb` | 5 seeds each |
| `33k_noise/ks_resnet` | `code/33k/noise/A5_noise_ks_resnet.ipynb` | input noise at 0.3x, 1x, 3x the one-step error |

Method names in file names: `mse` = continued training, `rand100` = RJM, `own100` = EDJM,
`noise1` = input noise at 1x, `base` = the starting model; `best` / `final` = validation-selected
and last checkpoint. Some 33k notebooks also ran variants that the paper does not report
(`own10`, `own1000`, `owncos100`, `both100`, `cos100`, `randcos100_*`); their results are
included but unused.

## From results to the paper

`analysis/make_all.sh` runs, in order:

| script | produces |
|---|---|
| `build_results.py` | Figures 1-3; every number in Section 5; Table 6 (`figures/results_facts.json`) |
| `build_fig_arms.py` | Figure 4 |
| `build_appendix_ledger.py` | Figures 5-6 |
| `build_tables.py` | Tables 4-5 (`figures/t_app_configs.tex`, `figures/t_app_bins.tex`) |
| `robust.py` | Table 7 (`figures/robust.json`) |

`load_all.py` and `res_common.py` load every result file into one record per seed.
The KS CNN at 8-10k parameters diverges under every method, including continued training,
and is excluded from all analyses.

## Requirements

See `requirements.txt`. Training ran with JAX 0.11.1 (Kaggle's preinstalled version), Equinox and
Optax; the solver runs in float64 and the emulators in float32.
