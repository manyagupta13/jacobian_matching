# 33k-parameter starting models

The pretrained emulators that every 33k fine-tuning run starts from: the APEBench presets
`Res;26;8;relu` (ResNet) and `FNO;12;18;4;gelu` (FNO) on four 1D chaotic PDEs, five seeds each.

```
weights_resnet/   <pde>__resnet__s<seed>__best.eqx   20 files
weights_fno/      <pde>__fno__s<seed>__best.eqx      20 files
trainlogs/        <pde>__<arch>__s<seed>__hist.json  pretraining loss and validation curves
analysis/         job0_fno_results.json, job1_resnet_results.json   per-model records, incl. the
                  Lyapunov exponent per step that the fine-tuning notebooks read
```

`<pde>` is one of `ks_L60`, `gks_d0.2`, `cgle_phaseL1`, `nik_r0.02`.
`best` is the pretraining checkpoint with the lowest validation **rollout** error
(geometric mean of the relative error over 100 steps, 16 validation trajectories).

Pretraining: 150k updates, batch 32, Adam, gradient-norm clipping at 1, learning rate warmed up
to 1e-3 over 2,000 updates and cosine-decayed to 1e-5. Ten ResNet runs stopped at 145k updates
when the compute budget ran out (all five `cgle_phaseL1` seeds, `gks_d0.2` s0, `ks_L60` s1–s3,
`nik_r0.02` s3); their `trainlogs` record the true final update count.
They were produced by `code/33k/pretrain/jacobian_study_JOB{0_fno,1_resnet}.py`, which differ only
in the `JOB` constant.

The Nikolaevskiy FNO did not learn the one-step map (relative one-step error about 39% on four of
five seeds) and is not fine-tuned in the paper.

Load a model with
```python
import jax, equinox as eqx, pdecore as pc            # pdecore.py is written by any of the notebooks
net = pc.make_net("fno", 1, jax.random.PRNGKey(0))  # template; nch = 2 for cgle_phaseL1
net = eqx.tree_deserialise_leaves("weights_fno/ks_L60__fno__s0__best.eqx", net)
```
