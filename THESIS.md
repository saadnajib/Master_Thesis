# Master thesis: Multi-HMR with the Anny body model, and ViT-S distillation

Author: Muhammad Saad Najib. Base code: Multi-HMR (Baradel et al., ECCV 2024, Naver Labs).
The upstream README.md documents the base method. This file documents what the thesis
adds, how the experiments chain together, and where the evidence lives.

## Research question

Can the single-shot multi-person mesh recovery model Multi-HMR be re-targeted from
SMPL-X to the **Anny** parametric body model (163 joints, 11 phenotype parameters)
without loss of accuracy, and can the resulting ViT-L model be **distilled** into a
ViT-S student that is usable on commodity hardware?

## Contributions (delta against upstream Multi-HMR)

1. **Anny body-model port.** `model.py` replaces the SMPL-X layer with an Anny layer and
   an 11-dimensional shape head; `loss.py` computes losses on Anny joints and vertices.
   `multi_hmr_anny/` holds a second encoder/head implementation used only to load the
   original-repo checkpoints in `demo.py` and `inspect_ckpt.py`. Training uses the
   `model.py` path.
2. **3DPW-to-Anny adapter and AnnyOne dataset.** `smpl_to_anny.py` fits Anny to SMPL
   ground truth; `npz_to_annyone.py` converts to the AnnyOne format. Fit quality is
   13.0 mm mean on reliable limbs, 40.9 mm over all 24 joints (see `adapter_check.md`).
   The AnnyOne training set has 579,960 samples (`shape_report.txt`).
3. **Two-stage fine-tuning recipe.** Pretrained backbone with fresh heads, then frozen
   backbone, then partial unfreezing of the last 4 blocks, then neck/hand loss boosting,
   then a shape-loss weight of 20. The final ViT-L model is `anny_s2_shape_v5/00099`,
   referred to below as **the teacher**.
4. **Knowledge distillation to ViT-S.** `distill.py` applies a KL divergence on backbone
   patch tokens (softmax over channels, temperature 4, scaled by T^2) through a linear
   384-to-1024 projector, plus an output-level L1 on rotation matrices, shape and
   distance from v3 onwards. Teacher heads are copied into the student from v3.

## Experiment chain (reconstructed from the scripts)

Each row starts from the checkpoint in "from". Flags are in the named script.

| Stage | Script | From | Change vs previous |
|---|---|---|---|
| Smoke test | `train_anny_test.sh` | scratch | small run |
| Full run v1 | `train_anny.sh` | scratch | lr 5e-6 |
| Medium | `train_anny_medium.sh` | full_run_v2/00299 | shape loss on, 2D losses from epoch 20 |
| Step 1 | `train_step1_pretrained_bb.sh` | backbone of multiHMR_672_L_anny | fresh heads, lr 1e-4 |
| Step 2 | `train_step2_frozen_bb.sh` | as step 1 | backbone frozen |
| s2 partial FT (v3) | `train_s2_partialft.sh` | s2_frozen_bb_v2/00299 | unfreeze last 4 blocks, helper-joint mask, lr 1e-5 |
| s2 boosted (v4) | `train_s2_boosted_v4.sh` | s2_partialft_v3/00054 | neck/hand weight 4 |
| **s2 shape (v5, teacher)** | `train_s2_shape_v5.sh` | s2_boosted_v4/00299 | alpha_shape 20, 100k iters |
| Distill v1 | `train_distill_vits.sh` | fresh ViT-S | token KD, lr 1e-4 |
| Distill v2 | `train_distill_vits_v2.sh` | distill_vits_v2/00199 | lr 2e-5 |
| Distill v3 | `train_distill_vits_v3.sh` | fresh ViT-S | heads from teacher, output KD 1.0 |
| Distill v4 | `train_distill_vits_v4.sh` | multiHMR_672_S backbone | colour jitter, batch 8 |
| Distill v5 | `train_distill_vits_v5.sh` | as v4 | heads frozen 5 epochs |
| Distill v6 | `train_distill_vits_v6.sh` | as v5 | head_dim 1024 |
| v7 scratch | `train_v7_scratch.sh` | step-1 recipe | helper mask + shape 20, no boost |
| v8 partial freeze | `train_v8_partial_freeze.sh` | multiHMR_672_L_anny | freeze + unfreeze 4, shape 20 |
| v8 resume | `train_v8_resume.sh` | anny_partial_freeze/00299 | boost 4 |

Known inconsistencies in the scripts are flagged in their header comments
(`train_s2_partialft.sh` writes to the v4 name; the two resume scripts read and write
the same folder).

## Results so far (from script comments; to be replaced by `results/*.csv`)

| Model | PVE (mm) | Source |
|---|---|---|
| ViT-L teacher, v5 | ~73.6 | comment in `train_s2_shape_v5.sh` |
| ViT-L, v3 partial FT | ~75 | comment in `train_s2_boosted_v4.sh` |
| ViT-S student, v3 | 193.4 | comment in `train_distill_vits_v3.sh` |
| ViT-S student, v5 | 315 | comment in `train_distill_vits_v6.sh` |

These numbers come from the last 100 AnnyOne samples, which were also the validation
split during training. They are validation scores, not test scores.

## Evaluation protocol (target)

- **Held-out AnnyOne test set:** `run_eval_test.sh` evaluates the last `--test_anny_n`
  samples; validation uses the `--val_anny_n` samples before them. Train excludes both.
  **Caveat:** every existing checkpoint, the v5 teacher included, was trained on all but
  the last 100 samples, and those 100 were used to pick checkpoints. A clean test number
  needs one retraining run with `--test_anny_n 500 --val_anny_n 100`. Until then,
  `train.py` records how many evaluated samples the checkpoint saw in training in the
  `n_seen_in_training` column.
- **3DPW:** `run_eval_3dpw.sh` runs the 3DPW path in `train.py` on the teacher.
- Metrics: PVE, PA-PVE, MPJPE, PA-MPJPE over 163 Anny joints, plus detection
  precision, recall and F1. Every evaluation appends a row to `results/<run>.csv`
  and `results/all_results.jsonl`.

## Open questions for the writing phase

1. The distilled student is 2.5 to 4 times worse than the teacher and got worse from
   v3 to v5. Either diagnose it (is the patch-token KL correlated with PVE at all? does
   the 384-to-1024 projector overfit?) or report distillation as a negative result.
2. `train_v7_scratch.sh` states that neck/hand boosting gave no visible improvement,
   yet v5 and v8-resume keep it. One ablation row settles this.
3. No 3DPW number exists yet. Without it there is no comparison to upstream Multi-HMR.

## Reproduction

- Cluster paths: data `/netscratch/najib/anydataset/`, 3DPW `/ds-av/public_datasets/3DPW/original`,
  checkpoints `/netscratch/najib/multi-hmr/models/multiHMR/`, logs
  `/netscratch/najib/multi-hmr/logs/anny_model/<run>/checkpoints/`,
  container `/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh`.
- The dataloader package `datasets/` must be copied from the cluster (see `datasets/README.md`).
- Environment: `conda.yaml` (Python 3.9, CUDA 11.7), `requirements.txt` (torch 2.0.1, xformers 0.0.20).
  The DINOv2 hub code needs the `| None` type-hint patch applied in the distill scripts.
