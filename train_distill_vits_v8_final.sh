#!/bin/bash
# =============================================================================
# DISTILLATION v8: the FINAL student = v6c recipe (batch 16) + mixed real data
# =============================================================================
# The two improvements so far live in different runs:
#   v6c  batch 16, synthetic only   -> AnnyOne holdout 114.5 / 84.6 (best synthetic-only)
#   v7   batch 8, +30% real 3DPW    -> 3DPW-val 107.0 / 77.1, holdout 120.1 / 89.4
# This run starts from v6c's last checkpoint (epoch 149, copied to
# models/multiHMR/anny_distill_vits_v6c_00149.pt) and continues with v7's
# mixed data at batch 16 on an 80 GB card: 150 epochs, ~24-30 h.
# Both evaluations every 5 epochs; judged on 3DPW-validation; 3DPW-test is
# evaluated ONCE at the end with run_eval_3dpw_test.sh.
#
# NOTE the checkpoint cleanup keeps only the 10 highest epochs: when the run
# is done, copy the last checkpoint out before anything else is launched.
# =============================================================================
#SBATCH --job-name=mhmr_distill_s_v8final
#SBATCH --partition=A100-80GB
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_distill_vits_v8_final_%j.out
#SBATCH --error=logs/train_distill_vits_v8_final_%j.err

COMMAND=$(cat <<'EOF'
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

export HF_HOME="/netscratch/najib/hf_cache"
export TORCH_HOME="/netscratch/najib/torch_cache"
export TORCH_EXTENSIONS_DIR="/netscratch/najib/torch_extensions"
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p $HF_HOME $TORCH_HOME $TORCH_EXTENSIONS_DIR $XDG_CACHE_HOME
mkdir -p /netscratch/najib/anny_cache "$HOME/.cache"
[ -e "$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "$HOME/.cache/anny"

cd /netscratch/najib/multi-hmr
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)" || true
find /netscratch/najib/torch_cache/hub/facebookresearch_dinov2_main/dinov2 -name "*.py" -exec sed -i 's/ | None//g' {} +

# Refuse to run on an unpatched train.py - the flags would be silently ignored
# by argparse and the run would be a repeat of v2.
if ! grep -q "init_heads_from_teacher" train.py; then
  echo "ERROR: train.py is not patched. Run:  python apply_student_fix.py train.py"
  exit 1
fi
if ! grep -q "freeze_heads_epochs" train.py; then
  echo "ERROR: train.py lacks the warm-up patch. Run:  python apply_student_fix_v2.py train.py"
  exit 1
fi
LABELS="/netscratch/najib/multi-hmr/threedpw_anny_full/labels"
if [ ! -f "$LABELS/index_train.txt" ] || [ ! -f datasets/threedpw_anny.py ] || ! grep -q "mix_3dpw_labels" train.py; then
  echo "ERROR: mixed-data setup missing. Need $LABELS/index_train.txt (run_build_labels.sh),"
  echo "       datasets/threedpw_anny.py, and train.py patched by apply_mixed_data_fix.py"
  exit 1
fi
if ! grep -q "teacher_bf16" train.py || ! grep -q "backbone_autocast_dtype" model.py; then
  echo "ERROR: speed patch missing. Run:  python apply_speed_fix.py   (patches train.py and model.py)"
  exit 1
fi
if ! grep -q "head_dim" train.py || ! grep -q "feat_proj" model.py; then
  echo "ERROR: v3 patch missing. Run:  python apply_student_fix_v3.py   (patches model.py and train.py)"
  exit 1
fi

TEACHER="/netscratch/najib/multi-hmr/models/multiHMR/anny_teacher_v9_mixed_00062.pt"
[ -f "$TEACHER" ] || { echo "ERROR: teacher not found: $TEACHER"; exit 1; }
echo "Teacher (frozen): $TEACHER"

# >>> SET to the highest epoch in anny_distill_vits_v6/checkpoints/ <<<
V6_EPOCH="00149"
V6_CKPT="/netscratch/najib/multi-hmr/models/multiHMR/anny_distill_vits_v6c_${V6_EPOCH}.pt"   # the saved copy of v6c's last epoch
if [ ! -f "$V6_CKPT" ]; then
  echo "ERROR: v6c checkpoint not found: $V6_CKPT"
  echo "Copy it first: cp logs/anny_model/anny_distill_vits_v6c/checkpoints/00149.pt models/multiHMR/anny_distill_vits_v6c_00149.pt"
  exit 1
fi
echo "Resuming v6c student (batch-16 recipe) from $V6_CKPT with mixed data"

# --tf32 1 --teacher_bf16 1 : apply_speed_fix.py. Expect ~0.3-0.5 s/it instead
#     of 1.24; check the tqdm rate in the .err file after a few minutes.
# Check the log for "[pretrained] loaded 244/244" (whole student incl. projector)
# and "[distill] projector-fed heads: backbone 384 -> heads 1024".
python train.py \
    --train_data AnnyOne \
    --person_center head \
    --freeze_backbone 0 \
    --mask_helper_joints_train 1 \
    --backbone dinov2_vits14 \
    --pretrained "$V6_CKPT" \
    --pretrained_remap 0 \
    --load_only_backbone 0 \
    --distill_teacher_ckpt "$TEACHER" \
    --head_dim 1024 \
    --mix_3dpw_labels "$LABELS" \
    --mix_3dpw_ratio 0.3 \
    --mix_3dpw_split train \
    --val_3dpw_split validation \
    --val_3dpw_n 300 \
    --tf32 1 \
    --teacher_bf16 1 \
    --lambda_kd 1.0 \
    --lambda_kd_out 1.0 \
    --kd_temperature 4.0 \
    --kd_softmax_dim channel \
    --batch_size 16 \
    --log_freq 50 \
    --img_size 672 \
    --n_iters_per_epoch 1000 \
    --max_iter 150000 \
    --num_workers 12 \
    --learning_rate 3e-5 \
    --lr_decay_every 60 \
    --lr_decay_gamma 0.5 \
    --amp 0 \
    --val_anny_n 100 \
    --eval_freq 5 \
    --use_anny_shape 1 \
    --start_2d_epoch 0 \
    --alpha_shape 20 \
    --brightness 0.2 \
    --contrast 0.2 \
    --saturation 0.2 \
    --hue 0.05 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_distill_vits_v8_final
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
