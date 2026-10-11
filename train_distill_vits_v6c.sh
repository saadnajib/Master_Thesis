#!/bin/bash
# =============================================================================
# DISTILLATION v6c: v6b + BATCH 16 on an 80 GB GPU
# =============================================================================
# v6b (job 3494692) runs at 0.47 s/it with TF32 + bf16 teacher backbone, i.e.
# the teacher no longer saturates the GPU. A bigger batch should therefore cost
# little extra time per iteration while doubling the images seen per epoch
# (16,000 vs 8,000 at 1000 it/epoch). Same v6 checkpoint as v6b, so v6b vs v6c
# is a clean batch-size comparison at equal epochs.
#   --batch_size 16      needs the 80 GB partition (batch 8 was sized for 40 GB)
#   --learning_rate 3e-5 v6b's 2e-5 scaled by ~sqrt(2) for the doubled batch
#   if the first iteration OOMs: --batch_size 12 --learning_rate 2.5e-5
# Everything else identical to v6b. BEFORE SUBMITTING: set V6_EPOCH to the
# highest checkpoint in anny_distill_vits_v6b/checkpoints/ (same as v6b).
# =============================================================================
#SBATCH --job-name=mhmr_distill_s_v6c
#SBATCH --partition=A100-80GB
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_distill_vits_v6c_%j.out
#SBATCH --error=logs/train_distill_vits_v6c_%j.err

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
if ! grep -q "teacher_bf16" train.py || ! grep -q "backbone_autocast_dtype" model.py; then
  echo "ERROR: speed patch missing. Run:  python apply_speed_fix.py   (patches train.py and model.py)"
  exit 1
fi
if ! grep -q "head_dim" train.py || ! grep -q "feat_proj" model.py; then
  echo "ERROR: v3 patch missing. Run:  python apply_student_fix_v3.py   (patches model.py and train.py)"
  exit 1
fi

TEACHER="/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt"
[ -f "$TEACHER" ] || { echo "ERROR: teacher not found: $TEACHER"; exit 1; }
echo "Teacher (frozen): $TEACHER"

# >>> SET to the highest epoch in anny_distill_vits_v6b/checkpoints/ <<<
V6_EPOCH="00140"
V6_CKPT="/netscratch/najib/multi-hmr/logs/anny_model/anny_distill_vits_v6b/checkpoints/${V6_EPOCH}.pt"
if [ ! -f "$V6_CKPT" ]; then
  echo "ERROR: v6 checkpoint not found: $V6_CKPT"
  echo "Check: ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_distill_vits_v6b/checkpoints/ | head"
  exit 1
fi
echo "Resuming v6 student from $V6_CKPT"

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
    --name anny_distill_vits_v6c
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
