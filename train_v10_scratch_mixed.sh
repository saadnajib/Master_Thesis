#!/bin/bash
# =============================================================================
# v10: FRESH model from multiHMR_672_L_anny on AnnyOne + converted 3DPW
# =============================================================================
# Identical to train_v8_partial_freeze.sh (the synthetic-only run
# anny_partial_freeze): DINOv2 ViT-L backbone from Naver's multiHMR_672_L_anny
# (--pretrained_remap 1 --load_only_backbone 1), heads from scratch, backbone
# frozen except the last 4 blocks, helper-joint mask, alpha_shape 20, LR 1e-5.
# The ONLY change: 30% of every batch is real 3DPW-train (converted to Anny by
# smpl_to_anny.py v2.2), and the model is evaluated on both the AnnyOne holdout
# and 300 converted 3DPW-validation images every 5 epochs.
# -> anny_partial_freeze vs anny_v10_scratch_mixed at equal epochs isolates
#    what real data does when it is present from the start.
#
# BUDGET: ~9-10 min/epoch at batch 4 with TF32 -> the 48 h limit covers
# ~280 epochs. --max_iter 300000 (300 epochs). If it is killed by the limit,
# resume the highest checkpoint under a NEW name (--pretrained_remap 0
# --load_only_backbone 0, same flags otherwise).
#
# Needs: train.py patched by apply_mixed_data_fix.py (+v2) and apply_speed_fix.py;
# datasets/threedpw_anny.py; threedpw_anny_full/labels/index_train.txt.
# =============================================================================
#SBATCH --job-name=mhmr_v10_scratch_mixed
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v10_scratch_mixed_%j.out
#SBATCH --error=logs/train_v10_scratch_mixed_%j.err

COMMAND=$(cat <<'EOF'
# 1. Initialize conda
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

# 2. Fix Cache Paths
export HF_HOME="/netscratch/najib/hf_cache"
export TORCH_HOME="/netscratch/najib/torch_cache"
export TORCH_EXTENSIONS_DIR="/netscratch/najib/torch_extensions"
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p $HF_HOME $TORCH_HOME $TORCH_EXTENSIONS_DIR $XDG_CACHE_HOME

mkdir -p /netscratch/najib/anny_cache
mkdir -p "$HOME/.cache"
[ -e "$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "$HOME/.cache/anny"

# 3. Navigate
cd /netscratch/najib/multi-hmr

# --- CRITICAL FIX FOR DEADLOCKS ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# ----------------------------------

echo "Ensuring DINOv2 is downloaded to cache..."
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)" || true

echo "Patching DINOv2 cache to remove Python 3.10+ type hints..."
find /netscratch/najib/torch_cache/hub/facebookresearch_dinov2_main/dinov2 -name "*.py" -exec sed -i 's/ | None//g' {} +

CKPT="/netscratch/najib/multi-hmr/models/multiHMR/multiHMR_672_L_anny.pt"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: base checkpoint not found: $CKPT"
  exit 1
fi
echo "Fresh heads + partial freeze, mixed data; backbone initialised from $CKPT"

LABELS="/netscratch/najib/multi-hmr/threedpw_anny_full/labels"
if [ ! -f "$LABELS/index_train.txt" ] || [ ! -f datasets/threedpw_anny.py ] || ! grep -q "mix_3dpw_labels" train.py; then
  echo "ERROR: mixed-data setup missing (labels, datasets/threedpw_anny.py, or apply_mixed_data_fix.py)"
  exit 1
fi

# 4. Run Training
python train.py \
    --train_data AnnyOne \
    --person_center head \
    --pretrained "$CKPT" \
    --pretrained_remap 1 \
    --load_only_backbone 1 \
    --freeze_backbone 1 \
    --unfreeze_last_n_blocks 4 \
    --mask_helper_joints_train 1 \
    --batch_size 4 \
    --log_freq 50 \
    --backbone dinov2_vitl14 \
    --img_size 672 \
    --n_iters_per_epoch 1000 \
    --max_iter 300000 \
    --num_workers 12 \
    --learning_rate 1e-5 \
    --lr_decay_every 30 \
    --lr_decay_gamma 0.5 \
    --amp 0 \
    --val_anny_n 100 \
    --eval_freq 5 \
    --use_anny_shape 1 \
    --start_2d_epoch 20 \
    --alpha_shape 20 \
    --tf32 1 \
    --mix_3dpw_labels "$LABELS" \
    --mix_3dpw_ratio 0.3 \
    --mix_3dpw_split train \
    --val_3dpw_split validation \
    --val_3dpw_n 300 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_v10_scratch_mixed
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"