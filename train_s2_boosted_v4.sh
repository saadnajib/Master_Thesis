#!/bin/bash
# =============================================================================
# STEP 2 FOLLOW-UP (v4): partial fine-tune + helper-joint masking
#                        + per-joint loss boosting (neck + hands)
# =============================================================================
# Continues from the partial-finetune run's LATEST checkpoint (v3 reached
# PVE 75 on holdout by epoch 54) with THREE
# changes on top of v3:
#
#   --unfreeze_last_n_blocks 4     last 4 of the ViT-L's 24 blocks (+ final
#                                  norm) stay trainable. Unchanged from v3.
#
#   --mask_helper_joints_train 1   the 88 Anny helper/deform bones stay
#                                  masked out of the rotmat loss. Unchanged
#                                  from v3.
#
#   --boost_neck_weight 4          NEW. Multiplies the rotmat + j3d loss on
#   --boost_hand_weight 4          neck-chain/head and finger/thumb/wrist
#                                  bones by 4x (renormalized to mean 1, so the
#                                  total loss scale is unchanged - this
#                                  REDISTRIBUTES the loss budget toward these
#                                  joints rather than inflating it).
#                                  Motivated by verified demo evidence: the
#                                  neck stays elongated and fingers render
#                                  claw-like even after v3's fixes, because
#                                  the loss never told the network these
#                                  joints matter more than any other.
#
# EXPECTATIONS:
#   - rotmat loss value is not directly comparable to v3 (different weighting).
#     Judge on holdout PVE/PA-PVE/MPJPE, and on the SAME crop-and-diff visual
#     check used to validate the v3 inference-time fixes.
#   - Once a checkpoint exists, re-run the A/B/C demo comparison from before
#     to see how much of the residual neck/hand error this closes.
# =============================================================================
#SBATCH --job-name=mhmr_s2_boosted
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=96:00:00
#SBATCH --output=logs/train_s2_boosted_%j.out
#SBATCH --error=logs/train_s2_boosted_%j.err

COMMAND=$(cat <<'EOF'
# 1. Initialize conda
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

# 2. Fix Cache Paths (routed to netscratch, away from the over-quota ~ )
export HF_HOME="/netscratch/najib/hf_cache"
export TORCH_HOME="/netscratch/najib/torch_cache"
export TORCH_EXTENSIONS_DIR="/netscratch/najib/torch_extensions"
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p $HF_HOME $TORCH_HOME $TORCH_EXTENSIONS_DIR $XDG_CACHE_HOME

# Route anny's hardcoded ~/.cache/anny onto netscratch so create_fullbody_model
# cannot die on the home-dir disk quota.
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

# Fail fast if the continuation checkpoint is missing.
# >>> IMPORTANT: continuing from v3 (which reached PVE 75 on holdout), NOT v2.
#     Check the actual latest epoch before submitting:
#     ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_s2_partialft_v3/checkpoints/ | head -3
#     and edit V3_EPOCH below to match.
V3_EPOCH=00054   # <-- SET to the highest epoch number found above
CKPT=$(printf "/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_partialft_v3/checkpoints/%s.pt" "$V3_EPOCH")
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  echo "Check: ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_s2_partialft_v3/checkpoints/"
  exit 1
fi
echo "Continuing from $CKPT"

# 4. Run Training
#   Continuing from v3 (own checkpoint, same Model class) - remap/backbone-only
#       stay OFF, same reasoning as the v3 launch: our own key names already
#       match, and we want the trained heads back, not just the backbone.
#   --unfreeze_last_n_blocks 4 / --mask_helper_joints_train 1 : unchanged from v3.
#   --boost_neck_weight 4 / --boost_hand_weight 4 : NEW - see header.
#   --learning_rate 1e-5 : unchanged from v3 (still touching backbone blocks
#       of an already-trained model).
#   --name : NEW name -> fresh checkpoint dir.

python train.py \
    --train_data AnnyOne \
    --person_center head \
    --pretrained "$CKPT" \
    --pretrained_remap 0 \
    --load_only_backbone 0 \
    --freeze_backbone 1 \
    --unfreeze_last_n_blocks 4 \
    --mask_helper_joints_train 1 \
    --batch_size 4 \
    --log_freq 50 \
    --backbone dinov2_vitl14 \
    --img_size 672 \
    --n_iters_per_epoch 1000 \
    --max_iter 300000 \
    --num_workers 8 \
    --learning_rate 1e-5 \
    --lr_decay_every 30 \
    --lr_decay_gamma 0.5 \
    --amp 0 \
    --val_anny_n 100 \
    --eval_freq 5 \
    --use_anny_shape 1 \
    --start_2d_epoch 20 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --boost_neck_weight 4 \
    --boost_hand_weight 4 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_s2_boosted_v4
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"