#!/bin/bash
# =============================================================================
# v7: FRESH TRAINING (not a continuation)
# =============================================================================
# Trains a NEW model instead of continuing the v3->v4->v5 chain, applying
# everything learned along the way FROM THE START rather than discovering it
# mid-run. Useful as a clean baseline: the current best model carries the
# history of several mid-course corrections, so its numbers are hard to
# attribute to any single choice.
#
# "From scratch" = the HEADS start fresh. The DINOv2 backbone is still
# initialised from multiHMR_672_L_anny, because a randomly initialised ViT-L
# needs far more data and time than this schedule allows - and the original
# authors did the same (load_only_backbone=1 in their saved args). For a truly
# random start, drop --pretrained entirely, but expect much worse results.
#
# APPLIED FROM THE START (each measured, not assumed):
#   --pretrained_remap 1        encoder.backbone.* -> backbone.encoder.*, the
#                               name mismatch that silently loaded 0 weights
#   --load_only_backbone 1      only the ViT transfers; the head architecture
#                               differs so its weights cannot
#   --freeze_backbone 0         full fine-tuning - this is a fresh run with the
#                               budget for it, unlike the frozen/partial
#                               experiments which were continuations
#   --mask_helper_joints_train 1  88 helper bones out of the rotmat loss; this
#                               fixed the chest crease artifact
#   --alpha_shape 20            the v5 fix. At the default 1.0 (vs alpha_v3d
#                               100.0) the shape head parks on dataset averages:
#                               age was off +0.199, height -0.154. With 20 those
#                               became -0.001 and -0.026.
#   --learning_rate 1e-4        right for randomly initialised heads; the
#                               5e-6/1e-5 values were for fine-tuning converged
#                               models
#   --lr_decay_every 30 / gamma 0.5   a constant LR on a partially trained model
#                               caused hundreds of NaN gradient events earlier
#   --start_2d_epoch 20         2D losses off early: with fresh heads the
#                               predictions are meaningless and reprojection
#                               gradients only add noise
#
# NOT APPLIED: --boost_neck_weight / --boost_hand_weight. Both were tested at 4x
# in v4 with no visible improvement, and the neck cause was later traced to
# skinning rather than pose. Left off to keep this baseline clean.
# =============================================================================
#SBATCH --job-name=mhmr_v7_scratch
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v7_scratch_%j.out
#SBATCH --error=logs/train_v7_scratch_%j.err

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

# Fresh run: the ORIGINAL repo checkpoint, backbone only.
CKPT="/netscratch/najib/multi-hmr/models/multiHMR/multiHMR_672_L_anny.pt"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: base checkpoint not found: $CKPT"
  exit 1
fi
echo "Fresh training; backbone initialised from $CKPT"

# 4. Run Training
#   Fresh run from the ORIGINAL repo checkpoint: --pretrained_remap 1 and
#       --load_only_backbone 1 load only the ViT-L backbone; heads start random.
#   --freeze_backbone 0 : whole backbone trains.
#   --mask_helper_joints_train 1 : helper/deform bones masked in the loss.
#   No --boost_* flags and no --unfreeze_last_n_blocks.
#   --learning_rate 1e-4 (halved every 30 epochs), 300 capped epochs.
#   --name anny_v7_scratch : fresh checkpoint dir.

python train.py \
    --train_data AnnyOne \
    --person_center head \
    --pretrained "$CKPT" \
    --pretrained_remap 1 \
    --load_only_backbone 1 \
    --freeze_backbone 0 \
    --mask_helper_joints_train 1 \
    --batch_size 4 \
    --log_freq 50 \
    --backbone dinov2_vitl14 \
    --img_size 672 \
    --n_iters_per_epoch 1000 \
    --max_iter 300000 \
    --num_workers 8 \
    --learning_rate 1e-4 \
    --lr_decay_every 30 \
    --lr_decay_gamma 0.5 \
    --amp 0 \
    --val_anny_n 100 \
    --eval_freq 5 \
    --use_anny_shape 1 \
    --start_2d_epoch 20 \
    --alpha_shape 20 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_v7_scratch
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
