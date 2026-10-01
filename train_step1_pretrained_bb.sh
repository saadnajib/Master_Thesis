#!/bin/bash
# =============================================================================
# TASK STEP 1: train a new model, initialised from multiHMR_672_L_anny
# =============================================================================
# multiHMR_672_L_anny.pt was trained with the ORIGINAL repo's Multi_HMR class
# (multi_hmr_anny/multi_hmr.py), not with this train.py's Model (model.py).
# Their module names differ, so train.py remaps the DINOv2 ViT keys
#     encoder.backbone.*   ->   backbone.encoder.*
# (--pretrained_remap 1) and keeps only those (--load_only_backbone 1).
#
# What transfers : the DINOv2 ViT-L encoder, 343 tensors / ~303M params, already
#                  fine-tuned on human data (annyone, bedlam, coco, mpii).
# What does NOT  : encoder.mlp_det / mlp_fov_unique (this Model takes K as input)
#                  and the decoder / mlp_pose / mlp_shape / mlp_dist stack, which
#                  is a different head architecture from x_attention_head.
#                  Those start from fresh init - hence "train a new model".
#
# Step 2 (frozen backbone) is the same script with --freeze_backbone 1;
# see train_step2_frozen_bb.sh. Keep every other hyperparameter identical
# between the two so the comparison isolates the freeze.
# =============================================================================
#SBATCH --job-name=mhmr_s1_bb
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_s1_pretrained_bb_%j.out
#SBATCH --error=logs/train_s1_pretrained_bb_%j.err

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
# (used when --use_anny_shape 1) cannot die on the home-dir disk quota.
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

# Fail fast if the base checkpoint is missing, rather than 20 minutes in.
CKPT="/netscratch/najib/multi-hmr/models/multiHMR/multiHMR_672_L_anny.pt"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: base checkpoint not found: $CKPT"
  exit 1
fi

# 4. Run Training
#
# SETTINGS SPECIFIC TO THIS TASK (differ from the earlier fine-tuning runs):
#   --pretrained          : the ORIGINAL repo checkpoint, not one of our runs
#   --pretrained_remap 1  : encoder.backbone.* -> backbone.encoder.* (see header)
#   --load_only_backbone 1: heads start fresh; only the ViT transfers
#   --freeze_backbone 0   : backbone TRAINS in this step (step 2 sets this to 1)
#   --learning_rate 1e-4  : RAISED from 5e-6. That value suited fine-tuning an
#                           already-converged model; here every head is randomly
#                           initialised and must learn from scratch, so a
#                           fine-tuning LR would crawl. Keep this IDENTICAL in
#                           step 2 so the comparison isolates the freeze.
#   --backbone/--img_size : must match the checkpoint (ViT-L @ 672) or the
#                           90% backbone-match guard in train.py will abort.
#   --start_2d_epoch 20   : keep 2D losses off early; with fresh heads the
#                           predictions are meaningless at first and 2D
#                           reprojection gradients would just add noise.
#   --name                : distinct per run -> distinct ckpt_dir. Never reuse a
#                           name whose folder holds higher-numbered checkpoints;
#                           the cleanup keeps the ten HIGHEST epoch numbers, so
#                           a restarted run's checkpoints get deleted instantly.

python train.py \
    --train_data AnnyOne \
    --person_center head \
    --pretrained "$CKPT" \
    --pretrained_remap 1 \
    --load_only_backbone 1 \
    --freeze_backbone 0 \
    --batch_size 4 \
    --log_freq 50 \
    --backbone dinov2_vitl14 \
    --img_size 672 \
    --n_iters_per_epoch 1000 \
    --max_iter 150000 \
    --num_workers 8 \
    --learning_rate 1e-4 \
    --amp 0 \
    --val_anny_n 100 \
    --eval_freq 5 \
    --use_anny_shape 1 \
    --start_2d_epoch 20 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_s1_pretrained_bb_v2
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
