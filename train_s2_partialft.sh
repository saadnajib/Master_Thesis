#!/bin/bash
# WARNING: header below describes the v3 partial-FT run, but the flags now pass --boost_neck/hand_weight 4 and --name anny_s2_boosted_v4 (same name as train_s2_boosted_v4.sh) - outputs land in the v4 folder.
# =============================================================================
# STEP 2 FOLLOW-UP (v3): partial fine-tune + helper-joint masking
# =============================================================================
# Continues from the frozen-backbone run's checkpoint 00299 with two changes:
#
#   --unfreeze_last_n_blocks 4     last 4 of the ViT-L's 24 blocks (+ final
#                                  norm) become trainable again. Early blocks
#                                  keep their generic features frozen; the
#                                  task-specific late blocks adapt to AnnyOne.
#
#   --mask_helper_joints_train 1   the 88 Anny helper/deform bones (breast,
#                                  spine/neck subdivisions, face micro-bones)
#                                  are forced to identity during training AND
#                                  in the GT targets, so the rotmat loss
#                                  concentrates entirely on the 75 learnable
#                                  joints - including the cervical neck chain
#                                  behind the remaining neck deformation.
#
# EXPECTATIONS:
#   - rotmat loss DROPS IMMEDIATELY at start: that is the 88 masked joints
#     leaving the sum, not sudden learning. Judge on holdout PVE/MPJPE.
#   - This is a THIRD configuration, not the "frozen" arm of the original
#     step1-vs-step2 comparison. Keep 00299's results as the step-2 answer.
#   - Speed will land between the frozen (3.8 it/s) and full (1.4 it/s) runs.
# =============================================================================
#SBATCH --job-name=mhmr_s2_partialft
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_s2_partialft_%j.out
#SBATCH --error=logs/train_s2_partialft_%j.err

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
CKPT="/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_frozen_bb_v2/checkpoints/00299.pt"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  echo "Check: ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_s2_frozen_bb_v2/checkpoints/"
  exit 1
fi
echo "Continuing from $CKPT"

# 4. Run Training
#   --pretrained_remap 0 / --load_only_backbone 0 : loading OUR OWN full
#       checkpoint (same class, same key names) - remapping would corrupt it,
#       backbone-only would throw away the trained heads.
#   --freeze_backbone 1 + --unfreeze_last_n_blocks 4 : freeze everything,
#       then re-enable the last 4 ViT blocks + final norm.
#   --learning_rate 1e-5 : lower than the heads-only 3e-5 because backbone
#       blocks of an already-trained model are now being touched.
#   --name : NEW name -> fresh checkpoint dir (never reuse a name whose folder
#       holds higher-numbered checkpoints; cleanup keeps the 10 HIGHEST).

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
