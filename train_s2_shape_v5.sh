#!/bin/bash
# =============================================================================
# STEP 2 FOLLOW-UP (v5): fix the SHAPE head
# =============================================================================
# WHY THIS RUN EXISTS
# Measured against AnnyOne ground truth (shape-check job 3362031), the shape
# head is systematically wrong on three phenotypes, by 4-6x the data's own
# standard deviation:
#
#     phenotype    GT mean    predicted    offset      GT std
#     age            0.553       0.752      +0.198      0.036
#     height         0.579       0.425      -0.154      0.026
#     firmness       0.757       0.590      -0.167      0.039
#
# It is NOT guessing per person - it is parked near a fixed value for everyone.
# The cause is visible in loss.py: alpha_shape=1.0 against alpha_v3d=100.0 and
# alpha_j3d=100.0, so shape is supervised ~100x more weakly than the mesh. The
# network can lower the total loss far more by nudging vertices than by getting
# body attributes right, so it stops learning shape.
#
#   --alpha_shape 20   raise shape supervision 20x (from the default 1.0).
#                      Deliberately below the 100 used for j3d/v3d: the goal is
#                      to make shape matter, not to dominate the geometry terms.
#
# WHAT THIS RUN WILL *NOT* FIX: the elongated neck. That was traced and every
# candidate eliminated by test - not the body model (--rest_pose renders a
# normal neck), not the helper-joint mask (all 4 neck bones active), not a
# domain gap (long necks appear on AnnyOne images too), and not shape
# ('proportions' is accurate to 0.018). What remains is linear blend skinning
# stretching across Anny's stacked neck01/02/03 + head bones, which is a
# rigging property, not something training can change.
#
# HOW TO JUDGE IT: rerun the shape check against a checkpoint from this run.
# Success = the age/height/firmness offsets shrink toward the GT means. Also
# watch holdout PVE does not regress (shape and geometry now compete more).
# =============================================================================
#SBATCH --job-name=mhmr_s2_shape
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_s2_shape_%j.out
#SBATCH --error=logs/train_s2_shape_%j.err

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
# >>> continuing from v4 (holdout PVE ~73.6).
#     Check the actual latest epoch before submitting:
#     ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_s2_boosted_v4/checkpoints/ | head -3
#     and edit V4_EPOCH below to match.
V4_EPOCH=00299   
CKPT=$(printf "/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_boosted_v4/checkpoints/%s.pt" "$V4_EPOCH")
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  echo "Check: ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_s2_boosted_v4/checkpoints/"
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
    --max_iter 100000 \
    --num_workers 8 \
    --learning_rate 1e-5 \
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
    --boost_neck_weight 4 \
    --boost_hand_weight 4 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_s2_shape_v5
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
