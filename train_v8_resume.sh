#!/bin/bash
# =============================================================================
# v6: CONTINUE from v5 for 300 more epochs
# =============================================================================
# v5 fixed the shape head. Measured in-domain against AnnyOne ground truth:
#
#     phenotype      GT      v4      v5     v4 err    v5 err
#     age          0.553   0.752   0.552    +0.199    -0.001   <- fixed
#     height       0.579   0.425   0.553    -0.154    -0.026   <- fixed
#     proportions  0.763   0.746   0.762    -0.017    -0.001   <- fixed
#     firmness     0.757   0.590   0.567    -0.167    -0.190   <- still off
#
# This run simply gives the same recipe more time (300 epochs). Everything is
# carried over unchanged from v5 so the comparison stays clean.
#
# NOTE ON THE NECK: unchanged and expected to stay unchanged. Four candidate
# causes were each eliminated by measurement - body model (--rest_pose renders
# a normal neck), helper-joint mask (all 4 neck bones active in the loss),
# domain gap (long necks appear on AnnyOne images too), and shape
# ('proportions' now accurate to 0.001). What remains is linear blend skinning
# across Anny's stacked neck01/02/03 + head bones: a rigging property, not
# something training changes. Judge this run on metrics, not on necks.
#
# HOW TO JUDGE: holdout PVE/PA-PVE/MPJPE in the log, and rerun the shape check
# against a v6 demo. Watch whether 'firmness' finally moves.
# =============================================================================
#SBATCH --job-name=mhmr_v6_resume
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v6_resume_%j.out
#SBATCH --error=logs/train_v6_resume_%j.err

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
# >>> continuing from v5 (the run that fixed the shape head).
#     Check the actual latest epoch before submitting:
#     ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_partial_freeze/checkpoints/ | head -3
#     and edit V5_EPOCH below to match.
V5_EPOCH=00299   # <-- SET to the highest epoch in anny_partial_freeze/checkpoints/
CKPT=$(printf "/netscratch/najib/multi-hmr/logs/anny_model/anny_partial_freeze/checkpoints/%s.pt" "$V5_EPOCH")
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  echo "Check: ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_partial_freeze/checkpoints/"
  exit 1
fi
echo "Continuing from $CKPT"

# 4. Run Training
#   Continuing from v5 (own checkpoint, same Model class) - remap/backbone-only
#       stay OFF, same reasoning as the v3 launch: our own key names already
#       match, and we want the trained heads back, not just the backbone.
#   --unfreeze_last_n_blocks 4 / --mask_helper_joints_train 1 : unchanged from v5.
#   --boost_neck_weight 4 / --boost_hand_weight 4 : carried over from v4/v5.
#   --learning_rate 1e-5 : unchanged from v5 (still touching backbone blocks
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
    --alpha_shape 20 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --boost_neck_weight 4 \
    --boost_hand_weight 4 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_v8_resume_anny_partial_freeze
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
