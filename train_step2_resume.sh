#!/bin/bash
# =============================================================================
# RESUME STEP 2: continue training from a checkpoint of anny_s2_frozen_bb_v2 (frozen backbone)
# =============================================================================
# Difference from the original launch:
#   --pretrained          : points at OUR OWN last checkpoint now, not the
#                           original repo's multiHMR_672_L_anny.pt
#   --pretrained_remap 0  : our own checkpoints already use this Model's key
#                           names (backbone.encoder.*), so remapping is not
#                           just unneeded but WRONG here - it would rename
#                           backbone.encoder.* to backbone.encoder.encoder.*
#                           and load nothing.
#   --load_only_backbone 0: load the WHOLE model this time (heads included) -
#                           they are no longer random, they are what the last
#                           run learned. Keeping load_only_backbone=1 would
#                           throw away that training.
#   --name                : anny_s2_frozen_bb_v2 - the SAME folder the checkpoint is
#                           loaded from. CAUTION: the checkpoint cleanup keeps
#                           the ten HIGHEST epoch numbers and epoch numbering
#                           restarts at 0 on each run, so new writes into this
#                           folder can be deleted immediately by the old high
#                           numbers already there (this happened once already).
#                           Use a fresh --name if that matters.
#
# >>> EDIT THIS LINE before submitting: set RESUME_EPOCH to the highest
#     checkpoint number found in anny_s2_frozen_bb_v2/checkpoints/ <<<
# =============================================================================
#SBATCH --job-name=mhmr_s2_resume
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=96:00:00
#SBATCH --output=logs/train_s2_resume_%j.out
#SBATCH --error=logs/train_s2_resume_%j.err

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

# >>> SET THIS to the highest epoch number in
#     logs/anny_model/anny_s2_frozen_bb_v2/checkpoints/ (e.g. 00149 -> 149)
RESUME_EPOCH=299   # <-- SET to S2 highest epoch, likely LARGER than S1 (S2 runs faster)
CKPT=$(printf "/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_frozen_bb_v2/checkpoints/%05d.pt" "$RESUME_EPOCH")
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  echo "Check: ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_s2_frozen_bb_v2/checkpoints/"
  exit 1
fi
echo "Resuming step 2 (frozen backbone) from $CKPT"

# 4. Run Training
#
# SETTINGS FOR THIS RESUME:
#   --pretrained_remap 0, --load_only_backbone 0 : see header - load everything,
#                           our own key names already match.
#   --freeze_backbone 1    : unchanged from step 2, backbone stays frozen.
#   --learning_rate 3e-5   : LOWERED from the original 1e-4 for the resume
#                           (anneal as training progresses); the step-1 and
#                           step-2 resumes use the same value so the comparison
#                           stays fair.
#   --max_iter 300000      : 300 capped epochs (1000 iters each); epoch
#                           numbering restarts at 0 on this run.
#   --name                 : v2, same dir as the loaded checkpoint (see header).

python train.py \
    --train_data AnnyOne \
    --person_center head \
    --pretrained "$CKPT" \
    --pretrained_remap 0 \
    --load_only_backbone 0 \
    --freeze_backbone 1 \
    --batch_size 4 \
    --log_freq 50 \
    --backbone dinov2_vitl14 \
    --img_size 672 \
    --n_iters_per_epoch 1000 \
    --max_iter 300000 \
    --num_workers 8 \
    --learning_rate 3e-5 \
    --lr_decay_every 30 \
    --lr_decay_gamma 0.5 \
    --amp 0 \
    --val_anny_n 100 \
    --eval_freq 5 \
    --use_anny_shape 1 \
    --start_2d_epoch 20 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_s2_frozen_bb_v2
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
