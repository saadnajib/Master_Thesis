#!/bin/bash
#SBATCH --job-name=multihmr_anny_med
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_anny_med_%j.out
#SBATCH --error=logs/train_anny_med_%j.err

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

# 4. Run Training
#
# KEY CHANGES FROM PREVIOUS RUN:
#   --pretrained        : resume from epoch 299 (00299.pt) of anny_full_run_v2.
#                         (An earlier version of this comment pointed at epoch
#                         19, the lowest PA-PVE=113.9; the flag now loads 299.)
#   --name              : new run name so logs/checkpoints don't overwrite the old run
#   --n_iters_per_epoch : FIXED - caps each epoch at 1000 iters so checkpoints
#                         and evals happen every ~15 min, not every 29 hours
#   --max_iter 300000   : 300 capped epochs of 1000 iters = ~75 hours of real training
#   --eval_freq 5       : evaluate every 5 epochs to keep overhead low
#   --start_2d_epoch 20 : FIX for offset explosion - wait until epoch 20 before
#                         switching on j2d/v2d losses (was 10, which destabilised
#                         the already-converged offset head)
#   --alpha_j2d 0.1     : FIX - reduce 2D loss weight from 1.0 to 0.1 so the
#   --alpha_v2d 0.1       first few 2D epochs don't shock the model with large
#                         gradients (can be increased in a future run once stable)

python train.py \
    --train_data AnnyOne \
    --person_center head \
    --pretrained /netscratch/najib/multi-hmr/logs/anny_model/anny_full_run_v2/checkpoints/00299.pt \
    --batch_size 4 \
    --log_freq 50 \
    --backbone dinov2_vitl14 \
    --img_size 672 \
    --n_iters_per_epoch 1000 \
    --max_iter 300000 \
    --num_workers 8 \
    --learning_rate 5e-6 \
    --amp 0 \
    --val_anny_n 100 \
    --eval_freq 5 \
    --use_anny_shape 1 \
    --start_2d_epoch 20 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_full_run_v2
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"