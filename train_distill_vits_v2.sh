#!/bin/bash
# =============================================================================
# DISTILLATION RESUME (v2): fix the LR decay schedule
# =============================================================================
# DIAGNOSIS from train_distill_vits_3386048.out:
#   The run genuinely converged in its first ~60-90 epochs (kd: 2679 -> ~90
#   within 2 epochs; bce: 222 -> ~1.0), then EFFECTIVELY STOPPED LEARNING for
#   the remaining ~160 epochs. Cause: --lr_decay_every 30 halves the LR every
#   30 epochs, which is fine for a short fine-tuning continuation but far too
#   aggressive for a 300-epoch from-scratch student:
#       epoch  60: lr = 2.50e-05
#       epoch 120: lr = 6.25e-06
#       epoch 180: lr = 1.56e-06
#       epoch 240: lr = 3.91e-07   <- 250x smaller than the start, no-op
#   kd/total/rotmat all plateau and noise around the same values from roughly
#   epoch 60 onward instead of continuing to improve - direct evidence the
#   optimiser had nothing left to work with.
#
# FIX: resume from the last checkpoint (do not discard the good early
# progress) with a much slower decay tuned to the remaining epoch budget, and
# a lower starting LR appropriate for a partially-trained model rather than a
# from-scratch one.
#
# >>> ASSUMPTION: resuming is what "retrain task A" meant. If you actually
#     want a clean from-scratch restart instead, remove the --pretrained line
#     below and change --learning_rate back to 1e-4. <<<
# =============================================================================
#SBATCH --job-name=mhmr_distill_s_v2
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=96:00:00
#SBATCH --output=logs/train_distill_vits_v2_%j.out
#SBATCH --error=logs/train_distill_vits_v2_%j.err

COMMAND=$(cat <<'EOF'
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

export HF_HOME="/netscratch/najib/hf_cache"
export TORCH_HOME="/netscratch/najib/torch_cache"
export TORCH_EXTENSIONS_DIR="/netscratch/najib/torch_extensions"
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p $HF_HOME $TORCH_HOME $TORCH_EXTENSIONS_DIR $XDG_CACHE_HOME
mkdir -p /netscratch/najib/anny_cache
mkdir -p "$HOME/.cache"
[ -e "$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "$HOME/.cache/anny"

cd /netscratch/najib/multi-hmr

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Ensuring DINOv2 is downloaded to cache..."
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)" || true
find /netscratch/najib/torch_cache/hub/facebookresearch_dinov2_main/dinov2 -name "*.py" -exec sed -i 's/ | None//g' {} +

# The STUDENT's own last checkpoint from the run that plateaued.
# >>> CHECK THE REAL LATEST EPOCH FIRST: <<<
#     ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_distill_vits_v2/checkpoints/ | head -5
STUDENT_EPOCH="00199"   # <-- SET to the highest epoch actually on disk
STUDENT_CKPT="/netscratch/najib/multi-hmr/logs/anny_model/anny_distill_vits_v2/checkpoints/${STUDENT_EPOCH}.pt"
if [ ! -f "$STUDENT_CKPT" ]; then
  echo "ERROR: student checkpoint not found: $STUDENT_CKPT"
  echo "Check: ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_distill_vits/checkpoints/"
  exit 1
fi
echo "Resuming student from $STUDENT_CKPT"

# The TEACHER (unchanged, always frozen - same one as before).
TEACHER="/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt"
if [ ! -f "$TEACHER" ]; then
  echo "ERROR: teacher checkpoint not found: $TEACHER"
  exit 1
fi
echo "Teacher (frozen): $TEACHER"

# --pretrained_remap 0 / --load_only_backbone 0 : loading OUR OWN student
#     checkpoint (same Model class, same key names) - remapping would corrupt
#     it, backbone-only would throw away the heads it already learned.
# --learning_rate 2e-5 : lower than the cold-start 1e-4, appropriate for a
#     model that is no longer randomly initialised.
# --lr_decay_every 60 / --lr_decay_gamma 0.5 : the actual fix. Over a 200-epoch
#     remaining budget this halves at 60/120/180 -> still a real LR (2.5e-6)
#     even near the end, instead of collapsing to noise by epoch 240.
# --start_2d_epoch 0 : the student already has meaningful predictions from the
#     first run; no need to repeat the original cold-start 2D warm-up delay.
# --name : NEW name -> fresh checkpoint dir (never reuse a name whose folder
#     holds higher-numbered checkpoints; cleanup keeps the 10 HIGHEST).

python train.py \
    --train_data AnnyOne \
    --person_center head \
    --pretrained "$STUDENT_CKPT" \
    --pretrained_remap 0 \
    --load_only_backbone 0 \
    --freeze_backbone 0 \
    --mask_helper_joints_train 1 \
    --backbone dinov2_vits14 \
    --distill_teacher_ckpt "$TEACHER" \
    --lambda_kd 1.0 \
    --kd_temperature 4.0 \
    --kd_softmax_dim channel \
    --batch_size 4 \
    --log_freq 50 \
    --img_size 672 \
    --n_iters_per_epoch 1000 \
    --max_iter 200000 \
    --num_workers 8 \
    --learning_rate 2e-5 \
    --lr_decay_every 60 \
    --lr_decay_gamma 0.5 \
    --amp 0 \
    --val_anny_n 100 \
    --eval_freq 5 \
    --use_anny_shape 1 \
    --start_2d_epoch 0 \
    --alpha_shape 20 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_distill_vits_v2_a
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
