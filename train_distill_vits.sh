#!/bin/bash
# =============================================================================
# KNOWLEDGE DISTILLATION: ViT-L teacher -> ViT-S student
# =============================================================================
# Trains a SMALL backbone to reproduce the LARGE backbone's patch-token
# representation, so the deployed model can be ~10x smaller at similar quality.
#
#   teacher : anny_s2_shape_v5 (ViT-L, 1024-dim, FROZEN, never updated)
#   student : dinov2_vits14    (ViT-S,  384-dim, trained here)
#   loss    : task_loss + lambda_kd * KL(teacher_tokens || student_tokens)
#
# The KL term is applied at the backbone patch tokens - immediately after patch
# decoding, BEFORE the projection heads - so the student matches the teacher's
# REPRESENTATION, not just its final predictions.
#
# Because ViT-S is 384-dim and ViT-L is 1024-dim, a learnable linear projector
# maps student->teacher width so KL is computed over the same support. That
# projector is training-only and is discarded at inference.
#
# IMPORTANT SCOPE LIMIT (raised by the supervisor, and correct):
# this run trains on SYNTHETIC AnnyOne data only. A student can drive KL to
# near-zero by matching the teacher on synthetic renders while both have simply
# latched onto render-specific cues that do not exist in real photographs.
# TREAT THIS RUN AS A SANITY CHECK - does the student converge, does KL behave
# sensibly, is the compression ratio right - NOT as the final distillation
# result. The reportable run is the one trained on mixed synthetic + converted
# 3DPW batches, which needs the SMPL->Anny conversion script first.
#
# For a ViT-B student instead, change --backbone to dinov2_vitb14 (768-dim).
# Comparing ViT-S vs ViT-B students is one of the requested ablations.
# =============================================================================
#SBATCH --job-name=mhmr_distill_s
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_distill_vits_%j.out
#SBATCH --error=logs/train_distill_vits_%j.err

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

# The FROZEN TEACHER: our best trained ViT-L checkpoint.
TEACHER_RUN="anny_s2_shape_v5"
TEACHER_EPOCH="00099"
TEACHER="/netscratch/najib/multi-hmr/logs/anny_model/${TEACHER_RUN}/checkpoints/${TEACHER_EPOCH}.pt"
if [ ! -f "$TEACHER" ]; then
  echo "ERROR: teacher checkpoint not found: $TEACHER"
  echo "Check: ls -lt /netscratch/najib/multi-hmr/logs/anny_model/${TEACHER_RUN}/checkpoints/"
  exit 1
fi
echo "Teacher (frozen): $TEACHER"
echo "Student: dinov2_vits14, trained from scratch heads"

# 4. Run Training
#   Student dinov2_vits14 from ImageNet weights with randomly initialised heads
#       (no --pretrained), fully trainable (--freeze_backbone 0).
#   --distill_teacher_ckpt / --lambda_kd 1.0 / --kd_temperature 4.0 /
#       --kd_softmax_dim channel : feature-level KD from the frozen teacher.
#   --mask_helper_joints_train 1 : helper/deform bones masked in the loss.
#   --learning_rate 1e-4 (halved every 30 epochs), 300 capped epochs.
#   --name anny_distill_vits : fresh checkpoint dir.

python train.py \
    --train_data AnnyOne \
    --person_center head \
    --freeze_backbone 0 \
    --mask_helper_joints_train 1 \
    --batch_size 4 \
    --log_freq 50 \
    --backbone dinov2_vits14 \
    --distill_teacher_ckpt "$TEACHER" \
    --lambda_kd 1.0 \
    --kd_temperature 4.0 \
    --kd_softmax_dim channel \
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
    --name anny_distill_vits
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
