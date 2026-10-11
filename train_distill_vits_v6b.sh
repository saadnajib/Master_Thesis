#!/bin/bash
# =============================================================================
# DISTILLATION v6b: CONTINUE v6 (projector-fed heads) for 150 more epochs
# =============================================================================
# v6 (job 3455531) hit the 48 h limit at epoch 138 with holdout PVE 152 /
# PA-PVE 105 - the best student so far (v4 156/117, v5 159/113, random-heads
# v2 193/128) and still improving (-12 mm over its last 30 epochs; kd and v3d
# losses still falling). 138 epochs x 8000 images is under two passes over
# AnnyOne, so the gap to the teacher (73.6 / 56.5) reads as undertraining.
#
# This run resumes v6's own checkpoint and simply trains longer:
#   --pretrained <v6 ckpt> --pretrained_remap 0 --load_only_backbone 0
#       our own checkpoint, same Model class, load everything.
#   --head_dim 1024   MUST stay: the checkpoint has the 384->1024 projector
#       and 1024-wide heads; without it the model is built at 384 and the
#       weights do not load.
#   no --init_heads_from_teacher / --freeze_heads_epochs: already done in v6.
#   --learning_rate 2e-5 : v6 ended at 1.75e-5 (7e-5 halved twice); keep that
#       level, decay every 60 again. --start_2d_epoch 0 (2D already active).
#   --name NEW (anny_distill_vits_v6b): never resume into the old folder.
# BEFORE SUBMITTING: set V6_EPOCH to the highest checkpoint in
#   /netscratch/najib/multi-hmr/logs/anny_model/anny_distill_vits_v6/checkpoints/
# =============================================================================
#SBATCH --job-name=mhmr_distill_s_v6b
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_distill_vits_v6b_%j.out
#SBATCH --error=logs/train_distill_vits_v6b_%j.err

COMMAND=$(cat <<'EOF'
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

export HF_HOME="/netscratch/najib/hf_cache"
export TORCH_HOME="/netscratch/najib/torch_cache"
export TORCH_EXTENSIONS_DIR="/netscratch/najib/torch_extensions"
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p $HF_HOME $TORCH_HOME $TORCH_EXTENSIONS_DIR $XDG_CACHE_HOME
mkdir -p /netscratch/najib/anny_cache "$HOME/.cache"
[ -e "$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "$HOME/.cache/anny"

cd /netscratch/najib/multi-hmr
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)" || true
find /netscratch/najib/torch_cache/hub/facebookresearch_dinov2_main/dinov2 -name "*.py" -exec sed -i 's/ | None//g' {} +

# Refuse to run on an unpatched train.py - the flags would be silently ignored
# by argparse and the run would be a repeat of v2.
if ! grep -q "init_heads_from_teacher" train.py; then
  echo "ERROR: train.py is not patched. Run:  python apply_student_fix.py train.py"
  exit 1
fi
if ! grep -q "freeze_heads_epochs" train.py; then
  echo "ERROR: train.py lacks the warm-up patch. Run:  python apply_student_fix_v2.py train.py"
  exit 1
fi
if ! grep -q "teacher_bf16" train.py || ! grep -q "backbone_autocast_dtype" model.py; then
  echo "ERROR: speed patch missing. Run:  python apply_speed_fix.py   (patches train.py and model.py)"
  exit 1
fi
if ! grep -q "head_dim" train.py || ! grep -q "feat_proj" model.py; then
  echo "ERROR: v3 patch missing. Run:  python apply_student_fix_v3.py   (patches model.py and train.py)"
  exit 1
fi

TEACHER="/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt"
[ -f "$TEACHER" ] || { echo "ERROR: teacher not found: $TEACHER"; exit 1; }
echo "Teacher (frozen): $TEACHER"

# >>> SET to the highest epoch in anny_distill_vits_v6/checkpoints/ <<<
V6_EPOCH="00137"
V6_CKPT="/netscratch/najib/multi-hmr/logs/anny_model/anny_distill_vits_v6/checkpoints/${V6_EPOCH}.pt"
if [ ! -f "$V6_CKPT" ]; then
  echo "ERROR: v6 checkpoint not found: $V6_CKPT"
  echo "Check: ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_distill_vits_v6/checkpoints/ | head"
  exit 1
fi
echo "Resuming v6 student from $V6_CKPT"

# --tf32 1 --teacher_bf16 1 : apply_speed_fix.py. Expect ~0.3-0.5 s/it instead
#     of 1.24; check the tqdm rate in the .err file after a few minutes.
# Check the log for "[pretrained] loaded 244/244" (whole student incl. projector)
# and "[distill] projector-fed heads: backbone 384 -> heads 1024".
python train.py \
    --train_data AnnyOne \
    --person_center head \
    --freeze_backbone 0 \
    --mask_helper_joints_train 1 \
    --backbone dinov2_vits14 \
    --pretrained "$V6_CKPT" \
    --pretrained_remap 0 \
    --load_only_backbone 0 \
    --distill_teacher_ckpt "$TEACHER" \
    --head_dim 1024 \
    --tf32 1 \
    --teacher_bf16 1 \
    --lambda_kd 1.0 \
    --lambda_kd_out 1.0 \
    --kd_temperature 4.0 \
    --kd_softmax_dim channel \
    --batch_size 8 \
    --log_freq 50 \
    --img_size 672 \
    --n_iters_per_epoch 1000 \
    --max_iter 150000 \
    --num_workers 12 \
    --learning_rate 2e-5 \
    --lr_decay_every 60 \
    --lr_decay_gamma 0.5 \
    --amp 0 \
    --val_anny_n 100 \
    --eval_freq 5 \
    --use_anny_shape 1 \
    --start_2d_epoch 0 \
    --alpha_shape 20 \
    --brightness 0.2 \
    --contrast 0.2 \
    --saturation 0.2 \
    --hue 0.05 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_distill_vits_v6b
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
