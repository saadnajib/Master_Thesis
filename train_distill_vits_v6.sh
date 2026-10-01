#!/bin/bash
# =============================================================================
# DISTILLATION v6: v5 + PROJECTOR-FED HEADS (--head_dim 1024)
# =============================================================================
# WHAT v3/v4/v5 SHOWED (jobs 3453670 / 3453700 / 3454122):
#   Every student with teacher-initialised heads was worse than the random-
#   heads student at the same epoch (v3 @ep45: PVE 274 vs v1 @ep0: 228), with
#   PA-PVE stuck at 155-199 vs 128. The v5 head freeze kept the transferred
#   weights intact (rotmat loss 90 vs v4's 300) and PVE was still 315 at the
#   unfreeze eval, PA-PVE unchanged. The heads are fine; their INPUT is not.
#   The 13 tensors that could not transfer are the whole input side of the
#   head (HPH token embedding, cross-attention K/V, detection/offset MLPs),
#   all random linear maps from 384-dim tokens. The KD projector that IS
#   trained to map student tokens onto the teacher's token distribution was
#   only used inside the loss and never fed to the heads.
#
# THE FIX (apply_student_fix_v3.py, patches model.py + train.py):
#   --head_dim 1024   a linear 384->1024 projector after the backbone; every
#                     head is built at 1024, so ALL 67 head tensors transfer
#                     ("heads initialised from teacher: 67/67"). The feature
#                     KL now compares the PROJECTED tokens with the teacher's
#                     (same width, no loss-side projector), i.e. it trains the
#                     projector towards exactly the input the heads expect.
#   --freeze_heads_epochs 5   warm-up: only backbone + projector train, guided
#                     by feature KD and by output KD through the teacher's own
#                     (frozen) heads. --start_2d_epoch 5 as in v5.
#
# Everything else is identical to v5 (HMR-trained ViT-S backbone from
# multiHMR_672_S, colour jitter, batch 8, feature + output KD, LR 7e-5).
# Student grows from 32.8M to ~37M params (still ~8.5x smaller than 318M).
#
# HOW TO JUDGE: the epoch-0 eval measures how well projector+backbone mimic
# the teacher's tokens with the teacher's own heads frozen on top; the epoch-5
# and epoch-10 evals after unfreezing are the decision. Success = clearly
# below 228 (random heads' first eval) and PA-PVE clearly below 128. If it is
# not, head transfer is dead: run the fallback (this script without
# --init_heads_from_teacher / --freeze_heads_epochs / --head_dim).
#
# BEFORE SUBMITTING (once):
#   cp model.py model.py.bak_before_v3 ; cp train.py train.py.bak_before_v3
#   python apply_student_fix_v3.py
# =============================================================================
#SBATCH --job-name=mhmr_distill_s_v6
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_distill_vits_v6_%j.out
#SBATCH --error=logs/train_distill_vits_v6_%j.err

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
if ! grep -q "head_dim" train.py || ! grep -q "feat_proj" model.py; then
  echo "ERROR: v3 patch missing. Run:  python apply_student_fix_v3.py   (patches model.py and train.py)"
  exit 1
fi

TEACHER="/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt"
[ -f "$TEACHER" ] || { echo "ERROR: teacher not found: $TEACHER"; exit 1; }
echo "Teacher (frozen): $TEACHER"

# Student BACKBONE init: the HMR-trained ViT-S from the Multi-HMR release.
# Download once on the login node (compute nodes have no internet):
#   cd /netscratch/najib/multi-hmr/models/multiHMR/
#   wget https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_S.pt
STUDENT_BB="/netscratch/najib/multi-hmr/models/multiHMR/multiHMR_672_S.pt"
if [ ! -f "$STUDENT_BB" ]; then
  echo "ERROR: student backbone checkpoint not found: $STUDENT_BB"
  echo "Download it on the login node first:"
  echo "  cd /netscratch/najib/multi-hmr/models/multiHMR/"
  echo "  wget https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_S.pt"
  exit 1
fi
echo "Student backbone init: $STUDENT_BB"

# --head_dim 1024 : projector-fed heads, see header. Check the log for
#     "[distill] projector-fed heads: backbone 384 -> heads 1024" and
#     "heads initialised from teacher: 67/67".
python train.py \
    --train_data AnnyOne \
    --person_center head \
    --freeze_backbone 0 \
    --mask_helper_joints_train 1 \
    --backbone dinov2_vits14 \
    --pretrained "$STUDENT_BB" \
    --pretrained_remap 1 \
    --load_only_backbone 1 \
    --distill_teacher_ckpt "$TEACHER" \
    --init_heads_from_teacher 1 \
    --head_dim 1024 \
    --freeze_heads_epochs 5 \
    --lambda_kd 1.0 \
    --lambda_kd_out 1.0 \
    --kd_temperature 4.0 \
    --kd_softmax_dim channel \
    --batch_size 8 \
    --log_freq 50 \
    --img_size 672 \
    --n_iters_per_epoch 1000 \
    --max_iter 150000 \
    --num_workers 8 \
    --learning_rate 7e-5 \
    --lr_decay_every 60 \
    --lr_decay_gamma 0.5 \
    --amp 0 \
    --val_anny_n 100 \
    --eval_freq 5 \
    --use_anny_shape 1 \
    --start_2d_epoch 5 \
    --alpha_shape 20 \
    --brightness 0.2 \
    --contrast 0.2 \
    --saturation 0.2 \
    --hue 0.05 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_distill_vits_v6
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
