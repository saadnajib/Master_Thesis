#!/bin/bash
# =============================================================================
# DISTILLATION v4: v3 + HMR-trained student backbone + augmentation + batch 8
# =============================================================================
# Runs IN PARALLEL with v3. v3 isolates the head fix (teacher-initialised
# heads + output distillation) on top of an ImageNet DINOv2-S backbone. v4
# stacks three further changes; v3-vs-v4 gives their combined effect.
#
#   1. STUDENT BACKBONE from multiHMR_672_S (naver/multi-hmr release):
#      a ViT-S already trained for human mesh recovery on
#      BEDLAM+AGORA+CUFFS+UBody - the exact ViT-S counterpart of the
#      multiHMR_672_L_anny checkpoint the TEACHER's backbone came from. The
#      earlier students started from ImageNet-only DINOv2-S, which has never
#      seen a body. Per the repo's table, a properly trained ViT-S trails
#      ViT-L by only ~9% on 3DPW (102.4 vs 94.1 PVE), not the 160% gap the
#      ImageNet-initialised student showed (193 vs 73.6 mm).
#      That checkpoint is an SMPL-X model in the original Multi_HMR class, so
#      only its BACKBONE transfers (--load_only_backbone 1, with the
#      encoder.backbone.* -> backbone.encoder.* remap); the HEADS still come
#      from the Anny teacher via --init_heads_from_teacher. Order of
#      operations in train.py guarantees this: --pretrained loads the backbone
#      at model build, then the head-init overwrites only non-backbone keys.
#
#   2. COLOUR-JITTER AUGMENTATION (brightness/contrast/saturation/hue): every
#      run so far used 0.0 for all four, i.e. pixel-perfect synthetic renders,
#      then was asked to handle real photos. The same augmented image goes to
#      both teacher and student, so distillation stays consistent.
#
#   3. BATCH 8 (was 4): the 4 was a ViT-L-with-gradients constraint. The
#      student is ViT-S and the teacher runs under no_grad, so memory is far
#      lower. Doubles samples per epoch (8000 vs 4000 at 1000 it/epoch).
#      LR scaled to 7e-5. If the first iteration OOMs, set --batch_size 6.
#
# EXPECTATION: first holdout eval well below v3's; the HMR-trained backbone
# should make the biggest single difference. Confirm both init lines in log:
#   [pretrained] loaded N/M ... (backbone from multiHMR_672_S)
#   [distill] heads initialised from teacher: N/M non-backbone tensors
# =============================================================================
#SBATCH --job-name=mhmr_distill_s_v4
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_distill_vits_v4_%j.out
#SBATCH --error=logs/train_distill_vits_v4_%j.err

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

# --learning_rate 5e-5 : heads are already trained (from the teacher), the
#     backbone is not; a middle value between the 1e-4 cold start and the
#     2e-5 fine-tune. --lr_decay_every 60 over 150 epochs -> 2 halvings.
# --start_2d_epoch 0 : heads already produce sensible geometry from step 0.
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
    --start_2d_epoch 0 \
    --alpha_shape 20 \
    --brightness 0.2 \
    --contrast 0.2 \
    --saturation 0.2 \
    --hue 0.05 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_distill_vits_v4
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
