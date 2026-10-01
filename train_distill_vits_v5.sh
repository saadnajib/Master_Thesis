#!/bin/bash
# =============================================================================
# DISTILLATION v5: v4 + HEAD WARM-UP FREEZE
# =============================================================================
# WHAT v3/v4 SHOWED (jobs 3453670 / 3453700):
#   Teacher-initialised heads made the FIRST eval WORSE than random heads
#   (377 / 298 mm vs 228 mm). The 13 tensors that cannot transfer from the
#   ViT-L teacher are exactly the head's ENTRY POINTS - the HPH input
#   projection (to_token_embedding), the cross-attention K/V projections, and
#   the detection/offset MLPs. Well-trained blocks fed through random
#   projections produce confident wrong poses, and the gradients from those
#   then damage the transferred weights before the projections have adapted.
#
# THE FIX (archive/patches/apply_student_fix_v2.py, already applied):
#   --freeze_heads_epochs 5   the 54 transferred tensors stay frozen for the
#                             first 5 epochs; only the 13 random projections,
#                             the backbone and the KD projector train, with
#                             output distillation giving them a direct target
#                             (match the teacher). Everything unfreezes at
#                             epoch 5.
#   --start_2d_epoch 5        v3/v4 used 0 on the (wrong) assumption that the
#                             heads produce sensible geometry from step 0; with
#                             random projections they do not, so reprojection
#                             gradients now wait for the warm-up too.
#
# Everything else is identical to v4 (HMR-trained ViT-S backbone from
# multiHMR_672_S, colour jitter, batch 8, teacher heads, feature + output KD).
#
# EXPECTATION: first eval (epoch 0) may still be high because the heads are
# frozen and the projections are learning; the eval at epoch 5 or 10, right
# after unfreezing, is the one that matters. It should be well BELOW 228.
# =============================================================================
#SBATCH --job-name=mhmr_distill_s_v5
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_distill_vits_v5_%j.out
#SBATCH --error=logs/train_distill_vits_v5_%j.err

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

# --learning_rate 7e-5 : heads are already trained (from the teacher), the
#     backbone is not; v3's 5e-5 scaled up for batch 8 (as in v4).
#     --lr_decay_every 60 over 150 epochs -> 2 halvings.
# --start_2d_epoch 5 : reprojection losses wait for the head warm-up (see header).
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
    --name anny_distill_vits_v5
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
