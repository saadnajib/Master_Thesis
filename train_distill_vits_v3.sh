#!/bin/bash
# =============================================================================
# DISTILLATION v3: teacher-initialised heads + output-level distillation
# =============================================================================
# DIAGNOSIS (from train_distill_vits_v2_3408961.out):
#   Student ViT-S holdout PVE 193.4 mm vs teacher ViT-L 73.6 mm -> 2.6x worse.
#   The Multi-HMR paper's own ViT-S vs ViT-B gap is only 80 vs 73 mm, so this
#   is NOT a capacity limit. The KD loss supervised only the backbone tokens;
#   the HPH / pose / shape heads were RANDOMLY initialised and learned from
#   scratch, while the teacher's heads had five stages of training. PVE was
#   still crawling (198 -> 193 over 80 epochs): a bad start, not convergence.
#
# TWO CHANGES (see apply_student_fix.py):
#   --init_heads_from_teacher 1   student starts with the teacher's trained
#                                 heads (same Model class; only embed_dim-
#                                 dependent input projections stay random)
#   --lambda_kd_out 1.0           student's predicted pose/shape/depth are
#                                 pulled toward the teacher's on every image,
#                                 in addition to the feature-level KL
#
# FRESH student backbone (ImageNet DINOv2-S), NOT resumed from v1/v2: those
# checkpoints carry heads that learned a bad solution, and we now have a far
# better initialisation for them.
#
# EXPECTATION: a large immediate drop in holdout PVE versus the 193-228 mm the
# earlier runs started from. Judge on PVE at the first few evals; if it is not
# well under 150 mm by epoch 10 something is wrong - stop and check the
# "[distill] heads initialised from teacher: N/M" line.
# =============================================================================
#SBATCH --job-name=mhmr_distill_s_v3
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_distill_vits_v3_%j.out
#SBATCH --error=logs/train_distill_vits_v3_%j.err

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
    --distill_teacher_ckpt "$TEACHER" \
    --init_heads_from_teacher 1 \
    --lambda_kd 1.0 \
    --lambda_kd_out 1.0 \
    --kd_temperature 4.0 \
    --kd_softmax_dim channel \
    --batch_size 4 \
    --log_freq 50 \
    --img_size 672 \
    --n_iters_per_epoch 1000 \
    --max_iter 150000 \
    --num_workers 8 \
    --learning_rate 5e-5 \
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
    --name anny_distill_vits_v3
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
