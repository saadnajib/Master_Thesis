#!/bin/bash
# =============================================================================
# TEACHER v9: continue the ViT-L teacher on MIXED AnnyOne + converted 3DPW
# =============================================================================
# The teacher (anny_s2_shape_v5, ep 99) scores 74.3 mm PVE on synthetic
# holdout but 127.1 mm on real 3DPW-validation (job 3502566): the same domain
# gap the student had, and 30% real data took the student from 149 to 106 mm
# on real photos. This run gives the teacher the same treatment.
#
# Same recipe as the v5/v6 continuations (backbone frozen except the last 4
# ViT blocks, helper-joint mask, alpha_shape 20, LR 1e-5), plus:
#   --mix_3dpw_labels / --mix_3dpw_ratio 0.3   30% of every batch is real 3DPW-train
#   --val_3dpw_split validation                evaluated on 300 converted 3DPW-val
#                                              images next to the AnnyOne holdout
#   --tf32 1                                   (apply_speed_fix.py) ~2x faster matmuls
# No distillation flags: this IS the teacher. Expect ~8-10 min/epoch at batch 4.
#
# WHY IT MATTERS DOWNSTREAM: a teacher that is good on real photos removes the
# tension in mixed distillation (KD terms pulling toward domain-gapped teacher
# predictions on 3DPW images). If this run beats 127 mm on 3DPW-val clearly,
# it becomes the teacher for the final student round.
#
# Needs: train.py patched by apply_mixed_data_fix.py (+v2), apply_speed_fix.py;
# datasets/threedpw_anny.py; threedpw_anny_full/labels/index_train.txt.
# =============================================================================
#SBATCH --job-name=mhmr_teacher_v9_mixed
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_teacher_v9_mixed_%j.out
#SBATCH --error=logs/train_teacher_v9_mixed_%j.err

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
#     ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/ | head -3
#     and edit V5_EPOCH below to match.
V5_EPOCH=00099   # <-- SET to the highest epoch in anny_s2_shape_v5/checkpoints/
CKPT=$(printf "/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/%s.pt" "$V5_EPOCH")
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  echo "Check: ls -lt /netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/"
  exit 1
fi
echo "Continuing from $CKPT"

LABELS="/netscratch/najib/multi-hmr/threedpw_anny_full/labels"
if [ ! -f "$LABELS/index_train.txt" ] || [ ! -f datasets/threedpw_anny.py ] || ! grep -q "mix_3dpw_labels" train.py; then
  echo "ERROR: mixed-data setup missing (labels, datasets/threedpw_anny.py, or apply_mixed_data_fix.py)"
  exit 1
fi

# 4. Run Training
#   --pretrained_remap 0 / --load_only_backbone 0 : our own checkpoint, load all.
#   --start_2d_epoch 0 : the model is converged; 2D losses on from the start.
#   --lr_decay_every 60 over 150 epochs -> LR 1e-5 / 5e-6 / 2.5e-6.
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
    --max_iter 150000 \
    --num_workers 12 \
    --learning_rate 1e-5 \
    --lr_decay_every 60 \
    --lr_decay_gamma 0.5 \
    --amp 0 \
    --val_anny_n 100 \
    --eval_freq 5 \
    --use_anny_shape 1 \
    --start_2d_epoch 0 \
    --alpha_shape 20 \
    --tf32 1 \
    --mix_3dpw_labels "$LABELS" \
    --mix_3dpw_ratio 0.3 \
    --mix_3dpw_split train \
    --val_3dpw_split validation \
    --val_3dpw_n 300 \
    --alpha_j2d 0.1 \
    --alpha_v2d 0.1 \
    --boost_neck_weight 4 \
    --boost_hand_weight 4 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_teacher_v9_mixed
EOF
)

srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"
