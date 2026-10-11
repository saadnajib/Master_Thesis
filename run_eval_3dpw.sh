#!/bin/bash
# =============================================================================
# EVALUATE ONE CHECKPOINT ON BOTH TEST SETS: Anny-One holdout + converted 3DPW
# =============================================================================
# Same metrics as run_eval_test.sh (PVE, PA-PVE, MPJPE, PA-MPJPE, precision,
# recall), reported twice: on the last --test_anny_n Anny-One samples and on
# --val_3dpw_n images of the converted 3DPW --val_3dpw_split (the labels
# written by build_threedpw_anny_labels.py). The 13-20 mm conversion residual
# is a floor on the 3DPW numbers.
#
# Needs train.py patched by apply_mixed_data_fix.py (and apply_student_fix_v3
# for --head_dim). Set HEAD_DIM=1024 for projector-fed students (v6 onwards),
# 0 for the teacher and for v1-v5 students. ~5-10 min per checkpoint.
#
# The two evaluations the thesis needs first:
#   RUN_NAME=anny_distill_vits_v6b  EPOCH=00140  HEAD_DIM=1024   (student before mixed training)
#   RUN_NAME=anny_s2_shape_v5       EPOCH=00099  HEAD_DIM=0      (teacher, never saw real photos)
# =============================================================================
#SBATCH --job-name=mhmr-eval-3dpw
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/eval-3dpw-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/eval-3dpw-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

# >>> WHICH CHECKPOINT TO EVALUATE <<<
RUN_NAME="anny_distill_vits_v6b"
EPOCH="00140"
HEAD_DIM=1024          # 1024 for projector-fed students (v6+), 0 for the teacher / v1-v5
BACKBONE="dinov2_vits14"   # dinov2_vitl14 for the teacher
# >>> HOW MANY TEST SAMPLES (100 = exactly the untrained holdout) <<<
TEST_N=100
LABELS="/netscratch/najib/multi-hmr/threedpw_anny_full/labels"
VAL_3DPW_SPLIT="validation"   # validation while training runs; test only for the final table
VAL_3DPW_N=300

COMMAND=$(cat <<EOF
set -euo pipefail

source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

export HF_HOME="/netscratch/najib/hf_cache"
export TORCH_HOME="/netscratch/najib/torch_cache"
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p \$HF_HOME \$TORCH_HOME \$XDG_CACHE_HOME
mkdir -p /netscratch/najib/anny_cache "\$HOME/.cache"
[ -e "\$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "\$HOME/.cache/anny"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /netscratch/najib/multi-hmr

python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)" || true
find /netscratch/najib/torch_cache/hub/facebookresearch_dinov2_main/dinov2 -name "*.py" -exec sed -i 's/ | None//g' {} + || true

CKPT="/netscratch/najib/multi-hmr/logs/anny_model/${RUN_NAME}/checkpoints/${EPOCH}.pt"
if [ ! -f "\$CKPT" ]; then
  echo "ERROR: checkpoint not found: \$CKPT"
  echo "Check: ls -lt /netscratch/najib/multi-hmr/logs/anny_model/${RUN_NAME}/checkpoints/"
  exit 1
fi
echo "Evaluating \$CKPT on the last ${TEST_N} AnnyOne samples and ${VAL_3DPW_N} 3DPW-${VAL_3DPW_SPLIT} images"
[ -f "${LABELS}/index_${VAL_3DPW_SPLIT}.txt" ] || { echo "ERROR: no 3DPW labels at ${LABELS}"; exit 1; }
grep -q "mix_3dpw_labels" train.py || { echo "ERROR: train.py lacks apply_mixed_data_fix.py"; exit 1; }

# --eval_only 1 : no training, just run evaluate() and print the metrics.
# --test_anny_n : builds the AnnyOne test split (see header).
# Architecture flags must MATCH the checkpoint or the weights will not load.
python train.py \\
    --eval_only 1 \\
    --train_data AnnyOne \\
    --test_anny_n ${TEST_N} \\
    --val_anny_n 100 \\
    --person_center head \\
    --pretrained "\$CKPT" \\
    --pretrained_remap 0 \\
    --load_only_backbone 0 \\
    --backbone ${BACKBONE} \\
    --head_dim ${HEAD_DIM} \\
    --mix_3dpw_labels "${LABELS}" \\
    --val_3dpw_split ${VAL_3DPW_SPLIT} \\
    --val_3dpw_n ${VAL_3DPW_N} \\
    --img_size 672 \\
    --batch_size 1 \\
    --num_workers 4 \\
    --amp 0 \\
    --use_anny_shape 1 \\
    --mask_helper_joints_train 1 \\
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \\
    --name eval3dpw_${RUN_NAME}_${EPOCH}
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"