#!/bin/bash
# =============================================================================
# FINAL TEST-SET EVALUATION: one checkpoint on Anny-One holdout + 3DPW-TEST
# =============================================================================
# usage:
#   sbatch run_eval_3dpw_test.sh <checkpoint.pt> <head_dim> <backbone> <label>
# e.g.
#   sbatch run_eval_3dpw_test.sh models/multiHMR/anny_distill_vits_v7_mixed_00130.pt 1024 dinov2_vits14 student_v7_ep130
#   sbatch run_eval_3dpw_test.sh models/multiHMR/anny_distill_vits_v6c_00149.pt   1024 dinov2_vits14 student_v6c_ep149
#   sbatch run_eval_3dpw_test.sh logs/anny_model/anny_s2_shape_v5/checkpoints/00099.pt 0 dinov2_vitl14 teacher_v5
#
# 3DPW-test (all 9,722 converted images, every third frame of the 24 test
# sequences) is used ONLY here, after every training decision was taken on
# the validation split. The 13-20 mm conversion residual is a floor on it.
# =============================================================================
#SBATCH --job-name=mhmr-eval-test3dpw
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/eval-test3dpw-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/eval-test3dpw-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

# >>> WHICH CHECKPOINT TO EVALUATE <<<
CKPT_ARG="${1:?checkpoint path}"
HEAD_DIM="${2:?head_dim (1024 students v6+, 0 teacher)}"
BACKBONE="${3:?backbone (dinov2_vits14 / dinov2_vitl14)}"
LABEL="${4:-eval}"
RUN_NAME="$LABEL"
EPOCH="test"
# >>> HOW MANY TEST SAMPLES (100 = exactly the untrained holdout) <<<
TEST_N=100
LABELS="/netscratch/najib/multi-hmr/threedpw_anny_full/labels"
VAL_3DPW_SPLIT="test"
VAL_3DPW_N=0        # 0 = all 9,722 test images

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

CKPT="${CKPT_ARG}"
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
    --name evaltest_${LABEL}
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"