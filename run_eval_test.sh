#!/bin/bash
# HELD-OUT TEST evaluation (train.py --eval_split test). NOTE: the previous version of this script (train_v3.py) evaluated on the VALIDATION samples, so its numbers were not a test result.
# =============================================================================
# AnnyOne TEST-SET EVALUATION - the 4 standard metrics
# =============================================================================
# Reports, per the evaluate() function in train.py:
#
#   PVE       per-vertex error, mm. Mean distance between predicted and GT mesh
#             vertices after centring both on the pelvis. Measures the whole
#             body surface, so it captures shape AND pose error together.
#   PA-PVE    the same after Procrustes alignment (optimal rotation, scale and
#             translation). Removes global orientation/scale error, so it
#             isolates how well the BODY ITSELF is reconstructed.
#   MPJPE     mean per-joint position error, mm, on Anny's own 163 joints,
#             pelvis-centred. The skeleton-level counterpart of PVE.
#   PA-MPJPE  the same after Procrustes alignment.
#
# Detection quality (precision / recall / F1) is printed alongside, because the
# metrics above are computed only over predictions that were MATCHED to a GT
# person - a model that detects few people can score well on the four metrics
# while missing most of the humans. Always report recall next to PVE.
#
# WHICH SAMPLES (train.py anny_split_ranges, N = AnnyOne size):
#     [0, N-600)       train
#     [N-600, N-500)   val   (--val_anny_n 100, the samples just BEFORE test)
#     [N-500, N)       test  (--test_anny_n 500)  <- evaluated here
# Runs trained with the same --test_anny_n/--val_anny_n never see val or test.
#
# WARNING: anny_s2_shape_v5 was trained BEFORE this split existed, with only
# --val_anny_n 100, i.e. on indices [0, N-100). So 400 of these 500 test
# samples were in its training set. train.py reads the training args stored
# in the checkpoint, prints the overlap, and records it in the
# n_seen_in_training column of results/<name>.csv. Only a model retrained with
# --test_anny_n 500 --val_anny_n 100 gets a clean test number here.
#
# Results: printed, and appended as a row to results/eval_<run>_<epoch>.csv
# and results/all_results.jsonl (see results/README.md).
# =============================================================================
#SBATCH --job-name=mhmr-eval-test
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/eval-test-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/eval-test-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

# >>> WHICH CHECKPOINT TO EVALUATE <<<
RUN_NAME="anny_s2_shape_v5"
EPOCH="00099"
# >>> SPLIT SIZES (must match what the evaluated model was trained with) <<<
TEST_N=100   # 100 = exactly the samples anny_s2_shape_v5 never trained on (but they were its validation set). Use 500 only for models retrained with --test_anny_n 500 --val_anny_n 100.
VAL_N=100

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
echo "Evaluating \$CKPT on the AnnyOne TEST split (last ${TEST_N} samples; val = the ${VAL_N} before them)"

# --eval_only 1     : no training, just run evaluate() and print the metrics.
# --eval_split test : evaluate the held-out test split, not validation.
# --test_anny_n / --val_anny_n : define the splits (see header).
# Architecture flags must MATCH the checkpoint or the weights will not load.
python train.py \\
    --eval_only 1 \\
    --eval_split test \\
    --train_data AnnyOne \\
    --test_anny_n ${TEST_N} \\
    --val_anny_n ${VAL_N} \\
    --person_center head \\
    --pretrained "\$CKPT" \\
    --pretrained_remap 0 \\
    --load_only_backbone 0 \\
    --backbone dinov2_vitl14 \\
    --img_size 672 \\
    --batch_size 1 \\
    --num_workers 4 \\
    --amp 0 \\
    --use_anny_shape 1 \\
    --mask_helper_joints_train 1 \\
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \\
    --name eval_${RUN_NAME}_${EPOCH}
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"