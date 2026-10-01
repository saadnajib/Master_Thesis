#!/bin/bash
# =============================================================================
# 3DPW EVALUATION of the Anny teacher (anny_s2_shape_v5 / 00099)
# =============================================================================
# Runs train.py --eval_only on the 3DPW --val_data path (datasets.threedpw.
# THREEDPW, SMPL ground truth). The result row is appended with split=3dpw to
# results/eval3dpw_<run>_<epoch>.csv and results/all_results.jsonl.
#
# WHAT IS AND IS NOT MEASURED - read before quoting numbers:
#   * precision / recall / F1 : computed. Predictions are matched to GT people
#     by 2D keypoint distance + bbox overlap (IoU >= 0.05). NOTE the matching
#     compares the first 45 projected ANNY joints to the 45 SMPL joints; the
#     joint sets do not correspond one-to-one, so this is effectively a loose
#     person-box match. Treat these as approximate detection numbers.
#   * PVE / PA-PVE / MPJPE / PA-MPJPE : NOT computed (left empty in the CSV).
#     3DPW GT is an SMPL mesh (6890 verts); the model predicts an Anny mesh.
#     train.py only has an SMPL-X->SMPL regressor, so it skips these metrics
#     with a warning instead of crashing (the old code crashed on the matmul).
#
# TODO(cluster): to get real 3D metrics on 3DPW, one of these is needed:
#   (a) an Anny->SMPL vertex regressor (6890 x V_anny) to apply to predictions
#       where train.py's evaluate() currently uses smplx2smpl_regressor, plus
#       models/smpl/J_regressor_h36m.npy for the 14-joint (H36M) MPJPE; or
#   (b) evaluate on the Anny-fitted 3DPW (run_npz_to_annyone.sh output under
#       /netscratch/najib/multi-hmr/annyone_3dpw/...) through the AnnyOne
#       path - but AnnyOne's data_folder is hard-coded to
#       /netscratch/najib/anydataset/ in train.py, so that needs a flag.
# TODO(cluster): THREEDPW's constructor is not in this repo. This script
#   assumes the upstream Multi-HMR signature THREEDPW(split, training,
#   img_size, subsample, n) with split 'test' and n=-1 meaning "all frames".
#   If it fails, check datasets/threedpw.py on the cluster (data path, split
#   names). threedpw_image_export.py prints the accepted arguments.
# =============================================================================
#SBATCH --job-name=mhmr-eval-3dpw
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/eval-3dpw-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/eval-3dpw-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

# >>> WHICH CHECKPOINT TO EVALUATE <<<
RUN_NAME="anny_s2_shape_v5"
EPOCH="00099"
# >>> 3DPW SPLIT / SUBSAMPLING (1 = every frame; e.g. 5 = every 5th, faster) <<<
TDPW_SPLIT="test"
TDPW_SUBSAMPLE=1

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
echo "Evaluating \$CKPT on 3DPW '${TDPW_SPLIT}' (subsample ${TDPW_SUBSAMPLE})"

# --eval_only 1 : no training, just run evaluate() on the --val_data loaders.
# --val_data THREEDPW --val_split/--val_subsample/--val_n : the 3DPW loader
#   built in train.py main() (one value per --val_data entry).
# --train_data is left at its default (BEDLAM) on purpose, so no AnnyOne
#   split is built and ONLY 3DPW is evaluated.
# Architecture flags must MATCH the checkpoint or the weights will not load.
python train.py \\
    --eval_only 1 \\
    --val_data THREEDPW \\
    --val_split ${TDPW_SPLIT} \\
    --val_subsample ${TDPW_SUBSAMPLE} \\
    --val_n -1 \\
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
    --name eval3dpw_${RUN_NAME}_${EPOCH}
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
