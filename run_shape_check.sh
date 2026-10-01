#!/bin/bash
# =============================================================================
# Diagnostic: is the SHAPE head over-predicting 'proportions' (the neck)?
# =============================================================================
# Compares GROUND-TRUTH vs PREDICTED phenotypes for the exact AnnyOne holdout
# samples that were rendered in the demo.
#
# Why this is the decisive test - everything else is already ruled out:
#   * NOT the body model      -> --rest_pose 1 renders a normal neck
#   * NOT the helper mask     -> check_mask_boost: all 4 neck bones active
#   * NOT a domain gap        -> long necks appear on AnnyOne images too
#   -> the shape head is what's left. Anny's 'proportions' phenotype controls
#      head/neck proportions, the demo predicts it at 0.65-0.83 for everyone,
#      and alpha_shape=1.0 vs alpha_v3d=100.0 means shape is barely supervised.
#
# Read-only. No training, no GPU work. ~2 minutes.
#
# You can also run this on the LOGIN NODE without slurm:
#     conda activate multihmr
#     cd /netscratch/najib/multi-hmr
#     grep "\[shape\] person" logs/demo-annyone-3356335.out > /tmp/pred_shape.txt
#     python compare_gt_pred_shape.py --pred_log /tmp/pred_shape.txt --start 579860 --n 10
# =============================================================================
#SBATCH --job-name=mhmr-shape-check
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/shape-check-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/shape-check-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

# >>> the demo log whose [shape] lines we compare against <<<
DEMO_LOG="/netscratch/najib/multi-hmr/logs/demo-annyone-3369500.out"
# first holdout index that was demoed, and how many samples
START_IDX=579860
N_SAMPLES=10

COMMAND=$(cat <<EOF
set -euo pipefail

source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

export TORCH_HOME=/netscratch/najib/torch_cache
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p "\$XDG_CACHE_HOME" /netscratch/najib/anny_cache "\$HOME/.cache"
[ -e "\$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "\$HOME/.cache/anny"

cd /netscratch/najib/multi-hmr

# Pull the predicted phenotypes out of the demo log.
if [ ! -f "${DEMO_LOG}" ]; then
  echo "ERROR: demo log not found: ${DEMO_LOG}"
  echo "Set DEMO_LOG at the top of this script to a demo .out file that"
  echo "contains '[shape] person' lines."
  exit 1
fi
grep "\\[shape\\] person" "${DEMO_LOG}" > /tmp/pred_shape.txt || true
echo "parsed \$(wc -l < /tmp/pred_shape.txt) predicted people from ${DEMO_LOG}"
if [ ! -s /tmp/pred_shape.txt ]; then
  echo "ERROR: no '[shape] person' lines in that log - was it run with the"
  echo "updated demo.py that prints predicted phenotypes?"
  exit 1
fi

python compare_gt_pred_shape.py \\
  --pred_log /tmp/pred_shape.txt \\
  --start ${START_IDX} \\
  --n ${N_SAMPLES}
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
