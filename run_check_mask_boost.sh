#!/bin/bash
# =============================================================================
# Diagnostic: do the helper-joint MASK and the per-joint loss BOOST conflict?
# =============================================================================
# Prints, for every neck/head and finger/thumb/wrist bone, whether it is
# MASKED (forced to identity -> loss exactly zero -> the x4 boost did nothing)
# or ACTIVE (loss live -> the boost applied).
#
# This is a read-only check. No GPU work, no training, ~1 minute.
#
# NOTE: you can also just run this on the LOGIN NODE without slurm:
#     conda activate multihmr
#     cd /netscratch/najib/multi-hmr
#     python check_mask_boost.py
# The slurm version exists in case the login node lacks the anny cache.
# =============================================================================
#SBATCH --job-name=mhmr-mask-check
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/mask-check-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/mask-check-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

COMMAND=$(cat <<'EOF'
set -euo pipefail

source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

export TORCH_HOME=/netscratch/najib/torch_cache
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p "$XDG_CACHE_HOME" /netscratch/najib/anny_cache "$HOME/.cache"
[ -e "$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "$HOME/.cache/anny"

cd /netscratch/najib/multi-hmr
python check_mask_boost.py
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"