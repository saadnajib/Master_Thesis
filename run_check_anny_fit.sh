#!/bin/bash
# Sanity-check the SMPL->Anny fit (.npz) - CPU only, ~1 min.
# Uses the same container + conda env as run_smpl_to_anny.sh so matplotlib/numpy match.
#SBATCH --job-name=anny-check
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/anny-check-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/anny-check-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

# usage: sbatch run_check_anny_fit.sh [fit.npz]   (defaults to the 60-frame pilot)
FIT_NPZ="${1:-/netscratch/najib/multi-hmr/threedpw_anny_fit_pilot60_s1.npz}"
OUT_DIR="/netscratch/najib/multi-hmr/anny_fit_check_$(basename "${FIT_NPZ}" .npz)"

COMMAND=$(cat <<EOF
set -euo pipefail
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr
cd /netscratch/najib/multi-hmr
python -c "import matplotlib" 2>/dev/null || pip install -q matplotlib
python check_anny_fit.py "${FIT_NPZ}" --out "${OUT_DIR}" --max-frames 12
EOF
)

srun -K \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/netscratch:/netscratch,/home/najib:/home/najib \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
