#!/bin/bash
# Convert a SMPL->Anny fit (.npz) into Anny-One style label files (.pkl), one per frame.
# CPU only, needs just numpy. Takes well under a minute for 60 frames.
#
# usage:
#   sbatch run_npz_to_annyone.sh                       # defaults: 60-frame pilot, person 0
#   sbatch run_npz_to_annyone.sh path/to/fit.npz       # another fit file
#   sbatch run_npz_to_annyone.sh path/to/fit.npz 1     # second person in the sequence
#
#SBATCH --job-name=npz2annyone
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/npz2annyone-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/npz2annyone-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

FIT_NPZ="${1:-/netscratch/najib/multi-hmr/threedpw_anny_fit_pilot60_s1.npz}"
PERSON="${2:-0}"
SEQ_PKL="/netscratch/najib/multi-hmr/data/3DPW/sequenceFiles/test/downtown_bar_00.pkl"
OUT_DIR="/netscratch/najib/multi-hmr/annyone_3dpw/$(basename "${FIT_NPZ}" .npz)_p${PERSON}"

COMMAND=$(cat <<EOF
set -euo pipefail
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr
cd /netscratch/najib/multi-hmr
python npz_to_annyone.py \
  --seq_pkl "${SEQ_PKL}" \
  --fit_npz "${FIT_NPZ}" \
  --out_dir "${OUT_DIR}" \
  --person "${PERSON}"
echo "Done. Label files written to: ${OUT_DIR}"
ls "${OUT_DIR}" | head
EOF
)

srun -K \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/netscratch:/netscratch,/home/najib:/home/najib \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
