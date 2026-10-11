#!/bin/bash
# 3DPW -> Anny, STEP 3 of 3: merge all fits into per-image label files. CPU only, ~1 min.
#SBATCH --job-name=3dpw-labels
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/3dpw-labels-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/3dpw-labels-%j.err

set -euo pipefail
FITS="/netscratch/najib/multi-hmr/threedpw_anny_full"
IMAGES="/netscratch/najib/multi-hmr/data/3DPW/imageFiles"
OUT="/netscratch/najib/multi-hmr/threedpw_anny_full/labels"

COMMAND=$(cat <<EOF
set -euo pipefail
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr
cd /netscratch/najib/multi-hmr
python build_threedpw_anny_labels.py --fits "${FITS}" --images "${IMAGES}" --out "${OUT}"
echo "unconverted tasks (no .npz yet):"
while read -r SPLIT SEQ PERSON N; do
  [ -f "${FITS}/\${SPLIT}/\${SEQ}_p\${PERSON}.npz" ] || echo "  \${SPLIT} \${SEQ} p\${PERSON}"
done < threedpw_tasks.txt
EOF
)

srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/netscratch:/netscratch,/home/najib:/home/najib \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
