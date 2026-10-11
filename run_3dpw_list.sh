#!/bin/bash
# =============================================================================
# 3DPW -> Anny, STEP 1 of 3: extract 3DPW once and list every conversion task
# =============================================================================
# Writes threedpw_tasks.txt, one line per (split, sequence, person):
#     train courtyard_arguing_00 0 1200
#     train courtyard_arguing_00 1 1200
#     ...
# Step 2 (run_smpl_to_anny_task.sh) is a Slurm ARRAY over these lines, so the
# array size is simply the number of lines. CPU only, a few minutes (the first
# time it also unzips imageFiles.zip, which is large).
# =============================================================================
#SBATCH --job-name=3dpw-list
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000,batch
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/3dpw-list-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/3dpw-list-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

ZIP_ROOT="/ds-av/public_datasets/3DPW/original"
EXTRACT_DIR="/netscratch/najib/multi-hmr/data/3DPW"
TASKS="/netscratch/najib/multi-hmr/threedpw_tasks.txt"

COMMAND=$(cat <<EOF
set -euo pipefail
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr
cd /netscratch/najib/multi-hmr
[ -d "${ZIP_ROOT}" ] || { echo "ERROR: ${ZIP_ROOT} not mounted in container"; exit 1; }
mkdir -p "${EXTRACT_DIR}"
[ -d "${EXTRACT_DIR}/sequenceFiles" ] || { echo "extracting sequenceFiles.zip"; unzip -q "${ZIP_ROOT}/sequenceFiles.zip" -d "${EXTRACT_DIR}"; }
[ -d "${EXTRACT_DIR}/imageFiles" ]    || { echo "extracting imageFiles.zip (large, once)"; unzip -q "${ZIP_ROOT}/imageFiles.zip" -d "${EXTRACT_DIR}"; }

python - <<'PYEOF'
import glob, os, pickle
root = "${EXTRACT_DIR}/sequenceFiles"
lines = []
for split in ("train", "validation", "test"):
    for pkl in sorted(glob.glob(os.path.join(root, split, "*.pkl"))):
        with open(pkl, "rb") as f:
            seq = pickle.load(f, encoding="latin1")
        n_people = len(seq["poses"])
        n_frames = len(seq["poses"][0])
        name = os.path.splitext(os.path.basename(pkl))[0]
        for p in range(n_people):
            lines.append(f"{split} {name} {p} {n_frames}")
with open("${TASKS}", "w") as f:
    f.write("\\n".join(lines) + "\\n")
by_split = {}
for l in lines:
    by_split[l.split()[0]] = by_split.get(l.split()[0], 0) + 1
print(f"wrote {len(lines)} tasks -> ${TASKS}   " + ", ".join(f"{k}: {v}" for k, v in by_split.items()))
print(f"total person-frames: {sum(int(l.split()[3]) for l in lines):,}")
PYEOF

echo ""
echo "Next:  N=\$(wc -l < ${TASKS});  sbatch --array=0-\$((N-1))%8 run_smpl_to_anny_task.sh"
EOF
)

srun -K --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/netscratch:/netscratch,/home/najib:/home/najib,/ds-av:/ds-av \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
