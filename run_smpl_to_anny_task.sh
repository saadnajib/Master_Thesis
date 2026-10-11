#!/bin/bash
# =============================================================================
# 3DPW -> Anny, STEP 2 of 3: convert ONE (sequence, person) - Slurm ARRAY task
# =============================================================================
# Submit as an array over the lines of threedpw_tasks.txt (written by step 1):
#     N=$(wc -l < threedpw_tasks.txt)
#     sbatch --array=0-$((N-1))%8 run_smpl_to_anny_task.sh
# %8 = at most 8 tasks on GPUs at once. A task that already has its .npz is
# skipped, so a failed/killed array can simply be resubmitted.
#
# Output: threedpw_anny_full/<split>/<seq>_p<person>.npz  (+ a log per task)
#
# COST: v2.1 measured 11 s/frame with the 40-frame settings (280 + 200 iters).
# Adjacent frames warm-start from each other, so far fewer iterations are
# needed here: 20/80/60 + 100 refine ~= 6 s/frame. At STRIDE=3 (10 fps, every
# third image; neighbours are near-duplicates anyway) a 1000-frame sequence
# takes ~35 min. STRIDE=1 converts every image (~3x longer).
#
# VALIDATE FIRST on one sequence you already know (outdoors_fencing_01 at the
# 40-frame settings gave 15.8 mm): run the array with --array=<its line>, and
# check the PER-JOINT table in its log is still ~16-20 mm. If it is worse,
# raise WARM_ITERS / REFINE_ITERS.
# =============================================================================
#SBATCH --job-name=smpl2anny-all
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/smpl2anny-all-%A_%a.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/smpl2anny-all-%A_%a.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

TASKS="/netscratch/najib/multi-hmr/threedpw_tasks.txt"
EXTRACT_DIR="/netscratch/najib/multi-hmr/data/3DPW"
SMPL_MODEL="/netscratch/najib/multi-hmr/models/smpl/SMPL_NEUTRAL.pkl"
OUT_ROOT="/netscratch/najib/multi-hmr/threedpw_anny_full"

# ---- conversion settings ----------------------------------------------------
STRIDE=3                      # 1 = every image; 3 = every third (10 fps)
ITERS="150 400 250"           # first frame of a sequence (cold start)
WARM_ITERS="20 80 60"         # every following frame (warm-started)
REFINE_ITERS=100              # pass 2 with the shared body
FREE_PHENOTYPES="height,proportions"
SET_PHENOTYPES="age=0.55"     # joints cannot measure adult age; AnnyOne mean
# -----------------------------------------------------------------------------

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(sed -n "$((TASK_ID + 1))p" "${TASKS}")
[ -n "${LINE}" ] || { echo "ERROR: no line $((TASK_ID + 1)) in ${TASKS}"; exit 1; }
read -r SPLIT SEQ PERSON NFRAMES <<< "${LINE}"
OUT="${OUT_ROOT}/${SPLIT}/${SEQ}_p${PERSON}.npz"
echo "task ${TASK_ID}: ${SPLIT}/${SEQ} person ${PERSON} (${NFRAMES} frames) -> ${OUT}"

if [ -f "${OUT}" ]; then
  echo "already converted - skipping"; exit 0
fi

COMMAND=$(cat <<EOF
set -uo pipefail
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr
export TORCH_HOME=/netscratch/najib/torch_cache
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p "\$XDG_CACHE_HOME" /netscratch/najib/anny_cache "\$HOME/.cache" "${OUT_ROOT}/${SPLIT}"
[ -e "\$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "\$HOME/.cache/anny"
cd /netscratch/najib/multi-hmr
[ -f "${SMPL_MODEL}" ] || { echo "ERROR: SMPL model missing: ${SMPL_MODEL}"; exit 1; }
PKL="${EXTRACT_DIR}/sequenceFiles/${SPLIT}/${SEQ}.pkl"
[ -f "\$PKL" ] || { echo "ERROR: \$PKL not found (run run_3dpw_list.sh first)"; exit 1; }

T0=\$(date +%s)
python smpl_to_anny.py \\
  --pkl "\$PKL" --person ${PERSON} \\
  --smpl_model_path "${SMPL_MODEL}" \\
  --out "${OUT}.tmp.npz" \\
  --max_frames 1000000 --frame_stride ${STRIDE} \\
  --iters ${ITERS} --warm_iters ${WARM_ITERS} --refine_iters ${REFINE_ITERS} \\
  --free_phenotypes "${FREE_PHENOTYPES}" --set_phenotypes "${SET_PHENOTYPES}" \\
  --shared_shape 1 --quiet 1
RC=\$?
if [ \$RC -eq 0 ] && [ -f "${OUT}.tmp.npz" ]; then
  mv "${OUT}.tmp.npz" "${OUT}"
  T1=\$(date +%s)
  echo "done: ${OUT}  (\$((T1-T0)) s)"
else
  rm -f "${OUT}.tmp.npz"
  echo "FAILED (rc=\$RC): ${SPLIT}/${SEQ} person ${PERSON}"
  exit 1
fi
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/netscratch:/netscratch,/home/najib:/home/najib,/ds-av:/ds-av \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
