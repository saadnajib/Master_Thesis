#!/bin/bash
# =============================================================================
# Task B, steps 1-3: convert several 3DPW scenes to Anny + render for checking
# =============================================================================
# For each scene in SCENES:
#   1. smpl_to_anny.py  --shared_shape 1
#        pass 1: free-shape fit on 8 frames -> MEDIAN phenotypes = the body
#        pass 2: refit ~40 frames with that body fixed (pose only)
#        -> threedpw_anny/<scene>.npz  (Anny params + camera bookkeeping)
#   2. render_anny_fit.py
#        -> threedpw_anny/renders/<scene>/<scene>_<frame>.jpg  (photo | overlay)
#
# Scenes chosen for VARIETY (supervisor asked for 3-4 diverse scenes):
#   downtown_walking_00   plain walking, one person, simple
#   outdoors_fencing_01   fast, wide limb motion
#   office_phoneCall_00   indoor, standing, subtle pose
#   flat_guitar_01        sitting, self-occlusion
#
# Output .npz files are what the mixed synthetic+real training (Task A/B
# merge) will consume. Frames with joint error > 80 mm are flagged in the log;
# filter on 'joint_err_mm' inside the .npz before training on them.
#
# Rendering is best-effort: if EGL/pyrender fails on a node, the fits are
# still saved and the script continues to the next scene.
# =============================================================================
#SBATCH --job-name=smpl2anny-scenes
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/smpl2anny-scenes-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/smpl2anny-scenes-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

# >>> scenes to convert (names of .pkl files in sequenceFiles/test, no ext) <<<
SCENES="downtown_walking_00 outdoors_fencing_01 office_phoneCall_00 flat_guitar_01"
FRAMES_PER_SCENE=40
FRAME_STRIDE=25

ZIP_ROOT="/ds-av/public_datasets/3DPW/original"
EXTRACT_DIR="/netscratch/najib/multi-hmr/data/3DPW"
SMPL_MODEL="/netscratch/najib/multi-hmr/models/smpl/SMPL_NEUTRAL.pkl"
OUT_DIR="/netscratch/najib/multi-hmr/threedpw_anny"

export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

COMMAND=$(cat <<EOF
set -uo pipefail
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

# EGL for offscreen rendering (same recipe as run_demo.sh)
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq libegl1 libglvnd0 libgl1 libgles2 >/dev/null 2>&1 || true
mkdir -p /usr/share/glvnd/egl_vendor.d
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \\
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0
pip install --quiet --exists-action i pyrender trimesh >/dev/null 2>&1 || true

export TORCH_HOME=/netscratch/najib/torch_cache
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p "\$XDG_CACHE_HOME" /netscratch/najib/anny_cache "\$HOME/.cache"
[ -e "\$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "\$HOME/.cache/anny"
cd /netscratch/najib/multi-hmr

[ -d "${ZIP_ROOT}" ] || { echo "ERROR: ${ZIP_ROOT} not mounted in container"; exit 1; }
[ -f "${SMPL_MODEL}" ] || { echo "ERROR: SMPL model missing: ${SMPL_MODEL}"; exit 1; }
mkdir -p "${EXTRACT_DIR}" "${OUT_DIR}/renders"
[ -d "${EXTRACT_DIR}/sequenceFiles" ] || { echo "extracting sequenceFiles.zip"; unzip -q "${ZIP_ROOT}/sequenceFiles.zip" -d "${EXTRACT_DIR}"; }
[ -d "${EXTRACT_DIR}/imageFiles" ]    || { echo "extracting imageFiles.zip (large, once)"; unzip -q "${ZIP_ROOT}/imageFiles.zip" -d "${EXTRACT_DIR}"; }

for SCENE in ${SCENES}; do
  echo ""
  echo "########################################################################"
  echo "# SCENE: \${SCENE}"
  echo "########################################################################"
  PKL="${EXTRACT_DIR}/sequenceFiles/test/\${SCENE}.pkl"
  [ -f "\$PKL" ] || PKL="${EXTRACT_DIR}/sequenceFiles/validation/\${SCENE}.pkl"
  if [ ! -f "\$PKL" ]; then
    echo "  not found in test/ or validation/: \${SCENE}.pkl - skipping"
    continue
  fi
  NPZ="${OUT_DIR}/\${SCENE}.npz"

  python smpl_to_anny.py \\
    --pkl "\$PKL" \\
    --smpl_model_path "${SMPL_MODEL}" \\
    --out "\$NPZ" \\
    --max_frames ${FRAMES_PER_SCENE} \\
    --frame_stride ${FRAME_STRIDE} \\
    --shared_shape 1 --shape_frames 8
  RC=\$?
  if [ \$RC -ne 0 ] || [ ! -f "\$NPZ" ]; then
    echo "  FIT FAILED for \${SCENE} (rc=\$RC) - continuing to next scene"
    continue
  fi

  echo "--- rendering \${SCENE} ---"
  python render_anny_fit.py \\
    --npz "\$NPZ" \\
    --img_dir "${EXTRACT_DIR}/imageFiles/\${SCENE}" \\
    --out "${OUT_DIR}/renders/\${SCENE}" \\
    || echo "  render failed for \${SCENE} (fits are still saved)"
done

echo ""
echo "=============================== SUMMARY ==============================="
python - <<'PYEOF'
import glob, numpy as np, os
for f in sorted(glob.glob("${OUT_DIR}/*.npz")):
    d = np.load(f, allow_pickle=True)
    e = d['joint_err_mm']
    bad = int((e > 80).sum())
    print(f"{os.path.basename(f):<30} frames={len(e):3d}  mean={e.mean():5.1f} mm  "
          f"max={e.max():5.1f}  >80mm: {bad}")
PYEOF
echo "renders in ${OUT_DIR}/renders/<scene>/ - open a few and check the bodies look right"
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm,/ds-av:/ds-av${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
