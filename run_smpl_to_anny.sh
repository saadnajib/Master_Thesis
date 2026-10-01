#!/bin/bash
# =============================================================================
# SMPL -> Anny conversion for 3DPW ground truth  (Track B)  -- v2.1 scripts
# =============================================================================
# 3DPW ships GT as SMPL parameters; our model speaks Anny. smpl_to_anny.py fits
# Anny's pose + shape to SMPL's joints by gradient descent.
#
# RUN IN STAGES:
#   MODE=list_bones      print Anny's real bone names
#   MODE=check_mapping   both rest skeletons side by side + the spine/hip bones
#                        the geometry check picks   <-- run this FIRST with v2
#   MODE=fit             convert frames of ${PKL}           -> ${OUT}
#   MODE=render          overlay an existing ${OUT} on the images (no refit)
#   MODE=fit_and_render  both
#
# v2 changes (after job 3454323): targets = 3DPW's own jointPositions, spine /
# hip bones chosen by rest-pose geometry, gender fixed from the .pkl, only
# age/height/proportions fit, light pose regulariser, per-joint error table,
# and the renderer uses the pose frame index as the image number (the old
# img_frame_ids field was 3DPW's 60 Hz index map, i.e. 2x the image number).
# =============================================================================
#SBATCH --job-name=smpl2anny
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTXA6000,RTXB6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/najib/multi-hmr/logs/smpl2anny-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/smpl2anny-%j.err

set -euo pipefail
mkdir -p /netscratch/najib/multi-hmr/logs

# >>> STEP: list_bones | check_mapping | fit | render | fit_and_render <<<
MODE="fit_and_render"

ZIP_ROOT="/ds-av/public_datasets/3DPW/original"
EXTRACT_DIR="/netscratch/najib/multi-hmr/data/3DPW"
SEQ="outdoors_fencing_01"                 # name of the .pkl in sequenceFiles/test (or validation)
PERSON=0                                  # subject index inside the .pkl
SMPL_MODEL="/netscratch/najib/multi-hmr/models/smpl/SMPL_NEUTRAL.pkl"

# ---- fit settings -----------------------------------------------------------
MAX_FRAMES=40          # 4 = smoke test, 40 = check, 100000 = whole sequence
FRAME_STRIDE=25        # 1 = adjacent frames; 25 = spread over the sequence
ITERS="150 400 250"    # root / root+pose / +shape iterations (first frame; later frames warm-start)
REFINE_ITERS=200       # pass 2 pose-only refit per frame with the shared body
POSE_REG=1e-4          # 0 = off (v1 behaviour)
TARGET="gt"            # gt = 3DPW jointPositions (recommended) | smpl = neutral SMPL forward
FREE_PHENOTYPES="age,height,proportions"   # or "all" for v1 behaviour
OUT_DIR="/netscratch/najib/multi-hmr/threedpw_anny_v21"   # v2 baseline stays in threedpw_anny/
OUT="${OUT_DIR}/${SEQ}.npz"
RENDER_DIR="${OUT_DIR}/renders/${SEQ}"
# -----------------------------------------------------------------------------

export NVIDIA_DRIVER_CAPABILITIES=all
DRI_MOUNT=""; [ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

COMMAND=$(cat <<EOF
set -uo pipefail
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

export TORCH_HOME=/netscratch/najib/torch_cache
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p "\$XDG_CACHE_HOME" /netscratch/najib/anny_cache "\$HOME/.cache"
[ -e "\$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "\$HOME/.cache/anny"
cd /netscratch/najib/multi-hmr

MODE="${MODE}"
[ -d "${ZIP_ROOT}" ] || { echo "ERROR: ${ZIP_ROOT} not mounted in container"; exit 1; }
[ -f "${SMPL_MODEL}" ] || { echo "ERROR: SMPL model missing: ${SMPL_MODEL}"; exit 1; }
mkdir -p "${EXTRACT_DIR}" "${OUT_DIR}"
[ -d "${EXTRACT_DIR}/sequenceFiles" ] || { echo "extracting sequenceFiles.zip"; unzip -q "${ZIP_ROOT}/sequenceFiles.zip" -d "${EXTRACT_DIR}"; }

PKL="${EXTRACT_DIR}/sequenceFiles/test/${SEQ}.pkl"
[ -f "\$PKL" ] || PKL="${EXTRACT_DIR}/sequenceFiles/validation/${SEQ}.pkl"

setup_egl() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq libegl1 libglvnd0 libgl1 libgles2 >/dev/null 2>&1 || true
  mkdir -p /usr/share/glvnd/egl_vendor.d
  printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \\
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
  export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
  export PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0
  pip install --quiet --exists-action i pyrender trimesh >/dev/null 2>&1 || true
  [ -d "${EXTRACT_DIR}/imageFiles" ] || { echo "extracting imageFiles.zip (large, once)"; unzip -q "${ZIP_ROOT}/imageFiles.zip" -d "${EXTRACT_DIR}"; }
}

do_fit() {
  [ -f "\$PKL" ] || { echo "ERROR: ${SEQ}.pkl not found in test/ or validation/"; exit 1; }
  echo "start: \$(date)"; T0=\$(date +%s)
  python smpl_to_anny.py \\
    --pkl "\$PKL" \\
    --person ${PERSON} \\
    --smpl_model_path "${SMPL_MODEL}" \\
    --target ${TARGET} \\
    --out "${OUT}" \\
    --max_frames ${MAX_FRAMES} \\
    --frame_stride ${FRAME_STRIDE} \\
    --iters ${ITERS} \\
    --pose_reg ${POSE_REG} \\
    --free_phenotypes "${FREE_PHENOTYPES}" \\
    --shared_shape 1 --refine_iters ${REFINE_ITERS}
  RC=\$?
  [ \$RC -eq 0 ] && [ -f "${OUT}" ] || { echo "FIT FAILED (rc=\$RC)"; exit 1; }
  T1=\$(date +%s)
  echo "end:   \$(date)"
  echo "wall time: \$((T1-T0)) s for ${MAX_FRAMES} frames = \$(( (T1-T0) / ${MAX_FRAMES} )) s/frame"
}

do_render() {
  [ -f "${OUT}" ] || { echo "ERROR: nothing to render, ${OUT} missing (run MODE=fit first)"; exit 1; }
  setup_egl
  python render_anny_fit.py \\
    --npz "${OUT}" \\
    --img_dir "${EXTRACT_DIR}/imageFiles/${SEQ}" \\
    --out "${RENDER_DIR}"
  echo "renders in ${RENDER_DIR}/ - red = SMPL GT joints, green = fitted Anny joints"
}

case "\$MODE" in
  list_bones)     python smpl_to_anny.py --list_bones ;;
  check_mapping)  python smpl_to_anny.py --check_mapping --smpl_model_path "${SMPL_MODEL}" ;;
  fit)            do_fit ;;
  render)         do_render ;;
  fit_and_render) do_fit; do_render ;;
  *) echo "ERROR: MODE must be list_bones, check_mapping, fit, render or fit_and_render (got '\$MODE')"; exit 1 ;;
esac
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm,/ds-av:/ds-av${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"
