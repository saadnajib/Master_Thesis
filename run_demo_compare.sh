#!/bin/bash

#!/bin/bash
# =============================================================================
# DEMO COMPARISON: the same images through v6b, v7 (mixed) and v6c
# =============================================================================
#   v6b  student, synthetic only, batch 8            (Anny-One 128.5 / 3DPW-val 149.0)
#   v7   student, + 30% real 3DPW, batch 8           (Anny-One ~121   / 3DPW-val ~107)
#   v6c  student, synthetic only, batch 16           (Anny-One 114.5)
# Every model is run with IDENTICAL settings on IMG_FOLDER, each into its own
# output folder  demo_out_<run>_<epoch>/  so the results can be put side by side
# (make_montage.py). A model whose checkpoint is missing is skipped, the others
# still run. Checkpoints are looked up in models/multiHMR/ first (the saved
# copies), then in logs/anny_model/<run>/checkpoints/<epoch>.pt, then the
# highest epoch still on disk (with a warning and the real epoch in the name).
# Needs model.py with the v3 patch (feat_proj), as for any v6+ student.
# =============================================================================
#SBATCH --partition=A100-40GB,A100-80GB,RTXA6000-AV,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=demo-compare
#SBATCH --output=/netscratch/najib/multi-hmr/logs/demo-compare-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/demo-compare-%j.err

set -euo pipefail

mkdir -p /netscratch/najib/multi-hmr/logs

export NVIDIA_DRIVER_CAPABILITIES=all

DRI_MOUNT=""
[ -d /dev/dri ] && DRI_MOUNT=",/dev/dri:/dev/dri"

COMMAND=$(cat <<'EOF'
set -euo pipefail

export PROJECT_ROOT="/netscratch/najib/multi-hmr"
export CONDA_SH="/netscratch/najib/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"

echo "======================================================"
echo "1. Installing Native EGL Graphics Routing..."
echo "======================================================"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libegl1 libglvnd0 libgl1 libgles2

mkdir -p /usr/share/glvnd/egl_vendor.d
cat << 'JSON' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
{
  "file_format_version" : "1.0.0",
  "ICD" : {
    "library_path" : "libEGL_nvidia.so.0"
  }
}
JSON
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d

echo "======================================================"
echo "2. Activating Env..."
echo "======================================================"
conda activate multihmr
pip install --quiet --exists-action i pyglet pyopengl

# torchvision is needed for --use_person_detector 1 (Faster R-CNN).
# Usually already installed with torch; this is a no-op safeguard.
python -c "import torchvision" 2>/dev/null || pip install --quiet torchvision

export TORCH_HOME=/netscratch/najib/torch_cache

# Route anny's hardcoded ~/.cache/anny onto netscratch (avoids disk-quota error)
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p "$XDG_CACHE_HOME" /netscratch/najib/anny_cache "$HOME/.cache"
[ -e "$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "$HOME/.cache/anny"

echo "======================================================"
echo "2b. Patching DINOv2 for Python 3.9 compatibility..."
echo "======================================================"
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)" || true
find /netscratch/najib/torch_cache/hub/facebookresearch_dinov2_main/dinov2 -name "*.py" -exec sed -i 's/ | None//g' {} + || true

echo "======================================================"
echo "2c. Checkpoints + 3. Running the demo on every model"
echo "======================================================"
cd "$PROJECT_ROOT"
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

grep -q "feat_proj" model.py || { echo "ERROR: model.py lacks the v3 patch (python apply_student_fix_v3.py)"; exit 1; }

# >>> EDIT HERE: images and the models to compare ("run_name|preferred_epoch") <<<
IMG_FOLDER="/netscratch/najib/multi-hmr/example_data"
FOV=55
MODELS=(
  "anny_distill_vits_v6b|00140"
  "anny_distill_vits_v7_mixed|00130"
  "anny_distill_vits_v6c|00149"
  # "anny_s2_shape_v5|00099"            # teacher, for the side-by-side figure
)
MODELS_DIR="$PROJECT_ROOT/models/multiHMR"
mkdir -p "$MODELS_DIR"

# resolve_ckpt RUN EPOCH -> prints the model name (run_epoch) whose .pt is in $MODELS_DIR, or nothing
resolve_ckpt() {
  local run="$1" ep="$2" logs="$PROJECT_ROOT/logs/anny_model/$1/checkpoints"
  if [ -f "$MODELS_DIR/${run}_${ep}.pt" ]; then echo "${run}_${ep}"; return; fi
  if [ -f "$logs/${ep}.pt" ]; then cp "$logs/${ep}.pt" "$MODELS_DIR/${run}_${ep}.pt"; echo "${run}_${ep}"; return; fi
  local last; last=$(ls -1 "$logs"/*.pt 2>/dev/null | sort | tail -1 || true)
  if [ -n "$last" ]; then
    local e; e=$(basename "$last" .pt)
    cp "$last" "$MODELS_DIR/${run}_${e}.pt"
    echo "WARNING: ${run} epoch ${ep} not found; using the latest on disk (${e})" >&2
    echo "${run}_${e}"; return
  fi
}

OK=(); FAILED=()
for entry in "${MODELS[@]}"; do
  run="${entry%%|*}"; ep="${entry##*|}"
  echo ""; echo "################ MODEL: ${run} (preferred epoch ${ep}) ################"
  name=$(resolve_ckpt "$run" "$ep")
  if [ -z "$name" ]; then echo "SKIPPED: no checkpoint for ${run} in models/multiHMR or logs"; FAILED+=("$run (no checkpoint)"); continue; fi
  echo "using model_name: $name"
  if python3.9 demo.py \
      --model_name "$name" \
      --img_folder "$IMG_FOLDER" \
      --out_folder "demo_out_${name}" \
      --alpha 0.9 \
      --fov "$FOV" \
      --use_person_detector 1 \
      --detector_thresh 0.8 \
      --crop_margin 0.35 \
      --anchor_to_box 1 \
      --max_persons 5 \
      --det_thresh 0.15 \
      --nms_kernel_size 5 \
      --iou_thresh 0.45 \
      --extra_views 0 \
      --save_rotating_video 0 \
      --person_frac 0.22 \
      --save_mesh 1 \
      --mask_helper_joints 1 \
      --relax_hands 1 \
      --rest_pose 0; then
    OK+=("$name")
  else
    FAILED+=("$name (demo.py failed)")
  fi
done

echo ""
echo "======================================================"
echo "SUMMARY"
echo "======================================================"
for n in "${OK[@]}"; do echo "  ok      demo_out_${n}/"; done
for n in "${FAILED[@]}"; do echo "  FAILED  ${n}"; done
[ ${#FAILED[@]} -eq 0 ] || exit 1
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"