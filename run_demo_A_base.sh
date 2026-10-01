#!/bin/bash
# =============================================================================
# DEMO A: BASELINE — no diagnostic toggles, current default behavior.
# Compare B and C against this to see what each toggle changes.
# =============================================================================
#SBATCH --partition=A100-40GB,A100-80GB,RTXA6000-AV,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=demo-A-base
#SBATCH --output=/netscratch/najib/multi-hmr/logs/demo-A-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/demo-A-%j.err

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
echo "2c. Preparing checkpoint for demo..."
echo "======================================================"
# Curated checkpoint (copied from logs/anny_model/anny_s2_partialft_v3/checkpoints/00054.pt)
CKPT_SRC="/netscratch/najib/multi-hmr/models/multiHMR/anny_s2_partialft_v3_00054.pt"
CKPT_NAME="anny_s2_partialft_v3_00054"
CKPT_CACHE_DIR="/netscratch/najib/torch_cache/multihmr"
mkdir -p "$CKPT_CACHE_DIR"
cp "$CKPT_SRC" "$CKPT_CACHE_DIR/${CKPT_NAME}.pt"
echo "Checkpoint copied to $CKPT_CACHE_DIR/${CKPT_NAME}.pt" 

echo "======================================================"
echo "3. Running Multi-HMR Demo..."
echo "======================================================" 
cd "$PROJECT_ROOT"
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

# MODE SWITCH:
#   Real photos (out-of-domain)     -> --use_person_detector 1 (detector-guided crops)
#   AnnyOne / synthetic (in-domain) -> --use_person_detector 0 (plain mode; the model
#     detects, places and scales by itself; also raise det_thresh back to 0.3)
python3.9 demo.py \
  --model_name "$CKPT_NAME" \
  --img_folder /netscratch/najib/multi-hmr/example_data \
  --out_folder demo_A_base \
  --alpha 0.9 \
  --fov 55 \
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
  --save_mesh 1

echo "======================================================"
echo "DEMO FINISHED SUCCESSFULLY!"
echo "======================================================"
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"