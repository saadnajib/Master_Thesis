#!/bin/bash
#SBATCH --partition=A100-40GB,A100-80GB,RTXA6000-AV,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=multi-hmr-demo-annyone
#SBATCH --output=/netscratch/najib/multi-hmr/logs/demo-annyone-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/demo-annyone-%j.err

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

export TORCH_HOME=/netscratch/najib/torch_cache
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p "$XDG_CACHE_HOME" /netscratch/najib/anny_cache "$HOME/.cache"
[ -e "$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "$HOME/.cache/anny"

echo "======================================================"
echo "2b. Patching DINOv2 for Python 3.9 compatibility..."
echo "======================================================"
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)" || true
find /netscratch/najib/torch_cache/hub/facebookresearch_dinov2_main/dinov2 -name "*.py" -exec sed -i 's/ | None//g' {} + || true

cd "$PROJECT_ROOT"

echo "======================================================"
echo "2c. Exporting images from the AnnyOne dataset..."
echo "======================================================"
# Pull sample images out of the dataset. 'holdout' = the last 100 samples,
# which --val_anny_n 100 reserved and the model never trained on.
# Switch to --split train to see performance on images it HAS seen.
IMG_DIR="/netscratch/najib/multi-hmr/annyone_demo_images"
rm -rf "$IMG_DIR"
# Capture the export output so we can read the TRUE field of view out of it.
# CRITICAL: AnnyOne images each have their OWN camera. Measured on the holdout
# set the FOVs were 67.9, 102.5, 57.5, 51.1 and 110.4 degrees. Passing one
# global --fov 97 for all of them put every mesh at the wrong depth and scale
# (renders came out as giant flat blobs filling the frame).
EXPORT_OUT=$(python3.9 anyone_image_export.py \
  --n 10 \
  --split holdout \
  --out "$IMG_DIR" 2>&1)
echo "$EXPORT_OUT"

# Pull the recommended mean FOV out of the export output; fall back to 78
# (the measured mean) if parsing fails.
FOV=$(echo "$EXPORT_OUT" | sed -n 's/.*pass  *--fov  *\([0-9][0-9]*\).*/\1/p' | head -1)
FOV=${FOV:-78}
echo "Using --fov $FOV (parsed from the export step)"
echo "WARNING: per-image FOV varies a lot in AnnyOne. A single global FOV is"
echo "         an approximation - meshes on images far from this value will"
echo "         still be mis-scaled. See notes below about --use_gt_K."

echo "Exported files:"
ls -la "$IMG_DIR"

echo "======================================================"
echo "2d. Preparing checkpoint..."
echo "======================================================"
# v4 = the latest boosted run (was pointing at the OLD anny_full_run_v2!)
V4_EPOCH="00099"
CKPT_SRC="/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/${V4_EPOCH}.pt"
CKPT_NAME="anny_s2_shape_v5_${V4_EPOCH}"
if [ ! -f "$CKPT_SRC" ]; then
  echo "ERROR: checkpoint not found: $CKPT_SRC"; exit 1
fi
CKPT_CACHE_DIR="/netscratch/najib/torch_cache/multihmr"
mkdir -p "$CKPT_CACHE_DIR"
cp "$CKPT_SRC" "$CKPT_CACHE_DIR/${CKPT_NAME}.pt"

echo "======================================================"
echo "3. Running Multi-HMR Demo on AnnyOne images..."
echo "======================================================"
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

# IN-DOMAIN SETTINGS (differ from run_demo.sh, which targets real photos):
#   --use_person_detector 0 : no Faster R-CNN crops. These images already match
#                             the training distribution in framing and scale, so
#                             detector-guided cropping would push them OFF the
#                             distribution the model learned.
#   --det_thresh 0.3        : back to the normal threshold. The low 0.15 in
#                             run_demo.sh compensated for out-of-domain photos;
#                             in-domain the detection head should be confident.
#   --fov                   : check the value printed by export_annyone_images.py
#                             above and set it to match the dataset's true FOV.
python3.9 demo.py \
  --model_name "$CKPT_NAME" \
  --img_folder "$IMG_DIR" \
  --out_folder demo_out_annyone_${CKPT_NAME} \
  --alpha 0.9 \
  --fov "$FOV" \
  --fov_json "/netscratch/najib/multi-hmr/annyone_fov.json" \
  --use_person_detector 0 \
  --max_persons 5 \
  --det_thresh 0.3 \
  --nms_kernel_size 3 \
  --extra_views 0 \
  --save_rotating_video 0 \
  --save_mesh 1 \
  --mask_helper_joints 1 \
  --relax_hands 0 \
  --rest_pose 0

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