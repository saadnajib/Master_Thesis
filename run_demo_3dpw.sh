#!/bin/bash
#SBATCH --partition=A100-40GB,A100-80GB,RTXA6000-AV,L40S-AV,A100-RP,A100-PCI,RTXA6000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=demo_3dpw_rand
#SBATCH --output=/netscratch/najib/multi-hmr/logs/demo-3dpw-rand-%j.out
#SBATCH --error=/netscratch/najib/multi-hmr/logs/demo-3dpw-rand-%j.err

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
apt-get install -y -qq libegl1 libglvnd0 libgl1 libgles2 unzip

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
echo "2. Activating Env & Patching DINOv2..."
echo "======================================================"
conda activate multihmr
pip install --quiet --exists-action i pyglet pyopengl torchvision

export TORCH_HOME=/netscratch/najib/torch_cache
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p "$XDG_CACHE_HOME" /netscratch/najib/anny_cache "$HOME/.cache"
[ -e "$HOME/.cache/anny" ] || ln -sfn /netscratch/najib/anny_cache "$HOME/.cache/anny"

python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)" || true
find /netscratch/najib/torch_cache/hub/facebookresearch_dinov2_main/dinov2 -name "*.py" -exec sed -i 's/ | None//g' {} + || true

echo "======================================================"
echo "3. Extracting 10 Random Images from 3DPW..."
echo "======================================================"
IMG_DIR="/netscratch/najib/multi-hmr/3dpw_demo_images_random"
rm -rf "$IMG_DIR"
mkdir -p "$IMG_DIR"

# 1. Read zip contents, 2. Filter for .jpg, 3. Randomly shuffle and pick 10
RANDOM_FILES=$(unzip -Z1 /ds-av/public_datasets/3DPW/original/imageFiles.zip | grep -i "\.jpg$" | shuf -n 10)

echo "Selected random files for extraction:"
echo "$RANDOM_FILES"

# Extract exactly those 10 random files
unzip -q /ds-av/public_datasets/3DPW/original/imageFiles.zip $RANDOM_FILES -d "$IMG_DIR"

echo "======================================================"
echo "4. Preparing Checkpoint..."
echo "======================================================"
V5_EPOCH="00099"
CKPT_SRC="/netscratch/najib/multi-hmr/logs/anny_model/anny_s2_shape_v5/checkpoints/${V5_EPOCH}.pt"
CKPT_NAME="anny_s2_shape_v5_${V5_EPOCH}"

CKPT_DEST_DIR="/netscratch/najib/multi-hmr/models/multiHMR"
mkdir -p "$CKPT_DEST_DIR"

if [ ! -f "$CKPT_SRC" ]; then
  echo "ERROR: Checkpoint not found: $CKPT_SRC"; exit 1
fi
cp "$CKPT_SRC" "$CKPT_DEST_DIR/${CKPT_NAME}.pt"

echo "======================================================"
echo "5. Running Multi-HMR Demo..."
echo "======================================================" 
cd "$PROJECT_ROOT"
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

# Run on the base imageFiles directory (demo.py recursively finds images inside sequence folders)
python3.9 demo.py \
  --model_name "$CKPT_NAME" \
  --img_folder "$IMG_DIR/imageFiles" \
  --out_folder demo_out_3dpw_random_${CKPT_NAME} \
  --alpha 0.9 \
  --fov 60 \
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
  --rest_pose 0

echo "======================================================"
echo "RANDOMIZED 3DPW DEMO FINISHED SUCCESSFULLY!"
echo "======================================================"
EOF
)

srun -K \
  --gpus=1 \
  --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
  --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm,/ds-av:/ds-av${DRI_MOUNT} \
  --container-workdir="/netscratch/najib/multi-hmr" \
  bash -c "$COMMAND"