#!/bin/bash
#SBATCH --job-name=multihmr_anny
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI,L40S,RTX3090,RTXA6000,V100-32GB,batch
#SBATCH --exclude=serv-3314
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --output=logs/train_anny_%j.out
#SBATCH --error=logs/train_anny_%j.err

# We wrap the entire training process in a COMMAND variable to pass into the container.
COMMAND=$(cat <<'EOF'
# 1. Initialize conda
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

# 2. Fix Cache Paths (Safely routed to netscratch)
export HF_HOME="/netscratch/najib/hf_cache"
export TORCH_HOME="/netscratch/najib/torch_cache"
export TORCH_EXTENSIONS_DIR="/netscratch/najib/torch_extensions"
mkdir -p $HF_HOME $TORCH_HOME $TORCH_EXTENSIONS_DIR

# 3. Navigate
cd /netscratch/najib/multi-hmr

# --- CRITICAL FIX FOR DEADLOCKS ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# ----------------------------------

# --- CRITICAL FIX FOR DINOv2 PYTHON 3.9 INCOMPATIBILITY ---
echo "Ensuring DINOv2 is downloaded to cache..."
# We trigger the download (it will fail on 3.9, but the files will be saved)
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)" || true
echo "Patching DINOv2 cache to remove Python 3.10+ type hints..."
# We patch the downloaded files to make them Python 3.9 compatible!
find /netscratch/najib/torch_cache/hub/facebookresearch_dinov2_main/dinov2 -name "*.py" -exec sed -i 's/ | None//g' {} +
# ----------------------------------------------------------

# 4. Run Training
python train.py \
    --train_data AnnyOne \
    --person_center head \
    --batch_size 4 \
    --backbone dinov2_vitl14 \
    --img_size 672 \
    --max_iter 200 \
    --n_iters_per_epoch 50 \
    --num_workers 0 \
    --learning_rate 5e-6 \
    --train_n 100 \
    --save_dir /netscratch/najib/multi-hmr/logs/anny_model \
    --name anny_small_run
EOF
)

# 5. Execute the command inside the DFKI Enroot/Pyxis Container
srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"