#!/bin/bash
#SBATCH --job-name=check_anny_shape
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/check_shape_%j.out
#SBATCH --error=logs/check_shape_%j.err

# We wrap the entire process in a COMMAND variable to pass into the container.
COMMAND=$(cat <<'EOF'
# 1. Initialize conda
source /netscratch/najib/miniconda3/etc/profile.d/conda.sh
conda activate multihmr

# 2. Fix Cache Paths (Safely routed to netscratch, away from the ~ quota)
export HF_HOME="/netscratch/najib/hf_cache"
export TORCH_HOME="/netscratch/najib/torch_cache"
export TORCH_EXTENSIONS_DIR="/netscratch/najib/torch_extensions"
export XDG_CACHE_HOME="/netscratch/najib/.cache"
mkdir -p $HF_HOME $TORCH_HOME $TORCH_EXTENSIONS_DIR $XDG_CACHE_HOME

# Route anny's hardcoded ~/.cache/anny onto netscratch (avoids disk quota error)
mkdir -p /netscratch/najib/anny_cache
mkdir -p "$HOME/.cache"
if [ ! -e "$HOME/.cache/anny" ]; then
    ln -sfn /netscratch/najib/anny_cache "$HOME/.cache/anny"
fi

# 3. Navigate
cd /netscratch/najib/multi-hmr

# --- CRITICAL FIX FOR DEADLOCKS ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# ----------------------------------

# Where the report is written (override by exporting SHAPE_LOG before sbatch)
export SHAPE_LOG="${SHAPE_LOG:-/netscratch/najib/multi-hmr/shape_report.txt}"

# 4. Run the shape inspection (writes only to $SHAPE_LOG, silent otherwise)
python check_shape.py

echo "check_shape.py finished. Report at: $SHAPE_LOG"
EOF
)

# 5. Execute the command inside the DFKI Enroot/Pyxis Container
srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"