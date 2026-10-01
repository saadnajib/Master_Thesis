#!/bin/bash
#SBATCH --job-name=test_anny_loader
#SBATCH --partition=A100-40GB,A100-80GB,L40S-AV,A100-RP,A100-PCI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_loader_%j.out
#SBATCH --error=logs/test_loader_%j.err

# We wrap the entire process in a COMMAND variable to pass into the container.
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

# 4. Run the Dataloader Test!
python test_loader.py

EOF
)

# 5. Execute the command inside the DFKI Enroot/Pyxis Container
srun \
   --container-image=/netscratch/najib/ContainerImages/ubuntu+20.04_v4.sqsh \
   --container-mounts=/fscratch:/fscratch,/netscratch:/netscratch,/home/najib:/home/najib,/dev/shm:/dev/shm \
   bash -c "$COMMAND"