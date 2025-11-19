#!/bin/bash

#================================================================#
# S B A T C H     D I R E C T I V E S
#================================================================#

# Set a name for your job
#SBATCH --job-name=attention-play-gpu

# --- 💡 EDIT THIS LINE 💡 ---
# Specify the account to charge
#SBATCH --account=stats_dept1

# Specify the partition (queue)
#SBATCH --partition=gpu

# Request resources
#SBATCH --nodes=1                   # 1 compute node
#SBATCH --ntasks-per-node=1         # 1 task
#SBATCH --cpus-per-task=8           # 8 CPU cores
#SBATCH --mem=32G                   # 32 Gigabytes of memory

# --- 👇 THIS IS THE KEY FIX 👇 ---
# Request 1 day and 12 hours of wall-clock time
#SBATCH --time=1-12:00:00

# Request 1 GPU
#SBATCH --gres=gpu:1

# Set up email notifications
#SBATCH --mail-user=sharkare@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#================================================================#
# J O B     C O M M A N D S
#================================================================#

cd $SLURM_SUBMIT_DIR

echo "Job started on $(hostname)"
echo "Running in directory: $(pwd)"

# 1. Load the modules
module purge  # Clear any old modules
module load python/3.13.2  # <-- Use the correct version you found
module load cuda/12.8.1  # <-- Load CUDA for the GPU

# 2. Set up the virtual environment
if [ ! -d "venv" ]; then
  echo "Creating new virtual environment..."
  python -m venv venv
fi
source venv/bin/activate

# 3. Install packages
echo "Installing packages..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run your Python script
echo "Starting Python script..."
python main_2.py

# 5. Deactivate
deactivate

echo "Job finished."
