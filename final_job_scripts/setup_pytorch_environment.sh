#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=setup_pytorch_env_v2
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=final_job_outputs/setup_pytorch_env_%A.out

# --- Setup: Activate environment and modules ---
module load 2023
module load Miniconda3/23.5.2-0
source /home/tchakravorty/.bashrc
conda activate code-switch

# --- Force reinstall PyTorch, Torchvision, and Torchaudio ---
echo "--- Uninstalling any existing PyTorch, Torchvision, Torchaudio ---"
pip uninstall -y torch torchvision torchaudio

echo "--- Force reinstalling PyTorch, Torchvision, Torchaudio for CUDA 11.8 ---"
# Using --force-reinstall to ensure all packages are updated correctly from the same source.
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "--- Installation complete. Verifying all packages ---"

# --- Verification Step ---
python -c "import torch; import torchvision; import torchaudio; print(f'PyTorch version: {torch.__version__}'); print(f'Torchvision version: {torchvision.__version__}'); print(f'Torchaudio version: {torchaudio.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo "--- Environment setup is complete. You can now run your analysis job. ---" 