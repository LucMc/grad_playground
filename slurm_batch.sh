#!/bin/bash
#SBATCH --job-name=multi_algo_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --time=02-23:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --array=0   # adjust based on num_algos * num_seeds - 1

# --- Configuration ---
VENV_DIR=".venv"
# script_name="${1:-slippery_ant.py}"   # defaults to slippery_ant.py if not provided

output_filename="task_${SLURM_ARRAY_TASK_ID}_${algo}_seed_${seed}.out"

# WandB settings (fill in your values)
# wandb_entity="lucmc"
# wandb_project="crl_experiments"

# --- Setup environment (uv or venv) ---
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python 3.12
fi
source "$VENV_DIR/bin/activate"

# Optionally install deps if not synced
uv sync
uv pip install -e ".[cuda12]"
uv pip install wandb
uv pip install -e .
uv pip install warp-lang
uv pip uninstall jax
uv pip install jax["cuda12"]

python -c "import jax;print(jax.devices())"
python -c "import warp;print(warp.is_cuda_available())"
# --- Run experiment ---
echo "Running"
python "learning/train_jax_ppo.py" --use_wandb --impl=warp --num_timesteps=800000000 > "$output_filename" 2>&1

