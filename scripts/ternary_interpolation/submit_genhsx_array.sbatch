#!/bin/bash
#SBATCH --job-name=genhsx_block
#SBATCH --mail-user=arauf@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=genhsx_out_%a_%A.log
#SBATCH --error=genhsx_err_%a_%A.log
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=whsun0
#SBATCH --mem=14G
#SBATCH --partition=standard

set -euo pipefail

module load python3.10-anaconda/2023.03

source $(conda info --base)/etc/profile.d/conda.sh
conda activate gliquid-env

# Update these paths before submission.
REPO_ROOT=/home/arauf/gliquid_python
EXEC_SCRIPT="$REPO_ROOT/scripts/ternary_interpolation/exec_genhsx.py"
BLOCK_LIST_FILE="$REPO_ROOT/all_dumps/hpc_runs/block_lists/temp_blocks_Al-Hf-Nb-W_100K.txt"

# Provide target system per job submission.
export SYSTEM_ELEMENTS="Al,Hf,Nb,W"

# Cluster defaults requested for these runs.
export USE_MP_CACHE="false"
export VERTICAL_SIMPLICES="false"
export INCLUDE_REF_SS="true"
export INCLUDE_POLYMORPHS="false"

# Optional tuning overrides.
# export GRID_DELTA="0.025"
# export TEMP_DELTA_K="5"
# export EQ_PROGRESS_EVERY="1"
# export PARAM_XLSX_PATH="$REPO_ROOT/data/high_component/ssol_fits_linear_model_legacy_refs-tau_penalty.xlsx"
# export REF_SS_OMEGAS_PATH="$REPO_ROOT/data/high_component/omegas.json"
# export OUTPUT_ROOT="$REPO_ROOT/all_dumps/hpc_runs"

if [[ ! -f "$BLOCK_LIST_FILE" ]]; then
  echo "ERROR: Block list file not found: $BLOCK_LIST_FILE"
  exit 1
fi

TEMP_BLOCK_K=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$BLOCK_LIST_FILE")
if [[ -z "${TEMP_BLOCK_K:-}" ]]; then
  echo "ERROR: No temperature block found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 1
fi
export TEMP_BLOCK_K

echo "============================================================"
echo "General HSX array run"
echo "Host: $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "System elements: $SYSTEM_ELEMENTS"
echo "Temperature block (K): $TEMP_BLOCK_K"
echo "Block list: $BLOCK_LIST_FILE"
echo "Exec script: $EXEC_SCRIPT"
echo "============================================================"

cd "$REPO_ROOT" || { echo "Could not enter $REPO_ROOT"; exit 1; }
python "$EXEC_SCRIPT"

echo "Finished task $SLURM_ARRAY_TASK_ID for TEMP_BLOCK_K=$TEMP_BLOCK_K"
