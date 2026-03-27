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
RUN_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

# Auto-detect the temp block list directly in submit directory.
# Expect exactly one file matching temp_blocks_*.txt.
shopt -s nullglob
block_files=("$RUN_ROOT"/temp_blocks_*.txt)
shopt -u nullglob

if [[ ${#block_files[@]} -eq 0 ]]; then
  echo "ERROR: No temp_blocks_*.txt found in submit directory: $RUN_ROOT"
  exit 1
fi

if [[ ${#block_files[@]} -gt 1 ]]; then
  echo "ERROR: Multiple temp_blocks_*.txt files found in submit directory: $RUN_ROOT"
  printf '  %s\n' "${block_files[@]}"
  echo "Keep only one temp_blocks_*.txt file in submit directory before submitting."
  exit 1
fi

BLOCK_LIST_FILE="${block_files[0]}"

# SYSTEM_ELEMENTS will be read from the block-list header line.

# Cluster defaults requested for these runs.
export USE_MP_CACHE="false"
export VERTICAL_SIMPLICES="false"
export INCLUDE_REF_SS="true"
export INCLUDE_POLYMORPHS="false"

# Force all generated dumps/caches into the submission directory (scratch).
export OUTPUT_ROOT="$RUN_ROOT/all_dumps/hpc_runs"
mkdir -p "$OUTPUT_ROOT"

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

SYSTEM_HEADER=$(sed -n "1p" "$BLOCK_LIST_FILE")
if [[ "$SYSTEM_HEADER" == \#SYSTEM_ELEMENTS=* ]]; then
  export SYSTEM_ELEMENTS="${SYSTEM_HEADER#\#SYSTEM_ELEMENTS=}"
else
  echo "ERROR: Block list is missing '#SYSTEM_ELEMENTS=...' header on line 1"
  exit 1
fi

if [[ -z "${SYSTEM_ELEMENTS:-}" ]]; then
  echo "ERROR: Parsed SYSTEM_ELEMENTS is empty from block list header"
  exit 1
fi

TEMP_BLOCK_K=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$BLOCK_LIST_FILE")
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
echo "Run root: $RUN_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "============================================================"

cd "$REPO_ROOT" || { echo "Could not enter $REPO_ROOT"; exit 1; }
python "$EXEC_SCRIPT"

echo "Finished task $SLURM_ARRAY_TASK_ID for TEMP_BLOCK_K=$TEMP_BLOCK_K"
