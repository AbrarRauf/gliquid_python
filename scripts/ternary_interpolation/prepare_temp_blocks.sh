#!/bin/bash
set -euo pipefail

# Prepare temperature blocks for General HSX array runs.
#
# Usage:
#   ./prepare_temp_blocks.sh "Al,Hf,Nb,W"
#   ./prepare_temp_blocks.sh "Al,Hf,Nb,W" 100
#   ./prepare_temp_blocks.sh "Al,Hf,Nb,W" 100 data/phase_transitions.json all_dumps/hpc_runs/block_lists

SYSTEM_ELEMENTS=${1:-${SYSTEM_ELEMENTS:-}}
BLOCK_SIZE_K=${2:-${BLOCK_SIZE_K:-100}}
PHASE_TRANSITIONS_FILE=${3:-${PHASE_TRANSITIONS_FILE:-data/phase_transitions.json}}
OUT_DIR=${4:-${OUT_DIR:-all_dumps/hpc_runs/block_lists}}

if [[ -z "$SYSTEM_ELEMENTS" ]]; then
  echo "ERROR: Provide system elements as comma-separated list, e.g. Al,Hf,Nb,W"
  exit 1
fi

python3 - "$SYSTEM_ELEMENTS" "$BLOCK_SIZE_K" "$PHASE_TRANSITIONS_FILE" "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

system_raw, block_raw, pt_path_raw, out_dir_raw = sys.argv[1:5]

block_size = float(block_raw)
if block_size <= 0:
    raise ValueError(f"BLOCK_SIZE_K must be > 0, got {block_size}")

els = [e.strip() for e in system_raw.split(',') if e.strip()]
if len(els) < 3:
    raise ValueError(f"Need at least 3 elements, got: {system_raw}")

pt_path = Path(pt_path_raw)
if not pt_path.exists():
    raise FileNotFoundError(f"phase_transitions file not found: {pt_path}")

with open(pt_path, 'r', encoding='utf-8') as f:
    raw = json.load(f)

elements_map = raw.get('elements', raw)

melt_t = {}
for el in els:
    if el not in elements_map:
        raise KeyError(f"Element {el} not found in phase_transitions data")
    found = None
    for phase in elements_map[el].get('phases', []):
        if phase.get('phase_type') != 'liquid':
            continue
        t = phase.get('transition_temperature_K')
        if t is None:
            continue
        found = float(t)
        break
    if found is None:
        raise ValueError(f"No liquid transition_temperature_K found for element {el}")
    melt_t[el] = found

min_tm = min(melt_t.values())
max_tm = max(melt_t.values())

# Lower bound policy requested by user:
# max(min(Tm)-500 K, 273.15 K)
tmin = max(min_tm - 500.0, 273.15)
tmax = max_tm + 500.0

# Round outward to 0.01 K for stable text output.
tmin = round(tmin, 2)
tmax = round(tmax, 2)

blocks = []
cur = tmin
while cur < tmax - 1e-12:
    nxt = min(cur + block_size, tmax)
    blocks.append((round(cur, 2), round(nxt, 2)))
    cur = nxt

if not blocks:
    blocks = [(tmin, tmax)]

system_tag = "-".join(sorted(els))
out_dir = Path(out_dir_raw)
out_dir.mkdir(parents=True, exist_ok=True)
block_file = out_dir / f"temp_blocks_{system_tag}_{int(block_size)}K.txt"

with open(block_file, 'w', encoding='utf-8') as f:
    for a, b in blocks:
        f.write(f"{a:.2f},{b:.2f}\n")

n = len(blocks)
array_expr = f"0-{n-1}"

print("=" * 72)
print("GENHSX TEMPERATURE BLOCK PREPARATION")
print("=" * 72)
print(f"System: {els}")
print(f"Canonical: {system_tag}")
print(f"phase_transitions file: {pt_path}")
print(f"Melt temperatures (K): {melt_t}")
print(f"Recommended full range (K): ({tmin:.2f}, {tmax:.2f})")
print(f"Block size (K): {block_size:.2f}")
print(f"Number of blocks: {n}")
print(f"SBATCH array argument: --array={array_expr}")
print(f"Block list file: {block_file}")
print("\nFirst 5 blocks:")
for i, (a, b) in enumerate(blocks[:5]):
    print(f"  {i}: {a:.2f},{b:.2f}")

print("\nExample line selection in sbatch:")
print('  TEMP_BLOCK_K=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "' + str(block_file) + '")')
print("=" * 72)
PY
