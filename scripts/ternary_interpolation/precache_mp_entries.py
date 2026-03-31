"""Pre-cache Materials Project chemsys entries for offline HPC General HSX runs.

This script fetches MP entries once on a machine with API access and writes
cache files matching GeneralInterpolation._get_mp_entries naming:

  <cache_dir>/<System>_entries.json

where <System> is alphabetically sorted, e.g. Hf-Nb-W-Zr.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

from emmet.core.utils import jsanitize
from mp_api.client import MPRester

try:
	from auth import mpapi_key as DEFAULT_MP_API_KEY
except Exception:
	DEFAULT_MP_API_KEY = None


# ==========================
# User-editable configuration
# ==========================
# Add one or more systems here. Each system is a list of element symbols.
SYSTEMS: List[List[str]] = [
	["Al", "Hf", "Nb", "W", "Zr"],
]

# Directory where <System>_entries.json files will be written.
CACHE_DIR = "data/mp_cache"

# Optional API key override. Keep None to use auth.mpapi_key.
MP_API_KEY_OVERRIDE = None

# If False, existing cache files are skipped.
FORCE_OVERWRITE = False


def _normalize_elements(elements: Sequence[str]) -> List[str]:
	elems = [str(t).strip() for t in elements if str(t).strip()]
	if len(elems) < 3:
		raise ValueError(f"Expected at least 3 elements, got: {elements}")
	return sorted(elems)


def _cache_one_system(cache_dir: Path, elements: Sequence[str], api_key: str, force: bool) -> None:
	elements_norm = _normalize_elements(elements)
	system_name = "-".join(elements_norm)
	cache_path = cache_dir / f"{system_name}_entries.json"

	if cache_path.exists() and not force:
		print(f"[SKIP] Cache exists: {cache_path}")
		return

	print(f"[FETCH] {system_name} -> {cache_path}")
	with MPRester(api_key) as mpr:
		entries = mpr.get_entries_in_chemsys(elements_norm)

	serialized = jsanitize(entries)
	with open(cache_path, "w", encoding="utf-8") as f:
		json.dump(serialized, f)

	print(f"[OK] {system_name}: fetched {len(serialized)} entries")


def main() -> None:
	cache_dir = Path(CACHE_DIR).resolve()
	cache_dir.mkdir(parents=True, exist_ok=True)

	api_key = MP_API_KEY_OVERRIDE or DEFAULT_MP_API_KEY
	if not api_key:
		raise ValueError(
			"No MP API key available. Set MP_API_KEY_OVERRIDE or ensure auth.mpapi_key is configured."
		)
	if not SYSTEMS:
		raise ValueError("SYSTEMS is empty. Add at least one element list to precache.")

	print("=" * 72)
	print("PRECACHE MATERIALS PROJECT ENTRIES")
	print("=" * 72)
	print(f"Cache dir: {cache_dir}")
	print(f"Systems to cache: {len(SYSTEMS)}")

	for sys_elems in SYSTEMS:
		_cache_one_system(
			cache_dir=cache_dir,
			elements=sys_elems,
			api_key=api_key,
			force=bool(FORCE_OVERWRITE),
		)

	print("=" * 72)


if __name__ == "__main__":
	main()
