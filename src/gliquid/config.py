from pathlib import Path
_DIR_STRUCT_OPTS = ['flat', 'nested']

project_root = Path.cwd()
# project_root = Path.cwd().parent
data_dir = Path(project_root / "data")
# data_dir = Path(project_root / "matrix_data")
dir_structure = _DIR_STRUCT_OPTS[0] # Change on push

fusion_enthalpies_file = data_dir / "fusion_enthalpies.json"
fusion_temps_file = data_dir / "fusion_temperatures.json"
vaporization_temps_file = data_dir / "vaporization_temperatures.json"
