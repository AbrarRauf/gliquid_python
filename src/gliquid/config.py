from pathlib import Path

project_root = Path.cwd() # .parent
data_dir = Path(project_root / "data")

fusion_enthalpies_file = data_dir / "fusion_enthalpies.json"
fusion_temps_file = data_dir / "fusion_temperatures.json"
vaporization_temps_file = data_dir / "vaporization_temperatures.json"
# Abrar, feel free to add more paths here as needed
