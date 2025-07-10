"""
Author: Joshua Willwerth
Last Modified: June 30, 2025
Description: This script provides functions to interface with the Materials Project (MP) APIs and locally cache DFT
calculated phase data. Publicly-availble data from the Materials Platform for Data Science (MPDS) may be downloaded and
processed using this script in order to autopopulate BinaryLiquid objects using the `from_cache` method.
GitHub: https://github.com/willwerj 
ORCID: https://orcid.org/0009-0004-6334-9426
"""
from __future__ import annotations

import os
import json
import numpy as np

from emmet.core.thermo import ThermoType
from mp_api.client import MPRester as MPRester
from pymatgen.core import Composition, Element, Structure
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram, CompoundPhaseDiagram
from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme
from sympy import comp
from gliquid.config import data_dir, fusion_enthalpies_file, fusion_temps_file, vaporization_temps_file

melt_enthalpies = json.load(open(fusion_enthalpies_file)) if os.path.exists(fusion_enthalpies_file) else {}
melt_temps = json.load(open(fusion_temps_file)) if os.path.exists(fusion_temps_file) else {}
boiling_temps = json.load(open(vaporization_temps_file)) if os.path.exists(vaporization_temps_file) else {}

missing_files = []
if not melt_enthalpies:
    missing_files.append("fusion_enthalpies.json")
if not melt_temps:
    missing_files.append("fusion_temperatures.json")
if not boiling_temps:
    missing_files.append("vaporization_temperatures.json")
if missing_files:
    # Get the last two directories in the data_dir path
    data_dir_parts = os.path.normpath(data_dir).split(os.sep)
    last_two_dirs = os.sep.join(data_dir_parts[-2:]) if len(data_dir_parts) >= 2 else data_dir
    raise FileNotFoundError(
        f"The following data files were not loaded correctly: {', '.join(missing_files)}. "
        f"Please ensure the files exist in the data directory '...{os.sep}{last_two_dirs}'."
    )

def validate_and_format_binary_system(input) -> tuple[list[str], str, bool]:
    """
    Args:
        input (str or list): System specification (e.g., 'A-B' or ['A', 'B']).
    
    Returns:
        tuple[list[str], str, bool]: Alphabetized list of components and hyphenated string continaing system information
        """
    if isinstance(input, str) and input.count('-') == 1:
        components = input.split('-')
        components_sorted = sorted(components)
    elif isinstance(input, list) and all(isinstance(c, str) for c in input) and len(input) == 2:
        components = input
        components_sorted = sorted(input)
    else:
        raise ValueError("Input must be a hyphenated string or list of two components.")

    # Validate components as valid composition objects
    [Composition(c) for c in components]
    return components, '-'.join(components), components_sorted != components


def shape_to_list(svgpath: str) -> list[list]:
    """Parse SVG path to a list of [X, T] pairs"""
    pairs = [s for s in svgpath.split(' ') if s not in ['L', 'M']]
    return [[float(p.split(',')[0]) / 100, float(p.split(',')[1]) + 273.15] for p in pairs]


def extract_digitized_liquidus(mpds_json: dict) -> tuple[list[list] | None, bool]:
    """Extracts digitized liquidus data from MPDS JSON.
    
    Args:
        mpds_json (dict): MPDS digitized phase equilibrium data for the system.

    Returns:
        list[list]: Digitized liquidus curve that is properly formatted for fitting purposes.
    """
    is_partial = False
    if mpds_json.get('reference') is None:
        print("No data in MPDS JSON.")
        return None, False

    data = next((b['svgpath'] for b in mpds_json['shapes'] if b.get('label') == 'L'), "")
    if not data:
        print("No liquidus data found.")
        return None, False

    liquidus = shape_to_list(data)

    def t_at_boundary(t, boundary):
        return t <= boundary[0] + 4 or t >= boundary[1] - 4

    liquidus = [pt for pt in liquidus if not t_at_boundary(pt[1] - 273.15, mpds_json['temp'])]

    if len(liquidus) < 3:
        print("Insufficient liquidus data.")
        return None, True

    def section_liquidus(points):
        """Splits liquidus into continuous sections."""
        sections, current_section, direction = [], [], None
        for i in range(len(points) - 1):
            x1, x2 = points[i][0], points[i + 1][0]
            new_direction = "increasing" if x2 > x1 else "decreasing" if x2 < x1 else None
            current_section.append(points[i])

            if new_direction != direction:
                if current_section:
                    sections.append(current_section)
                current_section = []
            direction = new_direction

        current_section.append(points[-1])
        sections.append(current_section)
        return sections

    sections = section_liquidus(liquidus)
    sections.sort(key=len, reverse=True)
    main_section = sorted(sections.pop(0))

    lhs = [0, melt_temps.get(mpds_json['chemical_elements'][0], 0)]
    rhs = [1, melt_temps.get(mpds_json['chemical_elements'][1], 0)]

    def within_tol_from_line(p1, p2, p3, tol):
        """Checks if a point is within tolerance from a line."""
        try:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            y_h = m * (p3[0] - p1[0]) + p1[1]
            return abs(p3[1] - y_h) <= tol
        except ZeroDivisionError:
            return abs(p2[1] - p1[1]) <= tol

    for sec in sections:
        sec.sort()
        if sec[-1][0] <= main_section[0][0] and within_tol_from_line(main_section[0], lhs, sec[-1], 250):
            main_section = sec + main_section
        elif sec[0][0] >= main_section[-1][0] and within_tol_from_line(main_section[-1], rhs, sec[0], 250):
            main_section += sec
        elif len(sec) == 2:
            if sec[0][0] < main_section[0][0] and within_tol_from_line(main_section[0], lhs, sec[0], 170):
                main_section = sec + main_section

            elif sec[-1][0] > main_section[-1][0] and within_tol_from_line(main_section[-1], rhs, sec[-1], 170):
                main_section += sec

    # If the liquidus does not have endpoints near the ends of the composition range, melting temps won't be good
    if main_section[0][0] > 0.03 or main_section[-1][0] < 0.97:
        print(f"MPDS liquidus does not span the entire composition range! "
              f"({100 * main_section[0][0]}-{100 * main_section[-1][0]})")
        is_partial = True

    mpds_liquidus = sorted(main_section)

    def fill_liquidus(p1, p2, max_interval):
        """Fills in points between two liquidus points (p1 and p2) based on a maximum interval."""
        num_points = int(np.ceil((p2[0] - p1[0]) / max_interval)) + 1  # Include endpoints
        filled_X = np.linspace(p1[0], p2[0], num_points)
        filled_T = np.linspace(p1[1], p2[1], num_points)
        return [[x, t] for x, t in zip(filled_X, filled_T)][1:-1]

    # Fill in ranges with missing points
    for i in reversed(range(len(mpds_liquidus) - 1)):
        if mpds_liquidus[i + 1][0] - mpds_liquidus[i][0] > 0.06:
            filler = fill_liquidus(mpds_liquidus[i], mpds_liquidus[i + 1], 0.03)
            for point in reversed(filler):
                mpds_liquidus.insert(i + 1, point)

    # Filter out duplicate values in the liquidus curve; greatly improves runtime efficiency
    for i in reversed(range(len(mpds_liquidus) - 1)):
        if mpds_liquidus[i][0] == 0 or mpds_liquidus[i][1] == 0:
            continue
        if abs(1 - mpds_liquidus[i + 1][0] / mpds_liquidus[i][0]) < 0.0005 and \
                abs(1 - mpds_liquidus[i + 1][1] / mpds_liquidus[i][1]) < 0.0005:
            del (mpds_liquidus[i + 1])

    return mpds_liquidus, is_partial


def load_mpds_data(input, verbose=True) -> tuple[dict, dict, list[list] | None]:
    """Retrieves MPDS data for a binary system.
    
    Args:
        input (str or list): System specification (e.g., 'A-B' or ['A', 'B']).
        verbose (bool): If True, outputs additional debugging information.

    Returns:
        tuple[dict, dict, list[list]]: A tuple containing the system MPDS JSON, component thermodynamic data, and
        digitized liquidus curve that is properly formatted for fitting purposes. Note that the MPDS json in the
        specified cache directory must follow the alphabetized, hyphenated naming convention (e.g. 'A-B.json')
    """
    components, system, _ = validate_and_format_binary_system(input) # TODO: determine if data should be flipped
    component_data = {
        comp: [melt_enthalpies.get(comp, 0), melt_temps.get(comp, 0), boiling_temps.get(comp, 0)]
        for comp in components
    }

    if verbose:
        for comp, data in component_data.items():
            print(f"{comp}: H_fusion = {data[0]} J/mol, T_fusion = {data[1]} K, T_vaporization = {data[2]} K")

    sys_file = os.path.join(data_dir, f"{system}.json")
    if os.path.exists(sys_file):
        with open(sys_file, 'r') as f:
            mpds_json = json.load(f)
        if verbose:
            print("\nReading MPDS json from entry at " + mpds_json['reference']['entry'] + "...\n")
    else:
        if verbose:
            print("\nNo cached phase data found; proceeding without it.")
        mpds_json = {"reference": None}

    return mpds_json, component_data, extract_digitized_liquidus(mpds_json)


def identify_mpds_phases(mpds_json: dict, verbose=False) -> list[dict]:
    """Identifies MPDS phases from JSON data.
    
    Args:
        mpds_json (dict): MPDS digitized phase equilibrium data for the system.
        verbose (bool): If True, outputs additional debugging information.

    Returns:
        list: A list of dictionaries containing information on equilibrium phase composition and temperature boundaries
    """
    if mpds_json.get('reference') is None:
        if verbose:
            print("System JSON does not contain any data!\n")
        return []

    phases = []
    for shape in mpds_json.get('shapes', []):
        if shape.get('nphases') == 1 and shape.get('is_solid') and 'label' in shape:
            data = shape_to_list(shape.get('svgpath', ""))
            if not data:
                if verbose:
                    print(f"No point data found for phase {shape['label']} in JSON!")
                continue

            data.sort(key=lambda x: x[1])  # Sort by temperature
            tbounds = [data[0], data[-1]]

            if shape.get('kind') == 'phase':
                data.sort(key=lambda x: x[0])  # Sort by composition
                cbounds = [data[0], data[-1]]
                if cbounds[-1][0] < 0.03 or cbounds[0][0] > 0.97:
                    continue
                phases.append({'type': 'ss', 'name': shape['label'].split()[0], 'comp': tbounds[1][0],
                               'cbounds': cbounds, 'tbounds': tbounds})
            else:  # Line compound
                phases.append({'type': 'lc', 'name': shape['label'].split()[0], 'comp': tbounds[1][0],
                               'tbounds': tbounds})

    if not phases and verbose:
        print("No phase data found in JSON!")

    return sorted(phases, key=lambda x: x['comp'])


def get_low_temp_phase_data(
        mpds_json: dict, dft_ch: PhaseDiagram) -> tuple[tuple[dict, dict, int | float], tuple[dict, dict, int | float]]:
    """Extracts low-temperature phase data from MPDS and MP convex hull.
    
    Args:
        mpds_json (dict): MPDS digitized phase equilibrium data for the system.
        dft_ch (PhaseDiagram): DFT convex hull data formatted with pymatgen.

    Returns:
        Tuples with low temperature phase data for both digitzed and computed phases. The returned data is in the
        following format: (mpds congruently melting phases, mpds incongruently melting phases, max phase decomp temp),
        (dft phase formation energies, dft phase energies below convex hull, minimum phase formation energy)
        """
    mpds_congruent_phases, mpds_incongruent_phases = {}, {}
    max_phase_temp = 0

    identified_phases = identify_mpds_phases(mpds_json)
    mpds_liquidus, is_partial = extract_digitized_liquidus(mpds_json)

    def phase_decomp_on_liq(phase, liq):
        """Determines if a solid phase decomposes on or near the liquidus."""
        if liq is None:
            return False
        for i in range(len(liq) - 1):
            if liq[i][0] == phase['tbounds'][1][0]:
                return abs(liq[i][1] - phase['tbounds'][1][1]) < 10
            # composition falls between two points:
            elif liq[i][0] < phase['tbounds'][1][0] < liq[i + 1][0]:
                return abs((liq[i][1] + liq[i + 1][1]) / 2 - phase['tbounds'][1][1]) < 10

    temp_range = mpds_json['temp'][1] - mpds_json['temp'][0]
    temp_threshold = (mpds_json['temp'][0] + 273.15) + temp_range * 0.10

    for phase in identified_phases:
        if phase['type'] in {'lc', 'ss'} and phase['tbounds'][0][1] < temp_threshold:
            if phase_decomp_on_liq(phase, mpds_liquidus):
                if phase['type'] == 'ss':
                    mpds_congruent_phases[phase['name']] = (
                        (phase['cbounds'][0][0], phase['cbounds'][1][0]), phase['tbounds'][1][1])
                else:
                    mpds_congruent_phases[phase['name']] = ((phase['comp'], phase['comp']), phase['tbounds'][1][1])
            else:
                if phase['type'] == 'ss':
                    mpds_incongruent_phases[phase['name']] = (
                        (phase['cbounds'][0][0], phase['cbounds'][1][0]), phase['tbounds'][1][1])
                else:
                    mpds_incongruent_phases[phase['name']] = (
                        (phase['comp'], phase['comp']), phase['tbounds'][1][1])
            max_phase_temp = max(phase['tbounds'][1][1], max_phase_temp)

    if max_phase_temp == 0 and mpds_liquidus:
        max_phase_temp = min(mpds_liquidus, key=lambda x: x[1])[1]

    dft_phases, dft_phases_ebelow = {}, {}
    min_form_e = 0

    for entry in dft_ch.stable_entries:
        comp_dict = entry.composition.fractional_composition.as_dict()
        if len(comp_dict) == 1:
            continue

        comp = comp_dict.get(dft_ch.elements[1].symbol, 0)
        form_e = dft_ch.get_form_energy_per_atom(entry)
        dft_phases[entry.name] = ((comp, comp), form_e)
        min_form_e = min(form_e, min_form_e)

        ch_copy = PhaseDiagram([e for e in dft_ch.stable_entries if e != entry])
        e_below_hull = -abs(dft_ch.get_hull_energy_per_atom(entry.composition) -
                            ch_copy.get_hull_energy_per_atom(entry.composition))
        dft_phases_ebelow[entry.name] = ((comp, comp), e_below_hull)

    return (mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp), (dft_phases, dft_phases_ebelow, min_form_e)


def _get_dft_entries_from_components(components: list[str], dft_type: str) -> list[dict]:
    """Fetches DFT entries for the specified components and DFT functional type."""
    entries = []

    def fetch_entries(api_key, client_class, thermo_type=None):
        """Helper function to fetch entries from API."""
        if not api_key:
            raise ValueError(f"Environment variable for {client_class.__name__} must contain a valid API key!")
        with client_class(api_key) as MPR:
            criteria = {'thermo_types': [thermo_type]} if thermo_type else {}
            return MPR.get_entries_in_chemsys(components, additional_criteria=criteria)

    new_mp_api_key = os.getenv('NEW_MP_API_KEY')
    scan_entries, ggau_entries = [], []

    if dft_type in {'R2SCAN', 'GGA/GGA+U/R2SCAN'}:
        scan_entries = fetch_entries(new_mp_api_key, MPRester, ThermoType.R2SCAN)
    if dft_type in {'GGA/GGA+U', 'GGA/GGA+U/R2SCAN'}:
        ggau_entries = fetch_entries(new_mp_api_key, MPRester, ThermoType.GGA_GGA_U)

    if dft_type == 'GGA/GGA+U/R2SCAN':
        entries = MaterialsProjectDFTMixingScheme().process_entries(scan_entries + ggau_entries, verbose=False)
    elif dft_type == 'GGA/GGA+U':
        entries = ggau_entries
    elif dft_type == 'R2SCAN':
        entries = scan_entries

    computed_entry_dicts = [e.as_dict() for e in entries]

    # Filter out Mg149 phase and remove run data to reduce cache size
    computed_entry_dicts = [e for e in computed_entry_dicts if e['composition'].get('Mg', 0) != 149]
    for e in computed_entry_dicts:
        e.pop('data', None)

    return computed_entry_dicts


def get_dft_convexhull(input, dft_type='GGA/GGA+U',
                       inc_structure_data=False, verbose=False) -> tuple[PhaseDiagram, dict]:
    """
    Returns the DFT convex hull of a given system with specified functionals.

    Args:
        input (str or list): System specification (e.g., 'A-B' or ['A', 'B']).
        dft_type (str): Functional type, e.g., 'GGA', 'GGA/GGA+U', etc.
        inc_structure_data (bool): Whether to include structural data.
        verbose (bool): Whether to print detailed output.

    Returns:
        A tuple of the phase diagram and a dictionary of stable entry atomic volumes.
    """
    components, sys_name, _ = validate_and_format_binary_system(input) # TODO: determine if data should be flipped

    supported_dft_types = ["GGA/GGA+U", "R2SCAN", "GGA/GGA+U/R2SCAN"]
    if dft_type not in supported_dft_types:
        raise SyntaxError(
            f"dft_type '{dft_type}' is not currently supported! "
            f"Please specify as one of the following: {', '.join(supported_dft_types)}"
        )

    if 'Yb' in components and dft_type == 'GGA/GGA+U':
        dft_type = 'GGA'
    if verbose:
        print(f"Using DFT entries solved with {dft_type} functionals.")

    dft_entries_file = os.path.join(
        data_dir, f"{sys_name}_ENTRIES_MP_{'_'.join(dft_type.split('/'))}.json"
    )
    use_compound_pd = any(len(Composition(c).elements) > 1 for c in components)

    if os.path.exists(dft_entries_file):
        with open(dft_entries_file, "r") as f:
            computed_entry_dicts = json.load(f)
        if verbose:
            print("Loading cached DFT entry data.")
    else:
        computed_entry_dicts = _get_dft_entries_from_components(components, dft_type)
        if verbose:
            print(f"Caching DFT entry data as {dft_entries_file}...")
        with open(dft_entries_file, "w") as f:
            json.dump(computed_entry_dicts, f)

    if use_compound_pd:
        pd = CompoundPhaseDiagram(
            terminal_compositions=[Composition(c) for c in components],
            entries=[ComputedEntry.from_dict(e) for e in computed_entry_dicts],
        )
    else:
        pd = PhaseDiagram(
            elements=[Element(c) for c in components],
            entries=[ComputedEntry.from_dict(e) for e in computed_entry_dicts],
        )
    if verbose:
        print(f"{len(pd.stable_entries) - 2} stable line compound(s) on the DFT convex hull.")

    stable_entry_atomic_volumes = {}

    if inc_structure_data:
        for entry in pd.stable_entries:
            entries_matching_composition = [
                e
                for e in computed_entry_dicts
                if Composition.from_dict(e["composition"]) == entry.composition
            ]
            hull_stable_entry = min(entries_matching_composition, key=lambda x: x["energy"])
            hull_stable_structure = Structure.from_dict(hull_stable_entry["structure"])
            ucell_volume = hull_stable_structure.volume  # Volume in cubic angstroms
            ucell_n_atoms = hull_stable_structure.num_sites  # Number of atoms per structure
            atomic_volume = ucell_volume / ucell_n_atoms  # Atomic volume in cubic angstroms per atom
            stable_entry_atomic_volumes[entry.composition.reduced_formula] = atomic_volume

    return pd, stable_entry_atomic_volumes
