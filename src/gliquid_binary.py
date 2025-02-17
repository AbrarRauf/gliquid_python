"""
Authors: Joshua Willwerth, Shibo Tan, Abrar Rauf
Last Modified: January 31, 2025
Description: This script is designed for the thermodynamic modeling of two-component systems.
It provides tools for fitting the non-ideal mixing parameters of the liquid phase from T=0K DFT-calculated phases and
digitized equilibrium phase boundary data. The data stored and produced may be visualized using the BLPlotter class
GitHub: https://github.com/willwerj 
ORCID: https://orcid.org/0009-0004-6334-9426
"""
from __future__ import annotations

import math
import time
import numpy as np
import pandas as pd
import sympy as sp

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
from itertools import combinations
from io import StringIO
from pymatgen.core import Composition
from phase_diagram import PDPlotter, PhaseDiagram, PDEntry  # Note that the PMG PDPlotter source code has been modified

import load_binary_data as lbd
from hsx import HSX

_x_step = 0.01  # Sets composition grid precision; has not been tested for values other than 0.01
_x_vals = np.arange(0, 1 + _x_step, _x_step)
_x_logs = np.log(_x_vals[1:-1])


def validate_binary_mixing_parameters(input) -> list[int | float]:
    """
    Args:
        input (list[int | float]): A list containing numerical values representing non-ideal mixing parameters.

    Returns:
        list[int | float]: A validated list of four numerical values representing non-ideal mixing parameters.
    """
    if isinstance(input, (list, tuple)):
        if len(input) == 0:
            return [0, 0, 0, 0] 
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in input) and len(input) == 4:
            return [i for i in input]  # Creates a copy of the input parameter list        
    raise ValueError("Parameters must be input as a list or tuple in the following format: [L0_a, L0_b, L1_a, L1_b]")


class BinaryLiquid:
    """
    Represents a binary liquid system for thermodynamic modeling and phase diagram generation.

    Attributes:
        sys_name (str): Binary system name.
        components (list): List of component names.
        component_data (dict): Thermodynamic data for components.
        mpds_json (dict): MPDS phase equilibrium data for the system.
        digitized_liq (list): Digitized liquidus data points.
        invariants (list): Identified invariant points.
        temp_range (list): Temperature range for calculations.
        comp_range (list): Composition range for calculations.
        ignored_comp_ranges (list): Ignored composition ranges.
        dft_type (str): Functional used for DFT calculations.
        dft_ch (PhaseDiagram): DFT convex hull data formatted with pymatgen.
        phases (list): List of phase data.
        params (list): Liquid non-ideal mixing parameters [L0_a, L0_b, L1_a, L1_b].
        guess_symbols (list): Sympy symbols for corresponding to guessed parameters.
        constraints (list): Sympy equations used to store parameter constraints.
        nmpath (np.ndarray): Nelder-Mead optimization path, stored after running Nelder-Mead for plotting purposes.
        hsx (HSX): HSX object for phase diagram calculations.
    """

    def __init__(self, sys_name, components, init_error=False, **kwargs):
        self.init_error = init_error
        self.sys_name = sys_name
        self.components = components
        self.component_data = kwargs.get('component_data', {})
        self.mpds_json = kwargs.get('mpds_json', {})
        self.digitized_liq = kwargs.get('digitized_liq', [])
        self.invariants = kwargs.get('invariants', [])
        self.temp_range = kwargs.get('temp_range', [])
        self.comp_range = kwargs.get('comp_range', [0, 100])
        self.ignored_comp_ranges = kwargs.get('ignored_comp_ranges', [])
        self.dft_type = kwargs.get('dft_type', "")
        self.dft_ch = kwargs.get('dft_ch', None)
        self.phases = kwargs.get('phases', [])
        self._params = kwargs.get('params', [0, 0, 0, 0])
        self.guess_symbols = None
        self.constraints = None
        self.nmpath = None
        self.hsx = None

    @classmethod
    def from_cache(cls, input, dft_type="GGA/GGA+U", params=[], reconstruction=False):
        """
        Initializes a BinaryLiquid object from cached data.

        Args:
            input (any): Binary system - can be either a list or hyphenated string
            dft_type (str): Type of DFT calculation.
            params (list): Initial fitting parameters.
            reconstruction (bool): Flag for liquidus reconstruction from prediction - 
                uses cached elemental data for melting points and referenced entropy instead of digitized liquidus.

        Returns:
            BinaryLiquid: Initialized BinaryLiquid object.
        """
        components, sys_name = lbd.validate_and_format_binary_system(input)
        params = validate_binary_mixing_parameters(params)

        ch, _ = lbd.get_dft_convexhull(components, dft_type)
        phases = []
        for entry in ch.stable_entries:
            composition = entry.composition.fractional_composition.as_dict().get(components[1], 0)
            phase = {
                'name': entry.name,
                'comp': composition,
                'points': [],
                'energy': 96485 * ch.get_form_energy_per_atom(entry),
            }
            phases.append(phase)

        phases.sort(key=lambda x: x['comp'])
        phases.append({'name': 'L', 'points': []})

        mpds_json, component_data, digitized_liq = lbd.load_mpds_data(components)
        if not reconstruction and digitized_liq:
            component_data[components[0]][1] = digitized_liq[0][1]
            component_data[components[-1]][1] = digitized_liq[-1][1]

        if 'temp' in mpds_json:
            temp_range = [mpds_json['temp'][0] + 273.15, mpds_json['temp'][1] + 273.15]
        else:
            comp_tms = [component_data[comp][1] for comp in components]
            temp_range = [min(comp_tms) - 50, max(comp_tms) * 1.1 + 50]

        if not digitized_liq:
            return cls(sys_name, components, True, mpds_json=mpds_json, component_data=component_data,
                       temp_range=temp_range, dft_type=dft_type, dft_ch=ch, phases=phases, params=params)

        return cls(sys_name, components, False, mpds_json=mpds_json, component_data=component_data,
                   digitized_liq=digitized_liq, temp_range=temp_range, dft_type=dft_type, dft_ch=ch, phases=phases,
                   params=params)
    

    def to_HSX(self, fmt="dict") -> dict | pd.DataFrame:
        """
        Converts phase data into HSX format for further calculations.

        Args:
            fmt (str): Output format ('dict' or 'dataframe').

        Returns:
            dict | pd.DataFrame: Data in HSX format.
        """
        data = {'X': list(_x_vals), 'S': [], 'H': [], 'Phase Name': ['L' for _ in _x_vals]}
        H_a_liq = self.component_data[self.components[0]][0]
        H_b_liq = self.component_data[self.components[1]][0]
        H_lc = (H_a_liq * _x_vals[-2:0:-1] +
                H_b_liq * _x_vals[1:-1])
        H_xs = _x_vals[1:-1] * _x_vals[-2:0:-1] * (self.get_L0_a() + self.get_L1_a() * (1 - 2 * _x_vals[1:-1]))
        data['H'] = list(H_lc + H_xs)
        data['H'].insert(0, H_a_liq)
        data['H'].append(H_b_liq)

        R = 8.314

        S_a_liq = self.component_data[self.components[0]][0] / self.component_data[self.components[0]][1]
        S_b_liq = self.component_data[self.components[1]][0] / self.component_data[self.components[1]][1]
        S_lc = (S_a_liq * _x_vals[-2:0:-1] +
                S_b_liq * _x_vals[1:-1])
        S_ideal = -R * (_x_vals[1:-1] * _x_logs + _x_vals[-2:0:-1] * _x_logs[::-1])
        S_xs = -_x_vals[1:-1] * _x_vals[-2:0:-1] * (self.get_L0_b() + self.get_L1_b() * (1 - 2 * _x_vals[1:-1]))
        data['S'] = list(S_lc + S_ideal + S_xs)
        data['S'].insert(0, S_a_liq)
        data['S'].append(S_b_liq)

        for x in _x_vals:
            for phase in self.phases:
                if phase['name'] == 'L':
                    continue
                if round(phase['comp'], 2) == round(x, 2):
                    data['X'].append(round(x, 2))
                    data['H'].append(phase['energy'])
                    data['S'].append(0)
                    data['Phase Name'].append(phase['name'])

        if fmt == "dict":
            return data
        if fmt == "dataframe":
            return pd.DataFrame(data)
        else:
            raise ValueError("kwarg 'fmt' must be either 'dict' or 'dataframe'!")
        
    def update_phase_points(self) -> None:
        """
        Calculates the phase points for given parameter values using the HSX class.

        This method converts phase data into the HSX form and uses HSX code to calculate the liquidus
        and low-temperature DFT phase boundaries.

        Returns:
            None
        """
        data = self.to_HSX()
        hsx_dict = {
            'data': data,
            'phases': [phase['name'] for phase in self.phases],
            'comps': self.components
        }
        self.hsx = HSX(hsx_dict, [self.temp_range[0] - 273.15, self.temp_range[-1] - 273.15])
        phase_points = self.hsx.get_phase_points()
        for phase in self.phases:
            phase['points'] = phase_points[phase['name']]

    def get_L0_a(self) -> int | float:
        return self._params[0]

    def get_L0_b(self) -> int | float:
        return self._params[1]

    def get_L1_a(self) -> int | float:
        return self._params[2]

    def get_L1_b(self) -> int | float:
        return self._params[3]
    
    def get_params(self) -> list[int | float]:
        """
        Get a copy of the current parameters such that the BinaryLiquid object will not be modified accidentally.
        
        Returns:
            list: A list of non-ideal mixing parameters in the following format: [L0_a, L0_b, L1_a, L1_b]
        """
        return [p for p in self._params]
    
    def update_params(self, input) -> None:
        """
        Update the non-ideal mixing parameters with validity checks, 
        then recalculate phase boundaries for the new parameters.

        Args:
            input (list[int | float]): A list containing numerical values representing non-ideal mixing parameters.

        Returns:
            None
        """
        self._params = validate_binary_mixing_parameters(input)
        self.update_phase_points()

    def find_invariant_points(self, verbose=False, t_tol=15) -> list[dict]:
        """
        Identifies invariant points in the MPDS data using the provided MPDS JSON and liquidus data.

        This function does not consider DFT phases, which may differ in composition from the MPDS data. It requires both 
        complete liquidus and JSON data for a binary system.

        Args:
            verbose (bool): If True, outputs additional debugging information.
            t_tol (int): Temperature tolerance for invariant point identification.

        Returns:
            list: A list of invariant points identified in the MPDS data.
        """
        if self.mpds_json['reference'] is None:
            print("System JSON does not contain any data!\n")
            return []

        # Identify phases from MPDS JSON
        phases = lbd.identify_mpds_phases(self.mpds_json, verbose=True)
        invariants = [phase for phase in phases if phase['type'] == 'mig']  # Miscibility gaps are not phases. 
        # They are also not really 'invariant points' either but we classify them as such for algorithm purposes.

        # Filter low-temperature phases
        mpds_lowt_phases = [
            phase for phase in phases
            if (
                phase['type'] in ['lc', 'ss'] and
                phase['tbounds'][0][1] < (self.mpds_json['temp'][0] + 273.15) +
                (self.mpds_json['temp'][1] - self.mpds_json['temp'][0]) * 0.10
            ) or '(' in phase['name']
        ]

        if verbose:
            print('--- Low temperature phases including component solid solutions ---')
            for phase in mpds_lowt_phases:
                print(phase)

        # Identify full composition solid solutions
        phase_labels = [label[0] for label in self.mpds_json['labels']]
        ss_label = f"({self.components[0]}, {self.components[1]})"
        ss_label_inv = f"({self.components[1]}, {self.components[0]})"
        ss_labels = [
            ss_label, f"{ss_label} ht", f"{ss_label} rt",
            ss_label_inv, f"{ss_label_inv} ht", f"{ss_label_inv} rt"
        ]
        full_comp_ss = bool([label for label in phase_labels if label in ss_labels])
        if full_comp_ss:
            print('Solidus processing not implemented!')
            self.init_error = True
            return invariants

        def find_local_minima(points):
            """
            Args:
                points (list of tuples): List of (x, y) points.

            Returns:
                list: Local minima points.
            """
            def is_lt_prev(index):
                return index > 0 and points[index][1] < points[index - 1][1]

            local_minima = []
            current_section = []

            for i in range(len(points)):
                if is_lt_prev(i):
                    current_section = [points[i]]
                elif current_section and current_section[-1][1] == points[i][1]:
                    current_section.append(points[i])
                elif current_section:
                    local_minima.append(current_section[len(current_section) // 2])
                    current_section = []

            return local_minima

        def find_local_maxima(points):
            """
            Args:
                points (list of tuples): List of (x, y) points.

            Returns:
                list: Local maxima points.
            """
            def is_gt_prev(index):
                return index > 0 and points[index][1] > points[index - 1][1]

            local_maxima = []
            current_section = []

            for i in range(len(points)):
                if is_gt_prev(i):
                    current_section = [points[i]]
                elif current_section and current_section[-1][1] == points[i][1]:
                    current_section.append(points[i])
                elif current_section:
                    local_maxima.append(current_section[len(current_section) // 2])
                    current_section = []

            return local_maxima

        # Locate maxima and minima in liquidus
        maxima = find_local_maxima(self.digitized_liq)
        minima = find_local_minima(self.digitized_liq)

        # Assign congruent melting points
        if mpds_lowt_phases:
            for coords in maxima[:]:
                mpds_lowt_phases.sort(key=lambda x: abs(x['comp'] - coords[0]))
                phase = mpds_lowt_phases[0]
                if (
                    phase['type'] in ['lc', 'ss'] and
                    abs(phase['comp'] - coords[0]) <= 0.02 and
                    phase['tbounds'][1][1] + t_tol >= coords[1]
                ):
                    phase['type'] = 'cmp'
                    invariants.append({
                        'type': phase['type'],
                        'comp': phase['comp'],
                        'temp': phase['tbounds'][1][1],
                        'phases': [phase['name']],
                        'phase_comps': [phase['comp']]
                    })
                    maxima.remove(coords)

        # Sort by descending temperature for peritectic identification
        mpds_lowt_phases.sort(key=lambda x: x['tbounds'][1][1], reverse=True)

        def find_adj_phases(point: list | tuple) -> tuple[dict, dict]:
            """
            Finds adjacent phases near a given point.

            Args:
                point (list | tuple): A point in composition-temperature space.

            Returns:
                tuple: Two nearest adjacent phases.
            """
            all_lowt_phases = (
                mpds_lowt_phases +
                [
                    {'name': self.components[0], 'comp': 0, 'type': 'lc',
                        'tbounds': [[], [0, self.component_data[self.components[0]][1]]]},
                    {'name': self.components[1], 'comp': 1, 'type': 'lc',
                        'tbounds': [[], [1, self.component_data[self.components[1]][1]]]},
                ]
            )
            all_lowt_phases = [p for p in all_lowt_phases if p['tbounds'][1][1] + t_tol >= point[1]]
            lhs_phases = [phase for phase in all_lowt_phases if phase['comp'] < point[0]]
            adj_lhs_phase = None if not lhs_phases else min(lhs_phases, key=lambda x: abs(x['comp'] - point[0]))
            rhs_phases = [phase for phase in all_lowt_phases  if phase['comp'] > point[0]]
            adj_rhs_phase = None if not rhs_phases else min(rhs_phases, key=lambda x: abs(x['comp'] - point[0]))
            return adj_lhs_phase, adj_rhs_phase

        # Identify liquid-liquid miscibility gap labels
        misc_gap_labels = []
        for label in self.mpds_json['labels']:
            delim_label = label[0].split(' ')
            if len(delim_label) == 3 and delim_label[0][0] == 'L' and delim_label[2][0] == 'L':
                misc_gap_labels.append([label[1][0] / 100.0, label[1][1] + 273.15])

        # Process miscibility gap labels and find the nearest two-phase region which each corresponds to
        for mgl in misc_gap_labels:
            if len(maxima) < 1:
                break
            nearest_maxima = min(maxima, key=lambda x: abs(x[0] - mgl[0]))

            tbounds = [None, nearest_maxima]
            cbounds = None
            phases = None
            phase_comps = None

            for shape in self.mpds_json['shapes']:
                if shape['nphases'] != 2:
                    continue
                data = lbd.shape_to_list(shape['svgpath'])
                if not data:
                    continue
                data.sort(key=lambda x: x[1])
                if not (abs(data[-1][1] - nearest_maxima[1]) < t_tol and abs(data[-1][0] - nearest_maxima[0]) < 0.05):
                    continue
                tbounds = [data[0], data[-1]]
                data.sort(key=lambda x: x[0])
                if not data[0][0] < nearest_maxima[0] < data[-1][0]:
                    continue
                cbounds = [data[0], data[-1]]
                break

            if len(minima) >= 1:
                # Adjacent monotectic should be a minima point at minimum distance in x-t space from the misc gap dome
                adj_mono = min(
                    minima,
                    key=lambda x: abs(tbounds[1][0] - x[0]) + 2 * (abs(tbounds[1][1] - x[1]) / self.temp_range[1])
                )
                tbounds[0] = [tbounds[1][0], adj_mono[1]]
                adj_phases = find_adj_phases(adj_mono)

                if adj_mono[0] < tbounds[1][0]:
                    if adj_phases[0] is not None:
                        phase_comps = [adj_phases[0]['comp']]
                        phases = [adj_phases[0]['name']]
                    if not cbounds:
                        lhs_ind = self.digitized_liq.index(adj_mono)
                        for i in range(lhs_ind + 1, len(self.digitized_liq) - 1):
                            if self.digitized_liq[i + 1][1] < adj_mono[1] <= self.digitized_liq[i][1]:
                                m = ((self.digitized_liq[i + 1][1] - self.digitized_liq[i][1]) /
                                        (self.digitized_liq[i + 1][0] - self.digitized_liq[i][0]))
                                rhs_comp = (adj_mono[1] - self.digitized_liq[i][1]) / m + self.digitized_liq[i][0]
                                cbounds = [adj_mono, [rhs_comp, adj_mono[1]]]
                                break
                elif adj_mono[0] > tbounds[1][0]:
                    if adj_phases[1] is not None:
                        phase_comps = [adj_phases[1]['comp']]
                        phases = [adj_phases[1]['name']]
                    if not cbounds:
                        rhs_ind = self.digitized_liq.index(adj_mono)
                        for i in reversed(range(1, rhs_ind - 1)):
                            if self.digitized_liq[i - 1][1] < adj_mono[1] <= self.digitized_liq[i][1]:
                                m = ((self.digitized_liq[i - 1][1] - self.digitized_liq[i][1]) /
                                        (self.digitized_liq[i - 1][0] - self.digitized_liq[i][0]))
                                lhs_comp = (adj_mono[1] - self.digitized_liq[i][1]) / m + self.digitized_liq[i][0]
                                cbounds = [[lhs_comp, adj_mono[1]], adj_mono]
                                break
                if cbounds[0][1] != cbounds[1][1]:
                    cbounds[0][1] = adj_mono[1]
                    cbounds[1][1] = adj_mono[1]
                minima.remove(adj_mono)

            if cbounds:
                invariants.append({
                    'type': 'mig',
                    'comp': tbounds[1][0],
                    'cbounds': cbounds,
                    'tbounds': tbounds,
                    'phases': phases,
                    'phase_comps': phase_comps
                })
                maxima.remove(nearest_maxima)
            break

        stable_phase_comps = []

        # Main loop for peritectic phase identification
        for phase in mpds_lowt_phases:
            if '(' in phase['name']:  # Ignore component SS phases
                continue

            # Congruent melting points will not be considered for peritectic formation but will limit comp search range
            if phase['type'] == 'cmp':
                stable_phase_comps.append(phase['comp'])
                continue

            sections, current_section = [], []
            phase_temp = phase['tbounds'][1][1]

            for i in range(len(self.digitized_liq) - 1):
                liq_point, next_liq_point = self.digitized_liq[i], self.digitized_liq[i + 1]
                liq_temp, next_liq_temp = liq_point[1], next_liq_point[1]

                # Liquidus point is above or equal to phase temp
                if liq_temp >= phase_temp:
                    current_section.append(liq_point)
                    if next_liq_temp >= phase_temp and i + 1 == len(self.digitized_liq) - 1:
                        current_section.append(next_liq_point)
                        sections.append(current_section)

                # Liquidus point is first point below phase temp
                elif current_section:
                    if abs(phase_temp - current_section[-1][1]) > abs(phase_temp - liq_temp):
                        current_section.append(liq_point)  # Add to section if closer to phase temp
                    sections.append(current_section)  # End section
                    current_section = []

                # Next liquidus point is above phase temp
                elif next_liq_temp >= phase_temp > liq_temp:
                    if abs(phase_temp - next_liq_temp) > abs(phase_temp - liq_temp):
                        current_section.append(liq_point)  # Add if below phase temp and closer than next point

            # Find endpoints of liquidus segments excluding the component ends
            endpoints = [
                section[i]
                for section in sections
                for i in [0, -1]
                if section[i] not in [self.digitized_liq[0], self.digitized_liq[-1]]
            ]

            # Filter endpoints if there exists a stable phase between the current phase and the liquidus
            for comp in stable_phase_comps:
                endpoints = [
                    ep for ep in endpoints
                    if abs(comp - ep[0]) > abs(phase['comp'] - ep[0])
                    or abs(comp - phase['comp']) > abs(phase['comp'] - ep[0])
                ]

            # Sort by increasing distance to liquidus to find the shortest distance
            endpoints.sort(key=lambda x: abs(x[0] - phase['comp']))

            # Take the closest liquidus point to the phase as the peritectic point
            if endpoints:
                invariants.append({
                    'type': 'per',
                    'comp': endpoints[0][0],
                    'temp': phase_temp,
                    'phases': [phase['name']],
                    'phase_comps': [phase['comp']]
                })

            stable_phase_comps.append(phase['comp'])

        # Identify eutectic points
        for coords in minima:
            adj_phases = find_adj_phases(coords)
            phases, phase_comps = zip(*[
                (None, None) if phase is None else (phase['name'], phase['comp'])
                for phase in adj_phases
            ])

            invariants.append({
                'type': 'eut',
                'comp': coords[0],
                'temp': coords[1],
                'phases': list(phases),
                'phase_comps': list(phase_comps)
            })

        invariants.sort(key=lambda x: x['comp'])
        invariants = [inv for inv in invariants if inv['type'] not in ['lc', 'ss']]
        if verbose:
            print('--- Identified invariant points ---')
            for inv in invariants:
                print(inv)
            print()
        return invariants

    def solve_params_from_constraints(self, guessed_vals: dict) -> None:
        """
        Updates the parameters of the object based on guessed values and constraints.

        Args:
            guessed_vals (dict): A dictionary containing guessed values for the parameters.
        """
        symbols = sp.symbols('a b c d')
        for ind, symbol in enumerate(symbols):
            try:
                if symbol in guessed_vals:
                    self._params[ind] = float(guessed_vals[symbol])
                elif self.constraints:
                    self._params[ind] = float(self.constraints[symbol].subs(guessed_vals))
            except TypeError:
                raise RuntimeError("Error in constraint equations!")

    def liquidus_is_continuous(self, tol=2 * _x_step) -> bool:
        """
        Checks if the liquidus line is continuous within a given tolerance.

        Args:
            tol (float): Tolerance for liquidus continuity. Default is twice the step size.

        Returns:
            bool: True if the generated liquidus line is compositionally-continuous, False otherwise.
        """
        last_coords = None
        for coords in self.phases[-1]['points']:
            if last_coords and coords[0] - last_coords[0] > tol:
                return False
            last_coords = coords
        return True

    def calculate_deviation_metrics(self, ignored_ranges=True, num_points=30) -> tuple[float, float]:
        """
        Calculates the deviation metrics between the digitized liquidus and the fitted liquidus.

        Args:
            ignored_ranges (bool): Whether to ignore specified composition ranges. Default is True.
            num_points (int): Number of points to sample for deviation metrics. Default is 30.

        Returns:
            tuple: Mean absolute error (MAE) and root mean square error (RMSE).
        """
        x1, T1 = zip(*self.digitized_liq)
        x2, T2 = zip(*self.phases[-1]['points'])

        # Generate evenly spaced points across the composition range
        x_coords = np.linspace(self.comp_range[0] / 100, self.comp_range[-1] / 100, num_points + 2)[1:-1]

        def within_boundaries(value):
            """Check if a value falls within any of the given boundary ranges."""
            for lower, upper in self.ignored_comp_ranges:
                if lower <= value <= upper:
                    return True
            return False

        if ignored_ranges:
            x_coords = [x for x in x_coords if not within_boundaries(x)]

        if len(x_coords) < 10:
            print(f"Warning: Large composition range filtered out (remaining num_points = {len(x_coords)})")
            return float('inf'), float('inf')

        Y1 = []
        Y2 = []

        for i in range(len(x_coords)):
            MPDS_ind = fit_ind = -1

            for j in range(len(self.digitized_liq) - 1):
                if x1[j] <= x_coords[i] < x1[j + 1]:
                    MPDS_ind = j
                    break
            for j in range(len(self.phases[-1]['points']) - 1):
                if x2[j] <= x_coords[i] < x2[j + 1]:
                    fit_ind = j
                    break

            if MPDS_ind != -1 and fit_ind != -1:
                m1 = (T1[MPDS_ind] - T1[MPDS_ind + 1]) / (x1[MPDS_ind] - x1[MPDS_ind + 1])
                b1 = (x1[MPDS_ind] * T1[MPDS_ind + 1] - x1[MPDS_ind + 1] * T1[MPDS_ind]) / (
                        x1[MPDS_ind] - x1[MPDS_ind + 1])
                y1 = m1 * x_coords[i] + b1
                m2 = (T2[fit_ind] - T2[fit_ind + 1]) / (x2[fit_ind] - x2[fit_ind + 1])
                b2 = (x2[fit_ind] * T2[fit_ind + 1] - x2[fit_ind + 1] * T2[fit_ind]) / (
                        x2[fit_ind] - x2[fit_ind + 1])
                y2 = m2 * x_coords[i] + b2
                Y1.append(y1)
                Y2.append(y2)

        # Calculate absolute and squared differences
        point_diffs = [abs(Y1[i] - Y2[i]) for i in range(len(Y2))]
        squared_point_diffs = [(Y1[i] - Y2[i]) ** 2 for i in range(len(Y2))]

        return np.mean(point_diffs), math.sqrt(np.mean(squared_point_diffs))

    def f(self, point: list | tuple) -> float:
        """
        Objective function for parameter fitting.

        Args:
            point (list | tuple): List of guessed parameter values to evaluate.

        Returns:
            float: Generated liquidus mean absolute error (MAE) for the given parameter values.
        """
        guess_dict = {symbol: guess for symbol, guess in zip(self.guess_symbols, point)}
        self.solve_params_from_constraints(guess_dict)  # Update parameter values
        try:
            self.update_phase_points()
        except (ValueError, TypeError) as e:
            print(e)
            return float('inf')
        if not self.liquidus_is_continuous():
            print(f'Liquidus not continuous for guess {guess_dict}')
            return float('inf')
        mae, _ = self.calculate_deviation_metrics()
        return mae

    def nelder_mead(self, max_iter=128, tol=5e-2, verbose=False,
                    initial_guesses=[[-20, -20], [-20, 20], [20, -20]]) -> tuple[float, float, np.ndarray]:
        """
        Nelder-Mead algorithm for fitting the liquid non-ideal mixing parameters.

        Args:
            max_iter (int): Maximum number of iterations. Default is 128.
            tol (float): Tolerance for algorithm convergence. Default is 0.05.
            verbose (bool): If True, print updates during optimization. Default is False.
            initial_guesses (list): Reasonable initial values for guessed parameters. Default values are for L_b params

        Returns:
            tuple: MAE, RMSE, and Nelder-Mead optimization path.
        """
        # Initial guesses for parameters
        x0 = np.array(initial_guesses, dtype=float)
        self.nmpath = np.empty((3, 3, max_iter), dtype=float)
        initial_time = time.time()

        print("--- Beginning Nelder-Mead optimization ---")

        for i in range(max_iter):
            start_time = time.time()
            if verbose:
                print("Iteration #", i)

            f_vals = np.array([self.f(x) for x in x0])
            self.nmpath[:, :2, i] = x0
            self.nmpath[:, 2:, i] = np.array([[f] for f in f_vals])
            iworst = np.argmax(f_vals)
            ibest = np.argmin(f_vals)
            centroid = np.mean(x0[f_vals != f_vals[iworst]], axis=0)
            xreflect = centroid + 1.0 * (centroid - x0[iworst, :])
            f_xreflect = self.f(xreflect)

            if iworst == ibest:
                self.nmpath = self.nmpath[:, :, :i]
                raise RuntimeError("Nelder-Mead algorithm is unable to find physical parameter values.")

            # Simplex reflection step
            if f_vals[iworst] <= f_xreflect < f_vals[2]:
                x0[iworst, :] = xreflect
            # Simplex expansion step
            elif f_xreflect < f_vals[ibest]:
                xexp = centroid + 2.0 * (xreflect - centroid)
                if self.f(xexp) < f_xreflect:
                    x0[iworst, :] = xexp
                else:
                    x0[iworst, :] = xreflect
            # Simplex contraction step
            else:
                if f_xreflect < f_vals[2]:
                    xcontract = centroid + 0.5 * (xreflect - centroid)
                    if self.f(xcontract) < self.f(x0[iworst, :]):
                        x0[iworst, :] = xcontract
                    else:  # Simplex shrink step
                        x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                        [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                        x0[iworst, :] = x0[imid, :] + 0.5 * (x0[imid, :] - x0[ibest, :])
                else:
                    xcontract = centroid + 0.5 * (x0[iworst, :] - centroid)
                    if self.f(xcontract) < self.f(x0[iworst, :]):
                        x0[iworst, :] = xcontract
                    else:  # Simplex shrink step
                        x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                        [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                        x0[imid, :] = x0[ibest, :] + 0.5 * (x0[imid, :] - x0[ibest, :])

            if verbose:
                print("Best guess:", x0[ibest, :], f_vals[ibest])
                print("Height of triangle =", 2 * np.max(np.abs(x0 - centroid)))
                print("--- %s seconds elapsed ---" % (time.time() - start_time))

            # Check convergence
            if np.max(np.abs(x0 - centroid)) < tol:
                self.f(x0[ibest, :])
                print("--- Nelder-Mead converged in %s seconds ---" % (time.time() - initial_time))
                mae, rmse = self.calculate_deviation_metrics()
                print("Mean temperature deviation per point between liquidus curves =", mae, '\n')
                self.nmpath = self.nmpath[:, :, :i]
                return mae, rmse, self.nmpath
        raise RuntimeError("Nelder-Mead algorithm did not converge within limit.")

    def fit_parameters(self, verbose=False, n_opts=1, t_tol=15) -> list[dict]:
        """
        Fit the liquidus non-ideal mixing parameters for a binary system.

        This function utilizes the Nelder-Mead algorithm to minimize the temperature deviation in the liquidus.

        Args:
        verbose (bool): If True, prints detailed progress and results.
        n_opts (int): Number of optimization attempts. Updates the BinaryLiquid object to reflect the lowest MAE fit
        t_tol (float): Temperature tolerance for invariant point identification.

        Returns:
        list[dict]: Parameter fitting data containing results of all optimization attempts.
        """

        if self.digitized_liq is None:
            print("System missing liquidus data! Ensure that 'BinaryLiquid.digitized_liq' is not empty!\n")
            return

        # Find invariant points
        if not self.invariants:
            self.invariants = self.find_invariant_points(verbose=verbose, t_tol=t_tol)
        
        if self.init_error:
            print("!!!WARNING!!!\n"
                  "The initialization of the BinaryLiquid object resulted in an initialization error! The fitting"
                  "algorithm may produce unintended results. For more information, please consult the documentation!\n")

        # Define thermodynamic symbols and equations
        xb, t, a, b, c, d = sp.symbols('x t a b c d')
        L0 = a + b * t
        L1 = c + d * t
        xa = 1 - xb

        R = 8.314
        Ga = (self.component_data[self.components[0]][0] -
              t * self.component_data[self.components[0]][0] / self.component_data[self.components[0]][1])
        Gb = (self.component_data[self.components[1]][0] -
              t * self.component_data[self.components[1]][0] / self.component_data[self.components[1]][1])

        G_ideal = R * t * (xa * sp.log(xa) + xb * sp.log(xb))
        G_xs = L0 * (xa * xb) + L1 * (xa * xb * (xa - xb))
        G_liq = Ga * xa + Gb * xb + G_ideal + G_xs

        G_prime = sp.diff(G_liq, xb)
        G_double_prime = sp.diff(G_prime, xb)

        def find_nearest_phase(composition, tol=0.02):
            """
            Args:
            composition (float): Target composition.
            tol (float): Tolerance for the distance.

            Returns:
            tuple: Nearest phase and deviation.
            """
            sorted_phases = sorted(self.phases[:-1], key=lambda x: abs(x['comp'] - composition))
            nearest = sorted_phases[0]
            deviation = abs(nearest['comp'] - composition)
            if deviation > tol:
                return {}, deviation
            return nearest, deviation

        eqs = []

        # Compare invariant points to self.phases to assess solving conditions
        for inv in self.invariants:
            if inv['type'] == 'mig':
                x1, t1 = inv['cbounds'][0]  # Bottom left of dome
                x2, t2 = inv['tbounds'][1]  # Top of dome
                x3, t3 = inv['cbounds'][1]  # Bottom right of dome

                eqn1 = sp.Eq(G_double_prime.subs({xb: x2, t: t2}), 0)
                eqn4 = sp.Eq(G_prime.subs({xb: x1, t: t1}), G_prime.subs({xb: x3, t: t3}))

                eqs.append(['mig', f'{round(x2, 2)} - 2nd order', t1, eqn1])
                eqs.append(['mig', f'{round(x1, 2)}-{round(x3, 2)} - 1st order', t1, eqn4])

            if inv['type'] == 'cmp':
                if '(' in inv['phases'][0]:
                    if inv['comp'] < 0.5:
                        self.ignored_comp_ranges.append([0, inv['comp']])
                    elif inv['comp'] > 0.5:
                        self.ignored_comp_ranges.append([inv['comp'], 1])
                    continue

                nearest_phase, _ = find_nearest_phase(inv['comp'])
                if not nearest_phase:
                    continue

                x1, t1 = nearest_phase['comp'], inv['temp']
                eqn = sp.Eq(G_liq.subs({xb: x1, t: t1}), nearest_phase['energy'])
                eqs.append(['cmp', f'{round(x1, 2)} - 0th order', t1, eqn])

            if inv['type'] == 'per':
                if '(' in inv['phases'][0]:
                    if inv['phase_comps'][0] < inv['comp']:
                        self.ignored_comp_ranges.append([0, inv['comp']])
                    elif inv['phase_comps'][0] > inv['comp']:
                        self.ignored_comp_ranges.append([inv['comp'], 1])
                    continue

                per_phase, _ = find_nearest_phase(inv['phase_comps'][0], tol=0.04)
                if not per_phase:
                    continue

                x1, t1 = inv['comp'], inv['temp']
                x2, g2 = per_phase['comp'], per_phase['energy']

                eqn1 = sp.Eq(G_liq.subs({xb: x1, t: t1}) + G_prime.subs({xb: x1, t: t1}) * (x2 - x1), g2)
                eqn2 = sp.Eq(G_liq.subs({xb: x1, t: t1}), g2)

                liq_point_at_phase = min(self.digitized_liq, key=lambda x: abs(x[0] - x2))
                temp_below_liq = liq_point_at_phase[1] - t1

                if temp_below_liq > t_tol:
                    eqs.append(['per', f'{round(x1, 2)} - 0th order', t1, eqn1])
                else:
                    eqs.append(['per', f'{round(x1, 2)} - pseudo CMP 0th order', t1, eqn2])

            if inv['type'] == 'eut':
                if None in inv['phase_comps']:
                    continue

                lhs_phase, _ = find_nearest_phase(inv['phase_comps'][0], tol=0.04)
                rhs_phase, _ = find_nearest_phase(inv['phase_comps'][1], tol=0.04)

                invalid_eut = False
                if not lhs_phase or lhs_phase['comp'] > inv['comp']:
                    self.ignored_comp_ranges.append([inv['phase_comps'][0], inv['comp']])
                    invalid_eut = True
                elif '(' in inv['phases'][0]:
                    invalid_eut = True
                if not rhs_phase or rhs_phase['comp'] < inv['comp']:
                    self.ignored_comp_ranges.append([inv['comp'], inv['phase_comps'][1]])
                    invalid_eut = True
                elif '(' in inv['phases'][1]:
                    invalid_eut = True
                if invalid_eut:
                    continue

                x1, g1 = lhs_phase['comp'], lhs_phase['energy']
                x2, t2 = inv['comp'], inv['temp']
                x3, g3 = rhs_phase['comp'], rhs_phase['energy']

                eqn1 = sp.Eq(G_prime.subs({xb: x2, t: t2}), (g3 - g1) / (x3 - x1))
                eqn2 = sp.Eq(G_liq.subs({xb: x2, t: t2}) + G_prime.subs({xb: x2, t: t2}) * (x1 - x2), g1)
                eqn3 = sp.Eq(G_liq.subs({xb: x2, t: t2}) + G_prime.subs({xb: x2, t: t2}) * (x3 - x2), g3)

                eqs.append(['eut', f'{round(x2, 2)} - 1st order', t2, eqn1])
                if g1 <= g3:
                    eqs.append(['eut', f'{round(x2, 2)} - 0th order lhs', t2, eqn2])
                else:
                    eqs.append(['eut', f'{round(x2, 2)} - 0th order rhs', t2, eqn3])

        max_liq_temp = max(self.digitized_liq, key=lambda x: x[1])[1]
        mean_liq_temp = (min(self.digitized_liq, key=lambda x: x[1])[1] + max_liq_temp) / 2
        eqs = [eq for eq in eqs if not eq[3] == False]

        initial_constrs = []
        if len(eqs) >= 2:
            self.guess_symbols = [b, d]
            highest_tm_eq = max(eqs, key=lambda x: x[2])
            for eq in eqs:
                if eq != highest_tm_eq:
                    self.constraints = sp.solve([eq[3], highest_tm_eq[3]], (a, c))
                    try:
                        init_mae = self.f([0, 0])
                        if init_mae != float('inf'):
                            initial_constrs.append([eq, highest_tm_eq, init_mae])
                    except RuntimeError:
                        continue
            initial_constrs.sort(key=lambda x: x[2])
        if not initial_constrs or len(eqs) < 2:
            print("Underconstrained system detected! Generating pseudo-constraints")

            self._params = [0, 0, 0, 0]
            self.guess_symbols = [a, c]
            self.constraints = None
            try:
                self.nelder_mead(tol=10, verbose=verbose)
            except RuntimeError:
                return []

            eqn1 = sp.Eq(L0.subs({t: mean_liq_temp}),
                         self.L0_a())
            eqn2 = sp.Eq(L1.subs({t: mean_liq_temp}),
                         self.L1_a())
            initial_constrs = [[['pseudo', f'L0_a + L0_b * t = {round(self.L0_a(), 2)}', mean_liq_temp, eqn1],
                                ['pseudo', f'L1_a + L1_b * t = {round(self.L1_a(), 2)}', mean_liq_temp, eqn2], []]]
            self.guess_symbols = [b, d]

        fitting_data = []
        for _ in range(n_opts):
            if not initial_constrs:
                break
            init_constr = initial_constrs.pop(0)[:-1]
            if verbose:
                print("--- Initial constraints ---")
                for (eq_type, score, temp, eq) in init_constr:
                    print(f"Invariant: {eq_type}, Type: {score}, Temperature: {round(temp, 1)}, Equation: {eq}")
                print(f"Maximum composition range fitted: {self.comp_range}")
                print(f"Ignored composition ranges: {self.ignored_comp_ranges}")
                print()

            selected_eqs = [eq[3] for eq in init_constr]
            self.constraints = sp.solve(selected_eqs, (a, c))
            guess_dict = {symbol: guess for symbol, guess in zip(self.guess_symbols, [0, 0])}
            self.solve_params_from_constraints(guess_dict)
            try:
                mae, rmse, path = self.nelder_mead(verbose=verbose, tol=5E-1)
            except RuntimeError as e:
                print('Nelder-Mead process encountered a fatal error: ', e)
                continue
            norm_mae = mae / max_liq_temp
            norm_rmse = rmse / max_liq_temp
            l0 = self.get_L0_a() + mean_liq_temp * self.get_L0_b()
            l1 = self.get_L1_a() + mean_liq_temp * self.get_L1_b()
            fit_invs = self.hsx.liquidus_invariants()[0]
            fitting_data.append({'mae': mae, 'rmse': rmse, 'norm_mae': norm_mae, 'norm_rmse': norm_rmse, 'nmpath': path,
                                 'L0_a': self.get_L0_a(), 'L0_b': self.get_L0_b(), 'L1_a': self.get_L1_a(), 'L1_b': self.get_L1_b(),
                                 'L0': l0, 'L1': l1, 'euts': fit_invs['Eutectics'], 'pers': fit_invs['Peritectics'],
                                 'cmps': fit_invs['Congruent Melting'], 'migs': fit_invs['Misc Gaps']})

        if fitting_data:
            best_fit = min(fitting_data, key=lambda x: x['mae'])
            self._params = [best_fit['L0_a'], best_fit['L0_b'], best_fit['L1_a'], best_fit['L1_b']]
            self.nmpath = best_fit['nmpath']
            self.update_phase_points()
        return fitting_data
    

class BLPlotter:
    """
    A plotting class for BinaryLiquid objects.

    This class contains methods to create various subfigures and visualizations for analyzing
    BinaryLiquid system data. It uses both static matplotlib and interactive Plotly plots.
    """

    def __init__(self, binaryliquid: BinaryLiquid, **plotkwargs):
        """
        Args:
            binaryliquid (BinaryLiquid): BinaryLiquid object containing the system data.
            plotkwargs (dict): Optional keyword arguments for plot customization (e.g., axis margins).
        """
        self._bl = binaryliquid
        self.plotkwargs = plotkwargs or {
            'axes': {'xmargin': 0.005, 'ymargin': 0}
        }

    def get_plot(self, plot_type: str, **kwargs) -> go.Figure | plt.Axes:
        """
        Generates the specified plot for the BinaryLiquid object.

        Args:
            plot_type (str): The type of plot to generate. Supported types include:
                - 'pc': Low-temperature phase comparison plot
                - 'ch', 'ch+g', 'vch': T=0K DFT convex hull plots
                - 'fit', 'fit+liq', 'pred', 'pred+liq': Generated phase diagram plots
                - 'nmp': Nelder-Mead path visualization plot
            kwargs: Additional keyword arguments for customization.

        Returns:
            go.Figure | plt.Axes: The generated plot object (Plotly or Matplotlib).
        """
        valid_plot_types = [
            'pc',
            'ch', 'ch+g', 'vch',
            'fit', 'fit+liq', 'pred', 'pred+liq',
            'nmp'
        ]
        if plot_type not in valid_plot_types:
            raise ValueError(f"Invalid plot type '{plot_type}'. Supported types: {valid_plot_types}")

        fig = None

        # Phase comparison plot
        if plot_type == 'pc':
            fig = self._generate_phase_comparison_plot()

        # Convex hull plots
        elif plot_type in ['ch', 'ch+g', 'vch']:
            fig = self._generate_convex_hull_plot(plot_type, **kwargs)

        # Liquidus fitting and prediction plots
        elif plot_type in ['fit', 'fit+liq', 'pred', 'pred+liq']:
            fig = self._generate_liquidus_fit_plot(plot_type, **kwargs)

        # Nelder-Mead path visualization
        elif plot_type == 'nmp':
            fig = self._generate_nelder_mead_path_plot()

        return fig

    def show(self, plot_type: str, **kwargs) -> None:
        """
        Displays the generated plot.

        Args:
            plot_type (str): The type of plot to generate. Supported types include:
                - 'pc': Low-temperature phase comparison plot
                - 'ch', 'ch+g', 'vch': T=0K DFT convex hull plots
                - 'fit', 'fit+liq', 'pred', 'pred+liq': Generated phase diagram plots
                - 'nmp': Nelder-Mead path visualization plot
            kwargs: Additional keyword arguments passed to `get_plot`.
        """
        fig = self.get_plot(plot_type, **kwargs)

        if isinstance(fig, go.Figure):
            fig.show()
        elif isinstance(fig, plt.Axes):
            plt.show()
            plt.close(fig)

    def write_image(self, plot_type: str, stream: str | StringIO, image_format: str = "svg", **kwargs) -> None:
        """
        Saves the generated plot as an image.

        Args:
            plot_type (str): The type of plot to save.
            stream (str | StringIO): The file path or stream to save the image.
            image_format (str): The format of the image (default is 'svg').
            kwargs: Additional keyword arguments passed to `get_plot`.
        """
        fig = self.get_plot(plot_type, **kwargs)

        if isinstance(fig, go.Figure):
            if plot_type in ['ch', 'ch+g', 'vch']:
                fig.write_image(stream, format=image_format, width=480 * 1.8, height=300 * 1.7)
            else:
                fig.write_image(stream, format=image_format)
        elif isinstance(fig, plt.Axes):
            fig.figure.savefig(stream, format=image_format)
            plt.close(fig)

    def _generate_phase_comparison_plot(self) -> plt.Figure:
        """
        Generates a phase comparison plot showing congruent and incongruent phases
        from MPDS and MP data. The plot consists of two subplots displaying phases
        in different temperature and magnitude ranges.

        Returns:
            plt.Figure: The generated phase comparison plot.
        """
        # Extract low-temperature phase data from MPDS and MP
        ([mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp],
        [mp_phases, mp_phases_ebelow, min_form_e]) = lbd.get_low_temp_phase_data(self._bl.mpds_json, self._bl.dft_ch)

        # Filter out phases containing parentheses
        mpds_congruent_phases = {key: value for key, value in mpds_congruent_phases.items() if '(' not in key}
        mpds_incongruent_phases = {key: value for key, value in mpds_incongruent_phases.items() if '(' not in key}

        # Create subplots with specific layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2), gridspec_kw={'hspace': 0})

        def plot_phases(ax, source, color, alpha=0.5):
            """
            Args:
                ax (matplotlib.axes.Axes): The axis to plot on.
                source (dict): Phase data with keys as phase names and values as bounds/magnitudes.
                color (str): Color for the phase fill.
                alpha (float): Transparency for the fill.
            """
            for _, ((lb, ub), mag) in source.items():
                # Ensure a minimum width for labeling
                if ub - lb < 0.026:
                    ave = (ub + lb) / 2
                    lb = ave - 0.013
                    ub = ave + 0.013
                ax.fill_betweenx([min(0, mag), max(0, mag)], lb, ub, color=color, alpha=alpha)
                ax.set_xlim(0, 1)
                ax.margins(x=0, y=0)

        # Plot phases for both subplots
        plot_phases(ax1, mpds_congruent_phases, 'blue')
        plot_phases(ax1, mpds_incongruent_phases, 'purple')
        plot_phases(ax2, mp_phases, 'orange')
        plot_phases(ax2, mp_phases_ebelow, 'red')

        # Check if MPDS phases exist
        mpds_phases = bool(mpds_congruent_phases or mpds_incongruent_phases)

        # Configure y-axis for the first subplot
        if mpds_phases:
            tick_range = np.linspace(0, max_phase_temp, 4)[1:]
            ax1.set_yticks(tick_range)
            ax1.set_yticklabels([format(tick, '.1e') for tick in tick_range])
            ax1.set_ylim(0, 1.1 * max_phase_temp)
        else:
            ax1.set_yticks([])

        ax1.set_ylabel('MPDS', fontsize=11, rotation=90, labelpad=5, fontweight='semibold')
        ax1.yaxis.set_label_position('right')
        ax1.set_xticks([])

        # Configure y-axis for the second subplot
        if mp_phases:
            tick_range = np.linspace(0, min_form_e, 4)
            ax2.set_yticks(tick_range)
            ax2.set_yticklabels([format(tick, '.1e') for tick in tick_range])
            ax2.set_ylim(1.1 * min_form_e, 0)
        elif mpds_phases:
            ax2.set_yticks([0])
            ax2.set_yticklabels([format(0, '.1e')])
            ax2.set_ylim(-1, 0)
        else:
            ax2.set_yticks([])

        ax2.set_ylabel('MP', fontsize=11, rotation=90, labelpad=5, fontweight='semibold')
        ax2.yaxis.set_label_position('right')
        ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax2.set_xticklabels([0, 20, 40, 60, 80, 100])

        # Add a title to the figure
        fig.suptitle('Low Temperature Phase Comparison', fontweight='semibold')

        return fig

    def _generate_convex_hull_plot(self, plot_type: str, **kwargs) -> go.Figure:
            """
            Generates a convex hull plot or phase diagram visualization.

            Args:
                plot_type (str): The type of plot to generate ('ch', 'vch', 'ch+g').
                kwargs: Additional arguments for customization, such as 't_vals' or 't_units' for temperature values.

            Returns:
                go.Figure: The generated Plotly figure.
            """
            if any(len(Composition(comp).as_dict()) > 1 for comp in self._bl.components):
                raise NotImplementedError("This feature is not presently supported for compound components")
            if plot_type == 'vch':
                # Generate volume-referenced convex hull
                ch, atomic_vols = lbd.get_dft_convexhull(self._bl.sys_name, self._bl.dft_type, inc_structure_data=True)

                new_entries = [PDEntry(
                    composition=e.composition,
                    energy=atomic_vols[e.composition.reduced_formula] * e.composition.num_atoms
                ) for e in ch.stable_entries]

                vch = PhaseDiagram(new_entries)
                pdp = PDPlotter(vch)
            else:
                # Use the standard convex hull
                pdp = PDPlotter(self._bl.dft_ch)

            if plot_type in ['ch', 'vch']:
                # Generate a basic convex hull plot
                fig = pdp.get_plot()

                if plot_type == 'vch':
                    fig.update_yaxes(title={'text': 'Referenced Atomic Volume (Å^3/atom)'})

            elif (not self._bl.component_data or not self._bl.digitized_liq) and 't_vals' not in kwargs:
                # Handle uninitialized BinaryLiquid data
                print("BinaryLiquid object phase diagram not initialized! Returning plot without liquid energy")
                fig = pdp.get_plot()

            else:
                # Generate convex hull plot with liquidus curves
                t_vals = kwargs.get('t_vals', [])
                if not isinstance(t_vals, list) or not all(isinstance(t, (int, float)) for t in t_vals):
                    raise ValueError("kwarg 't_vals' must be a list of valid temperatures, either as ints or floats!")
                t_units = kwargs.get('units', 'C')
                if not t_units or not isinstance(t_units, str) or t_units not in ['C', 'K']:
                    raise ValueError("kwarg 't_units' must be a string, either 'C' for Celsius or 'K' for Kelvin")
                if t_units and not t_vals:
                    print("No arguments specified for 't_vals', setting 't_units' to 'K'")

                max_phase_temp = 0
                if not t_vals:
                    t_units = 'K'

                    # Estimate temperature range from liquidus data
                    asc_temp = sorted(self._bl.digitized_liq, key=lambda x: x[1])
                    mpds_phases = lbd.identify_mpds_phases(self._bl.mpds_json)

                    if mpds_phases:
                        max_phase_temp = max(mpds_phases, key=lambda x: x['tbounds'][1][1])['tbounds'][1][1]
                    else:
                        max_phase_temp = asc_temp[0][1]
                if t_units == 'C':
                    t_vals = [t + 273.15 for t in t_vals if t >= 0]
                else:
                    t_vals = [t for t in t_vals if t >= 0]

                # Define thermodynamic equations
                xb, t, a, b, c, d = sp.symbols('x t a b c d')
                L0 = a + b * t
                L1 = c + d * t
                xa = 1 - xb

                R = 8.314
                Ga = (self._bl.component_data[self._bl.components[0]][0] -
                    t * self._bl.component_data[self._bl.components[0]][0] /
                    self._bl.component_data[self._bl.components[0]][1])
                Gb = (self._bl.component_data[self._bl.components[1]][0] -
                    t * self._bl.component_data[self._bl.components[1]][0] /
                    self._bl.component_data[self._bl.components[1]][1])

                G_ideal = R * t * (xa * sp.log(xa) + xb * sp.log(xb))
                G_xs = L0 * (xa * xb) + L1 * (xa * xb * (xa - xb))
                G_liq = Ga * xa + Gb * xb + G_ideal + G_xs

                params = self._bl._params

                def get_g_curve(A=0, B=0, C=0, D=0, T=0) -> go.Scatter:
                    """
                    Args:
                        A, B, C, D (float): Non-ideal mixing parameters.
                        T (float): Temperature in Kelvin.

                    Returns:
                        go.Scatter: Plotly scatter trace for the Gibbs free energy curve.
                    """
                    gliq_fx = sp.lambdify(xb, G_liq.subs({t: T, a: A, b: B, c: C, d: D}), 'numpy')
                    gliq_vals = gliq_fx(_x_vals[1:-1])
                    ga = np.float64(Ga.subs({t: T}) / 96485)
                    gb = np.float64(Gb.subs({t: T}) / 96485)
                    name = "Liquid T="
                    name += str(int(T)) + "K" if t_units == 'K' else str(int(T-273.15)) + "C"

                    return go.Scatter(
                        x=_x_vals,
                        y=[ga] + [g / 96485 for g in gliq_vals] + [gb],
                        mode='lines',
                        name=name
                    )

                # Generate liquidus curves
                if t_vals:
                    traces = [get_g_curve(A=params[0], B=params[1], C=params[2], D=params[3], T=temp)
                            for temp in reversed(t_vals)]
                else:
                    traces = [
                        get_g_curve(A=params[0], B=params[1], C=params[2], D=params[3], T=max_phase_temp),
                        get_g_curve(A=params[0], B=params[1], C=params[2], D=params[3])
                    ]

                # Add traces to convex hull plot
                fig = pdp.get_plot(data=traces)

            fig.update_layout(plot_bgcolor="white",
                              paper_bgcolor="white", 
                              xaxis=dict(title=dict(text="Composition (fraction)")))
            return fig

    def _generate_liquidus_fit_plot(self, plot_type: str) -> go.Figure:
        """
        Generates liquidus fitting and prediction plots.

        Args:
            plot_type (str): The type of liquidus plot to generate ('fit', 'fit+liq', 'pred', 'pred+liq').

        Returns:
            go.Figure: The generated plot object.
        """

        # Initialize variables for liquidus lines and gas temperature
        gas_temp = None

        # Check if the plot type includes the MPDS liquidus
        if plot_type in ['fit+liq', 'pred+liq'] and not self._bl.digitized_liq:
            print("Digitized_liquidus is not initialized! Returning plot without digitized liquidus")

        # Determine if prediction is required
        pred_pd = bool(plot_type in ['pred', 'pred+liq'])

        # If predicted phase diagram, calculate the minimum gas temperature as an advisory for liquidus accuracy
        if pred_pd:
            gas_temp = min([cd[2] for cd in self._bl.component_data.values()])

        # Ensure phase points are updated if not already done
        if self._bl.hsx is None:
            self._bl.update_phase_points()

        # Generate the plot using the HSX plot method
        fig = self._bl.hsx.plot_tx(
            digitized_liquidus=self._bl.digitized_liq if plot_type in ['fit+liq', 'pred+liq'] else None,
            pred=pred_pd,  # Determines generated liquidus color and temperature axis scaling
            gas_temp=gas_temp  # Include gas temperature if applicable
        )

        return fig

    def _generate_nelder_mead_path_plot(self) -> plt.Figure:
        """
        Generates a visualization of the Nelder-Mead optimization path.

        This method plots the progression of the Nelder-Mead optimization algorithm in the parameter
        space, using triangles to represent each iteration and color coding for iterations and errors.
        To use, BinaryLiquid object field 'nmpath' must be initialized.

        Returns:
            plt.Figure: The generated plot figure.
        """
        if self._bl.nmpath is None:
            raise ValueError("Underlying BinaryLiquid object has no Nelder-Mead path! Generate using `fit_parameters`")
        fig, ax = plt.subplots(figsize=(8, 5))
        num_iters = self._bl.nmpath.shape[2]
        fig.suptitle(
            f"{self._bl.sys_name} 2-Parameter Nelder-Mead Path",
            fontweight='semibold',
            fontsize=14
        )

        # Determine the range of temperature deviations (tdev_range)
        tdev_range = [None, None]
        for i in range(num_iters):
            path_i = self._bl.nmpath[:, :, i]  # Extract iteration data
            t_devs = [num for num in path_i[:, -1:] if num != float('inf')]
            if t_devs:
                tdev_range[0] = min(t_devs) if tdev_range[0] is None else min(tdev_range[0], min(t_devs))
                tdev_range[1] = max(t_devs) if tdev_range[1] is None else max(tdev_range[1], max(t_devs))

        # Triangle color mapping (iteration-based)
        sm1 = cm.ScalarMappable(cmap=cm.get_cmap('winter'), norm=LogNorm(vmin=1, vmax=num_iters))
        triangle_colors = sm1.to_rgba(np.arange(1, num_iters + 1, 1))
        ticks = [2 ** exp for exp in np.arange(0, math.ceil(np.log2(num_iters)), 1)]
        cbar1 = fig.colorbar(sm1, ax=ax, aspect=14)
        cbar1.minorticks_off()
        cbar1.set_ticks(ticks)
        cbar1.set_ticklabels(ticks)
        cbar1.set_label('Nelder-Mead Iteration', style='italic', labelpad=8, fontsize=12)

        # Marker color mapping (temperature deviation-based)
        sm2 = cm.ScalarMappable(cmap=cm.get_cmap('autumn'), norm=plt.Normalize(tdev_range[0], tdev_range[1]))
        marker_colors = sm2.to_rgba(np.arange(tdev_range[0], tdev_range[1], 1))
        cbar2 = fig.colorbar(sm2, ax=ax, aspect=14)
        cbar2.set_label(
            f"MAE From MPDS Liquidus ({chr(176)}C)",
            style='italic',
            labelpad=10,
            fontsize=12
        )

        plotted_points = []

        for i in range(num_iters):
            path_i = self._bl.nmpath[:, :, i]  # Extract iteration data
            triangle = path_i[:, :-1]  # Extract triangle vertices
            t_devs = path_i[:, -1:]  # Extract temperature deviations

            # Plot triangles connecting vertices
            coordinates = [triangle[j, :] for j in range(triangle.shape[0])]
            pair_combinations = list(combinations(coordinates, 2))
            for combo in pair_combinations:
                line = np.array(combo)
                ax.plot(
                    line[:, 0], line[:, 1],
                    color=triangle_colors[i],
                    linewidth=(2 - 1.7 * (i / num_iters)),
                    zorder=0
                )

            # Plot markers at triangle vertices
            for point, t_dev in zip(triangle, t_devs):
                if list(point) in plotted_points:
                    continue
                if t_dev != float('inf'):
                    c_ind = int(t_dev - tdev_range[0])
                    marker_color = marker_colors[c_ind]
                    ax.scatter(
                        point[0],
                        point[1],
                        s=(55 - 54.7 * (i / num_iters)),
                        color=marker_color,
                        marker='^',
                        edgecolor='black',
                        linewidth=0.3,
                        zorder=1
                    )
                else:
                    ax.scatter(
                        point[0],
                        point[1],
                        s=(45 - 44.7 * (i / num_iters)),
                        color='black',
                        label='Incalculable MAE',
                        marker='^',
                        zorder=1
                    )
                plotted_points.append(list(point))

        # Add legend and adjust axis labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

        # Adjust axis limits for better scaling
        ax.autoscale()
        ly, uy = ax.get_ylim()
        ax.set_ylim((uy + ly) / 2 - (uy - ly) / 2 * 1.1, (uy + ly) / 2 + (uy - ly) / 2 * 1.1)
        lx, ux = ax.get_xlim()
        ax.set_xlim((ux + lx) / 2 - (ux - lx) / 2 * 1.1, (ux + lx) / 2 + (ux - lx) / 2 * 1.1)

        ax.set_xlabel('L0_b', fontweight='semibold', fontsize=12)
        ax.set_ylabel('L1_b', fontweight='semibold', fontsize=12)
        fig.tight_layout()

        return fig
