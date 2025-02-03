"""
Authors: Abrar Rauf, Joshua Willwerth
Date: January 31, 2025
Description: This script takes the phase energy data in the form of enthalpy (H), entropy (S) and composition (X)
and performs transformations to composition-temperature (TX) phase diagrams with well-defined coexistence boundaries
GitHub: https://github.com/AbrarRauf
ORCID: https://orcid.org/0000-0001-5205-0075
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
from collections import defaultdict

class HSX:
    """Handles enthalpy (H), entropy (S), and composition (X) transformations for TX phase diagrams."""

    def __init__(self, data_dict: dict, conds: list[float]):
        """Initializes the HSX instance with provided phase data and conditions."""
        self.phases = data_dict['phases']
        self.comps = data_dict['comps']
        self.conds = conds
        self.df = pd.DataFrame(data_dict['data'])
        self.phase_color_remap = {}
        self.simplices = []
        self.final_phases = []
        self.df_tx = pd.DataFrame()
    
        # Data scaling
        s_scaler = 100
        h_scaler = 10000
        self.df.columns = ['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]', 'Phase']
        self.df['S [J/mol/K]'] /= s_scaler
        self.df['H [J/mol]'] /= h_scaler

        # Color Mapping
        color_array = px.colors.qualitative.Pastel
        inter_phases = [p for p in self.phases if p != 'L']
        self.color_map = {phase: color for phase, color in zip(inter_phases, color_array)}
        self.color_map['L'] = 'cornflowerblue'
        self.df['Colors'] = self.df['Phase'].map(self.color_map)

        # Data extraction for convex hull calculation
        df_inter = self.df[self.df['Phase'] != 'L']
        df_liq = self.df[self.df['Phase'] == 'L']
        self.liq_points = df_liq[['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]']].to_numpy()
        self.inter_points = df_inter[['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]']].to_numpy()
        self.points = self.df[['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]']].to_numpy()
        self.scaler = h_scaler / s_scaler

    def hull(self) -> np.ndarray:
        """Computes the lower convex hull of an N-dimensional Xi-S-H space."""
        dim = self.points.shape[1]
        
        # Initialize bounds for Xi
        x_list = [[1 if j == i - 1 else 0 for j in range(dim - 2)] for i in range(dim - 1)]
        x_list[0] = [0] * (dim - 2)
        
        # Compute S and H bounds
        s_min, s_extr = np.min(self.points[:, -2]), np.max([self.liq_points[0, -2], self.liq_points[-1, -2]])
        h_max = np.max(self.points[:, -1])
        upper_bound = 4 * h_max

        # Generate fictitious points
        liq_fict_coords = np.column_stack((self.liq_points[:, 0], self.liq_points[:, 1],
                                            np.full(len(self.liq_points), upper_bound)))
        fict_coords = np.vstack([
            np.append(x_list[i], [s_min, upper_bound]) for i in range(dim - 1)
        ] + [
            np.append(x_list[i], [s_extr, upper_bound]) for i in range(dim - 1)
        ])
        
        fict_points = np.vstack((fict_coords, liq_fict_coords))
        new_points = np.vstack((self.points, fict_points))
        
        # Compute convex hull
        new_hull = ConvexHull(new_points, qhull_options="Qt i")
        
        def check_common_rows(arr1: np.ndarray, arr2: np.ndarray) -> bool:
            """Checks if any row in arr1 exists in arr2."""
            return any((arr1 == row).all(axis=1).any() for row in arr2)
        
        # Filter hull simplices
        lower_hull_filter1 = [s for s in new_hull.simplices if not check_common_rows(new_points[s], fict_points)]
        lower_hull_filter2 = [s for s in lower_hull_filter1 
                              if sum((v == im).all() for v in self.points[s] for im in self.inter_points) < 3]
        self.simplices = np.array(lower_hull_filter2)
        return self.simplices

    def compute_tx(self) -> tuple[pd.DataFrame, list, np.ndarray, np.ndarray]:
        """Computes the TX phase diagram transformation."""
        self.simplices = self.hull()
        temps, valid_simplices, new_phases = [], [], []

        for simplex in self.simplices:
            A, B, C = self.points[simplex]
            n = np.cross(B - A, C - A).astype(float)
            T = (-n[1] / n[2]) * self.scaler
            if not np.isnan(T):
                temps.append(T)
                valid_simplices.append(simplex)
                new_phases.append([self.df.loc[simplex[i], 'Phase'] for i in range(3)])
        
        temps = np.array(temps)
        self.final_phases = np.array(new_phases)

        data = [
            [self.points[vertex][0], temps[i], labels[j], self.color_map.get(labels[j])]
            for i, simplex in enumerate(valid_simplices)
            for j, vertex in enumerate(simplex)
            for labels in [self.final_phases[i]]  # Extract labels once per simplex
        ]
        
        self.df_tx = pd.DataFrame(data, columns=['x', 't', 'label', 'color'])
        phase_remap = defaultdict(list)
        for entry in data:
            phase_remap[entry[2]].append([entry[0], entry[1]])
        self.phase_color_remap = dict(zip(self.df_tx['label'], self.df_tx['color']))
        return self.df_tx, self.final_phases, np.array(valid_simplices), temps
    

    def liquidus_invariants(self) -> tuple[dict, list, dict]:
        """Extracts eutectic, peritectic, and congruent melting points from the computed TX phase diagram."""
        self.df_tx, self.final_phases, final_simplices, final_temps = self.compute_tx()
        self.df_tx['t'] -= 273.15
        final_temps -= 273.15

        compositions = np.array([[vertex[0] for vertex in self.points[simplex]] for simplex in final_simplices])

        combined_list = []
        for i in range(len(compositions)):
            row_dict = {}
            for j in range(len(compositions[i])):
                row_dict[compositions[i][j]] = self.final_phases[i][j]
            if len(row_dict) == 2:
                for key in row_dict.keys():
                    if key == 0.0:
                        row_dict[key] = self.comps[0]
                    elif key == 1.0:
                        row_dict[key] = self.comps[1]
            combined_list.append([final_temps[i], row_dict])

        int_phases = [p for p in self.phases if p not in [self.comps[0], self.comps[1], 'L']]

        inv_points = {'Eutectics': [], 'Peritectics': [], 'Congruent Melting': [], 'Misc Gaps': []}
        peritectic_phases, non_triples = [], []

        for temp, comb_dict in combined_list:

            sorted_dict = dict(sorted(comb_dict.items()))
            comp, phase = list(sorted_dict.keys()), list(sorted_dict.values())

            if len(comp) == 3:
                if len(set(phase)) == 3:
                    if phase[1] == 'L':
                        inv_points['Eutectics'].append([temp, comp[1], comp, phase])
                    else:
                        inv_points['Peritectics'].append([temp, comp[1], comp, phase])
                        peritectic_phases.append(phase[1])
                else:
                    non_triples.append([temp, comp, phase])

        congruents_init = []
        for temp, comp, phase in non_triples:
            if phase[0] == 'L' and phase[2] != 'L':
                comp_diff = abs(comp[0] - comp[1])
                if comp_diff > 0.012:
                    inv_points['Misc Gaps'].append([temp, comp[1], comp, phase])
            elif phase[0] != 'L':
                comp_diff = abs(comp[1] - comp[2])
                if comp_diff > 0.012:
                    inv_points['Misc Gaps'].append([temp, comp[1], comp, phase])
            phase = [p for p in phase if p != 'L']
            if phase and phase[0] in int_phases and phase[0] not in peritectic_phases:
                congruents_init.append([temp, comp[0], comp, phase])

        grouped_data = defaultdict(list)
        for entry in congruents_init:
            grouped_data[entry[3][0]].append(entry)

        inv_points['Congruent Melting'] = [max(entries, key=lambda x: x[0]) for entries in grouped_data.values()]
        count_dict = {key: len(value) for key, value in inv_points.items()}

        return inv_points, combined_list, count_dict
    
    def plot_tx(self, pred: bool = False, digitized_liquidus: list = None, gas_temp: int | float = None) -> go.Figure:
        """Plots the binary phase diagram from computed phase boundaries and invariant points."""
        liq_inv = self.liquidus_invariants()
        inv_points, combined_list = liq_inv[:2]
        
        new_tx = []
        for comb in combined_list:
            temp = comb[0]
            comb_dict = comb[1]
            sorted_dict = {k: v for k, v in sorted(comb_dict.items())}
            comp = list(sorted_dict.keys())
            phase = list(sorted_dict.values())
            if len(comp) == 2:
                new_tx.append([temp, comp, phase])
            else:
                if phase[0] == 'L' and phase[1] == 'L':  # Liquid-Liquid-Solid or Liquid-Liquid-Liquid
                    comp.pop(0)
                    phase.pop(0)
                    new_tx.append([temp, comp, phase])

                elif phase[1] == 'L' and phase[2] == 'L':  # Solid-Liquid-Liquid
                    comp.pop(2)
                    phase.pop(2)
                    new_tx.append([temp, comp, phase])
                else:
                    new_tx.append([temp, comp, phase])
        
        temp_df_tx = [[x, t, phase[j], self.color_map.get(phase[j])] 
                      for t, comp, phase in new_tx for j, x in enumerate(comp)]
        new_df_tx = pd.DataFrame(temp_df_tx, columns=['x', 't', 'label', 'color'])
        new_df_tx['x'] *= 100
        
        liq_df = self.df_tx[self.df_tx['label'] == 'L'].copy()
        liq_df['x'] *= 100
        liq_df.sort_values(by=['x', 't'], inplace=True)
        liq_df.drop_duplicates(subset='x', keep='first', inplace=True)
        
        solid_df = new_df_tx[new_df_tx['label'] != 'L']
        element_df = pd.concat([
            solid_df[solid_df['x'] == 0].nlargest(1, 't', keep='last'),
            solid_df[solid_df['x'] == 100].nlargest(1, 't', keep='last')
        ])
        element_df['label'] = 'L'
        liq_df = pd.concat([liq_df, element_df]).sort_values(by='x')
        
        lhs_tm, rhs_tm = liq_df.iloc[0]['t'], liq_df.iloc[-1]['t']
        max_liq, min_liq = liq_df['t'].max(), liq_df['t'].min()

        if not pred or digitized_liquidus:
            self.conds[1] = max(min(self.conds[1] * 2 + 100, max_liq), self.conds[1])
        else:
            self.conds = [min_liq - 0.1 * (max_liq - min_liq), max_liq]
        
        trange = self.conds[1] - self.conds[0]
        yfactor = 0.36 if pred else 0.30
        if lhs_tm < rhs_tm:
            if lhs_tm + 0.3 * trange < self.conds[1]:  # higest temp at least 30% of range above lower tm
                self.conds[1] += 0.1 * trange
            else:
                self.conds[1] = lhs_tm + yfactor * trange
            legend = {'yanchor': 'top', 'y': 0.99, 'xanchor': 'left', 'x': 0.01, 'font': dict(size=18)}
        else:
            if rhs_tm + 0.3 * trange < self.conds[1]:  # higest temp at least 30% of range above lower tm
                self.conds[1] += 0.1 * trange
            else:
                self.conds[1] = rhs_tm + yfactor * trange
            legend = {'yanchor': 'top', 'y': 0.99, 'xanchor': 'right', 'x': 0.99, 'font': dict(size=18)}

        fig = go.Figure()
        if digitized_liquidus:
            fig.add_trace(
                go.Scatter(x=[p[0] * 100 for p in digitized_liquidus], y=[p[1] - 273.15 for p in digitized_liquidus],
                            mode='lines', line=dict(color='#B82E2E', dash='dash')))
        
        solid_phases = [p for p in self.phases if p not in [self.comps[0], self.comps[1], 'L']]
        for phase in solid_phases:
            phase_df = solid_df[solid_df['label'] == phase]
            if phase_df.empty:
                continue

            phase_decomp_temp = phase_df['t'].max()
            if phase_decomp_temp - 0.1 * trange < self.conds[0]:
                if phase_decomp_temp - 0.1 * trange < -273.15:
                    continue
                self.conds[0] = phase_decomp_temp - 0.1 * trange

        solid_comp_list = []
        idx_tracker = 0
        for phase in solid_phases:
            phase_df = solid_df[solid_df['label'] == phase]
            if phase_df.empty:
                continue

            solid_comp = phase_df['x'].values
            solid_comp = solid_comp[0] 
            solid_comp_list.append(solid_comp)

            new_row_df = pd.DataFrame(
                [{'x': solid_comp, 't': -273.15, 'label': phase, 'color': self.color_map.get(phase)}],
                  columns=phase_df.columns)
            phase_df = pd.concat([phase_df, new_row_df], ignore_index=True)
            line = px.line(phase_df, x='x', y='t', color='label', color_discrete_map=self.phase_color_remap)

            fig.add_trace(line.data[0])
            if idx_tracker == 0:
                if solid_comp < 5:
                    fig.add_annotation(
                        x=solid_comp + 1.5,
                        y=self.conds[0],
                        yanchor='bottom',
                        text=phase,
                        showarrow=False,
                        textangle=-90,
                        borderpad=5,
                        font=dict(
                            size=18,
                            color='black'
                        ))
                else:
                    fig.add_annotation(
                        x=solid_comp - 2.5,
                        y=self.conds[0],
                        yanchor='bottom',
                        text=phase,
                        showarrow=False,
                        textangle=-90,
                        borderpad=5,
                        font=dict(
                            size=18,
                            color='black'
                        ))
            else:
                if solid_comp < 5:
                    fig.add_annotation(
                        x=solid_comp + 1.5,
                        y=self.conds[0],
                        yanchor='bottom',
                        text=phase,
                        showarrow=False,
                        textangle=-90,
                        borderpad=5,
                        font=dict(
                            size=18,
                            color='black'
                        ))
                elif solid_comp_list[idx_tracker] - solid_comp_list[idx_tracker - 1] < 5:
                    fig.add_annotation(
                        x=solid_comp + 1.5,
                        y=self.conds[0],
                        yanchor='bottom',
                        text=phase,
                        showarrow=False,
                        textangle=-90,
                        borderpad=5,
                        font=dict(
                            size=18,
                            color='black'
                        ))
                else:
                    fig.add_annotation(
                        x=solid_comp - 2.5,
                        y=self.conds[0],
                        yanchor='bottom',
                        text=phase,
                        showarrow=False,
                        textangle=-90,
                        borderpad=5,
                        font=dict(
                            size=18,
                            color='black'
                        ))

        for key in inv_points.keys():
            if key in ['Eutectics', 'Peritectics', 'Misc Gaps']:
                for temp, _, comps, _ in inv_points[key]:
                    comps = [x * 100 for x in comps]
                    temps = [temp] * 3
                    line = px.line(x=comps, y=temps)
                    line.update_traces(line=dict(color='Silver'))
                    fig.add_trace(line.data[0])
        
        if pred:
            self.phase_color_remap['L'] = '#117733'
            fig.add_trace(px.line(liq_df, x='x', y='t', color='label',
                                  color_discrete_map=self.phase_color_remap).data[0])
        else:
            fig.add_trace(px.line(liq_df, x='x', y='t', color='label',
                                  color_discrete_map=self.phase_color_remap).data[0])
        fig.update_traces(line=dict(width=4), showlegend=False)
        if digitized_liquidus:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#B82E2E', dash='dash'),
                                     name='Digitized Liquidus', showlegend=True))
        if pred:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='#117733'),
                                     name='Predicted Liquidus', showlegend=True))
        else:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='cornflowerblue'),
                                     name='Fitted Liquidus', showlegend=True))
        if gas_temp and gas_temp - 273.15 < min(liq_df['t'].max(), self.conds[1]) and not digitized_liquidus:
            fig.add_trace(go.Scatter(x=[0, 100], y=[gas_temp - 273.15, gas_temp - 273.15],
                                     mode='lines', line=dict(color='#FFAE43', dash='dash'),
                                     name='Gas Phase Forms', showlegend=True))
        fig.update_layout(
            title=f"{self.comps[0]}-{self.comps[1]} {'Predicted' if pred else 'Fitted'} Binary Phase Diagram",
            xaxis=dict(range=[0, 100], title=f'{self.comps[1]} (at. %)'),
            yaxis=dict(range=[max(self.conds[0], -273), self.conds[1]], title='T (°C)', ticksuffix=" "),
            width=960,
            height=700,
            plot_bgcolor='white',
            font_size=22,
            showlegend=True,
            legend=legend
        )
        fig.update_xaxes(
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor='black',
            linewidth=2,
            tickcolor='black',
            tickformat=".0f"
        )
        fig.update_yaxes(
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor='black',
            linewidth=2,
            tickcolor='black'
        )
        fig.add_annotation(
            x=50,
            y=self.conds[1] - 0.08 * trange,
            text='L',
            showarrow=False,
            font=dict(
                size=18,
                color='black'
            )
        )

        return fig

    def get_phase_points(self) -> dict:
        """Extracts phase boundary points from the HSX object and converts to a list of dictionaries for BinaryLiquid"""

        def remove_duplicates(coordinates):
            """Removes duplicate x-coordinates while preserving order."""
            coordinates.sort(key=lambda x: (x[0], x[1]))
            filtered_coordinates = []
            current_x = None
            for coord in coordinates:
                if coord[0] != current_x:
                    filtered_coordinates.append(coord)
                    current_x = coord[0]
            return filtered_coordinates
        
        df_tx = self.compute_tx()[0]
        phase_points = {phase: df_tx[df_tx['label'] == phase][['x', 't']].values.tolist() for phase in self.phases}
        phase_points['L'] = remove_duplicates(phase_points['L'])
        return phase_points
