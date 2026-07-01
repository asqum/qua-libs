# %%
"""
Measure the coupling strength to position the current decouple offset.
* Make sure the poles will be close to the edge.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import extract_dominant_frequencies
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse
from scipy.fft import fft
import xarray as xr
from quam_libs.components.gates.two_qubit_gates import SWAP_Coupler_Gate
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# %% {Node_parameters}
qubit_pair_indexes = [1] 
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] =  ["coupler_q%s_q%s"%(i,i+1) for i in qubit_pair_indexes]
    num_averages: int = 880
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None

    # coupler_q1_q2:
    coupler_flux_min : float = -0.1
    coupler_flux_max : float = 0.3
    # coupler_q2_q3:
    # coupler_flux_min : float = -0.1 
    # coupler_flux_max : float = 0.3
    # q3_q4:
    # coupler_flux_min : float = 0.100 
    # coupler_flux_max : float = 0.170  
    # coupler_q4_q5:
    # coupler_flux_min : float = 0.269
    # coupler_flux_max : float = 0.450

    coupler_flux_step : float = 0.002
    idle_time_min : int = 16
    idle_time_max : int = 616
    idle_time_step : int = 4
    use_state_discrimination: bool = True
    operation:Literal["CZ","iSWAP"] = "CZ"
    

node = QualibrationNode(
    name="62_coupler_interaction_strength_calibration", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
# if any([qp.q1.z is None or qp.q2.z is None for qp in qubit_pairs]):
#     warnings.warn("Found qubit pairs without a flux line. Skipping")

num_qubit_pairs = len(qubit_pairs)

for qp in qubit_pairs:
    print("control: %s, target: %s" %(qp.qubit_control.name, qp.qubit_target.name))

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %%

####################
# Helper functions #
####################


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
cz:bool = True if node.parameters.operation.lower() == "cz" else False
flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
fluxes_coupler = np.arange(node.parameters.coupler_flux_min, node.parameters.coupler_flux_max, node.parameters.coupler_flux_step)
idle_times = np.arange(node.parameters.idle_time_min, node.parameters.idle_time_max, node.parameters.idle_time_step) // 4

with program() as CPhase_Oscillations:
    n = declare(int)
    flux_coupler = declare(float)
    comp_flux_qubit = declare(float)
    idle_time = declare(int)
    n_st = declare_stream()
    if node.parameters.use_state_discrimination:
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state = [declare(int) for _ in range(num_qubit_pairs)]
        state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st = [declare_stream() for _ in range(num_qubit_pairs)]
    else:
        I_control = [declare(float) for _ in range(num_qubit_pairs)]
        Q_control = [declare(float) for _ in range(num_qubit_pairs)]
        I_target = [declare(float) for _ in range(num_qubit_pairs)]
        Q_target = [declare(float) for _ in range(num_qubit_pairs)]
        I_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes("joint", qp)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)         
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(idle_time, idle_times)):
                    # reset
                    if node.parameters.reset_type == "active":
                            active_reset_simple(qp.qubit_control)
                            active_reset_simple(qp.qubit_target)
                            qp.align()
                    else:
                        wait(qp.qubit_control.thermalization_time * u.ns)
                        wait(qp.qubit_target.thermalization_time * u.ns)
                    
                    
                    if "coupler_qubit_crosstalk" in qp.extras:
                        assign(comp_flux_qubit, qp.detuning  +  qp.extras["coupler_qubit_crosstalk"] * flux_coupler )
                    else:
                        assign(comp_flux_qubit, qp.detuning)
                    qp.align()
                    
                    # setting both qubits ot the initial state
                    if cz:
                        qp.qubit_target.xy.play("x180")
                        qp.qubit_control.xy.play("x180")
                    else:
                        qp.qubit_control.xy.play("x180")

                    qp.qubit_control.z.wait(qp.qubit_control.xy.operations["x180"].length >> 2)
                    qp.coupler.wait(qp.qubit_control.xy.operations["x180"].length >> 2)             
                    
                    # Play the flux pulse on the qubit control and coupler
                    qp.qubit_control.z.play("const", amplitude_scale = comp_flux_qubit / qp.qubit_control.z.operations["const"].amplitude, duration = idle_time)
                    qp.coupler.play("const", amplitude_scale = flux_coupler / qp.coupler.operations["const"].amplitude, duration = idle_time)
                    
                    qp.align()
                    # readout
                    if node.parameters.use_state_discrimination:
                        readout_state(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                        assign(state[i], state_control[i]*2 + state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])
                        save(state[i], state_st[i])
                    else:
                        qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                        qp.qubit_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                        save(I_control[i], I_st_control[i])
                        save(Q_control[i], Q_st_control[i])
                        save(I_target[i], I_st_target[i])
                        save(Q_target[i], Q_st_target[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state_control{i + 1}")
                state_st_target[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state_target{i + 1}")
                state_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state{i + 1}")
            else:
                I_st_control[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"Q_control{i + 1}")
                I_st_target[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(CPhase_Oscillations)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {  "idle_time": idle_times, "flux_coupler": fluxes_coupler})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}

# %%
if not node.parameters.simulate:
    ds = ds.assign_coords(idle_time = ds.idle_time * 4)
    if node.parameters.use_state_discrimination:
        ds = ds.assign({"res_sum" : ds.state_control - ds.state_target})
    else:
        ds = ds.assign({"res_sum" : ds.I_control - ds.I_target})
    flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})    
# %%
if not node.parameters.simulate:
    # Add the dominant frequencies to the dataset
    ds['dominant_frequency'] = extract_dominant_frequencies(ds.res_sum)
    ds.dominant_frequency.attrs['units'] = 'GHz'


    # %%
    # Plot the dominant frequencies
    # Find the values of flux_coupler_full for which the dominant frequencies are max and min
    interaction_max = (ds.dominant_frequency * (ds.dominant_frequency<0.04)).max(dim='flux_coupler')
    coupler_flux_pulse = ds.flux_coupler.isel(flux_coupler=(ds.dominant_frequency * (ds.dominant_frequency<0.04)).argmax(dim='flux_coupler'))
    coupler_flux_min = ds.flux_coupler_full.isel(flux_coupler=ds.dominant_frequency.argmin(dim='flux_coupler'))

# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):     
        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_control.sel(qubit=qp['qubit'])
        else:
            values_to_plot = ds.Q_control.sel(qubit=qp['qubit'])
        values_to_plot.plot(ax = ax, cmap = 'viridis', y = 'idle_time', x = 'flux_coupler')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
    grid.fig.suptitle('I Control')
    plt.tight_layout()
    plt.show()
    node.results['figure_I_control'] = grid.fig
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_target.sel(qubit=qp['qubit'])
        else:
            values_to_plot = ds.Q_target.sel(qubit=qp['qubit'])
        values_to_plot.plot(ax = ax, cmap = 'viridis', y = 'idle_time', x = 'flux_coupler')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
    grid.fig.suptitle('I Target')
    plt.tight_layout()
    plt.show()
    node.results['figure_I_target'] = grid.fig
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        (1e3*ds.dominant_frequency.sel(qubit=qp['qubit'])).plot(ax = ax, marker = '.', ls = 'None', x = 'flux_coupler')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.axvline(x = qubit_pair.coupler.decouple_offset, color = 'black')
        ax.axvline(x = coupler_flux_pulse.sel(qubit=qp['qubit']), color = 'red', lw = 0.5, ls = '--')
        ax.axvline(x = coupler_flux_min.sel(qubit=qp['qubit']) - qubit_pair.coupler.decouple_offset, color = 'green', lw = 0.5, ls = '--')
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        ax.set_xlabel('Flux Coupler')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_yscale('log')
    grid.fig.suptitle('Dominant Frequency')
    plt.tight_layout()
    plt.show()
    node.results['figure_dominant_frequency'] = grid.fig

# %% {Test only - get the coupling strength from the current decouple_offset}
for ax, qp in grid_iter(grid):
    value = ds['dominant_frequency'].interp(flux_coupler=qubit_pair.coupler.decouple_offset).item()
    print(f"interpolated dominant_frequency at flux={qubit_pair.coupler.decouple_offset}: {value*1e3} MHz")

# %% {Test only - get the decouple offset from a desired coupling strength}
desired_coupling_strength_inMHz:float = 6.849
for ax, qp in grid_iter(grid):
    da = ds['dominant_frequency'].sel(qubit=qp['qubit'])
    idx = abs(da - desired_coupling_strength_inMHz*1e-3).argmin()

    flux_value = da.flux_coupler[idx].item()
    freq_value = da[idx].item()

    print("target freq =", desired_coupling_strength_inMHz)
    print("nearest dominant_frequency =", freq_value)
    print("corresponding flux_coupler =", flux_value)

# %% {Test }

desired_position:float|None = -0.24

grid = QubitPairGrid(grid_names, qubit_pair_names,size=7)  
for ax, qp in grid_iter(grid):
# Extract your data (1-D arrays)
    da = ds['dominant_frequency'].sel(qubit=qp['qubit'])
    x = da.flux_coupler.values
    y = da.values
    logy = np.log10(y + 1e-12)

    # --- key settings, might hardcoed ---
    peak_definitions = {"deviaton_percentage":1, "prominence_factor":1/8, "pts_between_peaks_factor":0.15, "boundary_bins":200}


    def extract_upper_envelope(x_data, y_data, bins=100):
        """
        Seperate x-axis get local maximums
        
        Args:
        x_data, y_data: Raw data,
        bins (int): The zone numbers we seperate x-axis, the more bins the more zones so that we can get a smooth boundaries.
        
        Return:
        x_env, y_env: The local zone boundaries
        """
        if len(x_data) == 0:
            return np.array([]), np.array([])
            
        x_min, x_max = np.min(x_data), np.max(x_data)
        bin_edges = np.linspace(x_min, x_max, bins + 1)
        
        x_env, y_env = [], []
        for i in range(bins):
            
            mask = (x_data >= bin_edges[i]) & (x_data < bin_edges[i+1])
            
            if np.any(mask):

                x_env.append((bin_edges[i] + bin_edges[i+1]) / 2) 
                y_env.append(np.max(y_data[mask]))
                
        return np.array(x_env), np.array(y_env)

    def estimate_period_by_valley_distance(x_data, y_data, peak_definitions):
        
        x_env, y_env_raw = extract_upper_envelope(x_data, y_data, bins=peak_definitions["boundary_bins"]) 
        
        if len(x_env) < 2:
            print("Error: insufficient data points on x-axis !")
            return None
        
        # Optional: smooth process
        y_env = gaussian_filter1d(y_env_raw, sigma=1) 
        
        # 2. invert Y to make velley to peak
        y_inverted = -y_env    
        
        # --- Key settings to find the velley ---
        y_range = np.max(y_env) - np.min(y_env)
        
        H_inverted = np.mean(y_inverted)  -  np.std(y_inverted)*peak_definitions['deviaton_percentage'] # expected height
        P = y_range * peak_definitions['prominence_factor']                  # prominence
        D_min_points = int(len(x_env) * peak_definitions['pts_between_peaks_factor']) # expected pts in a period
        
        
        # 3. Search velley
        valley_indices, properties = find_peaks(
            y_inverted, 
            height=H_inverted,           
            prominence=P,  
            distance=D_min_points 
        )
        
        if len(valley_indices) < 2:
            print("Velley Searching Error: We need at least 2 velleys but got less.")
            return None
        elif len(valley_indices) > 2:
            target_poles = []
            for ijk in valley_indices:
                valley_x_value = x_env[ijk]
                target_poles.append({"x_idx":ijk, "edge_distance":np.min(np.array([abs(valley_x_value-np.min(x_data)), abs(valley_x_value-np.max(x_data))]))})
                
            sorted_list = sorted(target_poles, key=lambda item: item['edge_distance'])[:2]
            valley_indices = [d['x_idx'] for d in sorted_list]          

        # 4. Calculate period
        valley_x_values = x_env[valley_indices]
        valley_distances = np.diff(valley_x_values)

        period_T = np.mean(valley_distances)
        std_dev = np.std(valley_distances)

        print("-" * 30)
        print(f"Velley coordinates on x-axis: {valley_x_values}")
        print(f"Distance between velleys: {valley_distances}")
        print(f"Period: {period_T:.4f}")
        print(f"Period deviation: {std_dev:.4f}")
        print("-" * 30)
        
        return abs(period_T), valley_x_values
    
    def estimate_period_by_peak_distance(x_data, y_data, peak_definitions):
        """
        Extract peaks to analysis the period
        """
        # increase bins can smoothen the boundaries
        x_env, y_env = extract_upper_envelope(x_data, y_data, bins=peak_definitions["boundary_bins"]) 
        
        if len(x_env) < 2:
            print("Error: Insufficient boundaries on x-axis !")
            return None

    
        y_range = np.max(y_env) - np.min(y_env)

        # 1. height
        H = np.mean(y_env) - peak_definitions["deviaton_percentage"] * np.std(y_env)
        
        # 2. prominence 
        P = y_range * peak_definitions["prominence_factor"]
        
        # 3. data pts between peaks
        D_min_points = int(len(x_env) * peak_definitions["pts_between_peaks_factor"]) 
        
        
        
        # 2. Match local maximums (peaks)
        peak_indices, properties = find_peaks(
            y_env, 
            height=H,           
            prominence=P,  
            distance=D_min_points 
        )
        
        
        if len(peak_indices) < 2:
            print("Can not match at least 2 peaks ... ")
            
            # only try on heiight and distance
            peak_indices, _ = find_peaks(
                y_env, 
                height=H,           
                distance=D_min_points 
            )
            
            if len(peak_indices) < 2:
                print("Error: Can NOT match at least 2 peaks.")
                return None

        peak_x_values = x_env[peak_indices]

        # 3. 計算間距
        peak_distances = np.diff(peak_x_values)

        # 4. 確定週期
        period_T = np.mean(peak_distances)
        std_dev = np.std(peak_distances)

        print("-" * 30)
        print(f"Poles: {peak_x_values}")
        print(f"distance between poles: {peak_distances}")
        print(f"Period: {period_T:.4f}")
        print(f"Period deviation: {std_dev:.4f}")
        print("-" * 30)
        
        return abs(period_T), peak_x_values
    
    T, poles = estimate_period_by_valley_distance(np.array(x),np.array(logy),peak_definitions)
    if T is None:
        T, poles = estimate_period_by_peak_distance(np.array(x),np.array(logy),peak_definitions)
    
    symmetric_axis = np.mean(poles)
    if desired_position is not None:
        desired_decouple_offset = desired_position*T + symmetric_axis


    
    (1e3*ds.dominant_frequency.sel(qubit=qp['qubit'])).plot(ax = ax, marker = '.', ls = 'None', x = 'flux_coupler')
    qubit_pair = machine.qubit_pairs[qp['qubit']]
    ax.axvline(x = qubit_pair.coupler.decouple_offset, color = 'black',label='Current_Decouple_Offset')
    ax.axvline(x = symmetric_axis, color = 'red',label="Symmetric Axis")
    if desired_position is not None:
        ax.axvline(x = desired_decouple_offset, color = 'cyan',label='Desired_Decouple_Offset')
    ax.axvline(x = np.min(poles), color = 'red', ls = '--',label="A period")
    ax.axvline(x = np.max(poles), color = 'red', ls = '--')
    if desired_position is None:
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}, about {round((qubit_pair.coupler.decouple_offset-symmetric_axis)*100/T,2)} % period to Symmetirc Axis", fontsize = 10)
    else:
        ax.set_title(f"{qp['qubit']}, desired coupler set point: {desired_decouple_offset} V")
    ax.set_xlabel('Flux Coupler')
    ax.set_ylabel('Frequency (MHz)')
    
    ax.set_yscale('log')
grid.fig.suptitle(f'Decouple offset positioning by coupling strength ({node.parameters.operation})')
plt.legend()
plt.tight_layout()
plt.show()
node.results['PositioningDecoupleOffset'] = grid.fig


# %% {Update state}
if not node.parameters.simulate:
    if desired_position is not None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                qp.coupler.decouple_offset = desired_decouple_offset


# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.save()
# %%
