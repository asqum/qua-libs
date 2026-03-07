"""
        T1 MEASUREMENT
The sequence consists in putting the qubit in the excited stated by playing the x180 pulse and measuring the resonator
after a varying time. The qubit T1 is extracted by fitting the exponential decay of the measured quadratures.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the qubit T1 in the state.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state, active_reset_simple
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):
    coupler:int = 1
    num_averages: int = 300
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 90016
    wait_time_step_in_ns: int = 900
    c_bias_max:float = 0.05
    c_bias_min:float = -0.05
    bias_pts:int = 2
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent", "arbitrary"] = "independent"
    reset_type: Literal["active", "thermal"] = "thermal"
    use_state_discrimination: bool = True
    timeout: int = 100
    load_data_id: Optional[int] = None
    simulation_time:int|None = 5000 # Simulate if it was given

node = QualibrationNode(name="05_T1", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP

qmm = machine.connect()

# Get coupler
qp = machine.qubit_pairs["coupler_q%s_q%s"%(node.parameters.coupler,node.parameters.coupler+1)]
# Get qubits
qubits = [machine.qubits[f"q{node.parameters.coupler}"], machine.qubits[f"q{node.parameters.coupler+1}"]]
num_qubits = len(qubits)
print("qubits", [q.name for q in qubits])

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)
flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary  # 'independent' or 'joint'
fluxes_coupler = np.linspace(node.parameters.c_bias_min-qp.coupler.decouple_offset, node.parameters.c_bias_max-qp.coupler.decouple_offset, node.parameters.bias_pts) # to absolute voltage

with program() as coupler_t1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    flux_coupler = declare(float)
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

    # Do NOT apply any bias for simulation purposes    
    if node.parameters.simulation_time is None:
        machine.apply_all_couplers_to_min()
        for i, qubit in enumerate(qubits):

            # Bring the active qubits to the desired frequency point
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            qubit.z.settle()

    
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        align()
        with for_(*from_array(flux_coupler, fluxes_coupler)):
            align()
            with for_(*from_array(t, idle_times)):
                for i, qubit in enumerate(qubits):
                    if node.parameters.reset_type == "active":
                        # active_reset(qubit, "readout")
                        active_reset_simple(qubit, "readout")
                    else:
                        if node.parameters.simulation_time is None:
                            qubit.wait(qubit.thermalization_time * u.ns)
                            
                        else:
                            qubit.wait(16 * u.ns)
                
                align()    
                # start play coupler bias  
                for i, qubit in enumerate(qubits):  
                    qubit.xy.play("x180")   
                
                align()
                # May need to insert XY-Z delay here
                qp.coupler.play(
                    "const", 
                    amplitude_scale = flux_coupler / qp.coupler.operations["const"].amplitude, 
                    duration = t
                )
                # wait(t)

                align()
                # qp.coupler.wait(16*u.ns)
                # align()
                for i, qubit in enumerate(qubits):
                    # Measure the state of the resonators
                    if node.parameters.use_state_discrimination:
                        readout_state(qubit, state[i])
                        save(state[i], state_st[i])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                align()
    
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulation_time is not None:
    if node.parameters.load_data_id is None:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=node.parameters.simulation_time//4)  # In clock cycles = 4ns
        job = qmm.simulate(config, coupler_t1, simulation_config)
        # Get the simulated samples and plot them for all controllers

        samples = job.get_simulated_samples()
        samples.con1.plot()
        node.results = {"figure": plt.gcf()}
        wf_report = job.get_simulated_waveform_report()
        wf_report.create_plot(samples, plot=True, save_path=None)
        node.machine = machine
        node.save()
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
        node.results = {"ds": ds}
else:
    if node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(coupler_t1)
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                # Fetch results
                n = results.fetch_all()[0]
                # Progress bar
                progress_counter(n, n_avg, start_time=results.start_time)

        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times, "flux_coupler": fluxes_coupler})
        # Convert IQ data into volts
        if not node.parameters.use_state_discrimination:
            ds = convert_IQ_to_V(ds, qubits)
        # Convert time into µs
        ds = ds.assign_coords(idle_time=4 * ds.idle_time / u.us)  # convert to µs
        ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    node.results = {"ds": ds}

# %% {Data_analysis}

ds = ds.assign_coords(flux_mV=ds.flux_coupler * 1e3)
fit_results = {}
# Fit the exponential decay
if node.parameters.use_state_discrimination:
    fit_data = fit_decay_exp(ds.state, "idle_time")
else:
    fit_data = fit_decay_exp(ds.I, "idle_time")
fit_data.attrs = {"long_name": "time", "units": "µs"}
# Fitted decay
fitted = decay_exp(
    ds.idle_time,
    fit_data.sel(fit_vals="a"),
    fit_data.sel(fit_vals="offset"),
    fit_data.sel(fit_vals="decay"),
)
# Decay rate and its uncertainty
decay = fit_data.sel(fit_vals="decay")
decay.attrs = {"long_name": "decay", "units": "MHz"}
decay_res = fit_data.sel(fit_vals="decay_decay")
decay_res.attrs = {"long_name": "decay", "units": "MHz"}
# T1 and its uncertainty
tau = -1 / fit_data.sel(fit_vals="decay")
tau.attrs = {"long_name": "T1", "units": "µs"}
tau_error = -tau * (np.sqrt(decay_res) / decay)
tau_error.attrs = {"long_name": "T1 error", "units": "µs"}

for q in qubits:
        fit_results[q.name] = {}
        fit_results[q.name]["T1_vs_coupler_flux"] = tau.sel(qubit=q.name).values.tolist()
        fit_results[q.name]["T1err_vs_coupler_flux"] = tau_error.sel(qubit=q.name).values.tolist()
node.results["fit_results"] = fit_results

# %% {Plotting}
# ---- RAW PLOT ----
grid = QubitGrid(ds, [q.grid_location for q in qubits])
grid.fig.set_size_inches(12, 3 * len(qubits))

for ax, qubit in grid_iter(grid):
    qname = qubit["qubit"]

    if node.parameters.use_state_discrimination:
        im = ds.sel(qubit=qname).state.plot(
            ax=ax, x="flux_mV", y="idle_time",add_colorbar=False,
            cmap="viridis"
        )
        ax.set_ylabel("Idle time (µs)")
    else:
        im = ds.sel(qubit=qname).I.plot(
            ax=ax, x="flux_mV", y="idle_time",add_colorbar=False,
            cmap="viridis"
        )
        ax.set_ylabel("Idle time (µs)")

    ax.set_title(f"{qname}")
    ax.set_xlabel("Coupler flux (mV)")
    cb = grid.fig.colorbar(im, ax=ax)
    cb.set_label(("state" if node.parameters.use_state_discrimination else "I"))


grid.fig.suptitle("Raw")
plt.tight_layout()
plt.show()

node.results["figure_raw_coupler"] = grid.fig

# ---- FIT PLOT ----
grid = QubitGrid(ds, [q.grid_location for q in qubits])
grid.fig.set_size_inches(12, 3 * len(qubits))

for ax, qubit in grid_iter(grid):
    qname = qubit["qubit"]

    im = fitted.sel(qubit=qname).plot(
        ax=ax, x="flux_mV", y="idle_time", add_colorbar=False,
            cmap="viridis"
    )

    ax.set_title(f"{qname}")
    ax.set_xlabel("Coupler flux (mV)")
    ax.set_ylabel("Idle time (µs)")
    cb = grid.fig.colorbar(im, ax=ax)
    cb.set_label("Fitted" + ("state" if node.parameters.use_state_discrimination else "I"))

grid.fig.suptitle("Fit")
plt.tight_layout()
plt.show()

node.results["figure_fit_coupler"] = grid.fig

# ---- T1 PLOT ----
grid = QubitGrid(ds, [q.grid_location for q in qubits])
grid.fig.set_size_inches(12, 3 * len(qubits))

for ax, qubit in grid_iter(grid):
    qname = qubit["qubit"]

    T1 = tau.sel(qubit=qname).values
    T1err = tau_error.sel(qubit=qname).values

    # Ignore negative/unphysical values
    mask = T1 > 0
    T1 = np.where(mask, T1, np.nan)
    T1err = np.where(mask, T1err, np.nan)

    ax.errorbar(
        ds.flux_mV.values,
        T1,
        yerr=T1err,
        fmt="o-",
        capsize=3,
    )

    ax.set_title(f"{qname}")
    ax.set_xlabel("Coupler flux (mV)")
    ax.set_ylabel("T1 (µs)")

grid.fig.suptitle("T1 vs Coupler Flux")
plt.tight_layout()
plt.show()

node.results["figure_T1_coupler"] = grid.fig


# %% {Save_results}
if node.parameters.load_data_id is None:
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %% {run all}