"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit.wait(qubit.thermalization_time * u.ns) spectroscopy, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state.
    - Update the g -> e thresholds (threshold & rus_threshold) in the state.
    - Update the confusion matrices in the state.
    - Save the current state
"""


# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Literal, Optional, List
from qualibrate import QualibrationNode
from quam_libs.components import QuAM
from qualang_tools.loops import from_array
from qualibrate import NodeParameters, QualibrationNode
from quam_libs.lib.fit import fit_oscillation, oscillation
from quam_libs.macros import qua_declaration, active_reset, split_bipolar_macro
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *


# %% {Node_parameters}


class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ['q4']
    num_runs: int = 1000
    operation_x180_or_any_90: Literal["x180", "x90", "-x90", "y90", "-y90"] = "x180"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type_thermal_or_active: Literal["active", "thermal"] = "active"
    simulate: bool = True
    max_number_rabi_pulses_per_sweep:int = 2
    simulation_duration_ns: int = 10_000
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    neg_amp_ratio_min:float = 0.0
    neg_amp_ratio_max:float = 1.0
    neg_amp_ratio_pts:int = 100
    additional_wait_ns:int = 0

node = QualibrationNode(name="BipolarOpimize", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load()
# machine.network["port"] = int(access_port)

# print(f"Machine access port :{access_port}")
if node.parameters.load_data_id is None:
    qmm = machine.connect()

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

config = machine.generate_config()

# %% {QUA_program}
n_avg = node.parameters.num_runs  # The number of averages
N_pi = (
    node.parameters.max_number_rabi_pulses_per_sweep
)  # Number of applied Rabi pulses sweep
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
operation = node.parameters.operation_x180_or_any_90  # The qubit operation to play
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.linspace(
    node.parameters.neg_amp_ratio_min,
    node.parameters.neg_amp_ratio_max,
    node.parameters.neg_amp_ratio_pts,
)

if operation == "x180":
    N_pi_vec = np.arange(1, N_pi, 2).astype("int")
elif operation in ["x90", "-x90", "y90", "-y90"]:
    N_pi_vec = np.arange(2, N_pi, 4).astype("int")
else:
    raise ValueError(f"Unrecognized operation {operation}.")


with program() as power_rabi:
    I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(bool) for _ in range(num_qubits)]
    state_stream = [declare_stream() for _ in range(num_qubits)]
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    for i, qubit in enumerate(qubits):
        if not node.parameters.simulate:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            if "c" in qubit.id: qubit.z.set_dc_offset(qubit.z.joint_offset) # for coupler-test case
            qubit.z.settle()
            qubit.align() 

            # Wait for the flux bias to settle
            for qb in qubits:
                wait(1000, qb.z.name)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(npi, N_pi_vec)):
                with for_(*from_array(a, amps)):
                    # Initialize the qubits
                    if reset_type == "active":
                        active_reset(qubit)
                        
                    else:
                        if not node.parameters.simulate:
                            qubit.wait(qubit.thermalization_time * u.ns)
                        else:
                            qubit.wait(16 * u.ns)
                    
                    align(*[q.xy.name for q in qubits] +
                            [q.resonator.name for q in qubits] +
                            [q.z.name for q in qubits])


                    split_bipolar_macro(qubit, amplitude_scale=0.2/qubit.z.operations['flattopV2'].amplitude, neg_pole_amp_ratio=a, debug=False)
                    qubit.align()
                    # Loop for error amplification (perform many qubit pulses)
                    with for_(count, 0, count < npi, count + 1):
                        qubit.xy.play(operation)
                    align(qubit.xy.name, qubit.resonator.name)
                    #        [q.z.name for q in qubits]))
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    assign(state[i], I[i] > qubit.resonator.operations["readout"].threshold)
                    save(state[i], state_stream[i])

                    if node.parameters.additional_wait_ns > 16:
                        wait(node.parameters.additional_wait_ns//4)

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            if operation == "x180":
                state_stream[i].boolean_to_int().buffer(len(amps)).buffer(
                    np.ceil(N_pi / 2)
                ).average().save(f"state{i + 1}")
            elif operation in ["x90", "-x90", "y90", "-y90"]:
                state_stream[i].boolean_to_int().buffer(len(amps)).buffer(
                    np.ceil(N_pi / 4)
                ).average().save(f"state{i + 1}")
            else:
                raise ValueError(f"Unrecognized operation {operation}.")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000//4)  # In clock cycles = 4ns
    job = qmm.simulate(config, power_rabi, simulation_config)
    samples = job.get_simulated_samples()
    samples.con1.plot()
    node.results = {"figure": plt.gcf()}
    wf_report = job.get_simulated_waveform_report()
    wf_report.create_plot(samples, plot=True, save_path=None)
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(power_rabi)

        # %% {Live_plot}
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
# Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
ds = fetch_results_as_xarray(
    job.result_handles, qubits, {"amp": amps, "N": N_pi_vec}
)
# Add the qubit pulse absolute amplitude to the dataset
ds = ds.assign_coords(
    {
        "abs_amp": (
            ["qubit", "amp"],
            np.array([amps for q in qubits]),
        )
    }
)
# Add the dataset to the node
node.results = {"ds": ds}

# %% {Data_analysis}
fit_results = {}
if N_pi == 1:
    # Fit the power Rabi oscillations
    fit = fit_oscillation(ds.state, "amp")
    fit_evals = oscillation(
        ds.amp,
        fit.sel(fit_vals="a"),
        fit.sel(fit_vals="f"),
        fit.sel(fit_vals="phi"),
        fit.sel(fit_vals="offset"),
    )
    
# Save fitting results
    for q in qubits:
        fit_results[q.name] = {}
        f_fit = fit.loc[q.name].sel(fit_vals="f")
        phi_fit = fit.loc[q.name].sel(fit_vals="phi")
        phi_fit = phi_fit - np.pi * (phi_fit > np.pi / 2)
        factor = float(1.0 * (np.pi - phi_fit) / (2 * np.pi * f_fit))
        new_pi_amp = q.xy.operations[operation].amplitude * factor
        if new_pi_amp < 0.3:  # TODO: 1 for OPX1000 MW
            print(f"amplitude for Pi pulse is modified by a factor of {factor:.2f}")
            print(
                f"new amplitude is {1e3 * new_pi_amp:.2f} mV \n"
            )  # TODO: 1 for OPX1000 MW
            fit_results[q.name]["Pi_amplitude"] = float(new_pi_amp)
        else:
            print(f"Fitted amplitude too high, new amplitude is 300 mV \n")
            fit_results[q.name]["Pi_amplitude"] = 0.3  # TODO: 1 for OPX1000 MW
    node.results["fit_results"] = fit_results

elif N_pi > 1:
    # Get the average along the number of pulses axis to identify the best pulse amplitude
    I_n = ds.state.mean(dim="N")
    data_max_idx = I_n.argmax(dim="amp")
    
# Save fitting results
    for q in qubits:
        new_pi_amp = ds.abs_amp.sel(qubit=q.name)[data_max_idx.sel(qubit=q.name)]
        fit_results[q.name] = {}
        if new_pi_amp < 1:  # TODO: 1 for OPX1000 MW
            fit_results[q.name]["Pi_amplitude"] = float(new_pi_amp)
            print(
                f"amplitude for Pi pulse is modified by a factor of {I_n.idxmax(dim='amp').sel(qubit = q.name):.2f}"
            )
            print(
                f"new amplitude is {1e3 * new_pi_amp:.2f} mV \n"
            )  # TODO: 1 for OPX1000 MW
        else:
            print(f"Fitted amplitude too high, new amplitude is 1000 mV \n")
            fit_results[q.name]["Pi_amplitude"] = 1  # TODO: 1 for OPX1000 MW
    node.results["fit_results"] = fit_results

# %% {Plotting}
grid_names = [q.grid_location for q in qubits]
grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):
    if N_pi == 1:
        ds.assign_coords(amp_mV=ds.abs_amp).loc[qubit].state.plot(
            ax=ax, x="amp_mV"
        )
        ax.plot(ds.abs_amp.loc[qubit], fit_evals.loc[qubit][0])
        ax.set_ylabel("Trans. amp. I [mV]")
    elif N_pi > 1:
        ds.assign_coords(amp_mV=ds.abs_amp).loc[qubit].state.plot(
            ax=ax, x="amp_mV", y="N"
        )
        ax.axvline(ds.abs_amp.loc[qubit][data_max_idx.loc[qubit]], color="r")
        ax.set_ylabel("num. of pulses")
    ax.set_xlabel("Negative pole amplitude factor")
    ax.set_title(qubit["qubit"])
grid.fig.suptitle("Rabi : I vs. Bipolar negative pole amplitude")
plt.tight_layout()
plt.show()
node.results["figure"] = grid.fig

# %% {Update_state}
with node.record_state_updates():
    for q in qubits:
        q.xy.operations[operation].amplitude = fit_results[q.name]["Pi_amplitude"]

# %% {Save_results}
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()
