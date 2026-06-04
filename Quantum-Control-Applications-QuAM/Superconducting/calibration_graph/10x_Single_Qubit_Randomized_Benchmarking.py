"""
        SINGLE QUBIT RANDOMIZED BENCHMARKING
The program consists in playing random sequences of Clifford gates and measuring the state of the resonator afterward.
Each random sequence is derived on the FPGA for the maximum depth (specified as an input) and played for each depth
asked by the user (the sequence is truncated to the desired depth). Each truncated sequence ends with the recovery gate,
found at each step thanks to a preloaded lookup table (Cayley table), that will bring the qubit back to its ground state.

If the readout has been calibrated and is good enough, then state discrimination can be applied to only return the state
of the qubit. Otherwise, the 'I' and 'Q' quadratures are returned.
Each sequence is played n_avg times for averaging. A second averaging is performed by playing different random sequences.

The data is then post-processed to extract the single-qubit gate fidelity and error per gate
.
Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR and state discrimination.
    - Set the desired flux bias.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import (
    qua_declaration,
    active_reset,
    active_reset_simple,
    active_reset_gef,
    readout_state_gef,
)
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
from quam_libs.experiments.node_parameters import (
    QubitsExperimentNodeParameters,
    SimulatableNodeParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters,
)
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# %% {Node_parameters}
class Parameters(
    NodeParameters,
    SimulatableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
    QubitsExperimentNodeParameters,
):
    qubits: Optional[List[str]] = None
    use_state_discrimination: bool = True
    readout_mode: Literal["ge", "gef"] = "ge" #gef will plot the probability of being in g,e and outside of the qubit subspace
    use_strict_timing: bool = False
    num_random_sequences: int = 100 # Number of random sequences
    num_averages: int = 50
    max_circuit_depth: int = 800  # Maximum circuit depth
    delta_clifford: int = 40
    seed: int = None
    reset_type_thermal_or_active: Literal["thermal", "active", "active_gef"] =  "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False

node = QualibrationNode(name="10a_Single_Qubit_Randomized_Benchmarking", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations

config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)


# %% {QUA_program_parameters}
num_of_sequences = node.parameters.num_random_sequences  # Number of random sequences
# Number of averaging loops for each random sequence
n_avg = node.parameters.num_averages
max_circuit_depth = node.parameters.max_circuit_depth  # Maximum circuit depth
if node.parameters.delta_clifford < 1:
    raise NotImplementedError("Delta clifford < 2 is not supported.")
#  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
delta_clifford = node.parameters.delta_clifford
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
readout_mode = node.parameters.readout_mode
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
num_depths = max_circuit_depth // delta_clifford + 1
seed = node.parameters.seed  # Pseudo-random number generator seed
# Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
state_discrimination = node.parameters.use_state_discrimination
strict_timing = node.parameters.use_strict_timing
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]


# %% {Utility functions}
def power_law(power, a, b, p):
    return a * (p**power) + b


def generate_sequence():
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=max_circuit_depth + 1)
    inv_gate = declare(int, size=max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < max_circuit_depth, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


def play_sequence(sequence_list, depth, qubit: Transmon):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                qubit.xy.play("x180", amplitude_scale=0.0)
            with case_(1):  # x180
                qubit.xy.play("x180")
            with case_(2):  # y180
                qubit.xy.play("y180")
            with case_(3):  # Z180
                qubit.xy.play("y180")
                qubit.xy.play("x180")
            with case_(4):  # Z90 X180 Z-180
                qubit.xy.play("x90")
                qubit.xy.play("y90")
            with case_(5):  # Z-90 Y-90 Z-90
                qubit.xy.play("x90")
                qubit.xy.play("-y90")
            with case_(6):  # Z-90 X180 Z-180
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
            with case_(7):  # Z-90 Y90 Z-90
                qubit.xy.play("-x90")
                qubit.xy.play("-y90")
            with case_(8):  # X90 Z90
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(9):  # X-90 Z-90
                qubit.xy.play("y90")
                qubit.xy.play("-x90")
            with case_(10):  # z90 X90 Z90
                qubit.xy.play("-y90")
                qubit.xy.play("x90")
            with case_(11):  # z90 X-90 Z90
                qubit.xy.play("-y90")
                qubit.xy.play("-x90")
            with case_(12):  # x90
                qubit.xy.play("x90")
            with case_(13):  # -x90
                qubit.xy.play("-x90")
            with case_(14):  # y90
                qubit.xy.play("y90")
            with case_(15):  # -y90
                qubit.xy.play("-y90")
            with case_(16):  # Z90
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(17):  # -Z90
                qubit.xy.play("-x90")
                qubit.xy.play("-y90")
                qubit.xy.play("x90")
            with case_(18):  # Y-90 Z-90
                qubit.xy.play("x180")
                qubit.xy.play("y90")
            with case_(19):  # Y90 Z90
                qubit.xy.play("x180")
                qubit.xy.play("-y90")
            with case_(20):  # Y90 Z-90
                qubit.xy.play("y180")
                qubit.xy.play("x90")
            with case_(21):  # Y-90 Z90
                qubit.xy.play("y180")
                qubit.xy.play("-x90")
            with case_(22):  # x90 Z-90
                qubit.xy.play("x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(23):  # -x90 Z90
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("-x90")


def build_gef_state_probabilities(ds: xr.Dataset) -> xr.DataArray:
    required_vars = ["state_g", "state_e", "state_outside"]
    missing_vars = [var for var in required_vars if var not in ds]
    if missing_vars:
        raise ValueError(f"GEF readout data is missing {missing_vars}")

    probabilities = xr.concat(
        [ds["state_g"], ds["state_e"], ds["state_outside"]],
        dim="state",
    )
    probabilities = probabilities.assign_coords(state=("state", ["0", "1", "outside"]))
    probabilities.attrs = {"long_name": "state probability"}
    return probabilities


def plot_gef_state_probabilities(
    probabilities: xr.DataArray,
    qubits: List[Transmon],
    num_random_sequences: int,
):
    grid = QubitGrid(probabilities, [q.grid_location for q in qubits])
    style_map = {
        "0": {"color": "tab:blue", "marker": "o", "linewidth": 2.0},
        "1": {"color": "tab:orange", "marker": "s", "linewidth": 2.0},
        "outside": {"color": "black", "marker": "x", "linewidth": 2.0, "linestyle": "--"},
    }

    for ax, qubit in grid_iter(grid):
        for state_label in probabilities.state.values:
            state_probabilities = probabilities.sel(qubit=qubit["qubit"], state=state_label)
            mean_prob = state_probabilities.mean(dim="sequence")
            error_bars = state_probabilities.std(dim="sequence")
            ax.errorbar(
                mean_prob.m,
                mean_prob,
                yerr=error_bars,
                label=state_label,
                capsize=2,
                elinewidth=0.5,
                **style_map[str(state_label)],
            )
        ax.set_title(qubit["qubit"], pad=12)
        ax.set_xlabel("Circuit depth")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid("all")
        ax.legend(
            framealpha=0,
            title="State",
            fontsize=7,
            title_fontsize=8,
            handlelength=1.0,
            markerscale=0.7,
            labelspacing=0.2,
            borderpad=0.2,
            borderaxespad=0.2,
        )

    grid.fig.suptitle(
        f"GEF readout state probabilities\nRandom gate number per depth = {num_random_sequences}"
    )
    grid.fig.tight_layout()
    return grid.fig


# %% {QUA_program}
with program() as randomized_benchmarking:
    depth = declare(int)  # QUA variable for the varying depth
    # QUA variable for the current depth (changes in steps of delta_clifford)
    depth_target = declare(int)
    # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
    saved_gate = declare(int)
    m = declare(int)  # QUA variable for the loop over random sequences
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    # The relevant streams
    m_st = declare_stream()
    # state_st = declare_stream()
    state_st = [declare_stream() for _ in range(num_qubits)]
    if readout_mode == "gef":
        state_not_g = [declare(int) for _ in range(num_qubits)]
        state_g = [declare(int) for _ in range(num_qubits)]
        state_e = [declare(int) for _ in range(num_qubits)]
        state_outside = [declare(int) for _ in range(num_qubits)]
        state_g_st = [declare_stream() for _ in range(num_qubits)]
        state_e_st = [declare_stream() for _ in range(num_qubits)]
        state_outside_st = [declare_stream() for _ in range(num_qubits)]

    # QUA for_ loop over the random sequences
    for multiplexed_qubits in qubits.batch():
        with for_(m, 0, m < num_of_sequences, m + 1):
            # Generate the random sequence of length max_circuit_depth
            sequence_list, inv_gate_list = generate_sequence()
            assign(depth_target, 0)  # Initialize the current depth to 0

            with for_(depth, 1, depth <= max_circuit_depth, depth + 1):
                # Replacing the last gate in the sequence with the sequence's inverse gate
                # The original gate is saved in 'saved_gate' and is being restored at the end
                assign(saved_gate, sequence_list[depth])
                assign(sequence_list[depth], inv_gate_list[depth - 1])
                # Only played the depth corresponding to target_depth
                with if_((depth == 1) | (depth == depth_target)):
                    with for_(n, 0, n < n_avg, n + 1):
                        # Bring the active qubits to the desired frequency point.
                        if flux_point == "independent":
                            machine.apply_all_flux_to_min()
                            for qubit in multiplexed_qubits.values():
                                qubit.z.to_independent_idle()
                        elif flux_point == "joint":
                            machine.apply_all_flux_to_joint_idle()
                        else:
                            raise ValueError(f"Unrecognized flux point {flux_point}")

                        align(*([q.xy.name for q in multiplexed_qubits.values()] +
                                [q.resonator.name for q in multiplexed_qubits.values()] +
                                [q.z.name for q in multiplexed_qubits.values()]))

                        # Initialize the qubits.
                        for qubit in multiplexed_qubits.values():
                            if reset_type == "active":
                                # qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                                active_reset(qubit)
                                # qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                            elif reset_type == "active_gef":
                                active_reset_gef(qubit)
                            else:
                                qubit.resonator.wait(qubit.thermalization_time * u.ns)

                        align(*([q.xy.name for q in multiplexed_qubits.values()] +
                                [q.resonator.name for q in multiplexed_qubits.values()] +
                                [q.z.name for q in multiplexed_qubits.values()]))

                        # Play the RB sequence on all qubits in the batch.
                        for qubit in multiplexed_qubits.values():
                            if strict_timing:
                                with strict_timing_():
                                    play_sequence(sequence_list, depth, qubit)
                            else:
                                play_sequence(sequence_list, depth, qubit)

                        align(*([q.xy.name for q in multiplexed_qubits.values()] +
                                [q.resonator.name for q in multiplexed_qubits.values()] +
                                [q.z.name for q in multiplexed_qubits.values()]))

                        # Read out all qubits in the batch with aligned resonators.
                        for i, qubit in multiplexed_qubits.items():
                            if readout_mode == "gef":
                                readout_state_gef(qubit, state[i])
                                assign(state_not_g[i], Cast.to_int(state[i] > 0))
                                assign(state_g[i], Cast.to_int(state[i] == 0))
                                assign(state_e[i], Cast.to_int(state[i] == 1))
                                assign(state_outside[i], Cast.to_int(state[i] > 1))
                                save(state_not_g[i], state_st[i])
                                save(state_g[i], state_g_st[i])
                                save(state_e[i], state_e_st[i])
                                save(state_outside[i], state_outside_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                assign(
                                    state[i],
                                    Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold),
                                )
                                save(state[i], state_st[i])

                    # Go to the next depth
                    assign(depth_target, depth_target + delta_clifford)
                # Reset the last gate of the sequence back to the original Clifford gate
                # (that was replaced by the recovery gate at the beginning)
                assign(sequence_list[depth], saved_gate)
            # Save the counter for the progress bar
            save(m, m_st)

    with stream_processing():
        m_st.save("iteration")
        for i in range(num_qubits):
            state_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_of_sequences).save(
                f"state{i + 1}"
            )
            if readout_mode == "gef":
                state_g_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(
                    num_of_sequences
                ).save(f"state_g{i + 1}")
                state_e_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(
                    num_of_sequences
                ).save(f"state_e{i + 1}")
                state_outside_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(
                    num_of_sequences
                ).save(f"state_outside{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=100_000)  # in clock cycles
    job = qmm.simulate(config, randomized_benchmarking, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    node.results["figure"] = plt.gcf()
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    # Prepare data for saving
    node.results = {}
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(randomized_benchmarking)
        results = fetching_tool(job, ["iteration"], mode="live")
        while results.is_processing():
            # Fetch results
            m = results.fetch_all()[0]
            # Progress bar
            progress_counter(m, num_of_sequences, start_time=results.start_time)


    # %% {Data_fetching_and_dataset_creation}
    if node.parameters.load_data_id is None:
        depths = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
        depths[0] = 1
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(
        job.result_handles,
        qubits,
            {"depths": depths, "sequence": np.arange(num_of_sequences)},
        )
    else:
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    # Add the dataset to the node
    node.results = {"ds": ds}
    # %% {Data_analysis}
    if readout_mode == "gef":
        gef_state_probabilities = build_gef_state_probabilities(ds)
        gef_state_probabilities = gef_state_probabilities.assign_coords(
            depths=gef_state_probabilities.depths - 1
        )
        gef_state_probabilities = gef_state_probabilities.rename(depths="m")
        gef_state_probabilities.m.attrs = {"long_name": "no. of Cliffords"}
        node.results["gef_state_probabilities"] = gef_state_probabilities.to_dataset(name="probability")
        da_state = ds["state_g"].mean(dim="sequence")
    else:
        gef_state_probabilities = None
        da_state = 1 - ds["state"].mean(dim="sequence")
    da_state: xr.DataArray
    da_state.attrs = {"long_name": "p(0)"}
    da_state = da_state.assign_coords(depths=da_state.depths - 1)
    da_state = da_state.rename(depths="m")
    da_state.m.attrs = {"long_name": "no. of Cliffords"}
    # Fit the exponential decay
    da_fit = None
    EPC = None
    EPG = None
    fit_error = None
    fit_successful = False
    node.results["fit_results"] = {}
    try:
        da_fit = fit_decay_exp(da_state, "m")
        # Extract the decay rate
        alpha = np.exp(da_fit.sel(fit_vals="decay"))
        # average_gate_per_clifford = 45/24 = 1.875
        average_gate_per_clifford = (1 * 3 + 9 * 2 + 1 * 4 + 2 * 3 + 4 * 2 + 2 * 3) / 24
        # EPC from here: https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html#Step-5:-Fit-the-results
        EPC = (1 - alpha) - (1 - alpha) / 2
        EPG = EPC / average_gate_per_clifford
        fit_successful = True
    except Exception as e:
        da_fit = None
        fit_error = str(e)
        print(f"RB fit failed: {fit_error}")
        print("Proceeding with raw data plot only.")

    # Save fitting results
    for q in qubits:
        node.results["fit_results"][q.name] = {"fit_successful": fit_successful}
        if not fit_successful:
            node.results["fit_results"][q.name]["fit_error"] = fit_error
            continue
        try:
            node.results["fit_results"][q.name]["EPC"] = EPC.sel(qubit=q.name).values
            node.results["fit_results"][q.name]["EPG"] = EPG.sel(qubit=q.name).values
            print(f"{q.name}: EPC={EPC.sel(qubit=q.name).values}")
            print(f"{q.name}: EPG={EPG.sel(qubit=q.name).values}")
        except Exception as e:
            node.results["fit_results"][q.name]["fit_successful"] = False
            node.results["fit_results"][q.name]["fit_error"] = str(e)
            print(f"RB fit result extraction failed for {q.name}: {e}")


# %% {Plotting}
if not node.parameters.simulate:
    grid_names = [q.grid_location for q in qubits]
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        da_state_qubit = da_state.sel(qubit=qubit["qubit"])
        if readout_mode == "gef":
            da_state_std = ds["state_g"].std(dim="sequence").sel(qubit=qubit["qubit"])
        else:
            da_state_std = ds["state"].std(dim="sequence").sel(qubit=qubit["qubit"])
        ax.errorbar(
            da_state_qubit.m,
            da_state_qubit,
            yerr=da_state_std,
            fmt=".",
            capsize=2,
            elinewidth=0.5,
        )
        ax.grid("all")
        m = da_state.m.values
        ax.set_title(qubit["qubit"], pad=22)
        ax.set_xlabel("Circuit depth")
        fit_result = node.results["fit_results"].get(qubit["qubit"], {})
        if fit_result.get("fit_successful", False):
            try:
                fit_dict = {k: da_fit.sel(**qubit).sel(fit_vals=k).values for k in da_fit.fit_vals.values}
                ax.plot(m, decay_exp(m, **fit_dict), "r--", label="fit")
                ax.text(
                    0.0,
                    1.07,
                    f"1Q gate fidelity = {1 - EPG.sel(**qubit).values:.5f}",
                    transform=ax.transAxes,
                )
            except Exception as e:
                fit_result["fit_successful"] = False
                fit_result["fit_error"] = str(e)
                print(f"RB fit plotting failed for {qubit['qubit']}: {e}")
    plt.suptitle(f"SQ RB\n Random gate number per depth = {node.parameters.num_random_sequences}")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    if readout_mode == "gef" and gef_state_probabilities is not None:
        gef_state_fig = plot_gef_state_probabilities(
            gef_state_probabilities,
            qubits,
            node.parameters.num_random_sequences,
        )
        gef_state_fig.show()
        node.results["figure_gef_state_probabilities"] = gef_state_fig

    # %% {Save_results}
    successful_fit_qubits = [
        q for q in qubits if node.results["fit_results"].get(q.name, {}).get("fit_successful", False)
    ]
    if not node.parameters.simulate and successful_fit_qubits:
        with node.record_state_updates():
            for q in successful_fit_qubits:
                q.extras["EPG"] = EPG.sel(qubit=q.name).item()
                q.extras["EPC"] = EPC.sel(qubit=q.name).item()
                
    if not node.parameters.simulate:
        node.outcomes = {
            q.name: (
                "successful"
                if node.results["fit_results"].get(q.name, {}).get("fit_successful", False)
                else "failed"
            )
            for q in qubits
        }
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()


# %%
