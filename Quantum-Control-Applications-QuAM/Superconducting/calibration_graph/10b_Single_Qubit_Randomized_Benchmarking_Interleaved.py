"""
SINGLE QUBIT INTERLEAVED RANDOMIZED BENCHMARKING (for gates >= 40ns)
The program consists in playing random sequences of Clifford gates and measuring the state of the resonator afterwards.
Each random sequence is derived on the FPGA for the maximum depth (specified as an input) and played for each depth
asked by the user (the sequence is truncated to the desired depth). Each truncated sequence ends with the recovery gate,
found at each step thanks to a preloaded lookup table (Cayley table), that will bring the qubit back to its ground state.
In this version, a Clifford gate chosen by the user is interleaved between each random gate in the sequence. This allows
to characterize the fidelity of a specific gate.

If the readout has been calibrated and is good enough, then state discrimination can be applied to only return the state
of the qubit. Otherwise, the 'I' and 'Q' quadratures are returned.
Each sequence is played n_avg times for averaging. A second averaging is performed by playing different random sequences.

The data is then post-processed to extract the single-qubit gate fidelity and error per gate
.
Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.components.batchable_list import BatchableList
from quam_libs.macros import active_reset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import time


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q4","q5"]
    use_state_discrimination: bool = True
    use_strict_timing: bool = True
    interleaved_gate_index: int = 2
    num_random_sequences: int = 50  # Number of random sequences
    num_averages: int = 20
    max_circuit_depth: int = 1200  # Maximum circuit depth
    delta_clifford: int = 40
    seed: int = 345324
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    multiplexed: bool = True


node = QualibrationNode(name="10b_Single_Qubit_Randomized_Benchmarking_Interleaved", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
node.machine = machine
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

qubits = BatchableList(
    [machine.qubits[q] for q in node.parameters.qubits],
    multiplexed=node.parameters.multiplexed,
)
num_qubits = len(qubits)


# %% {QUA_program}
num_of_sequences = node.parameters.num_random_sequences  # Number of random sequences
n_avg = node.parameters.num_averages  # Number of averaging loops for each random sequence
max_circuit_depth = node.parameters.max_circuit_depth  # Maximum circuit depth
if node.parameters.delta_clifford < 1:
    raise NotImplementedError("Delta clifford < 2 is not supported.")
delta_clifford = (
    node.parameters.delta_clifford
)  #  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
num_depths = max_circuit_depth // delta_clifford + 1
seed = node.parameters.seed  # Pseudo-random number generator seed
# Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
state_discrimination = node.parameters.use_state_discrimination
strict_timing = node.parameters.use_strict_timing
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
# index of the gate to interleave from the play_sequence() function defined below
# Correspondence table:
#  0: identity |  1: x180 |  2: y180
# 12: x90      | 13: -x90 | 14: y90 | 15: -y90 |
interleaved_gate_index = node.parameters.interleaved_gate_index


# %% {Utility functions}
def get_interleaved_gate(gate_index):
    if gate_index == 0:
        return "I"
    elif gate_index == 1:
        return "x180"
    elif gate_index == 2:
        return "y180"
    elif gate_index == 12:
        return "x90"
    elif gate_index == 13:
        return "-x90"
    elif gate_index == 14:
        return "y90"
    elif gate_index == 15:
        return "-y90"
    else:
        raise ValueError(f"Interleaved gate index {gate_index} doesn't correspond to a single operation")


def power_law(power, a, b, p):
    return a * (p**power) + b


def guess_power_law(x, y):
    """Initial guess for A * p**x + B from the measured curve.

    The tail estimates the asymptote B. A = y[0] - B is negative when the
    curve rises (excited-state population) and positive when it decays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_tail = max(3, len(y) // 5)
    b = float(np.median(y[-n_tail:]))
    a = float(y[0] - b)
    if abs(a) < 1e-4:
        return [0.0, float(np.median(y)), 0.99]
    ratio = (y - b) / a
    mask = (x > 0) & (ratio > 0.02) & (ratio < 0.98)
    if not np.any(mask):
        p = 0.99
    else:
        p = float(np.exp(np.median(np.log(ratio[mask]) / x[mask])))
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return [a, b, p]


def generate_sequence(interleaved_gate_index):
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=2 * max_circuit_depth + 1)
    inv_gate = declare(int, size=2 * max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < 2 * max_circuit_depth, i + 2):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])
        # interleaved gate
        assign(step, interleaved_gate_index)
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i + 1], step)
        assign(inv_gate[i + 1], inv_list[current_state])

    return sequence, inv_gate


def play_sequence(sequence_list, depth, qubit: Transmon):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                qubit.xy.wait(qubit.xy.operations["x180"].length // 4)
            with case_(1):
                qubit.xy.play("x180")
            with case_(2):
                qubit.xy.play("y180")
            with case_(3):
                qubit.xy.play("y180")
                qubit.xy.play("x180")
            with case_(4):
                qubit.xy.play("x90")
                qubit.xy.play("y90")
            with case_(5):
                qubit.xy.play("x90")
                qubit.xy.play("-y90")
            with case_(6):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
            with case_(7):
                qubit.xy.play("-x90")
                qubit.xy.play("-y90")
            with case_(8):
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(9):
                qubit.xy.play("y90")
                qubit.xy.play("-x90")
            with case_(10):
                qubit.xy.play("-y90")
                qubit.xy.play("x90")
            with case_(11):
                qubit.xy.play("-y90")
                qubit.xy.play("-x90")
            with case_(12):
                qubit.xy.play("x90")
            with case_(13):
                qubit.xy.play("-x90")
            with case_(14):
                qubit.xy.play("y90")
            with case_(15):
                qubit.xy.play("-y90")
            with case_(16):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(17):
                qubit.xy.play("-x90")
                qubit.xy.play("-y90")
                qubit.xy.play("x90")
            with case_(18):
                qubit.xy.play("x180")
                qubit.xy.play("y90")
            with case_(19):
                qubit.xy.play("x180")
                qubit.xy.play("-y90")
            with case_(20):
                qubit.xy.play("y180")
                qubit.xy.play("x90")
            with case_(21):
                qubit.xy.play("y180")
                qubit.xy.play("-x90")
            with case_(22):
                qubit.xy.play("x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(23):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("-x90")


def batch_elements(multiplexed_qubits):
    return (
        [q.xy.name for q in multiplexed_qubits.values()]
        + [q.resonator.name for q in multiplexed_qubits.values()]
        + [q.z.name for q in multiplexed_qubits.values()]
    )


def get_rb_interleaved_program():
    with program() as rb:
        depth = declare(int)  # QUA variable for the varying depth
        depth_target = declare(int)  # QUA variable for the current depth (changes in steps of delta_clifford)
        # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
        saved_gate = declare(int)
        m = declare(int)  # QUA variable for the loop over random sequences
        n = declare(int)  # QUA variable for the averaging loop
        I = [declare(fixed) for _ in range(num_qubits)]
        Q = [declare(fixed) for _ in range(num_qubits)]
        state = [declare(bool) for _ in range(num_qubits)]
        # The relevant streams
        m_st = declare_stream()
        I_st = [declare_stream() for _ in range(num_qubits)]
        Q_st = [declare_stream() for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

        # multiplexed=False runs one qubit at a time; multiplexed=True plays the batch in parallel.
        for multiplexed_qubits in qubits.batch():
            # Bring the active qubits to the minimum frequency point
            if flux_point == "independent":
                machine.apply_all_flux_to_min()
                for qubit in multiplexed_qubits.values():
                    qubit.z.to_independent_idle()
                machine.apply_all_couplers_to_min()
            elif flux_point == "joint":
                machine.apply_all_flux_to_joint_idle()
            else:
                machine.apply_all_flux_to_zero()

            # Wait for the flux bias to settle
            for qb in qubits:
                wait(1000, qb.z.name)

            align(*batch_elements(multiplexed_qubits))

            with for_(m, 0, m < num_of_sequences, m + 1):  # QUA for_ loop over the random sequences
                # Generates the RB sequence with a gate interleaved after each Clifford
                sequence_list, inv_gate_list = generate_sequence(interleaved_gate_index=interleaved_gate_index)
                # Depth_target is used to always play the gates by pairs [(random_gate-interleaved_gate)^depth/2-inv_gate]
                assign(depth_target, 0)  # Initialize the current depth to 0

                with for_(depth, 1, depth <= 2 * max_circuit_depth, depth + 1):
                    # Replacing the last gate in the sequence with the sequence's inverse gate
                    # The original gate is saved in 'saved_gate' and is being restored at the end
                    assign(saved_gate, sequence_list[depth])
                    assign(sequence_list[depth], inv_gate_list[depth - 1])
                    # Only played the depth corresponding to target_depth
                    with if_((depth == 2) | (depth == depth_target)):
                        with for_(n, 0, n < n_avg, n + 1):
                            # Initialize the qubits
                            for qubit in multiplexed_qubits.values():
                                if reset_type == "active":
                                    active_reset(qubit, "readout")
                                else:
                                    qubit.resonator.wait(qubit.thermalization_time * u.ns)
                            align(*batch_elements(multiplexed_qubits))
                            # The strict_timing ensures that the sequence will be played without gaps
                            for qubit in multiplexed_qubits.values():
                                if strict_timing:
                                    with strict_timing_():
                                        play_sequence(sequence_list, depth, qubit)
                                else:
                                    play_sequence(sequence_list, depth, qubit)
                            # Align the elements to measure after playing the circuit.
                            align(*batch_elements(multiplexed_qubits))
                            for i, qubit in multiplexed_qubits.items():
                                # Make sure you updated the ge_threshold and angle if you want to use state discrimination
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                if state_discrimination:
                                    assign(
                                        state[i],
                                        I[i] > qubit.resonator.operations["readout"].threshold,
                                    )
                                    save(state[i], state_st[i])
                                else:
                                    save(I[i], I_st[i])
                                    save(Q[i], Q_st[i])
                        # always play the random gate followed by the interleaved gate. The factor of 2 is there to always
                        # play the gates by pairs [(random_gate-interleaved_gate)^depth/2-inv_gate]
                        assign(depth_target, depth_target + 2 * delta_clifford)
                    # Reset the last gate of the sequence back to the original Clifford gate
                    # (that was replaced by the recovery gate at the beginning)
                    assign(sequence_list[depth], saved_gate)
                # Save the counter for the progress bar
                save(m, m_st)

            align()

        with stream_processing():
            m_st.save("iteration")
            for i in range(num_qubits):
                if state_discrimination:
                    # saves a 2D array of depth and random pulse sequences in order to get error bars along the random sequences
                    state_st[i].boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(
                        num_of_sequences
                    ).save(f"state{i + 1}")
                else:
                    I_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_of_sequences).save(
                        f"I{i + 1}"
                    )
                    Q_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_of_sequences).save(
                        f"Q{i + 1}"
                    )

    return rb


def analyze_qubit(qubit, value_avg, error_avg, x):
    p0 = guess_power_law(x, value_avg)
    print(f"{qubit.name} initial guess: A = {p0[0]:.3}, B = {p0[1]:.3}, p = {p0[2]:.6}")
    pars, cov = curve_fit(
        f=power_law,
        xdata=x,
        ydata=value_avg,
        p0=p0,
        bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 1.0]),
        maxfev=10000,
    )
    stdevs = np.sqrt(np.diag(cov))

    print("#########################")
    print(f"### Fitted Parameters ({qubit.name}) ###")
    print("#########################")
    print(
        f"A = {pars[0]:.3} ({stdevs[0]:.1}), B = {pars[1]:.3} ({stdevs[1]:.1}), p = {pars[2]:.3} ({stdevs[2]:.1})"
    )
    print("Covariance Matrix")
    print(cov)

    one_minus_p = 1 - pars[2]
    r_c = one_minus_p * (1 - 1 / 2**1)
    r_g = r_c / 1.875  # 1.875 is the average number of gates in clifford operation
    r_c_std = stdevs[2] * (1 - 1 / 2**1)
    r_g_std = r_c_std / 1.875

    print("#########################")
    print(f"### Useful Parameters ({qubit.name}) ###")
    print("#########################")
    print(
        f"Error rate: 1-p = {np.format_float_scientific(one_minus_p, precision=2)} ({stdevs[2]:.1})\n"
        f"Clifford set infidelity: r_c = {np.format_float_scientific(r_c, precision=2)} ({r_c_std:.1})\n"
        f"Gate infidelity: r_g = {np.format_float_scientific(r_g, precision=2)}  ({r_g_std:.1})"
    )
    fig_analysis = plt.figure()
    plt.errorbar(x, value_avg, yerr=error_avg, marker=".")
    plt.plot(x, power_law(x, *pars), linestyle="--", linewidth=2)
    plt.xlabel("Number of Clifford gates")
    plt.ylabel("Sequence Fidelity")
    plt.title(f"Single qubit interleaved RB for {qubit.name} ({get_interleaved_gate(interleaved_gate_index)})")

    node.results[qubit.name] = {
        f"{qubit.name}_depth": x,
        f"{qubit.name}_fidelity": value_avg,
        f"{qubit.name}_error": error_avg,
        f"{qubit.name}_fit": power_law(x, *pars),
        f"{qubit.name}_covariance_par": pars,
        f"{qubit.name}_error_rate": one_minus_p,
        f"{qubit.name}_clifford_set_infidelity": r_c,
        f"{qubit.name}_gate_infidelity": r_g,
    }
    node.results[f"{qubit.name}_figure"] = fig_analysis
    node.results[f"{qubit.name}_figure_analysis"] = fig_analysis


# %% {Simulate_or_execute}
rb_program = get_rb_interleaved_program()
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=100_000)  # in clock cycles
    job = qmm.simulate(config, rb_program, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    node.results["figure"] = plt.gcf()
else:
    node.results = {}
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(rb_program)
        results = fetching_tool(job, data_list=["iteration"], mode="live")
        while results.is_processing():
            iteration = results.fetch_all()[0]
            progress_counter(iteration, num_of_sequences, start_time=results.get_start_time())
            time.sleep(1)

        if state_discrimination:
            data_list = [f"state{i + 1}" for i in range(num_qubits)]
        else:
            data_list = [name for i in range(num_qubits) for name in (f"I{i + 1}", f"Q{i + 1}")]
        fetched = fetching_tool(job, data_list=data_list).fetch_all()

    x = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
    x[0] = 1  # to set the first value of 'x' to be depth = 1 as in the experiment
    for i, qubit in enumerate(qubits):
        if state_discrimination:
            values = fetched[i]
        else:
            values = fetched[2 * i]
        analyze_qubit(qubit, np.mean(values, axis=0), np.std(values, axis=0), x)


node.results["initial_parameters"] = node.parameters.model_dump()
node.save()

# %%
