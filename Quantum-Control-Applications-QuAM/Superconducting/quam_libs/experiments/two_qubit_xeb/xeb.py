import json
import warnings
from typing import Union, List, Optional, Dict, Tuple, Literal
import numpy as np
from qm.qua import *
from .macros import (
    qua_declaration,
    reset_qubit,
    binary,
    exponential_decay,
    fit_exponential_decay,
    get_parallel_gate_combinations as gate_combinations,
    align_transmon_pair,
    generate_circuits,
    compute_log_fidelity,
    evaluate_log_fidelity,
    update_record,
    update_data_frame,
    cross_entropy,
    two_qubit_unitary_from_fsim_params,
)
from quam_libs.macros import readout_state_gef
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from qiskit_aer import AerJob
import pandas as pd
from .xeb_config import XEBConfig
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Statevector
from qualang_tools.results import DataHandler

from quam_libs.components import QuAM, Transmon
from qm import SimulationConfig, QuantumMachinesManager, generate_qua_script
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.simulated_job import SimulatedJob
import seaborn as sns
from warnings import warn

from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


def _resolve_readout_modes(xeb_config: XEBConfig) -> List[int]:
    """
    Returns the number of readout levels (2 or 3) to use for each qubit in `xeb_config.qubits`.

    For the standard (single qubit-pair) XEB use case, `xeb_config.qubits` is ordered as
    [qubit_control, qubit_target], so the first qubit uses `control_readout_mode` and the
    second uses `target_readout_mode`. Any other configuration (0 or >1 qubit pairs) defaults
    to 2-level (G/E) readout for every qubit, since leakage tracking is only implemented for the
    single qubit-pair XEB use case.
    """
    modes = [2] * xeb_config.n_qubits
    if len(xeb_config.qubit_pairs) == 1 and xeb_config.n_qubits == 2:
        modes = [xeb_config.control_readout_mode, xeb_config.target_readout_mode]
    return modes


# Minimum mean of (sum_i p_i^2 - 1/D)^2 over the circuits of a depth for the linear XEB fidelity to be defined.
_MIN_XEB_SIGNAL = 1e-9


def _fit_decay_zero(depths, values, min_depth: int = 6) -> Dict:
    """
    Fit ``values = amplitude * base**depth`` (no offset) with bounded least squares.

    Non-finite points are ignored. Depths below ``min_depth`` are dropped when at least four deeper points remain.
    The same routine is used for linear XEB, purity and the unitary-fit comparison so that their decay bases are
    directly comparable.

    Returns a dict with ``amplitude``, ``base``, ``amplitude_stderr``, ``base_stderr`` and the ``depths`` used.
    """
    x = np.asarray(depths, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask & (x >= min_depth)) >= 4:
        mask &= x >= min_depth
    if np.count_nonzero(mask) < 4:
        raise ValueError("Need at least four depth points for the decay fit.")

    def model(depth, amplitude, base):
        return amplitude * base**depth

    amplitude_0 = float(np.clip(np.max(y[mask]), 1e-3, 1.9))
    popt, pcov = curve_fit(
        model,
        x[mask],
        y[mask],
        p0=[amplitude_0, 0.98],
        bounds=([0.0, 0.0], [2.0, 1.0]),
        maxfev=5000,
    )
    stderr = np.sqrt(np.diag(pcov))
    return {
        "amplitude": float(popt[0]),
        "base": float(popt[1]),
        "amplitude_stderr": float(stderr[0]),
        "base_stderr": float(stderr[1]),
        "depths": x[mask],
    }


class XEB:
    def __init__(self, xeb_config: XEBConfig, machine: QuAM):
        """
        Initialize the XEB experiment
        Args:
            xeb_config: XEBConfig object containing the parameters of the experiment
            machine: Machine object containing the Quantum Machine configuration
        """
        self.xeb_config = xeb_config
        self.machine = machine
        self.qubit_dict: Dict = {i: qubit for i, qubit in enumerate(self.qubits)}

        if len(self.qubit_pairs) == 0:
            warn("No qubit pairs provided. The experiment will run with single qubit gates only.")

        try:
            self.qubit_drive_channels = [qubit.xy for qubit in self.qubits]
            self.readout_channels = [qubit.resonator for qubit in self.qubits]
        except AttributeError:
            raise AttributeError(
                "Qubit objects must have 'xy' and 'resonator' attributes, "
                "Contact CS Team if your QuAM structure is different."
            )

        # Create CouplingMap from QuAM qubit pairs
        qubit_dict = {qubit: i for i, qubit in enumerate(self.qubits)}
        coupling_map = CouplingMap()
        for qubit in range(len(self.qubits)):
            coupling_map.add_physical_qubit(qubit)
        for qubit_pair in self.qubit_pairs:
            if qubit_pair.qubit_control not in self.qubits or qubit_pair.qubit_target not in self.qubits:
                raise ValueError("Qubit pairs must be formed by qubits present in the qubits list")
            coupling_map.add_edge(qubit_dict[qubit_pair.qubit_control], qubit_dict[qubit_pair.qubit_target])
        self._coupling_map = coupling_map
        self._available_combinations = gate_combinations(self.coupling_map)
        self.xeb_config.available_combinations = self.available_combinations
        self.xeb_config.coupling_map = self.coupling_map
        self.data_handler = DataHandler(name="XEB", root_data_folder=xeb_config.save_dir)

    @property
    def qubit_pairs(self):
        """
        Returns the qubit pairs for the XEB experiment
        """
        return self.xeb_config.qubit_pairs

    @property
    def qubits(self):
        """
        Returns the qubits for the XEB experiment
        """
        return self.xeb_config.qubits

    @property
    def readout_qubits(self):
        """
        Returns the readout qubits for the XEB experiment
        """
        return self.xeb_config.readout_qubits

    @property
    def available_combinations(self):
        """
        Returns the available combinations of qubit pairs for the XEB experiment
        """
        return self._available_combinations

    @property
    def coupling_map(self) -> CouplingMap:
        """
        Returns the coupling map for the XEB experiment
        """
        return self._coupling_map

    def _assign_amplitude_matrix(self, gate_idx, amp_matrix, amp_stream=None):
        """
        Assign the amplitude matrix of a gate based on the gate index

        Args:
            gate_idx (QUA int): Index of the gate
            amp_matrix (List): Amplitude matrix of the gate
            amp_stream (QUA stream): Stream to save the amplitude matrix
        """
        with switch_(gate_idx):
            for i in range(len(self.xeb_config.gate_set)):
                with case_(i):
                    for j in range(4):
                        assign(amp_matrix[j], self.xeb_config.gate_set[i].amp_matrix[j])
        if amp_stream is not None:  # Save the amplitude matrix to a stream
            for j in range(4):
                save(
                    amp_matrix[j],
                    amp_stream,
                )

    def _play_random_sq_gate(self, qubit: Transmon, gate_idx, amp_matrix: Optional[List] = None):
        """
        Play a random single qubit gate on a given qubit element.

        This macro plays a random single qubit gate on a given qubit element, by modulating
        the amplitude matrix of a baseline calibrated X/2 (SX) pulse if the gate set
        is set up to run through amplitude matrix modulation.
        Otherwise, it plays the gate through a switch case over the gate index.

        Args:
            qubit (Transmon): Qubit element on which to play the gate.
            gate_idx (QUA int): Index of the gate to play.
            amp_matrix (List): Amplitude matrix of the gate.
        """
        if self.xeb_config.gate_set.run_through_amp_matrix_modulation and amp_matrix is not None:
            # Play all gates through real-time amplitude matrix modulation
            qubit.xy.play(self.xeb_config.baseline_gate_name, amplitude_scale=amp_matrix)
            # qubit.xy.play('x180') # NOTE: for debugging purposes
            # qubit.xy.play('x90') # NOTE: for debugging purposes
            # pass # NOTE: for debugging purposes
        else:
            # Play all gates through switch case over the gate index
            with switch_(gate_idx, unsafe=True):
                for i in range(len(self.xeb_config.gate_set)):
                    with case_(i):
                        self.xeb_config.gate_set[i].gate_macro(qubit)

    def _readout_modes(self) -> List[int]:
        """
        Returns the number of readout levels (2 or 3) to use for each qubit in `self.qubits`.
        """
        return _resolve_readout_modes(self.xeb_config)

    def _xeb_prog(self, simulate: bool = False):
        """
        Generate the QUA program for the XEB experiment
        Args:
            simulate: Indicate if output should be simulated or not
        Returns: QUA program for the XEB experiment
        """
        n_qubits = self.xeb_config.n_qubits
        dim = self.xeb_config.dim
        random_gates = len(self.xeb_config.gate_set)
        readout_modes = self._readout_modes()
        ge_thresholds = [
            readout_element.operations[self.xeb_config.readout_pulse_name].threshold
            for readout_element in self.readout_channels
        ]

        with program() as xeb_prog:
            # Declare QUA variables
            I, I_st, Q, Q_st = qua_declaration(n_qubits=n_qubits, readout_elements=self.readout_channels)
            depth, depth_, n, s, tot_state_ = [declare(int) for _ in range(5)]
            gate = [
                declare(int, size=self.xeb_config.depths[-1]) for _ in range(n_qubits)
            ]  # Gate indices list for both qubits
            two_qubit_gate_pattern = declare(int, value=0)
            if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                amp_matrix = [
                    [declare(fixed, size=self.xeb_config.depths[-1]) for _ in range(4)] for _ in range(n_qubits)
                ]
            counts = declare(int, value=[0] * dim)  # Counts for all possible bit-strings (00, 01, 10, 11)
            # Qubit states: 0/1 for 2-level (G/E) readout, 0/1/2 for 3-level (G/E/F) readout
            state = [declare(int) for _ in range(n_qubits)]
            leak = [declare(bool) for _ in range(n_qubits)]  # True if qubit q was measured in |f> this shot
            any_leak = declare(bool)
            leak_counts = declare(int, value=[0] * n_qubits)  # Number of leakage (|f>) events per qubit, per depth
            # Declare streams
            counts_st = [
                declare_stream() for _ in range(dim)
            ]  # Stream for counts of all possible bit-strings (00, 01, 10, 11)
            state_st = [declare_stream() for _ in range(n_qubits)]  # Stream for individual qubit states
            leak_counts_st = [declare_stream() for _ in range(n_qubits)]  # Stream for per-qubit leakage counts
            gate_st = [
                declare_stream() for _ in range(n_qubits)
            ]  # Stream for gate indices (enabling circuit reconstruction in post-processing)
            amp_st = [declare_stream() for _ in range(n_qubits)]  # Stream for amplitude matrices
            # Bring the active qubits to the idle points:
            self.machine.apply_all_flux_to_min()
            self.machine.apply_all_couplers_to_min()
            # for q, qubit in enumerate(self.qubits):
            #     qubit.z.to_independent_idle()

            # Setting seed for reproducibility
            r = Random(seed=self.xeb_config.seed)

            # If simulating, update the frequency to 0 to visualize sequence
            if simulate:
                for qubit in self.qubit_drive_channels:
                    update_frequency(qubit.name, 0)

            # Generate the random sequences
            with for_(s, 0, s < self.xeb_config.seqs, s + 1):
                # Generate random gate sequence for each qubit
                for q in range(n_qubits):
                    # Iteration 0: Start the sequence with a random gate
                    assign(gate[q][0], r.rand_int(random_gates))
                    save(gate[q][0], gate_st[q])

                    # Map indices into amplitude matrix arguments
                    # (each index corresponds to a random gate)

                    if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                        self._assign_amplitude_matrix(
                            gate[q][0],
                            [amp_matrix[q][i][0] for i in range(4)],
                            amp_st[q],
                        )

                with for_(depth_, 1, depth_ < self.xeb_config.depths[-1], depth_ + 1):
                    for q in range(n_qubits):
                        assign(gate[q][depth_], r.rand_int(random_gates))
                        with while_(
                            gate[q][depth_] == gate[q][depth_ - 1]
                        ):  # Make sure same gate is not applied twice in a row
                            assign(gate[q][depth_], r.rand_int(random_gates))

                        save(gate[q][depth_], gate_st[q])

                        # Map indices into amplitude matrix arguments
                        # (each index corresponds to a random gate)
                        if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                            self._assign_amplitude_matrix(
                                gate[q][depth_],
                                [amp_matrix[q][i][depth_] for i in range(4)],
                                amp_st[q],
                            )

                # Run the XEB sequence
                with for_each_(depth, self.xeb_config.depths):  # Truncate depth to each value in depths
                    with for_(n, 0, n < self.xeb_config.n_shots, n + 1):
                        # Play all cycles generated for sequence s of depth d
                        with for_(depth_, 0, depth_ < depth, depth_ + 1):
                            for q, qubit in enumerate(self.qubits):  # Play single qubit gates on all qubits
                                self._play_random_sq_gate(
                                    qubit,
                                    gate[q][depth_],
                                    (
                                        [amp_matrix[q][i][depth_] for i in range(4)]
                                        if self.xeb_config.gate_set.run_through_amp_matrix_modulation
                                        else None
                                    ),
                                )

                            # Insert your two-qubit gate macro here
                            if self.xeb_config.two_qb_gate is not None and len(self.qubit_pairs) > 0:
                                if len(self.qubit_pairs) > 1:  # Multi-qubit XEB case
                                    with switch_(two_qubit_gate_pattern):
                                        for i, combination in enumerate(self.available_combinations):
                                            with case_(i):
                                                for pair in combination:
                                                    ctrl_idx, tgt_idx = pair
                                                    # qubit_ctrl = self.qubit_dict[ctrl_idx]
                                                    # qubit_tgt = self.qubit_dict[tgt_idx]
                                                    if tgt_idx < ctrl_idx:
                                                        ctrl_idx, tgt_idx = tgt_idx, ctrl_idx
                                                    qubit_pair = self.machine.qubit_pairs["coupler_q{}_q{}".format(ctrl_idx+1, tgt_idx+1)]
                                                    # align_transmon_pair(qubit_ctrl @ qubit_tgt)
                                                    align_transmon_pair(qubit_pair)
                                                    # Two qubit gate macro
                                                    # self.xeb_config.two_qb_gate.gate_macro(qubit_ctrl @ qubit_tgt)
                                                    self.xeb_config.two_qb_gate.gate_macro(qubit_pair)
                                                    # align_transmon_pair(qubit_ctrl @ qubit_tgt)
                                                    align_transmon_pair(qubit_pair)

                                    with if_(two_qubit_gate_pattern == len(self.available_combinations) - 1):
                                        assign(two_qubit_gate_pattern, 0)
                                    with else_():
                                        assign(two_qubit_gate_pattern, two_qubit_gate_pattern + 1)
                                else:  # Two-qubit XEB case (no need for switch case)
                                    qubit_pair = self.qubit_pairs[0]
                                    # align_transmon_pair(qubit_pair)
                                    self.xeb_config.two_qb_gate.gate_macro(qubit_pair)
                                    # align_transmon_pair(qubit_pair)

                        # Measure the state
                        wait(150 * u.ns)
                        align()
                        assign(any_leak, False)
                        for q_idx, qubit in enumerate(self.qubits):
                            # Play the readout on the other resonator to measure in the same condition as when optimizing readout
                            for other_qubit in self.readout_qubits:
                                if other_qubit.resonator != qubit.resonator:
                                    other_qubit.resonator.play("readout")

                            if readout_modes[q_idx] == 3:
                                # 3-level (G/E/F) readout: state[q_idx] in {0, 1, 2}. The |f> outcome (2) is
                                # flagged as leakage and excluded from the computational-subspace bitstring count.
                                readout_state_gef(qubit, state[q_idx], pulse_name=self.xeb_config.readout_pulse_name)
                                assign(leak[q_idx], state[q_idx] == 2)
                            else:
                                qubit.resonator.measure(
                                    self.xeb_config.readout_pulse_name,
                                    qua_vars=(I[q_idx], Q[q_idx]),
                                )
                                # State Estimation: returned as an integer, to be later converted to bit-strings
                                assign(state[q_idx], Cast.to_int(I[q_idx] > ge_thresholds[q_idx]))
                                save(I[q_idx], I_st[q_idx])
                                save(Q[q_idx], Q_st[q_idx])
                                assign(leak[q_idx], False)
                            save(state[q_idx], state_st[q_idx])
                            assign(any_leak, any_leak | leak[q_idx])
                            with if_(leak[q_idx]):
                                assign(leak_counts[q_idx], leak_counts[q_idx] + 1)
                            assign(
                                tot_state_,
                                tot_state_ + 2**q_idx * Cast.to_int(state[q_idx] == 1),
                            )

                            reset_qubit(
                                self.xeb_config.reset_method,
                                qubit,
                                threshold=ge_thresholds[q_idx],
                                readout_pulse_name=self.xeb_config.readout_pulse_name,
                                **self.xeb_config.reset_kwargs,
                            )
                        if len(self.qubit_pairs) > 1:  # Multi-qubit XEB case (Reset pattern)
                            assign(two_qubit_gate_pattern, 0)

                        # Only accumulate the bitstring counts (used for the XEB fidelity estimate) when
                        # neither qubit leaked out of the computational subspace this shot.
                        with if_(~any_leak):
                            with switch_(tot_state_):
                                for i in range(dim):  # Bitstring conversion
                                    with case_(i):
                                        assign(counts[i], counts[i] + 1)  # counts for 00, 01, 10 and 11
                        assign(tot_state_, 0)  # Resetting the state
                    for i in range(dim):  # Resetting Bitstring collection
                        save(counts[i], counts_st[i])
                        assign(counts[i], 0)
                    for q in range(n_qubits):  # Resetting per-depth leakage collection
                        save(leak_counts[q], leak_counts_st[q])
                        assign(leak_counts[q], 0)

            # Save the results
            with stream_processing():
                for q in range(n_qubits):
                    gate_st[q].buffer(self.xeb_config.depths[-1]).save_all(f"g{q}")
                    # I_st/Q_st are only ever saved to for 2-level (G/E) readout (see the measurement
                    # block above); for 3-level (G/E/F) readout, these streams stay empty, and QOP
                    # errors out ("cannot process an empty stream") if they are referenced here.
                    if readout_modes[q] != 3:
                        I_st[q].buffer(self.xeb_config.n_shots).map(FUNCTIONS.average()).buffer(
                            len(self.xeb_config.depths)
                        ).save_all(f"I{q}")
                        Q_st[q].buffer(self.xeb_config.n_shots).map(FUNCTIONS.average()).buffer(
                            len(self.xeb_config.depths)
                        ).save_all(f"Q{q}")
                    state_st[q].buffer(self.xeb_config.n_shots).map(FUNCTIONS.average()).buffer(
                        len(self.xeb_config.depths)
                    ).save_all(f"state{q}")
                    leak_counts_st[q].buffer(len(self.xeb_config.depths)).save_all(f"leak{q}")
                    amp_st[q].buffer(self.xeb_config.depths[-1], 2, 2).save_all(f"amp_matrix_q{q}")
                for i in range(dim):
                    string = "s" + binary(i, n_qubits)
                    counts_st[i].buffer(len(self.xeb_config.depths)).save_all(string)

        return xeb_prog

    def run(
        self,
        simulate: bool = False,
        simulation_config: Optional[SimulationConfig] = None,
        qmm_cloud_simulator: Optional[QuantumMachinesManager] = None,
        **simulate_kwargs,
    ):
        """
        Run QUA program for the XEB experiment
        Args:
            simulate: Indicate if output should be simulated or not
            simulation_config: SimulationConfig object containing the parameters of the simulation
            qmm_cloud_simulator: QuantumMachinesManager object to simulate the experiment
            simulate_kwargs: Optional additional keyword arguments passed to `qm.simulate`

        Returns: XEBJob object containing the information about the experiment (including results)

        """
        # Compile the QUA program

        config = self.machine.generate_config()
        if simulation_config is None:
            simulation_config = SimulationConfig(duration=10_000)
        xeb_prog = self._xeb_prog(simulate=simulate)  # set simulate=True to get the amplitude matrix
        if simulate and qmm_cloud_simulator is not None:
            qmm = qmm_cloud_simulator
        else:
            qmm = self.machine.connect()
        qm = qmm.open_qm(config, close_other_machines=False)
        if simulate:
            with open("debug.py", "w+") as f:
                f.write(generate_qua_script(xeb_prog, config))
            job = qm.simulate(xeb_prog, simulate=simulation_config, **simulate_kwargs)
        elif self.xeb_config.generate_new_data:
            job = qm.execute(xeb_prog)
        else:
            warnings.warn("Running deactivated. Set generate_new_data to True to run the experiment."
                          "Use XEBResult.from_data() method to load data from a previous run.")
            return 

        return XEBJob(
            job,
            self.xeb_config,
            self.data_handler,
            self.available_combinations,
            False,
            simulate,
            qm=qm,
        )

    def simulate(self, backend: BackendV2):
        """
            Simulate the XEB experiment: To simulate it, you must provide an AerBackend object
             with a noise model corresponding to your experiments parameters.
            For instance,
        ```python
        from qiskit import Aer
        from qiskit.providers.aer import AerSimulator
        from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
        error1q = 0.07
        error2q = 0.03
        depol_error1q = depolarizing_error(error1q, 1)
        depol_error2q = depolarizing_error(error2q, 2)
        sq_gate_set = ["h", "t", "rx", "ry", "sw"] # Specify which gates are subject to noise
        noise_model = NoiseModel(basis_gates = sq_gate_set)
        noise_model.add_all_qubit_quantum_error(depol_error2q, ["cz"])
        noise_model.add_all_qubit_quantum_error(depol_error1q, sq_gate_set)
        backend = AerSimulator(noise_model=noise_model, method="density_matrix", basis_gates=noise_model.basis_gates)
        ```
            Args:
                backend: AerBackend object to simulate the experiment. Note that it should carry a noise model to see
                a fidelity decay.

            Returns: XEBJob object containing the information about the experiment (including results)

        """
        from qiskit_aer.backends.aerbackend import AerBackend

        assert isinstance(backend, AerBackend), "The backend should be an AerBackend object"
        num_qubits = len(self.qubits)
        random_gates = len(self.xeb_config.gate_set)
        sq_gates, counts_list, states_list, circuits_list = [], [], [], []
        two_qubit_gate_pattern = 0
        # Generate sequences
        for s in range(self.xeb_config.seqs):  # For each sequence
            circuits_list.append([])
            sq_gates.append(np.zeros((num_qubits, self.xeb_config.depths[-1]), dtype=int))
            for q in range(num_qubits):  # For each qubit
                # Generate random single qubit gates
                # Start the sequence with a random gate
                sq_gates[s][q][0] = np.random.randint(random_gates)
            for d_ in range(1, self.xeb_config.depths[-1]):  # Generate sequences of max_depth, to be truncated later
                for q in range(num_qubits):  # For each qubit
                    sq_gates[s][q][d_] = np.random.randint(random_gates)
                    # Make sure that the same gate is not applied twice in a row
                    while sq_gates[s][q][d_] == sq_gates[s][q][d_ - 1]:
                        sq_gates[s][q][d_] = np.random.randint(random_gates)
            for i, d in enumerate(self.xeb_config.depths):  # For each maximum depth
                # Define the circuit
                qc = QuantumCircuit(num_qubits)
                for d_ in range(d):  # Apply layers
                    for q in range(num_qubits):  # For each qubit, append single qubit gates
                        qc.append(self.xeb_config.gate_set[sq_gates[s][q][d_]].gate, [q])
                    qc.barrier()
                    # Apply two-qubit gates
                    if num_qubits > 2 and self.xeb_config.two_qb_gate is not None and len(self.qubit_pairs) > 0:
                        for i, combination in enumerate(self.available_combinations):
                            if i == two_qubit_gate_pattern:
                                for pair in combination:
                                    qc.append(self.xeb_config.two_qb_gate.gate, pair)
                                qc.barrier()
                                break
                        if two_qubit_gate_pattern == len(self.available_combinations) - 1:
                            two_qubit_gate_pattern = 0
                        else:
                            two_qubit_gate_pattern += 1
                        # qc.append(self.xeb_config.two_qb_gate.gate, [0, 1])

                two_qubit_gate_pattern = 0
                qc.save_density_matrix()  # Actual state, subject to noise simulation
                circuits_list[s].append(qc)

                # Simulate the circuit
                # Execute circuit (transpiled) and store counts
        circ_list = [
            circuits_list[s][i].measure_all(inplace=False)
            for s in range(self.xeb_config.seqs)
            for i in range(len(self.xeb_config.depths))
        ]
        transpiled_circs = circ_list
        job = backend.run(transpiled_circs, shots=self.xeb_config.n_shots)

        return XEBJob(job, self.xeb_config, self.data_handler, self.available_combinations, True)


class XEBJob:
    def __init__(
        self,
        running_job: Union[SimulatedJob, RunningQmJob, AerJob],
        xeb_config: XEBConfig,
        data_handler: DataHandler,
        available_combinations: List[Tuple[Tuple[int, int]]],
        simulate=False,
        hardware_simulate=False,
        qm=None,
    ):
        self.job = running_job
        self.available_combinations = available_combinations
        self._simulate = simulate
        self._hardware_simulate = hardware_simulate
        self._qm = qm
        self._qm_closed = False
        self._result_handles = self.job.result() if isinstance(running_job, AerJob) else self.job.result_handles
        if not isinstance(running_job, AerJob):
            self._result_handles.wait_for_all_values()
        self.xeb_config = xeb_config
        self.data_handler = data_handler
        self._gate_indices = np.zeros(
            (self.xeb_config.seqs, self.xeb_config.n_qubits, self.xeb_config.depths[-1]), dtype=int
        )
        self._circuits = self._get_circuits()

    def _get_circuits(self):
        """
        Returns the circuits generated by the XEB experiment
        Returns:
            List of lists of QuantumCircuit objects representing the circuits generated by the XEB experiment.
            The list is formatted as follows:
            - The first dimension corresponds to the sequence index
            - The second dimension corresponds to the depth index

        """
        circuits = []
        if self._simulate:
            assert isinstance(self.job, AerJob), "The job should be an AerJob object"
            idx = 0
            for s in range(self.xeb_config.seqs):
                circuits.append([])
                for _ in range(len(self.xeb_config.depths)):
                    circuits[s].append(self.job.circuits()[idx].remove_final_measurements(inplace=False))
                    circuits[s][-1].data.pop(-1)  # Remove save_density_matrix instruction
                    circuits[s][-1].measure_all(inplace=True)
                    idx += 1

        else:
            max_depth = self.xeb_config.depths[-1]
            n_qubits = self.xeb_config.n_qubits
            g = [self._result_handles.get(f"g{q}").fetch_all()["value"] for q in range(n_qubits)]

            for s in range(self.xeb_config.seqs):
                for d in range(max_depth):
                    for q in range(n_qubits):
                        self._gate_indices[s, q, d] = g[q][s, d]

            circuits = generate_circuits(self.xeb_config, self.gate_indices, self.available_combinations)
        return circuits

    def result(self, disjoint_processing: Optional[bool] = None):
        """
        Returns the results of the XEB experiment

        Args:
            disjoint_processing: Indicate if disjoint processing should be applied to the results
        Returns: XEBResult object containing the results of the experiment

        """

        if disjoint_processing is not None:
            assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
            self.xeb_config.disjoint_processing = disjoint_processing
        try:
            if self._simulate:
                result = self.job.result()
                counts = result.get_counts()
                dms = np.array([result.data(i)["density_matrix"].data for i in range(len(counts))])
                for count in counts:  # Fill in missing bit-strings with 0 counts
                    for key in [binary(i, self.xeb_config.n_qubits) for i in range(self.xeb_config.dim)]:
                        if key not in count.keys():
                            count[key] = 0
                states = [{f"state_{qubit.name}": 0.0 for qubit in self.xeb_config.qubits} for _ in range(len(counts))]
                for c, count in enumerate(counts):
                    for key, value in count.items():
                        for i, bit in enumerate(reversed(key)):
                            if bit == "1":
                                states[c][f"state_{self.xeb_config.qubits[i].name}"] += value
                states = {
                    key: np.reshape(
                        [state[key] / self.xeb_config.n_shots for state in states],
                        (self.xeb_config.seqs, len(self.xeb_config.depths)),
                    )
                    for key in states[0].keys()
                }

                counts = {
                    key: np.reshape(
                        [count[key] for count in counts],
                        (self.xeb_config.seqs, len(self.xeb_config.depths)),
                    )
                    for key in counts[0].keys()
                }

                saved_data = {"counts": counts, "states": states, "density_matrices": dms}
            else:

                gate_indices, states, counts, quadratures, amp_st, leak_counts = {}, {}, {}, {}, {}, {}
                result = self._result_handles
                readout_modes = _resolve_readout_modes(self.xeb_config)
                for q, qubit in enumerate(self.xeb_config.qubits):
                    gate_indices[f"g_{qubit.name}"] = result.get(f"g{q}").fetch_all()["value"]
                    # I/Q quadratures are only streamed for qubits read out in 2-level (G/E) mode.
                    if readout_modes[q] != 3:
                        quadratures[f"I_{qubit.name}"] = result.get(f"I{q}").fetch_all()["value"]
                        quadratures[f"Q_{qubit.name}"] = result.get(f"Q{q}").fetch_all()["value"]
                    states[f"state_{qubit.name}"] = result.get(f"state{q}").fetch_all()["value"]
                    leak_counts[f"leak_{qubit.name}"] = result.get(f"leak{q}").fetch_all()["value"]
                    if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                        amp_st[f"amp_matrix_{qubit.name}"] = result.get(f"amp_matrix_q{q}").fetch_all()["value"]

                n_qubits = self.xeb_config.n_qubits
                for i in range(self.xeb_config.dim):
                    counts[binary(i, n_qubits)] = result.get(f"s{binary(i, n_qubits)}").fetch_all()["value"]

                saved_data = {
                    **quadratures,
                    **states,
                    **counts,
                    **amp_st,
                    **leak_counts,
                    "gate_indices": self.gate_indices,
                    "gate_sequences": self.gate_sequences,
                }

            return XEBResult(
                self.xeb_config,
                self.circuits,
                counts,
                states,
                saved_data,
                self.data_handler if self.xeb_config.should_save_data else None,
            )
        finally:
            self.close()

    @property
    def circuits(self):
        """
        Returns the circuits generated by the XEB experiment
        Circuits are formatted as follows:
        - The first dimension corresponds to the sequence index
        - The second dimension corresponds to the depth index
        Returns:
            List of lists of QuantumCircuit objects representing the circuits generated by the XEB experiment.

        """
        return self._circuits

    @property
    def simulate(self):
        return self._simulate

    @property
    def hardware_simulate(self):
        return self._hardware_simulate

    @property
    def gate_indices(self):
        """
        Returns the gate indices of the XEB experiment in the form of a 3D numpy array (sequence, qubit, depth)
        """
        return self._gate_indices

    @property
    def gate_sequences(self):
        """
        Returns the gate sequences of the XEB experiment in the form of a 3D numpy array (sequence, qubit, depth)
        """
        gate_sequences = np.zeros(
            (self.xeb_config.seqs, self.xeb_config.n_qubits, self.xeb_config.depths[-1]), dtype=str
        )
        for s in range(self.xeb_config.seqs):
            for q in range(self.xeb_config.n_qubits):
                for d in range(self.xeb_config.depths[-1]):
                    gate_sequences[s, q, d] = self.xeb_config.gate_set[self.gate_indices[s, q, d]].name

        return gate_sequences

    def plot_simulated_samples(self):
        if self.hardware_simulate:
            samples = self.job.get_simulated_samples()
            plt.subplots(nrows=len(samples.keys()), sharex=True)
            for i, con in enumerate(samples.keys()):
                plt.subplot(len(samples.keys()), 1, i + 1)
                samples[con].plot()
                plt.title(con)
            plt.tight_layout()
            plt.show()
        else:
            warnings.warn("Simulated samples are not available because the job was run and not hardware-simulated.")

    def close(self):
        if self._qm is not None and not self._qm_closed:
            self._qm.close()
            self._qm_closed = True


class XEBResult:
    def __init__(
        self, xeb_config: XEBConfig, circuits, counts: Dict, states: Dict, saved_data, data_handler: DataHandler = None
    ):
        self.xeb_config = xeb_config
        self.circuits: List[List[QuantumCircuit]] = circuits
        self.counts = counts
        self.states = states
        self.data = saved_data
        self.data_handler = data_handler
        self.readout_modes = _resolve_readout_modes(xeb_config)
        # Leakage (|f>) counts per qubit, shape (seqs, len(depths)). Only populated for qubits
        # actually read out with `control_readout_mode`/`target_readout_mode` == 3 (for 2-level
        # qubits, the corresponding stream is always zero since leakage is never checked for them).
        self.leak_counts = {
            qubit_name: np.asarray(saved_data[f"leak_{qubit_name}"])
            for qubit_name, mode in zip(self.qubit_names, self.readout_modes)
            if mode == 3 and f"leak_{qubit_name}" in saved_data
        }
        (
            self._joint_measured_probs,
            self._disjoint_measured_probs,
            self._joint_expected_probs,
            self._disjoint_expected_probs,
            self._records,
            self._log_fidelities,
            self._linear_fidelities,
            self._singularities,
            self._outliers,
        ) = self.retrieve_data()

        self.data.update(
            {
                "joint_measured_probs": self._joint_measured_probs,
                "disjoint_measured_probs": self._disjoint_measured_probs,
                "joint_expected_probs": self._joint_expected_probs,
                "disjoint_expected_probs": self._disjoint_expected_probs,
                "valid_shots": self._valid_shots,
                "log_fidelities": self._log_fidelities,
                "linear_fidelities": (
                    np.array(self.linear_fidelities["fidelity"])
                    if not self.xeb_config.disjoint_processing
                    else np.array([fidelity["fidelity"] for fidelity in self.linear_fidelities])
                ),
                "singularities": self._singularities,
                "outliers": self._outliers,
            }
        )

        if self.xeb_config.should_save_data and self.data_handler is not None:
            save_data = {}
            for key, value in self.data.items():  # Remove the amplitude matrices from the saved data
                if "amp_matrix" in key:
                    continue
                save_data[key] = value
            self.data_handler.save_data(
                save_data,
                self.xeb_config.data_folder_name,
                metadata=self.xeb_config.as_dict(),
            )

    @classmethod
    def from_data(
        cls, directory: str, disjoint_processing: Optional[bool] = None, data_handler: Optional[DataHandler] = None
    ):
        """
        Retrieve the XEBResult object from a saved data file (JSON format)

        Args:
            directory: Directory of the saved data files (should contain data.json and node.json)
            disjoint_processing: Indicate if disjoint processing should be applied to the results
            data_handler: DataHandler object to handle the data
        """
        data: Dict = json.load(open(directory + "/data.json", "r"))
        arrays: Dict = np.load(directory + "/arrays.npz")
        metadata: Dict = json.load(open(directory + "/node.json", "r"))
        xeb_config = XEBConfig.from_dict(metadata["metadata"])
        if disjoint_processing is not None:
            assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
            xeb_config.disjoint_processing = disjoint_processing
        gate_indices = arrays["gate_indices"]
        circuits = generate_circuits(xeb_config, gate_indices, xeb_config.available_combinations)

        new_data = {"states": {}, "counts": {}, "quadratures": {}, "amp_st": {}}

        for key, value in data.items():
            if "state" in key:
                new_data["states"][key] = arrays[key]
            elif key.isnumeric():
                new_data["counts"][key] = arrays[key]
            elif "amp_matrix" in key:
                new_data["amp_st"][key] = arrays[key]
            elif key.startswith("I") or key.startswith("Q"):
                new_data["quadratures"][key] = arrays[key]
            else:
                if key in arrays:
                    new_data[key] = arrays[key]
                else:
                    new_data[key] = value

        return cls(xeb_config, circuits, new_data["counts"], new_data["states"], new_data, data_handler)

    @classmethod
    def from_node_results(
        cls,
        results: Dict,
        machine: QuAM,
        apply_confusion_matrix: Optional[bool] = None,
        disjoint_processing: Optional[bool] = None,
    ):
        """
        Rebuild an XEBResult from the ``node.results`` of a saved (hardware) XEB node, so the analysis can be
        re-run without a new measurement.

        Only the raw data (bitstring counts, qubit states, leakage counts, gate indices) is used; probabilities,
        fidelities and purity are recomputed, so analysis options can be changed freely.

        Args:
            results: ``node.results`` of the loaded node (must contain ``xeb_config``, ``gate_indices``, the
                bitstring counts ``"00"``, ``"01"``, ... and ``"state_<qubit>"``). Results saved from a
                ``qiskit_simulation`` run do not contain these keys and cannot be reloaded; run them again.
            machine: QuAM machine of the loaded node (used to resolve qubits and qubit pairs).
            apply_confusion_matrix: Overrides the saved setting when not None.
            disjoint_processing: Overrides the saved setting when not None.
        """

        def raw_array(key):
            if key not in results:
                raise KeyError(
                    f"'{key}' is missing from the loaded node results. Only results from a hardware XEB run "
                    "contain the raw data needed for re-analysis."
                )
            array = np.asarray(results[key])
            if array.dtype.kind not in "biuf":
                raise ValueError(f"'{key}' was not restored as a numeric array from the saved node.")
            return array

        if "xeb_config" not in results:
            raise KeyError("'xeb_config' is missing from the loaded node results.")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No gate macro or amp_matrix provided")
            xeb_config = XEBConfig.from_dict(dict(results["xeb_config"]), machine=machine)
        if apply_confusion_matrix is not None:
            xeb_config.apply_confusion_matrix = bool(apply_confusion_matrix)
        if disjoint_processing is not None:
            xeb_config.disjoint_processing = bool(disjoint_processing)

        # Creating the XEB object (no program is built or run) fills in the coupling map and the parallel gate
        # combinations that are needed to rebuild the circuits.
        xeb = XEB(xeb_config, machine=machine)

        n_qubits = xeb_config.n_qubits
        gate_indices = raw_array("gate_indices").astype(int)
        circuits = generate_circuits(xeb_config, gate_indices, xeb.available_combinations)

        counts = {binary(i, n_qubits): raw_array(binary(i, n_qubits)) for i in range(xeb_config.dim)}
        qubit_names_ = [qubit.name for qubit in xeb_config.qubits]
        states = {f"state_{name}": raw_array(f"state_{name}") for name in qubit_names_}

        saved_data = {**counts, **states, "gate_indices": gate_indices}
        for name in qubit_names_:
            for prefix in ("leak", "I", "Q"):
                key = f"{prefix}_{name}"
                if key in results:
                    saved_data[key] = np.asarray(results[key])

        return cls(xeb_config, circuits, counts, states, saved_data, None)

    @staticmethod
    def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
        corrected = np.clip(np.real_if_close(probabilities), 0, None).astype(float)
        norm = corrected.sum()
        if norm <= 0:
            return corrected
        return corrected / norm

    def _get_single_qubit_confusion_matrix(self, qubit_index: int) -> Optional[np.ndarray]:
        """
        Returns the 2x2 readout matrix ``M[measured, prepared]`` (columns sum to 1) of one qubit, so that
        ``p_measured = M @ p_true``.

        ``resonator.confusion_matrix`` (07b_IQ_Blobs) and ``resonator.gef_confusion_matrix`` (08d_IQ_Blobs_G_E_F)
        are stored as ``[prepared][measured]``, hence they are transposed here. For 3-level (G/E/F) readout the
        |f> outcome is discarded in the QUA program, so only the G/E block is used, renormalized per prepared state.
        """
        qubit = self.xeb_config.qubits[qubit_index]
        resonator = getattr(qubit, "resonator", None)
        if resonator is None:
            return None

        if self.readout_modes[qubit_index] == 3:
            gef = getattr(resonator, "gef_confusion_matrix", None)
            if gef is None:
                return None
            gef = np.asarray(gef, dtype=float)
            if gef.shape != (3, 3):
                return None
            block = gef[:2, :2]
            row_sums = block.sum(axis=1, keepdims=True)
            if np.any(row_sums <= 0):
                return None
            return (block / row_sums).T

        confusion = getattr(resonator, "confusion_matrix", None)
        if confusion is None:
            return None

        confusion = np.asarray(confusion, dtype=float)
        if confusion.shape != (2, 2):
            return None
        return confusion.T

    def _get_joint_confusion_matrix(self) -> Optional[np.ndarray]:
        """
        Returns the 4x4 readout matrix ``M[measured, prepared]`` in the bitstring order used by this experiment
        (index = b0 + 2*b1, i.e. ``xeb_config.qubits[0]`` is the least significant bit, as in Qiskit and in the QUA
        program), so that ``p_measured = M @ p_true``.

        ``qubit_pair.confusion`` (34_2Q_confusion_matrix) is stored as ``[measured][prepared]`` with the index
        ``2 * control + target`` (control is the most significant bit), so it is reordered when the first XEB qubit
        is the control qubit.

        With 3-level readout (or when the pair matrix is missing), the matrix is built from the single-qubit matrices
        (this ignores readout correlations between the two qubits).
        """
        if len(self.xeb_config.qubit_pairs) != 1 or self.xeb_config.n_qubits != 2:
            return None

        qp = self.xeb_config.qubit_pairs[0]

        if all(mode == 2 for mode in self.readout_modes):
            confusion = getattr(qp, "confusion", None)
            if confusion is not None:
                confusion = np.asarray(confusion, dtype=float)
                if confusion.shape == (4, 4):
                    first_qubit_name = getattr(self.xeb_config.qubits[0], "name", self.xeb_config.qubits[0])
                    control = getattr(qp, "qubit_control", None)
                    control_name = getattr(control, "name", control)
                    if first_qubit_name == control_name:
                        perm = [0, 2, 1, 3]
                        confusion = confusion[np.ix_(perm, perm)]
                    return confusion

        single_qubit_matrices = [self._get_single_qubit_confusion_matrix(q) for q in range(2)]
        if all(matrix is not None for matrix in single_qubit_matrices):
            return np.kron(single_qubit_matrices[1], single_qubit_matrices[0])
        return None

    def _apply_joint_confusion_correction(self, measured_probs: np.ndarray) -> np.ndarray:
        confusion = self._get_joint_confusion_matrix()
        if confusion is None:
            return measured_probs

        try:
            corrected = np.linalg.solve(confusion, measured_probs)
        except np.linalg.LinAlgError:
            warnings.warn("Joint confusion matrix is singular. Falling back to raw measured probabilities.")
            return measured_probs
        return self._normalize_probabilities(corrected)

    def _apply_single_qubit_confusion_correction(self, qubit_index: int, measured_probs: np.ndarray) -> np.ndarray:
        confusion = self._get_single_qubit_confusion_matrix(qubit_index)
        if confusion is None:
            return measured_probs

        try:
            corrected = np.linalg.solve(confusion, measured_probs)
        except np.linalg.LinAlgError:
            warnings.warn(
                f"Single-qubit confusion matrix for {self.qubit_names[qubit_index]} is singular. "
                "Falling back to raw measured probabilities."
            )
            return measured_probs
        return self._normalize_probabilities(corrected)

    def retrieve_data(self):
        """
        Retrieve the data from the XEB experiment

        Returns:
            measured_probs: Measured probabilities of the states
            expected_probs: Expected probabilities of the states
            records: Records of the experiment
            log_fidelities: Logarithmic fidelities
            linear_fidelities: Linear fidelities
            singularities: Singularities
            outliers: Outliers
        """
        dim = 2 ** len(self.xeb_config.qubits)
        n_qubits = len(self.xeb_config.qubits)
        seqs = self.xeb_config.seqs
        depths = self.xeb_config.depths
        counts = self.counts
        states = self.states

        if self.xeb_config.disjoint_processing and any(mode == 3 for mode in self.readout_modes):
            raise ValueError(
                "disjoint_processing is not supported with 3-level (G/E/F) readout: the per-qubit state stream "
                "contains the |f> outcome. Set control_readout_mode/target_readout_mode to 2 or disable "
                "disjoint_processing."
            )

        existing_data = "joint_expected_probs" in self.data.keys()
        # Number of shots that contribute to the bitstring counts (shots with |f> on either qubit are discarded)
        if existing_data and "valid_shots" in self.data:
            valid_shots = np.asarray(self.data["valid_shots"], dtype=float)
        else:
            valid_shots = np.full((seqs, len(depths)), float(self.xeb_config.n_shots))

        if not existing_data:
            joint_expected_probs = np.zeros((seqs, len(depths), dim))
            joint_measured_probs = np.zeros((seqs, len(depths), dim))

            disjoint_expected_probs = np.zeros((n_qubits, seqs, len(depths), 2))
            disjoint_measured_probs = np.zeros((n_qubits, seqs, len(depths), 2))
        else:
            joint_expected_probs = self.data["joint_expected_probs"]
            joint_measured_probs = self.data["joint_measured_probs"]

            disjoint_expected_probs = self.data["disjoint_expected_probs"]
            disjoint_measured_probs = self.data["disjoint_measured_probs"]

        if not self.xeb_config.disjoint_processing:
            records, singularity, outlier = [], [], []
            incoherent_distribution = np.ones(dim) / dim
            log_fidelities = np.zeros((seqs, len(depths)))

        else:
            records = [[] for _ in range(n_qubits)]
            singularity = [[] for _ in range(n_qubits)]
            outlier = [[] for _ in range(n_qubits)]
            incoherent_distribution = np.ones(2) / 2
            log_fidelities = np.zeros((n_qubits, seqs, len(depths)))

        for s in range(seqs):
            for d_, depth in enumerate(depths):
                qc = self.circuits[s][d_].remove_final_measurements(inplace=False)
                if not existing_data:
                    statevector = Statevector(qc)
                    joint_expected_probs[s, d_] = statevector.probabilities()
                    joint_raw_counts = np.array(
                        [counts[binary(i, n_qubits)][s][d_] for i in range(dim)], dtype=float
                    )
                    # Normalize over the shots that were actually counted: shots where a qubit was found in |f>
                    # are not part of the bitstring counts. Leakage is reported separately (leakage_probs).
                    valid_shots[s, d_] = joint_raw_counts.sum()
                    if valid_shots[s, d_] > 0:
                        joint_raw_measured_probs = joint_raw_counts / valid_shots[s, d_]
                    else:
                        joint_raw_measured_probs = np.zeros(dim)
                    if self.xeb_config.apply_confusion_matrix:
                        joint_measured_probs[s, d_] = self._apply_joint_confusion_correction(joint_raw_measured_probs)
                    else:
                        joint_measured_probs[s, d_] = joint_raw_measured_probs

                    for q in range(n_qubits):
                        disjoint_expected_probs[q, s, d_] = statevector.probabilities([q])
                        qubit_state = states[f"state_{self.qubit_names[q]}"][s, d_]
                        disjoint_raw_measured_probs = np.array([1 - qubit_state, qubit_state])
                        if self.xeb_config.apply_confusion_matrix:
                            disjoint_measured_probs[q, s, d_] = self._apply_single_qubit_confusion_correction(
                                q, disjoint_raw_measured_probs
                            )
                        else:
                            disjoint_measured_probs[q, s, d_] = disjoint_raw_measured_probs

                if not self.xeb_config.disjoint_processing:
                    # Calculate the cross-entropy fidelities (logarithmic)
                    f_xeb = compute_log_fidelity(
                        incoherent_distribution, joint_expected_probs[s, d_], joint_measured_probs[s, d_]
                    )
                    log_fidelities[s, d_] = evaluate_log_fidelity(f_xeb, singularity, outlier, s, int(depth))

                    # Store records for linear XEB post-processing
                    records = update_record(
                        records, s, depth, joint_expected_probs[s, d_], joint_measured_probs[s, d_], dim
                    )

                else:
                    for q, qubit_name in enumerate(self.qubit_names):
                        # Calculate the cross-entropy fidelities (logarithmic)
                        f_xeb = compute_log_fidelity(
                            incoherent_distribution,
                            disjoint_expected_probs[q, s, d_],
                            disjoint_measured_probs[q, s, d_],
                        )
                        log_fidelities[q, s, d_] = evaluate_log_fidelity(f_xeb, singularity[q], outlier[q], 
                                                                         s, int(depth))
                        # Store records for linear XEB post-processing
                        records[q] = update_record(
                            records[q],
                            s,
                            depth,
                            disjoint_expected_probs[q, s, d_],
                            disjoint_measured_probs[q, s, d_],
                            2,
                        )

        def per_cycle_depth(df):
            denominator = df["denominator"].sum()
            # Depths whose ideal distribution is uniform for every circuit (e.g. depth 1: sqrt-gates on |0> give
            # 50/50 and CZ only adds a phase) carry no XEB signal: 0/0 up to float noise -> undefined.
            if denominator < _MIN_XEB_SIGNAL * len(df):
                return pd.Series({"fidelity": np.nan})
            fid_lsq = df["numerator"].sum() / denominator
            return pd.Series({"fidelity": fid_lsq})

        if not self.xeb_config.disjoint_processing:
            df = update_data_frame(pd.DataFrame(records))
            linear_fidelities = df.groupby("depth").apply(per_cycle_depth).reset_index()
        else:
            df, linear_fidelities = [], []
            for q in range(n_qubits):
                df_q = update_data_frame(pd.DataFrame(records[q]))
                linear_fidelities.append(df_q.groupby("depth").apply(per_cycle_depth).reset_index())
                df.append(df_q)

        if np.isnan(log_fidelities).all():
            warnings.warn("All fidelities computed from log-entropies are singularities.")

        self._valid_shots = valid_shots

        return (
            joint_measured_probs,
            disjoint_measured_probs,
            joint_expected_probs,
            disjoint_expected_probs,
            df,
            log_fidelities,
            linear_fidelities,
            singularity,
            outlier,
        )

    def get_layer_fidelity(self, fidelity_metric: Literal["log", "linear"] = "linear", disjoint_processing: bool = None):
        """
        Returns the layer fidelities for the XEB experiment
        Args:
            fidelity_metric: Indicate which fidelity metric should be computed: "log" or "linear"
            disjoint_processing: Indicate if disjoint processing should be applied to the results
        """
        if disjoint_processing is not None:
            assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
        else:
            disjoint_processing = self.xeb_config.disjoint_processing
        
        if disjoint_processing:
            if fidelity_metric == "log":
                Fxeb = np.nanmean(self.log_fidelities, axis=1)
            else:
                Fxeb = np.array([fidelity["fidelity"] for fidelity in self.linear_fidelities])
            
            a = [None] * len(self.qubit_names)
            layer_fid = [None] * len(self.qubit_names)
            for q, qubit in enumerate(self.qubit_names):
                a[q], layer_fid[q], *_ = fit_exponential_decay(self.xeb_config.depths, Fxeb[q])
        else:
            if fidelity_metric == "log":
                Fxeb = np.nanmean(self.log_fidelities, axis=0)
            else:
                Fxeb = np.array(self.linear_fidelities["fidelity"])
            a, layer_fid, *_ = fit_exponential_decay(self.xeb_config.depths, Fxeb)
        return layer_fid


    def plot_fidelities(self, fit_linear: bool = True, fit_log_entropy: bool = True, separate_plots: bool = False):
        """
        Plot the cross-entropy fidelities for the XEB experiment
        Args:
            fit_linear: Indicate if the linear XEB data should be fitted
            fit_log_entropy: Indicate if the log-entropy XEB data should be fitted
            separate_plots: Indicate if the fidelities should be plotted on separate plots (one per qubit, relevant
                only when disjoint_processing is True)
        Returns:
        """
        figs = [plt.figure()]
        plt.rcParams["text.usetex"] = False

        def plot_fidelity_data(xx, depths, linear_fidelities, Fxeb, qubit_label=""):
            try:
                if fit_linear:
                    a_lin, layer_fid_lin, *_ = fit_exponential_decay(
                        linear_fidelities["depth"], linear_fidelities["fidelity"]
                    )
                    plt.plot(
                        xx,
                        exponential_decay(xx, a_lin, layer_fid_lin),
                        label=f"Fit (Linear XEB{qubit_label}), layer_fidelity={layer_fid_lin * 100:.1f}%",
                    )
            except Exception:
                warnings.warn("Fit for Linear XEB data failed")

            try:
                if fit_log_entropy:
                    a_log, layer_fid_log, *_ = fit_exponential_decay(depths, Fxeb)
                    plt.plot(
                        xx,
                        exponential_decay(xx, a_log, layer_fid_log),
                        label=f"Fit (Log XEB{qubit_label}), layer_fidelity={layer_fid_log * 100:.1f}%",
                        linewidth=5.7, color="red",
                    )
            except Exception:
                warnings.warn("Fit for Log XEB data failed")

            if fit_linear:
                mask_lin = (linear_fidelities["fidelity"] > 0) & (linear_fidelities["fidelity"] < 1)
                plt.scatter(
                    linear_fidelities["depth"][mask_lin],
                    linear_fidelities["fidelity"][mask_lin],
                    label=f"Linear XEB Data {qubit_label}",
                )

            if fit_log_entropy and not np.isnan(Fxeb).all():
                mask_log = (Fxeb > 0) & (Fxeb < 1)
                plt.scatter(depths[mask_log], Fxeb[mask_log], label=f"Log XEB Data {qubit_label}", 
                            s=13.5, c="blue")
            else:
                warnings.warn(f"Log XEB data for {qubit_label} is a singularity.")
        ret_figs = []
        if self.xeb_config.disjoint_processing:
            for q, qubit in enumerate(self.qubit_names):
                if separate_plots and q > 0:
                    figs.append(plt.figure())
                linear_fidelities = self.linear_fidelities[q]
                xx = np.linspace(0, linear_fidelities["depth"].max())
                Fxeb = np.nanmean(self.log_fidelities[q], axis=0)
                plot_fidelity_data(xx, self.xeb_config.depths, linear_fidelities, Fxeb, f" {qubit}")
                plt.ylabel("Circuit fidelity", fontsize=20)
                plt.xlabel("Cycle Depth $d$", fontsize=20)
                plt.title("XEB Fidelity")
                plt.legend(loc="best")
                if separate_plots:
                    plt.tight_layout()
                    plt.show()
                ret_figs.append(plt.gcf())
        else:
            xx = np.linspace(0, self.linear_fidelities["depth"].max())
            Fxeb = np.nanmean(self.log_fidelities, axis=0)
            plot_fidelity_data(xx, self.xeb_config.depths, self.linear_fidelities, Fxeb)
            plt.ylabel("Circuit fidelity", fontsize=20)
            plt.xlabel("Cycle Depth $d$", fontsize=20)
            plt.title("XEB Fidelity")
            plt.legend(loc="best")
            plt.tight_layout()
            ret_figs.append(plt.gcf())
            plt.show()

        return ret_figs
    
    def plot_records(self):
        """
        Plot the records for the XEB experiment
        Returns:

        """
        depths = self.xeb_config.depths
        colors = sns.cubehelix_palette(n_colors=len(depths))
        colors = {k: colors[i] for i, k in enumerate(depths)}
        _lines = []

        def per_cycle_depth(df, _lines=None):
            fid_lsq = df["numerator"].sum() / df["denominator"].sum()
            cycle_depth = df.name
            xx = np.linspace(0, df["x"].max())
            (l,) = plt.plot(xx, fid_lsq * xx, color=colors[cycle_depth])
            plt.scatter(df["x"], df["y"], color=colors[cycle_depth])
            _lines += [l]  # for legend
            return pd.Series({"fidelity": fid_lsq})

        ret_figs = []
        if not self.xeb_config.disjoint_processing:
            plt.figure()
            fids = self.records.groupby("depth").apply(per_cycle_depth, _lines).reset_index()
            plt.xlabel(r"$e_U - u_U$", fontsize=18)
            plt.ylabel(r"$m_U - u_U$", fontsize=18)
            _lines = np.asarray(_lines)
            plt.legend(_lines[[0, -1]], depths[[0, -1]], loc="best", title="Cycle depth")
            title = "Fxeb_linear = %s" % [fids["fidelity"][x] for x in [0, 1]]
            plt.title(title)
            plt.tight_layout()
            ret_figs.append(plt.gcf())
            
        else:
            fids = []
            for i, q in enumerate(self.qubit_names):
                _lines = []
                plt.figure()
                fids.append(self.records[i].groupby("depth").apply(per_cycle_depth, _lines).reset_index())
                plt.xlabel(r"$e_U - u_U$", fontsize=18)
                plt.ylabel(r"$m_U - u_U$", fontsize=18)
                _lines = np.asarray(_lines)
                plt.legend(_lines[[0, -1]], depths[[0, -1]], loc="best", title="Cycle depth")
                plt.title(
                    "q-%s: Fxeb_linear = %s"
                    % (
                        q,
                        [fids[i]["fidelity"][x] for x in [0, 1]],
                    )
                )
                ret_figs.append(plt.gcf())
                plt.show()
        
        return ret_figs

    def plot_state_heatmap(self):
        """
        Plot a comparison between expected and actual probability distributions for all sequences.

        This method creates subplots within a single figure to display the state heatmaps. If the number of plots exceeds
        the subplot capacity, it generates a new figure containing the remaining subplots.

        The method handles both disjoint and non-disjoint processing cases.

        Returns:
            None
        """
        titles, data = [], []
        if not self.xeb_config.disjoint_processing:
            for i in range(self.xeb_config.dim):
                titles.append(f"<{binary(i, self.xeb_config.n_qubits)}> Measured")
                titles.append(f"<{binary(i, self.xeb_config.n_qubits)}> Expected")
                data.append(self.measured_probs[:, :, i])
                data.append(self.expected_probs[:, :, i])
        else:
            for i, q in enumerate(self.qubit_names):
                for j in range(2):
                    titles.append(f"{q}<{j}> Measured")
                    titles.append(f"{q}<{j}> Expected")
                    data.append(self.disjoint_measured_probs[i, :, :, j])
                    data.append(self.disjoint_expected_probs[i, :, :, j])

        num_plots = len(titles)
        plots_per_fig = 4  # Adjust this value based on the desired grid size (e.g., 2x2 grid)
        num_figs = (num_plots + plots_per_fig - 1) // plots_per_fig
        ret_figs = []
        for fig_idx in range(num_figs):
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Adjust the grid size as needed
            axs = axs.flatten()
            start_idx = fig_idx * plots_per_fig
            end_idx = min(start_idx + plots_per_fig, num_plots)

            for plot_idx in range(start_idx, end_idx):
                ax = axs[plot_idx - start_idx]
                ax.pcolor(
                    self.xeb_config.depths,
                    range(self.xeb_config.seqs),
                    np.abs(data[plot_idx]),
                    vmin=0,
                    vmax=1,
                    cmap="viridis",
                )
                ax.set_title(titles[plot_idx])
                ax.set_xlabel("Circuit depth")
                ax.set_ylabel("Sequences")
                ax.set_xticks(self.xeb_config.depths)
                ax.set_yticks(np.arange(1, self.xeb_config.seqs + 1))
                fig.colorbar(
                    ax.pcolor(
                        self.xeb_config.depths,
                        range(self.xeb_config.seqs),
                        np.abs(data[plot_idx]),
                        vmin=0,
                        vmax=1,
                    ),
                    ax=ax,
                )

            plt.tight_layout()
            plt.show()
            ret_figs.append(fig)
        return ret_figs

    @property
    def measured_probs(self):
        """
        Measured probabilities of the states
        Returns: Measured probabilities of the states
        """
        return self._joint_measured_probs

    @property
    def disjoint_measured_probs(self):
        """
        Measured probabilities of the states for disjoint processing
        Returns: Measured probabilities of the states for disjoint processing
        """
        return self._disjoint_measured_probs

    @property
    def expected_probs(self):
        """
        Expected probabilities of the states
        Returns: Expected probabilities of the states
        """
        return self._joint_expected_probs

    @property
    def disjoint_expected_probs(self):
        """
        Expected probabilities of the states for disjoint processing
        Returns: Expected probabilities of the states for disjoint processing
        """
        return self._disjoint_expected_probs

    @property
    def records(self):
        """
        Records of the experiment
        Returns: Records of the experiment
        """
        return self._records

    @property
    def log_fidelities(self):
        """
        Logarithmic fidelities
        Returns: Logarithmic fidelities
        """
        return self._log_fidelities

    @property
    def linear_fidelities(self):
        """
        Linear fidelities
        Returns: Linear fidelities
        """
        if self.xeb_config.disjoint_processing:
            fidelities = [self._linear_fidelities[q] for q in range(self.xeb_config.n_qubits)]
        else:
            fidelities = self._linear_fidelities
        return fidelities

    @property
    def singularities(self):
        """
        Singularities
        Returns: Singularities
        """
        return self._singularities

    @property
    def outliers(self):
        """
        Outliers
        Returns: Outliers
        """
        return self._outliers

    @property
    def purities(self):
        """
        Estimated purities of final states, computed from the variance of the measured probabilities
        Returns: Purities
        """
        dim = 2**self.xeb_config.n_qubits
        var_pt = (dim - 1) / (dim**2 * (dim + 1))
        purities = np.var(self.measured_probs, axis=-1) / var_pt
        return purities

    def _shot_noise_variance(self) -> np.ndarray:
        """
        Expected contribution of finite-shot (multinomial) noise to the pooled variance of the measured
        probabilities, per depth. The raw counts have covariance (diag(p) - p p^T) / N; when the confusion
        matrix correction is applied, this covariance is propagated through the inverse confusion matrix
        (ignoring the clipping/renormalization of negative entries).

        Returns: array of shape (len(depths),)
        """
        probs = np.asarray(self.measured_probs, dtype=float)
        n_seqs, n_depths, dim = probs.shape
        valid_shots = np.maximum(np.asarray(self._valid_shots, dtype=float)[:n_seqs, :n_depths], 1.0)

        inverse = np.eye(dim)
        if self.xeb_config.apply_confusion_matrix:
            confusion = self._get_joint_confusion_matrix()
            if confusion is not None:
                try:
                    inverse = np.linalg.inv(confusion)
                except np.linalg.LinAlgError:
                    pass

        cov = np.einsum("sdi,ij->sdij", probs, np.eye(dim)) - np.einsum("sdi,sdj->sdij", probs, probs)
        cov = cov / valid_shots[:, :, None, None]
        noise_diag = np.einsum("ij,sdjk,ik->sdi", inverse, cov, inverse)
        return noise_diag.mean(axis=(0, 2))

    def calculate_speckle_purity(self, correct_shot_noise: bool = True) -> np.ndarray:
        """
        Speckle purity at each depth, Arute et al., Nature 574, 505 (2019), Eq. (49):
        purity = Var(p) * D^2 * (D + 1) / (D - 1).

        The variance is pooled across sequences and bitstrings. The mean of a normalized distribution is exactly
        1/D, so the variance is computed around 1/D. sqrt(purity) decays with the incoherent (coherence-limited)
        error; linear XEB decays with the total error.

        Args:
            correct_shot_noise: Subtract the expected finite-shot noise contribution to Var(p). Without it, the
                raw estimator is biased upwards (about 1% purity for D=4 and 500 shots) and never decays below
                that floor. With the correction the estimate is unbiased but can be slightly negative at large
                depth, so fit it directly (not through its square root).
        """
        if self.xeb_config.disjoint_processing:
            raise ValueError("Speckle purity uses the joint bitstring distribution.")
        probs = np.asarray(self.measured_probs)
        n_depths = probs.shape[1]
        dim = 2**self.xeb_config.n_qubits
        flat = probs.transpose(0, 2, 1).reshape(-1, n_depths)
        var_p = np.mean((flat - 1.0 / dim) ** 2, axis=0)
        if correct_shot_noise:
            var_p = var_p - self._shot_noise_variance()
        purity = var_p * (dim**2 * (dim + 1) / (dim - 1))
        self.average_state_purity = purity
        return purity

    def coherence_limit_comparison(self, min_depth: int = 6, correct_shot_noise: bool = True):
        """
        Separate the per-cycle error into incoherent and coherent parts.

        Two decays are fitted with the same model (``A * base**depth``), the same depth range and the same bounds:

        - linear XEB          -> ``F_total``  (total error, ``1 - F_total``)
        - speckle purity      -> ``base = f_cycle**2``, i.e. ``f_cycle = sqrt(base)`` (incoherent error,
          ``1 - f_cycle``). Purity is fitted directly because the shot-noise-corrected estimate can be negative
          at large depth, which would bias a fit of ``sqrt(purity)``.

        Unitary (coherent) errors such as a wrong CZ conditional phase or uncompensated local Z phases lower the
        overlap with the ideal circuit (linear XEB) but not the purity, so
        ``coherent_error_per_cycle = f_cycle - F_total`` (= total error - incoherent error).
        Decoherence, leakage and stochastic noise lower both and cancel in the difference.

        Depths below ``min_depth`` are dropped from both fits when at least four deeper points remain.

        Returns (summary, figure).
        """
        purity = np.asarray(self.calculate_speckle_purity(correct_shot_noise=correct_shot_noise), dtype=float)
        xs = np.asarray(self.xeb_config.depths, dtype=float)
        linear = self.linear_fidelities
        xeb_depths = np.asarray(linear["depth"], dtype=float)
        xeb_fidelity = np.asarray(linear["fidelity"], dtype=float)

        # Depths where linear XEB is undefined (uniform ideal distribution, e.g. depth 1) carry no information for
        # the purity either (it is just shot noise there): leave them out of the purity fit and the plot.
        signal_depths = np.isin(xs, xeb_depths[np.isfinite(xeb_fidelity)])
        purity_valid = np.where(signal_depths, purity, np.nan)

        fit_xeb = _fit_decay_zero(xeb_depths, xeb_fidelity, min_depth=min_depth)
        fit_purity = _fit_decay_zero(xs, purity_valid, min_depth=min_depth)
        curve_depths = xs[signal_depths]

        f_total = fit_xeb["base"]
        f_total_stderr = fit_xeb["base_stderr"]
        purity_base = fit_purity["base"]
        f_cycle = float(np.sqrt(purity_base))
        # delta method: d sqrt(g) / dg = 1 / (2 sqrt(g))
        f_cycle_stderr = float(fit_purity["base_stderr"] / (2.0 * f_cycle)) if f_cycle > 0 else float("inf")
        coherent_error = f_cycle - f_total
        coherent_error_stderr = float(np.hypot(f_total_stderr, f_cycle_stderr))
        summary = {
            "estimator": "pooled Arute purity" + (", shot-noise corrected" if correct_shot_noise else ", raw"),
            "min_depth": min_depth,
            "fit_depths": fit_xeb["depths"],
            "F_total": f_total,
            "F_total_stderr": f_total_stderr,
            "f_cycle": f_cycle,
            "f_cycle_stderr": f_cycle_stderr,
            "f_cycle_minus_F_total": coherent_error,
            "total_error_per_cycle": 1.0 - f_total,
            "total_error_stderr": f_total_stderr,
            "incoherent_error_per_cycle": 1.0 - f_cycle,
            "incoherent_error_stderr": f_cycle_stderr,
            "coherent_error_per_cycle": coherent_error,
            "coherent_error_stderr": coherent_error_stderr,
            "purity": purity,
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(xeb_depths, xeb_fidelity, "o", color="C0", label="Linear XEB")
        ax.plot(
            curve_depths,
            fit_xeb["amplitude"] * f_total**curve_depths,
            "-",
            color="C0",
            label=f"F_total = {f_total:.5f} (err {1 - f_total:.2e})",
        )
        ax.plot(
            curve_depths,
            np.sqrt(np.clip(purity[signal_depths], 0.0, None)),
            "s",
            color="C1",
            label=r"$\sqrt{\mathrm{Purity}}$",
        )
        ax.plot(
            curve_depths,
            np.sqrt(fit_purity["amplitude"]) * f_cycle**curve_depths,
            "--",
            color="C1",
            label=f"f_cycle = {f_cycle:.5f} (err {1 - f_cycle:.2e})",
        )
        ax.set_xlabel("Cycle depth")
        ax.set_ylabel(r"XEB fidelity, $\sqrt{\mathrm{Purity}}$")
        ax.set_title(
            f"coherent error = f_cycle - F_total = {coherent_error:+.2e} +/- {coherent_error_stderr:.1e}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return summary, fig

    @property
    def qubit_names(self):
        """
        Qubit names
        """
        return [qubit.name if isinstance(qubit, Transmon) else qubit for qubit in self.xeb_config.qubits]

    @property
    def leakage_probs(self) -> Dict[str, np.ndarray]:
        """
        Measured leakage (|f>) probability per qubit, depth and sequence, for qubits configured with
        3-level (G/E/F) readout (`control_readout_mode`/`target_readout_mode` == 3).

        Returns:
            Dict mapping qubit name to a (seqs, len(depths)) array of leakage probabilities. Only
            includes qubits for which 3-level readout data is available.
        """
        return {name: counts / self.xeb_config.n_shots for name, counts in self.leak_counts.items()}

    def plot_leakage(self):
        """
        Plot the leakage (|f>) probability vs. circuit depth, averaged over sequences (with the
        standard error of the mean as error bars), for every qubit configured with 3-level readout.

        Returns: Matplotlib figure, or None if no qubit was configured with 3-level readout.
        """
        if not self.leak_counts:
            warnings.warn(
                "No leakage data available: set control_readout_mode/target_readout_mode to 3 to enable it."
            )
            return None

        depths = self.xeb_config.depths
        fig = plt.figure()
        for qubit_name, probs in self.leakage_probs.items():
            mean = np.mean(probs, axis=0)
            sem = np.std(probs, axis=0) / np.sqrt(max(probs.shape[0], 1))
            plt.errorbar(depths, mean, yerr=sem, marker="o", capsize=3, label=f"{qubit_name} leakage to |f>")
        plt.xlabel("Cycle Depth $d$", fontsize=14)
        plt.ylabel("Leakage probability", fontsize=14)
        plt.title("XEB leakage vs. depth")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
        return fig

    def plot_leakage_heatmap(self):
        """
        Plot a heatmap (sequences vs. depths) of the leakage (|f>) probability for every qubit
        configured with 3-level readout.

        Returns: List of matplotlib figures (one per qubit with 3-level readout data), or an empty
            list if no qubit was configured with 3-level readout.
        """
        if not self.leak_counts:
            warnings.warn(
                "No leakage data available: set control_readout_mode/target_readout_mode to 3 to enable it."
            )
            return []

        ret_figs = []
        for qubit_name, probs in self.leakage_probs.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            mesh = ax.pcolor(self.xeb_config.depths, range(self.xeb_config.seqs), probs, vmin=0, vmax=1, cmap="magma")
            ax.set_title(f"{qubit_name}: leakage to |f> probability")
            ax.set_xlabel("Circuit depth")
            ax.set_ylabel("Sequences")
            ax.set_xticks(self.xeb_config.depths)
            fig.colorbar(mesh, ax=ax)
            plt.tight_layout()
            plt.show()
            ret_figs.append(fig)
        return ret_figs

    def estimate_2q_unitary(
        self,
        x0: Tuple[float, float, float, float] = (0.0, np.pi, 0.0, 0.0),
        method: str = "Nelder-Mead",
        max_depth: Optional[int] = None,
        min_fit_depth: int = 6,
        **minimize_kwargs,
    ) -> Dict:
        """
        Fit an *effective* two-qubit unitary to the measured XEB data.

        The two-qubit gate is modeled as a generic fSim gate with local Z corrections (see
        `two_qubit_unitary_from_fsim_params`), parameterized by (theta_iswap, phi_cphase, phi_rz1,
        phi_rz2). The ideal CZ gate corresponds to (0, pi, 0, 0), which is also the default starting
        point of the fit. The fit minimizes the average cross-entropy between the measured
        probabilities and the probabilities predicted by re-simulating every recorded random circuit
        with this parameterized two-qubit gate in place of the nominal one (e.g. the ideal CZ gate).

        This enables distinguishing *coherent* two-qubit gate miscalibration (captured by a
        significant deviation of the fitted parameters from (0, pi, 0, 0)) from *incoherent* error
        sources (decoherence, leakage, etc., which still show up as an exponential fidelity decay with
        depth even after fitting the best possible unitary).

        Args:
            x0: Initial guess for (theta_iswap, phi_cphase, phi_rz1, phi_rz2), in radians.
            method: Optimization method passed to `scipy.optimize.minimize` (default "Nelder-Mead").
            max_depth: Only depths up to ``max_depth`` enter the cross-entropy loss. The model has no
                depolarization parameter, so deep circuits (whose outcome is close to uniform) carry almost no
                information about the unitary and mostly add noise. ``None`` uses all depths.
            min_fit_depth: Shallowest depth used to fit the layer fidelity decay of the nominal/fitted unitary
                (same rule as ``coherence_limit_comparison``).
            minimize_kwargs: Additional keyword arguments passed to `scipy.optimize.minimize`.

        Returns:
            Dict with the fitted parameters (in radians and degrees), the fitted/nominal average
            cross-entropy loss, the `OptimizeResult.success` flag, and the re-simulated expected
            probabilities under the fitted unitary (`fitted_probs`, shape (seqs, len(depths), 4)),
            which can be passed to `plot_fidelity_with_unitary_fit` for visualization. It also contains the
            linear-XEB decay bases for the nominal gate (`layer_fid_nominal`) and the fitted unitary
            (`layer_fid_fitted`); their difference `coherent_error_unitary_fit` is the error that a unitary
            correction of the gate could remove. `nominal_consistency_error` is the largest deviation between
            the internal simulation of the nominal gate and the Qiskit expected probabilities and should be ~0.

        Bit order: the matrices are built with ``qubits[0]`` as the most significant bit (the convention of
        `two_qubit_unitary_from_fsim_params`) and the simulated probabilities are converted to the bitstring order
        of the measured data (``qubits[0]`` is the least significant bit) before they are compared.
        """
        if self.xeb_config.disjoint_processing:
            raise ValueError("2Q unitary estimation requires joint (not disjoint) processing.")
        if self.xeb_config.two_qb_gate is None:
            raise ValueError("2Q unitary estimation requires a two-qubit gate to be configured in the experiment.")
        if self.xeb_config.n_qubits != 2:
            raise ValueError("2Q unitary estimation only supports the standard 2-qubit XEB use case.")

        gate_set = self.xeb_config.gate_set
        sq_matrices = [gate_set[i].gate.to_matrix() for i in range(len(gate_set))]
        gate_indices = np.asarray(self.data["gate_indices"])  # shape (seqs, n_qubits, max_depth)
        depths = self.xeb_config.depths
        seqs = self.xeb_config.seqs
        measured = self.measured_probs  # shape (seqs, len(depths), 4)

        # Index permutation from "qubits[0] is the MSB" (kron order) to "qubits[0] is the LSB" (measured data)
        msb_to_lsb = [0, 2, 1, 3]

        depths_arr = np.asarray(depths)
        if max_depth is None:
            n_fit_depths = len(depths_arr)
        else:
            n_fit_depths = int(np.count_nonzero(depths_arr <= max_depth))
            if n_fit_depths == 0:
                raise ValueError(f"No depth is <= max_depth={max_depth}.")

        def simulate(theta, phi, rz1, rz2, n_depths=len(depths)):
            u2 = two_qubit_unitary_from_fsim_params(theta, phi, rz1, rz2)
            probs = np.zeros((seqs, len(depths), 4))
            for s in range(seqs):
                psi = np.zeros(4, dtype=complex)
                psi[0] = 1.0
                ptr = 0
                for d_, depth in enumerate(depths[:n_depths]):
                    while ptr < depth:
                        g0, g1 = gate_indices[s, 0, ptr], gate_indices[s, 1, ptr]
                        psi = np.kron(sq_matrices[g0], sq_matrices[g1]) @ psi
                        psi = u2 @ psi
                        ptr += 1
                    probs[s, d_] = (np.abs(psi) ** 2)[msb_to_lsb]
            return probs

        nominal_params = np.array([0.0, np.pi, 0.0, 0.0])
        nominal_probs = simulate(*nominal_params)
        nominal_consistency_error = float(np.max(np.abs(nominal_probs - np.asarray(self.expected_probs))))
        if nominal_consistency_error > 1e-3:
            warnings.warn(
                "The internal simulation of the nominal CZ circuit differs from the Qiskit expected probabilities "
                f"(max deviation {nominal_consistency_error:.2e}); the 2Q unitary fit is not reliable."
            )

        def loss(params):
            probs = simulate(*params, n_depths=n_fit_depths)
            return float(
                np.mean(
                    [cross_entropy(measured[s, d_], probs[s, d_]) for s in range(seqs) for d_ in range(n_fit_depths)]
                )
            )

        loss_nominal = loss(nominal_params)
        opt_result = minimize(loss, x0=np.asarray(x0, dtype=float), method=method, **minimize_kwargs)
        theta, phi, rz1, rz2 = opt_result.x
        if not opt_result.success:
            warnings.warn(f"2Q unitary fit did not converge: {opt_result.message}")
        fitted_probs = simulate(theta, phi, rz1, rz2)

        # Decay bases of linear XEB for the nominal gate and the fitted unitary (same fit as F_total)
        layer_fid_nominal, layer_fid_fitted = np.nan, np.nan
        try:
            depths_nom, fid_nom, _ = self._linear_xeb_with_stderr(self.expected_probs)
            depths_fit, fid_fit, _ = self._linear_xeb_with_stderr(fitted_probs, signal_probs=self.expected_probs)
            layer_fid_nominal = _fit_decay_zero(depths_nom, fid_nom, min_depth=min_fit_depth)["base"]
            layer_fid_fitted = _fit_decay_zero(depths_fit, fid_fit, min_depth=min_fit_depth)["base"]
        except Exception as e:
            warnings.warn(f"Could not fit the layer fidelity decay for the unitary fit comparison: {e}")

        fit_summary = {
            "theta_iswap": theta,
            "phi_cphase": phi,
            "phi_rz1": rz1,
            "phi_rz2": rz2,
            "theta_iswap_deg": np.degrees(theta),
            "cphase_error_deg": np.degrees(phi - np.pi),
            "rz1_deg": np.degrees(rz1),
            "rz2_deg": np.degrees(rz2),
            "loss_fitted": float(opt_result.fun),
            "loss_nominal": loss_nominal,
            "success": bool(opt_result.success),
            "fit_max_depth": max_depth,
            "min_fit_depth": min_fit_depth,
            "nominal_consistency_error": nominal_consistency_error,
            "layer_fid_nominal": layer_fid_nominal,
            "layer_fid_fitted": layer_fid_fitted,
            "coherent_error_unitary_fit": float(layer_fid_fitted - layer_fid_nominal),
            "fitted_probs": fitted_probs,
        }
        self._unitary_fit = fit_summary
        return fit_summary

    def _linear_xeb_with_stderr(self, expected_probs: np.ndarray, signal_probs: Optional[np.ndarray] = None):
        """
        Pooled linear XEB versus depth, with the standard error from the per-sequence residuals.

        Args:
            expected_probs: Ideal probabilities used as the model (e.g. nominal CZ or fitted unitary).
            signal_probs: Probabilities that decide which depths carry XEB signal. Defaults to ``expected_probs``.
                Pass the nominal probabilities when evaluating a fitted unitary, so that both series are undefined at
                the same depths (a fitted unitary with a small iSWAP angle or Z phases is no longer exactly uniform
                at depth 1, which would otherwise give a noisy ratio of two tiny numbers).
        """
        measured = np.asarray(self.measured_probs)
        expected = np.asarray(expected_probs)
        signal = expected if signal_probs is None else np.asarray(signal_probs)
        limit = min(measured.shape[1], expected.shape[1], signal.shape[1], len(self.xeb_config.depths))
        measured = measured[:, :limit]
        expected = expected[:, :limit]
        signal = signal[:, :limit]
        depths = np.asarray(self.xeb_config.depths[:limit])

        e_u = np.sum(expected**2, axis=-1)
        u_u = np.sum(expected, axis=-1) / self.xeb_config.dim
        m_u = np.sum(measured * expected, axis=-1)
        x = e_u - u_u
        y = m_u - u_u
        denominator = (x**2).sum(axis=0)
        # Depths without XEB signal (uniform ideal distribution, e.g. depth 1) are undefined
        x_signal = np.sum(signal**2, axis=-1) - np.sum(signal, axis=-1) / self.xeb_config.dim
        defined = ((x_signal**2).sum(axis=0) >= _MIN_XEB_SIGNAL * x.shape[0]) & (
            denominator >= _MIN_XEB_SIGNAL * x.shape[0]
        )
        fidelity = np.full(limit, np.nan)
        fidelity[defined] = (x * y).sum(axis=0)[defined] / denominator[defined]

        std_errs = []
        for i in range(limit):
            if not defined[i]:
                std_errs.append(np.nan)
                continue
            residuals = y[:, i] - fidelity[i] * x[:, i]
            residual_variance = np.sum(residuals**2) / max(len(x[:, i]) - 1, 1)
            std_errs.append(np.sqrt(residual_variance / np.sum(x[:, i] ** 2)))
        return depths, fidelity, np.asarray(std_errs)

    def plot_fidelity_with_unitary_fit(self, fit_summary: Optional[Dict] = None):
        """
        Linear XEB versus depth for the nominal gate and the fitted unitary, with speckle purity
        and the fitted angles printed in the upper-right corner.
        """
        if fit_summary is None:
            fit_summary = getattr(self, "_unitary_fit", None)
        if fit_summary is None:
            raise ValueError("Call estimate_2q_unitary() first, or pass its returned dict explicitly.")

        min_fit_depth = fit_summary.get("min_fit_depth", 6)
        # Both series are undefined at the depths where the nominal circuits carry no XEB signal (e.g. depth 1)
        depths_init, fid_init, err_init = self._linear_xeb_with_stderr(self.expected_probs)
        depths_opt, fid_opt, err_opt = self._linear_xeb_with_stderr(
            fit_summary["fitted_probs"], signal_probs=self.expected_probs
        )

        def fit_curve(depths, fidelity):
            """Same fit (and depth range) as F_total / the numbers printed by the node."""
            try:
                fit = _fit_decay_zero(depths, fidelity, min_depth=min_fit_depth)
            except Exception:
                return None, np.nan
            valid_depths = depths[np.isfinite(fidelity)]
            return (valid_depths, fit["amplitude"] * fit["base"] ** valid_depths), fit["base"]

        curve_init, layer_init = fit_curve(depths_init, fid_init)
        curve_opt, layer_opt = fit_curve(depths_opt, fid_opt)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(
            depths_init,
            fid_init,
            yerr=err_init,
            fmt="o",
            color="C0",
            alpha=0.6,
            label="Linear XEB Data (Initial)",
        )
        if curve_init is not None:
            ax.plot(
                *curve_init,
                "--",
                color="C0",
                label=f"Fit (Initial): {layer_init * 100:.2f}% (err: {1 - layer_init:.1e})",
            )
        ax.errorbar(
            depths_opt,
            fid_opt,
            yerr=err_opt,
            fmt="o",
            color="C1",
            label="Linear XEB Data (Optimized)",
        )
        if curve_opt is not None:
            ax.plot(
                *curve_opt,
                "-",
                color="C1",
                label=f"Fit (Opt): {layer_opt * 100:.2f}% (err: {1 - layer_opt:.1e})",
            )

        # Purity only where XEB carries signal (depth 1 is uniform for every circuit, purity is just shot noise)
        sqrt_purity = np.sqrt(np.clip(self.calculate_speckle_purity(), 0.0, None))
        signal_depths = np.isfinite(fid_init)
        purity_depths = np.asarray(self.xeb_config.depths)[signal_depths]
        sqrt_purity = sqrt_purity[signal_depths]
        ax.plot(
            purity_depths,
            sqrt_purity,
            "s",
            color="gray",
            alpha=0.5,
            zorder=-1,
            label=r"$\sqrt{\mathrm{Purity}}$ (Speckle)",
        )
        ax.plot(purity_depths, sqrt_purity, "--", color="gray", linewidth=1, alpha=0.5, zorder=-1)

        ax.set_xlabel("Cycle Depth")
        ax.set_ylabel(r"$\sqrt{\mathrm{Purity}}$, Circuit Fidelity")
        ylim = ax.get_ylim()
        ax.set_ylim(max(ylim[0] - 0.2 * (ylim[1] - ylim[0]), -0.05), min(ylim[1], 1.05))
        ax.legend(loc="lower left", fontsize="small")
        ax.grid(True, alpha=0.3)

        param_text = (
            "Estimated Param.\n"
            + r"$\theta_\mathrm{iSWAP}=$"
            + f"{fit_summary['theta_iswap_deg']:.2f}"
            + r"$^\circ$"
            + "\n"
            + r"$\phi_\mathrm{CZ}=$"
            + f"{np.degrees(fit_summary['phi_cphase']):.2f}"
            + r"$^\circ$"
            + "\n"
            + r"$\phi_\mathrm{RZ1}=$"
            + f"{fit_summary['rz1_deg']:.2f}"
            + r"$^\circ$"
            + "\n"
            + r"$\phi_\mathrm{RZ2}=$"
            + f"{fit_summary['rz2_deg']:.2f}"
            + r"$^\circ$"
        )
        ax.text(
            0.98,
            0.98,
            param_text,
            ha="right",
            va="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        fig.tight_layout()
        plt.show()
        return fig
