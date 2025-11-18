from qiskit.circuit.library import get_standard_gate_name_mapping, Reset
from qm.qua import *
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import Target
from quam_libs.components import QuAM, Transmon
from typing import List, Optional
import numpy as np
from qm import generate_qua_script

def create_target(machine: QuAM):
    qubit_pairs_mapping = {qubit_pair.name: (machine.active_qubits.index(qubit_pair.qubit_control), machine.active_qubits.index(qubit_pair.qubit_target)) for qubit_pair in machine.active_qubit_pairs}
    
    target = Target('quam', len(machine.active_qubits),
    dt=1e-9, granularity=4, min_length=16)
    gate_map = get_standard_gate_name_mapping()
    single_qubit_prop = {(i,): None for i in range(len(machine.active_qubits))}
    two_qubit_prop = {qubit_pairs_mapping[pair.name]: None for pair in machine.active_qubit_pairs}
    for instr in ["sx", "x", "rz", "measure", "reset"]:
        target.add_instruction(gate_map[instr], single_qubit_prop)
    target.add_instruction(gate_map["cz"], two_qubit_prop)
    return target


def qiskit_to_qua_macro(circuit: QuantumCircuit, machine: QuAM, target_qubits: List[Transmon] | None = None, optimization_level: Optional[int] = None):
    initial_layout = [machine.active_qubits.index(qubit) for qubit in target_qubits] if target_qubits is not None else None
    if optimization_level is not None:
        target = create_target(machine)
        qc = transpile(circuit, target=target, initial_layout=initial_layout, optimization_level=optimization_level)
    else:
        qc = circuit
    qubit_indices = {qubit: qc.find_bit(qubit).index for i, qubit in enumerate(qc.qubits)}
    print(qc)
    cregs = {creg.name: declare(bool, value= [False] * creg.size) for creg in qc.cregs}
    
    for instruction in qc.data:
        try:
            qubits = instruction.qubits
            if instruction.operation.name == "barrier":
               continue
            if instruction.operation.name == "multiplexed_measurement":
                involved_qubits = [machine.active_qubits[qubit_indices[q]] for q in qubits]
                clbits_indices = [qc.find_bit(clbit).index for clbit in instruction.clbits]
                all_qubits = machine.active_qubits
                all_qubits[0].align(*all_qubits[1:])
                results = {}
                
                for q, qubit in enumerate(all_qubits):
                    if qubit in involved_qubits:
                        index = involved_qubits.index(qubit)
                        result_q = qubit.apply('measure')
                        results[index] = result_q

                    else:
                        qubit.resonator.play('readout')
                all_qubits[0].align(*all_qubits[1:])
                for clbit in instruction.clbits:
                    registers = qc.find_bit(clbit).registers
                    if len(registers) > 1:
                        raise ValueError(f"Multiple registers found for clbit: {clbit}")
                    creg, index = registers[0]
                    assign(cregs[creg.name][clbits_indices[index]], results[index])
                continue

            if len(qubits) == 2:
                qubit_control = machine.active_qubits[qubit_indices[qubits[0]]]
                qubit_target = machine.active_qubits[qubit_indices[qubits[1]]]
                qubit_pair = qubit_control @ qubit_target
                qubit_pair.align()
                qubit_pair.apply(instruction.operation.name, *instruction.operation.params)
                qubit_pair.align()
            elif len(qubits) == 1:
                qubit = machine.active_qubits[qubit_indices[qubits[0]]]
                qubit.align()
                result = qubit.apply(instruction.operation.name, *instruction.operation.params)
                qubit.align()
                if instruction.clbits:
                    for clbit in instruction.clbits:
                        registers = qc.find_bit(clbit).registers
                        if len(registers) > 1:
                            raise ValueError(f"Multiple registers found for clbit: {clbit}")
                        creg, index = registers[0]
                        assign(cregs[creg.name][index], result)    
                        
            else:
                raise ValueError(f"Unsupported number of qubits: {len(qubits)}")
        except Exception as e:
            print(f"Error processing instruction: {instruction}")
            raise e
    
    return cregs

def has_reset_at_boundary(circuit: QuantumCircuit) -> bool:
    """Check if each qubit in the QuantumCircuit has a reset at the start or end."""
    instructions = circuit.data
    qubits = circuit.qubits

    if not instructions:
        return True  # Empty circuit means all qubits are in reset state

    # Create per-qubit instruction lists
    qubit_instructions = {q: [] for q in qubits}
    for inst in instructions:
        for q in inst.qubits:
            qubit_instructions[q].append(inst)

    # Check each qubit's first and last operations
    for qubit, qubit_insts in qubit_instructions.items():
        if not qubit_insts:
            continue  # No instructions means qubit remained in reset state
            
        # Check first operation on this qubit
        has_start_reset = qubit_insts[0].operation.name == "reset"
        
        # Check last operation on this qubit
        has_end_reset = qubit_insts[-1].operation.name == "reset"
        
        if not (has_start_reset or has_end_reset):
            return False

    return True
def ensure_resets_for_active_qubits(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Ensures that every active (non-idle) qubit in the circuit has a reset
    either at the beginning or at the end of its usage. If not, a reset is
    prepended at the beginning of its activity.

    Args:
        circuit (QuantumCircuit): The input circuit to modify in place.

    Returns:
        QuantumCircuit: The modified circuit.
    """
    qc = circuit.copy()
    used_qubits = {q: [] for q in circuit.qubits}

    # Collect instruction indices where each qubit is used
    for idx, (instr, qargs, _) in enumerate(circuit.data):
        for q in qargs:
            used_qubits[q].append(idx)

    for qubit, use_indices in used_qubits.items():
        # Skip idle qubits
        if not use_indices:
            continue

        first_instr = circuit.data[use_indices[0]][0]
        last_instr = circuit.data[use_indices[-1]][0]

        has_reset_at_start_or_end = (
            first_instr.name == "reset" or last_instr.name == "reset"
        )

        # If qubit is active and has no reset at start or end, prepend one
        if not has_reset_at_start_or_end:
            qc.compose(Reset(), qubits=[qubit], inplace=True, front=True)

    return qc


def design_qua_program_from_qiskit(
    circuit: QuantumCircuit,
    machine: "QuAM",
    target_qubits: list["Transmon"] | None = None,
    n_shots: int = 1024,
    optimization_level: Optional[int] = None,
):
    """
    Constructs a QUA program for a given Qiskit QuantumCircuit.

    Args:
        circuit (QuantumCircuit): The Qiskit QuantumCircuit to run.
        machine (QuAM): The QuAM machine to run the circuit on.
        target_qubits (List[Transmon], optional): The qubits to target for the circuit execution. Defaults to None.
        n_shots (int, optional): The number of shots to run the circuit. Defaults to 1024.
        optimization_level (int, optional): The optimization level to use for the circuit transpilation. Defaults to 1.

    Returns:
        program: The constructed QUA program.
    """
    if not circuit.cregs:
        raise ValueError("The circuit does not have any classical registers.")
    if not target_qubits and circuit.num_qubits != len(machine.active_qubits):
        raise ValueError("The target qubits are not specified and the circuit does not have the same number of qubits as the machine.")
    if optimization_level not in [0, 1, 2, 3]:
        raise ValueError("The optimization level must be 0, 1, 2, or 3.")
    if n_shots <= 0:
        raise ValueError("The number of shots must be greater than 0.")

    if target_qubits is not None:
        target_qubits = [machine.active_qubits[i] for i in range(circuit.num_qubits)]

    with program() as prog:
        shot = declare(int)
        cregs_streams = {creg.name: declare_stream() for creg in circuit.cregs}

        machine.apply_all_flux_to_joint_idle()

        with for_(shot, 0, shot < n_shots, shot + 1):
            if not has_reset_at_boundary(circuit):
                circuit = ensure_resets_for_active_qubits(circuit)
            cregs = qiskit_to_qua_macro(circuit, machine, target_qubits, optimization_level)
            for creg in circuit.cregs:
                for index in range(creg.size):
                    save(cregs[creg.name][index], cregs_streams[creg.name])

        with stream_processing():
            for creg, stream in zip(circuit.cregs, cregs_streams.values()):
                stream.boolean_to_int().buffer(creg.size).save_all(creg.name)
    
    # print("Generated QUA program:")
    # print(generate_qua_script(prog))
    return prog


def run_qua_program_and_return_results(
    prog,
    machine: "QuAM",
    circuit: QuantumCircuit,
    n_shots: int,
    qm=None,
):
    """
    Executes a given QUA program on the hardware and returns processed results as a dictionary.

    Args:
        prog: The QUA program to execute.
        machine (QuAM): The QuAM machine to run the program on.
        circuit (QuantumCircuit): The original circuit corresponding to the program.
        n_shots (int): Number of shots to expect.
        qm (QuantumMachine, optional): If provided, use this QM to run instead of opening a new one.

    Returns:
        dict: Dictionary of the results per creg.
    """
    if qm is None:
        qmm = machine.connect()
        qm = qmm.open_qm(machine.generate_config(), close_other_machines=True)
    job = qm.execute(prog)

    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    results = {}
    binary = lambda n, size: bin(n)[2:].zfill(size)
    for creg in circuit.cregs:
        results[creg.name] = {binary(int(i), creg.size): 0 for i in range(2**creg.size)}
        c_reg_result = result_handles.get(creg.name).fetch_all()['value']

        for shot in range(n_shots):
            c_reg_result_shot = c_reg_result[shot].tolist()
            state_int = sum(c_reg_result_shot[i] * (1 << i) for i in range(creg.size))
            results[creg.name][binary(int(state_int), creg.size)] += 1

    return results