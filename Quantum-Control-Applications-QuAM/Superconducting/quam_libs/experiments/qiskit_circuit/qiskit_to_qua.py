from qiskit.circuit.library import get_standard_gate_name_mapping
from qm.qua import *
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import Target
from quam_libs.components import QuAM, Transmon
from typing import List


def qiskit_to_qua_macro(circuit: QuantumCircuit, machine: QuAM, target_qubits: List[Transmon] | None = None, optimization_level: int = 1):
    qubit_pairs_mapping = {qubit_pair.name: (machine.active_qubits.index(qubit_pair.qubit_control), machine.active_qubits.index(qubit_pair.qubit_target)) for qubit_pair in machine.active_qubit_pairs}
    initial_layout = [machine.active_qubits.index(qubit) for qubit in target_qubits] if target_qubits is not None else None
    target = Target('quam', len(machine.active_qubits),
    dt=1e-9, granularity=4, min_length=16)
    gate_map = get_standard_gate_name_mapping()
    single_qubit_prop = {i: None for i in range(len(machine.active_qubits))}
    two_qubit_prop = {qubit_pairs_mapping[pair.name]: None for pair in machine.active_qubit_pairs}
    for instr in ["sx", "x", "rz"]:
        target.add_instruction(gate_map[instr], single_qubit_prop)
    target.add_instruction(gate_map["cz"], two_qubit_prop)
    qc = transpile(circuit, target=target, initial_layout=initial_layout, optimization_level=optimization_level)
    qubit_indices = {qubit: qc.find_bit(qubit).index for i, qubit in enumerate(qc.qubits)}
    
    cregs = {creg.name: declare(int, size=creg.size) for creg in qc.cregs}
    
    for instruction in qc.data:
        try:
            qubits = instruction.qubits
            if len(qubits) == 2:
                qubit_control = machine.active_qubits[qubit_indices[qubits[0]]]
                qubit_target = machine.active_qubits[qubit_indices[qubits[1]]]
                qubit_pair = qubit_control @ qubit_target
                qubit_pair.apply(instruction.operation.name, *instruction.operation.params)
            elif len(qubits) == 1:
                qubit = machine.active_qubits[qubit_indices[qubits[0]]]
                result = qubit.apply(instruction.operation.name, *instruction.operation.params)
                if instruction.clbits:
                    for clbit in instruction.clbits:
                        creg, index = qc.find_bit(clbit)
                        assign(cregs[creg.name][index], result)    
                        
            else:
                raise ValueError(f"Unsupported number of qubits: {len(qubits)}")
        except Exception as e:
            print(f"Error processing instruction: {instruction}")
            raise e
    
    return cregs

def has_reset_at_boundary(circuit: QuantumCircuit) -> bool:
    """Check if the QuantumCircuit has a reset at the start or end."""
    instructions = circuit.data

    if not instructions:
        return False

    # Check first instruction
    first = instructions[0].operation.name == "reset"
    # Check last instruction
    last = instructions[-1].operation.name == "reset"

    return first or last

def run_qiskit_to_qua_program(circuit: QuantumCircuit, machine: QuAM, target_qubits: List[Transmon] | None = None, n_shots: int = 1024, optimization_level: int = 1):
    """
    Run a Qiskit QuantumCircuit on a QuAM machine and return the results.
    Args:
        circuit (QuantumCircuit): The Qiskit QuantumCircuit to run.
        machine (QuAM): The QuAM machine to run the circuit on.
        target_qubits (List[Transmon], optional): The qubits to target for the circuit execution. Defaults to None.
        n_shots (int, optional): The number of shots to run the circuit. Defaults to 1024.
        optimization_level (int, optional): The optimization level to use for the circuit transpilation. Defaults to 1.

    Returns:
        dict: A dictionary of the results.
    """
    if not circuit.cregs:
        raise ValueError("The circuit does not have any classical registers.")
    if not target_qubits:
        raise ValueError("The target qubits are not specified.")
    if optimization_level not in [0, 1, 2, 3]:
        raise ValueError("The optimization level must be 0, 1, 2, or 3.")
    if n_shots <= 0:
        raise ValueError("The number of shots must be greater than 0.")
        
        
    with program() as prog:

        shot = declare(int)
        cregs_streams = {creg.name: declare_stream() for creg in circuit.cregs}

        with for_(shot, 0, shot < n_shots, shot + 1):
            if not has_reset_at_boundary(circuit):
                for qubit in target_qubits:
                    qubit.apply('reset')
            cregs = qiskit_to_qua_macro(circuit, machine, target_qubits, optimization_level)
            
            for creg in cregs:
                for index in range(creg.size):
                    save(cregs[creg.name][index], cregs_streams[creg.name])
        
        with stream_processing():
            for creg, stream in zip(circuit.cregs, cregs_streams):
                stream.buffer(creg.size).boolean_to_int().save_all(creg.name)
    
    qmm = machine.connect()
    qm = qmm.open_qm(machine.generate_config())
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