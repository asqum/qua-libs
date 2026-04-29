import numpy as np
# %%
def calculate_avg_2q_clifford_time(native_sq_gate_time_ns:int, cz_gate_time_ns:int):
    """
    - Returns: Averaged 2Q Clifford gate time。
    - ref: https://www.nature.com/articles/nature13171
    """
    sq_layers_per_2q_clifford = 5
    '''
    In John's paper (Supplemetary information), they get averaged 2Q Clifford gate length is 160ns from the calculation with CZ time 40ns, native SQ gate length 20ns.
    Because 2Q is parallel and CZ is 2q shared, 160-1.5*40 = 100 ns. Means 2 single qubit share 100 ns in parallel.
    100 / 20 = 5 layers (doing SQ gate in parallel). 

    '''
    # 1. 計算雙量子位元操作的平均總時間
    total_2q_time = cz_gate_time_ns * (3/2)
    
    # 2. 計算單量子位元操作的平均總時間
    total_1q_time = native_sq_gate_time_ns * sq_layers_per_2q_clifford
    
    # 3. 加總得出平均 Clifford 時間
    avg_clifford_time = total_2q_time + total_1q_time
    
    return avg_clifford_time


def coherence_limit(nQ=2, T1_list=None, T2_list=None, gatelen=0.1):
    """
    The error per gate (1-average_gate_fidelity) given by the T1,T2 limit.

    Args:
        nQ (int): number of qubits (1 and 2 supported).
        T1_list (list): list of T1's (Q1,...,Qn).
        T2_list (list): list of T2's (as measured, not Tphi).
            If not given assume T2=2*T1 .
        gatelen (float): length of the gate.

    Returns:
        float: coherence limited error per gate.

    Raises:
        ValueError: if there are invalid inputs
    """
    # https://github.com/qiskit-community/qiskit-ignis/blob/stable/0.3/qiskit/ignis/verification/randomized_benchmarking/rb_utils.py

    T1 = np.array(T1_list)

    if T2_list is None:
        T2 = 2 * T1
    else:
        T2 = np.array(T2_list)

    if len(T1) != nQ or len(T2) != nQ:
        raise ValueError("T1 and/or T2 not the right length")

    coherence_limit_err = 0

    if nQ == 1:
        coherence_limit_err = 0.5 * (1.0 - 2.0 / 3.0 * np.exp(-gatelen / T2[0]) - 1.0 / 3.0 * np.exp(-gatelen / T1[0]))

    elif nQ == 2:
        T1factor = 0
        T2factor = 0

        for i in range(2):
            T1factor += 1.0 / 15.0 * np.exp(-gatelen / T1[i])
            T2factor += 2.0 / 15.0 * (np.exp(-gatelen / T2[i]) + np.exp(-gatelen * (1.0 / T2[i] + 1.0 / T1[1 - i])))

        T1factor += 1.0 / 15.0 * np.exp(-gatelen * np.sum(1 / T1))
        T2factor += 4.0 / 15.0 * np.exp(-gatelen * np.sum(1 / T2))

        coherence_limit_err = 0.75 * (1.0 - T1factor - T2factor)

    else:
        raise ValueError("Not a valid number of qubits")

    return coherence_limit_err

def SRB_confidence_estimation(clifford_2q_fidelity, clifford_1q_fidelity, measured_cz_fidelity):
    """
    Check the Clifford error with John's estimation !
    * REF: https://www.nature.com/articles/nature13171 (Supplementary Infromation, euqations S10) 
    """
    sq_native_gate_error = (1 - clifford_1q_fidelity)/1.875
    clifford_2q_error = 1 - clifford_2q_fidelity
    cz_error = 1 - measured_cz_fidelity

    theo = 1.5*cz_error + 8.25*sq_native_gate_error

    print(f"Measured fidelity is deviated from the theory about {(theo-clifford_2q_error)/theo:.2%} !")
    print(f"John thought {np.abs(0.0173-0.0189)/0.0173:.2%} is close enough.")
    print(f"THE. Clifford error = {theo:.2%}")
    print(f"EXP. Clifford error = {clifford_2q_error:.2%}")
    


def IRB_confidence_estimation(interleave_fidelity, clifford_1q_fidelity, cz_fidelity):
    """
    Check the Interleaved error with John's estimation !
    * REF: https://www.nature.com/articles/nature13171 (Supplementary Infromation, euqations S11) 
    """
    sq_native_gate_error = (1 - clifford_1q_fidelity)/1.875
    cz_error = 1 - cz_fidelity
    exp_error = 1 - interleave_fidelity

    theo_error = 5*(cz_error)/2 + 33*(sq_native_gate_error)/4

    print(f"Measured fidelity is deviated from the theory about {(theo_error-exp_error)/theo_error:.2%} !")
    print(f"John thought {np.abs(0.0233-0.0244)/0.0233:.2%} is close enough.")
    print(f"THE. Interleaved error = {theo_error:.2%}")
    print(f"EXP. Interleaved error = {exp_error:.2%}")
    



#%%
clifford_2q_fidelity = 0.9519
clifford_1q_fidelity = 0.9895
cz_fidelity = 0.996
interleaved_fidelity = 0.9482

print("SRB confidence: ")
SRB_confidence_estimation(clifford_2q_fidelity, clifford_1q_fidelity, cz_fidelity)
print("IRB confidence: ")
IRB_confidence_estimation(interleaved_fidelity, clifford_1q_fidelity, cz_fidelity)

# %%
# Example usage
T1s = [40e3, 40e3]  # in ns
T2s = [9e3, 4e3]  # in ns
CZ_gate_length = 40  # in ns
x180_gate_time = 16
gate_length = calculate_avg_2q_clifford_time(x180_gate_time, CZ_gate_length)
print(f"Averaged 2Q Clifford gate duration = {gate_length} ns")
error = coherence_limit(nQ=2, T1_list=T1s, T2_list=T2s, gatelen=gate_length)
# error = coherence_limit(nQ=1, T1_list=T1s[:1], T2_list=T2s[:1], gatelen=gate_length)
print(f"Coherence limited error per gate: {error:.2%}")
print(f"Coherence limited Fidelity per gate: {1-error:.1%}")
# %%

#
