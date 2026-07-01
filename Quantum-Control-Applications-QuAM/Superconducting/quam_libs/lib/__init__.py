from quam_libs.components.transmon_pair import TransmonPair
from typing import List

def find_c_with_q(qubit_list:List[str], coupler_list:List[TransmonPair])->List[TransmonPair]:
    """
    With given qubits name, filter the connected couplers.
    """
    # 1. turn list into set
    valid_qubits = set(qubit_list)
    
    result = []
    
    # 2. ask every c in coupler_list
    for coupler in coupler_list:
        parts = coupler.name.split('_')
        
        # check the format
        if len(parts) == 3:
            q_a = parts[1] # first q
            q_b = parts[2] # second q
            
            # 3. final check 
            if q_a in valid_qubits and q_b in valid_qubits:
                result.append(coupler)
                
    return result