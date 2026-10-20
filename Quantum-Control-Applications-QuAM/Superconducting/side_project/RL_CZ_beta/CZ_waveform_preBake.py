""" Bakes more waveform in Config for CZ if it is CosineFlattop, and with >= 1ns grid for flat_length
    - Warnings:
        1. MUST have 'Cz' operation in qp.gates
        2. It will generate more Cz_idx gates with the flat_length varing by >= 1ns arbitarary number.
        3. The varing flat_length boundary will be [1, length]
        4. *** If the gate length > 100 ns, please use it with caution (~1e4 operations will be generated and saved into state.json)

    - Modes:
        * append: Generating waveforms
        * delete: Deleting waveform with the name in the rule: "Cz_{idx}"

    - Outcomes:
        1. Many Cz operations in qp.gates
        2. The total generated number will be provided in qp.extras with the key name 'CZ_preBaked_num'

"""
from quam_libs.components import QuAM
from quam_libs.lib.pulses import *
from typing import Literal
import copy
from numpy import arange
from numpy.typing import ArrayLike
# %%
target_qubit_pair: str = 'coupler_q4_q5'
mode:Literal['delete', 'append'] = 'append'
# Global: will generate from 1 to len by step = grid, try costomize it below and set it by None
flat_length_grid_ns:int|None = None

# costomize 
coupler_flat_len_cadidates: None|ArrayLike = arange(4,32,2).tolist()
qubit_flat_len_candidates: None|ArrayLike = arange(4,32,2).tolist()



# %%
machine = QuAM.load()
# An initial value
qp = machine.qubit_pairs[target_qubit_pair]
target_cz_operation_name = "Cz"



if mode == 'append':

    if target_cz_operation_name != "Cz":
        raise KeyError("Target Operation name must be 'Cz' !")
    
    OP = qp.gates[target_cz_operation_name]
    idx = 0

    has_cp = isinstance(OP.coupler_flux_pulse, CosineFlatTopPulse)
    has_ctrl = isinstance(OP.flux_pulse_control, CosineFlatTopPulse)

    if has_cp and has_ctrl:
        target_coupler_flux_pulse_length = OP.coupler_flux_pulse.length
        c_amp = OP.coupler_flux_pulse.amplitude
        target_control_flux_pulse_length = OP.flux_pulse_control.length

        if flat_length_grid_ns is None:
            if coupler_flat_len_cadidates is None or qubit_flat_len_candidates is None:
                raise ValueError("Got no the candidates !")
            cp_range = coupler_flat_len_cadidates
            qb_range = qubit_flat_len_candidates
        else:
            cp_range = range(1, target_coupler_flux_pulse_length, flat_length_grid_ns)
            qb_range = range(1, target_control_flux_pulse_length, flat_length_grid_ns)


        for cp_flat_len in cp_range:
            for ctrl_flat_len in qb_range:
                new_op = copy.copy(OP)

                # 複製脈衝物件，解除舊 parent 綁定後再賦值
                cp_pulse = copy.copy(OP.coupler_flux_pulse)
                cp_pulse.parent = None
                cp_pulse.flat_length = cp_flat_len
                new_op.coupler_flux_pulse = cp_pulse

                ctrl_pulse = copy.copy(OP.flux_pulse_control)
                ctrl_pulse.parent = None
                ctrl_pulse.flat_length = ctrl_flat_len
                new_op.flux_pulse_control = ctrl_pulse

                qp.gates[f"{target_cz_operation_name}_{idx}"] = new_op
                idx += 1

        qp.extras['CZ_preBaked_num'] = idx

    elif has_cp:
        target_coupler_flux_pulse_length = OP.coupler_flux_pulse.length
        c_amp = OP.coupler_flux_pulse.amplitude

        if flat_length_grid_ns is None:
            if coupler_flat_len_cadidates is None :
                raise ValueError("Got no the candidates !")
            cp_range = coupler_flat_len_cadidates
        else:
            cp_range = range(1, target_coupler_flux_pulse_length, flat_length_grid_ns)
            
        for cp_flat_len in cp_range:
            new_op = copy.copy(OP)

            cp_pulse = copy.copy(OP.coupler_flux_pulse)
            cp_pulse.parent = None
            cp_pulse.flat_length = cp_flat_len
            new_op.coupler_flux_pulse = cp_pulse

            qp.gates[f"{target_cz_operation_name}_{idx}"] = new_op
            idx += 1

        qp.extras['CZ_preBaked_num'] = idx

    elif has_ctrl:
        target_control_flux_pulse_length = OP.flux_pulse_control.length
        if flat_length_grid_ns is None:
            if qubit_flat_len_candidates is None:
                raise ValueError("Got no the candidates !")
            qb_range = qubit_flat_len_candidates
        else:
            qb_range = range(1, target_control_flux_pulse_length, flat_length_grid_ns)
        

        for ctrl_flat_len in qb_range:
            new_op = copy.copy(OP)

            ctrl_pulse = copy.copy(OP.flux_pulse_control)
            ctrl_pulse.parent = None
            ctrl_pulse.flat_length = ctrl_flat_len
            new_op.flux_pulse_control = ctrl_pulse

            qp.gates[f"{target_cz_operation_name}_{idx}"] = new_op
            idx += 1

        qp.extras['CZ_preBaked_num'] = idx

    else:
        qp.extras['CZ_preBaked_num'] = 0

else:
    op_names = list(qp.gates.keys())
    name_2_delete = []
    for name in op_names:
        if len(name.split("_")) == 2:
            if name.split("_")[0] == "Cz":
                try:
                    idx = int(name.split("_")[-1])
                    name_2_delete.append(name)
                except:
                    pass

    for name in name_2_delete:
        qp.gates.pop(name, None)
    
    qp.extras.pop('CZ_preBaked_num', None)

# %%
machine.save()
# %%