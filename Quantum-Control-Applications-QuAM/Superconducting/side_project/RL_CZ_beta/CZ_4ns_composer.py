""" 
Currently, it only supports initialize the parameters in gates['composed_Cz'] with the given tot_cz_length.
    - Introductions for composed_Cz:
        * breaking the CZ wavefrom into multiple 4ns segments with play including coupler and qubit

    - Prerequisites:
        1. 'composed_Cz' in qubit_pairs.gates with the class `quam_libs.components.gates.two_qubit_gates.compsed_CZGate`

    - What will happened:
        1. the amplitude for control_flux_pulse and coupler_flux_pulse will be initialized.
        2. total CZ duration will be initialized.
        3. the qb_amps and cp_amps lists will be initialized.

"""


from quam_libs.components import QuAM
from quam_libs.lib.pulses import *
from quam_libs.components.gates.two_qubit_gates import composed_CZGate

# %%

target_qubit_pair: str = 'coupler_q4_q5'
tot_cz_length: int = 60
cp_init_amp: float = -0.1
qb_init_amp: float = 0.1
seg_length:int = 4 # must be a multiple of 4



# %%
machine = QuAM.load()
# An initial value
qp = machine.qubit_pairs[target_qubit_pair]
target_cz_operation_name = "composed_Cz"

OP:composed_CZGate = qp.gates[target_cz_operation_name]

if seg_length%4 != 0 :
    raise ValueError("seg_length must be multiples of 4") 

OP.tot_len = tot_cz_length
OP.flux_pulse_control.amplitude = qb_init_amp
OP.flux_pulse_control.length = seg_length
OP.coupler_flux_pulse.amplitude = cp_init_amp
OP.coupler_flux_pulse.length = seg_length

amps_num = OP.tot_len // seg_length

OP.extras['cp_amps'], OP.extras['qb_amps'] = [], []
for idx in range(amps_num):
    OP.extras['cp_amps'].append(1.0)
    OP.extras['qb_amps'].append(1.0)

OP.extras['seg_num'] = amps_num

# %%
machine.save()


