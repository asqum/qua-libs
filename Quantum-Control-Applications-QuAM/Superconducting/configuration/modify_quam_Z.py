# %%
import json
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.quam_builder.machine import save_machine
import numpy as np


path = "/home/ratiswu/Documents/GitHub/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration/quam_state/as-qpu-10qV2_q1q5"
machine = QuAM.load()
u = unit(coerce_to_integer=True)


# %% {Show current offsets}
for i, q in enumerate(machine.qubits):
    print(f"{q}'s independent_offset: {machine.qubits[q].z.independent_offset}")
for i, c in enumerate(machine.qubit_pairs):
    print(f"{c}'s decouple_offset: {machine.qubit_pairs[c].coupler.decouple_offset}")



# %% {modify offsets}
inde_or_decouple_offset_scenarios:dict = {"coupler_q2_q3":-0.4,"coupler_q3_q4":-0.4,"coupler_q4_q5":0.2,"coupler_q5_q6":-0.6,}
for i, ele in enumerate(inde_or_decouple_offset_scenarios):
    if ele in machine.qubits:
        print(f"{ele}'s independent offset -> {inde_or_decouple_offset_scenarios[ele]} V")
        machine.qubits[ele].z.independent_offset = inde_or_decouple_offset_scenarios[ele]
    if ele in machine.qubit_pairs:
        print(f"{ele}'s decoupler offset -> {inde_or_decouple_offset_scenarios[ele]} V")
        machine.qubit_pairs[ele].coupler.decouple_offset = inde_or_decouple_offset_scenarios[ele]

   
# %%
# save into state.json
save_machine(machine, path)

# %%


