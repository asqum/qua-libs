"""
State transformation from different branches like `as_quam_qualibrate_readoutCoupler` and `as_quam_qualibrate`. 
Beause the aSWAP pulse is not supported in the branch `as_quam_qualibrate`, this script will remove the pulse from the target state json.

- Warnings:
    1. Removing the aSWAP pulse anyway.

- Outcomes:
    1. The state and wiring json will be saved into a new folder with a new name, which is the original folder name with a suffix "_NOaSWAP". The original state and wiring json will be kept unchanged.
"""
# %%
from quam_libs.components import QuAM
from quam_libs.lib.pulses import aSWAPPulse
import pathlib

# %% {Input Zone}

state_PATH:str|None = None


# %% {Processing}
if state_PATH is not None:
    if len(state_PATH) > 0:
        machine = QuAM.load(state_PATH)
    else:
        machine = QuAM.load()
else:
    machine = QuAM.load()

original_PATH = machine.get_quam_state_path()
after_PATH = pathlib.Path(original_PATH).parent / f"{pathlib.Path(original_PATH).name}_NOaSWAP"



for q_name in machine.qubits:
    qubit = machine.qubits[q_name]
    for pulse in list(qubit.z.operations.keys()):
        if isinstance(qubit.z.operations[pulse], aSWAPPulse):

            print(f"Removing aSWAP from {q_name}")
            del qubit.z.operations[pulse]

for c_name in machine.qubit_pairs:
    c = machine.qubit_pairs[c_name]
    for pulse in list(c.coupler.operations.keys()):
        if isinstance(c.coupler.operations[pulse], aSWAPPulse):

            print(f"Removing aSWAP from {c_name}")
            del c.coupler.operations[pulse]


# %% {Update and save}
machine.save(after_PATH)
# %%