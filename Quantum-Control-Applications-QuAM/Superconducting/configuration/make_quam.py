# %%
from quam_libs.components import QuAM
from quam_libs.quam_builder.machine import build_quam

path = "C:/Users/JackChao/OneDrive - QM Machines LTD/Customer/windows/AS/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration/quam_state/test"

machine = QuAM.load(path)

# Make the QuAM object and save it
quam = build_quam(machine, quam_state_path=path, octave_names=["oct1"])

# %%
