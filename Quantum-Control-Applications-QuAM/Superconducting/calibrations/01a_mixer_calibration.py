"""
A simple program to calibrate Octave mixers for all qubits and resonators
"""

from pathlib import Path
from quam_libs.components import QuAM

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Instantiate the QuAM class from the state file

# Instantiate the QuAM class from the state file
import os
os.environ["quam_state_path"] = "/home/dean/src/qm/asqum/Quantum-Control-Applications-QuAM/Superconducting/configuration/quam_state"
machine = QuAM.load()

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()
import json
with open("qua_config_calibration_db.json", "w+") as f:
   json.dump(config, f, indent=4)

for qubit in machine.active_qubits:
    qm = qmm.open_qm(config)
    qubit.calibrate_octave(qm)
