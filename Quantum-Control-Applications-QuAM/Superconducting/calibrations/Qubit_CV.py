from pathlib import Path
# from qm.qua import *
from qualang_tools.units import unit
from quam_libs.components import QuAM
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("TKAgg")

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
# config = machine.generate_config()
# Open Communication with the QOP
# qmm = machine.connect()

# Get the relevant QuAM components
qubits = machine.active_qubits
num_qubits = len(qubits)

for q in qubits: print("%s: %s" %(q.name,q.xy.RF_frequency))

x_data = [q.name for q in qubits]
y_data = [q.xy.RF_frequency for q in qubits]

QPU_Map = plt.figure()
plt.suptitle("qubit frequencies")
plt.xlabel("qubit.name")
plt.ylabel("qubit.xy.frequency (GHz)")
plt.plot(x_data, y_data, marker='o', color='red')

for i, (x, y) in enumerate(zip(x_data, y_data)):
    plt.annotate(f"{y*1e-9:.4f}GHz", (x, y), textcoords="offset points", xytext=(21, 12), ha="center") 

plt.show()

