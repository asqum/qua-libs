# %%

"""
A simple program to close all other open QMs.
"""

from typing import Optional, List
from qualibrate import QualibrationNode, NodeParameters

from quam_libs.components import QuAM


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None


node = QualibrationNode(name="00_Close_other_QMs", parameters=Parameters())

# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Generate the OPX and Octave configurations
config = machine.generate_config()

# Open Communication with the QOP
qmm = machine.connect() # Use this line if you want to connect to the QOP defined in the state file

# from qm import QuantumMachinesManager
# qmm = QuantumMachinesManager(host='10.21.19.201', cluster_name='QPX1000_2',port = 2001, log_level="DEBUG")
# qmm = QuantumMachinesManager(host='10.31.19.131', cluster_name='QPX1000_2',port = 2002, log_level="DEBUG")
# qmm = QuantumMachinesManager(host='192.168.1.27', cluster_name='QPX1000_2',port = 8002, log_level="DEBUG")
# qmm = QuantumMachinesManager(host='192.168.1.225', cluster_name='QPX1000_2',port = 9510, log_level="DEBUG")

qmm.close_all_qms()
#%%