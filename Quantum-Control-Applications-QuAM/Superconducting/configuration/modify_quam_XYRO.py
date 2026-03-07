# %%
import json
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.quam_builder.machine import save_machine
import numpy as np


def get_band(freq):
    if 50e6 <= freq < 5.5e9:
        return 1
    elif 4.5e9 <= freq < 7.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(f"The specified frequency {freq} HZ is outside of the MW fem bandwidth [50 MHz, 10.5 GHz]")


path = "/home/ratiswu/Documents/GitHub/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration/quam_state/10Q9Cv2_12_260119_q6q10"
machine = QuAM.load()
u = unit(coerce_to_integer=True)

# %%
# Change active qubits
# machine.active_qubit_names = ["q0"]

# for i in range(len(machine.qubits.items())):
#     machine.qubits[f"q{i+1}"].grid_location = f"{i},0"

## Update frequencies (unit: GHz)
rr_freq: dict = {"q5":5.85, "q6":6.0, "q7":6.1, "q8":5.95, "q9":6.17, "q10":6.05}
rr_LO = 6.025

## Readout power, allowed_powers = [-11, -8, -5, -2, 1, 4, 7, 10, 13, 16]
rr_max_power_dBm:int|None = None

## Readout len (unit: us)
rr_length_global:int|None = None

## Drivin frequency (unit: GHz)
xy_freq:dict = {}#{"q5":3.85, "q6":4.0, "q7":3.7, "q8":3.64, "q9":3.64, "q10":3.55}
xy_LO:dict = {}#{"q5":4.1, "q6":4.25, "q7":3.95, "q8":3.9, "q9":3.9, "q10":3.8}

## Driving power, allowed_powers = [-11, -8, -5, -2, 1, 4, 7, 10, 13, 16]
xy_max_power_dBm:int|None = 7

## Drivin length (unit: ns)
xy_length:dict = {}

# NOTE: be aware of coupled ports for bands
for i, q in enumerate(machine.qubits):
    ## Update qubit rr freq and power
    if rr_max_power_dBm is not None:
        print(f"Modified {q} rr power ...")
        machine.qubits[q].resonator.opx_output.full_scale_power_dbm = rr_max_power_dBm
    if len(list(rr_freq.keys())) != 0:
        print(f"Modified {q} rr freq ...")
        if q in rr_freq:
            machine.qubits[q].resonator.opx_output.upconverter_frequency = rr_LO*u.GHz
            machine.qubits[q].resonator.opx_input.downconverter_frequency = rr_LO*u.GHz
            machine.qubits[q].resonator.opx_input.band = get_band(rr_LO*u.GHz)
            machine.qubits[q].resonator.opx_output.band = get_band(rr_LO*u.GHz)
            machine.qubits[q].resonator.intermediate_frequency = (rr_freq[q] - rr_LO)*u.GHz

    ## Update qubit xy freq and power
    if xy_max_power_dBm is not None:
        print(f"Modified {q} driving power ...")
        machine.qubits[q].xy.opx_output.full_scale_power_dbm = xy_max_power_dBm
    if len(list(rr_freq.keys())) != 0:
        print(f"Modified {q} driving freq ...")
        if q in xy_LO and q in xy_freq:
            machine.qubits[q].xy.opx_output.upconverter_frequency = xy_LO[q]*u.GHz
            machine.qubits[q].xy.opx_output.band = get_band(xy_LO[q]*u.GHz)
            machine.qubits[q].xy.intermediate_frequency = (xy_freq[q] - xy_LO[q]) *u.GHz


    ## Update pulses
    # Readout
    machine.qubits[q].resonator.operations["readout"].length = 1.5 * u.us
    # machine.qubits[q].resonator.operations["readout"].amplitude = 0.25
    # Qubit saturation
    # machine.qubits[q].xy.operations["saturation"].length = 20 * u.us
    # machine.qubits[q].xy.operations["saturation"].amplitude = 0.5
    # Single qubit gates - DragCosine
    
        
    machine.qubits[q].xy.operations["x180_DragCosine"].length = 16
    # machine.qubits[q].xy.operations["x180_DragCosine"].amplitude = 0.8
    # machine.qubits[q].xy.operations["x90_DragCosine"].amplitude = (
    #     machine.qubits[q].xy.operations["x180_DragCosine"].amplitude / 2
    # )
    # Single qubit gates - Square
    machine.qubits[q].xy.operations["x180_Square"].length = 40
    # machine.qubits[q].xy.operations["x180_Square"].amplitude = 0.1
    # machine.qubits[q].xy.operations["x90_Square"].amplitude = (
    #    machine.qubits[q].xy.operations["x180_Square"].amplitude / 2
    # )

# %%
# for Active Reset?
# machine.qubits["q1"].xy.thread = "a"
# machine.qubits["q2"].xy.thread = "b"
# machine.qubits["q3"].xy.thread = "c"
# machine.qubits["q4"].xy.thread = "d"
# machine.qubits["q5"].xy.thread = "e"

# machine.qubits["q1"].resonator.thread = "b"
# machine.qubits["q2"].resonator.thread = "c"
# machine.qubits["q3"].resonator.thread = "d"
# machine.qubits["q4"].resonator.thread = "e"
# machine.qubits["q5"].resonator.thread = "a"

# Default: 
# machine.qubits["q1"].xy.thread = "a"
# machine.qubits["q2"].xy.thread = "b"
# machine.qubits["q3"].xy.thread = "c"
# machine.qubits["q4"].xy.thread = "d"
# machine.qubits["q5"].xy.thread = "e"

# machine.qubits["q1"].resonator.thread = "a"
# machine.qubits["q2"].resonator.thread = "b"
# machine.qubits["q3"].resonator.thread = "c"
# machine.qubits["q4"].resonator.thread = "d"
# machine.qubits["q5"].resonator.thread = "e"


# %%
# Setting readout and driving durations:
for i, q in enumerate(machine.qubits):
    if rr_length_global is not None:
        print(f"Modified {q} rr length ...")
        machine.qubits[q].resonator.operations["readout"].length = rr_length_global * u.us

    if q in xy_length:
        print(f"Modified {q} dirving length  ...")
        machine.qubits[q].xy.operations["x180_DragCosine"].length = xy_length[q]

# %%
# save into state.json
save_machine(machine, path)

# %%
# View the corresponding "raw-QUA" config
# with open("./Quantum-Control-Applications-QuAM/Superconducting/configuration/qua_config.json", "w+") as f:
#     json.dump(machine.generate_config(), f, indent=4)

# %%


