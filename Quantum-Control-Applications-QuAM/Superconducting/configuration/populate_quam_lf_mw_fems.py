# %%
import json
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.lib.mw_power_utils import optimal_mw_power_settings
from quam_libs.quam_builder.machine import save_machine
import numpy as np

path = "YOUR_PATH"
machine = QuAM.load(path)

# %%                                 QUAM loading and auxiliary functions
########################################################################################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
def get_band(freq):
    """Determine the MW fem DAC band corresponding to a given frequency.

    Args:
        freq (float): The frequency in Hz.

    Returns:
        int: The Nyquist band number.
            - 1 if 50 MHz <= freq < 5.5 GHz
            - 2 if 4.5 GHz <= freq < 7.5 GHz
            - 3 if 6.5 GHz <= freq <= 10.5 GHz

    Raises:
        ValueError: If the frequency is outside the MW fem bandwidth [50 MHz, 10.5 GHz].
    """
    if 50e6 <= freq < 5.5e9:
        return 1
    elif 4.5e9 <= freq < 7.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(f"The specified frequency {freq} Hz is outside of the MW fem bandwidth [50 MHz, 10.5 GHz]")


def get_full_scale_power_dBm_and_amplitude(desired_power: float, max_amplitude: float = 0.5) -> tuple[int, float]:
    """Get full_scale_power_dbm and amplitude for a target output power.

    Optimal combination: lowest allowed full_scale_power_dbm + highest amplitude within
    max_amplitude that still gives the desired power.

        P_out [dBm] = full_scale_power_dbm + 20 * log10(wf_amplitude)

    `full_scale_power_dbm` is the MW-FEM output power at normalized waveform amplitude 1.0.
    To convert to voltage:
        power_mw = 10**(full_scale_power_dbm / 10)
        max_voltage_amp = np.sqrt(2 * power_mw * 50 / 1000)
        amp_in_volts = waveform * max_voltage_amp

    Allowed full_scale_power_dbm values are opx1000_full_scale_powers_dbm (1 dBm steps from -11 to +16 dBm).

    Args:
        desired_power: Desired output power in dBm.
        max_amplitude: Maximum allowed normalized waveform amplitude. Default is 0.5.

    Returns:
        full_scale_power_dbm and waveform amplitude realizing the desired power.
    """
    settings = optimal_mw_power_settings(desired_power, max_amplitude=max_amplitude)
    return settings.full_scale_power_dbm, settings.amplitude

########################################################################################################################
# %%
# Change active qubits
machine.active_qubit_names = ["q1","q2","q3","q4","q5"]  #change

for i in range(len(machine.qubits.items())):
    machine.qubits[f"q{i+1}"].grid_location = f"{i},0"

# Update frequencies
rr_freq = np.array([5932987219.0, 6023928729.0, 5866936123.0, 6079048431.0, 5971697831.0]) #* u.GHz #change
rr_LO = 5.95 * u.GHz #change
rr_if = rr_freq - rr_LO 
assert np.all(np.abs(rr_if) < 400 * u.MHz), (
    "The resonator intermediate frequency must be within [-400; 400] MHz. \n"
    f"Readout frequencies: {rr_freq} \n"
    f"Readout LO frequency: {rr_LO} \n"
    f"Readout IF frequencies: {rr_if} \n"
)

# Desired output power in dBm - Must be within [-80, 16] dBm
readout_power = -40 #change
# Get the full_scale_power_dBm and waveform amplitude corresponding to the desired powers
rr_full_scale, rr_amplitude = get_full_scale_power_dBm_and_amplitude(
    readout_power, max_amplitude= 1.0 / len(machine.qubits)
)

xy_freq = np.array([5108604110.9, 4834229255.6, 5146263353.0, 4674709204.1, 4880175329.7]) #* u.GHz #change
xy_LO = np.array([4.9, 4.9, 4.9, 4.9, 4.9]) * u.GHz #change
# xy_LO = np.array([4.9]*5) * u.GHz
xy_if = xy_freq - xy_LO
assert np.all(np.abs(xy_if) < 400 * u.MHz), (
    "The xy intermediate frequency must be within [-400; 400] MHz. \n"
    f"Qubit drive frequencies: {xy_freq} \n"
    f"Qubit drive LO frequencies: {xy_LO} \n"
    f"Qubit drive IF frequencies: {xy_if} \n"
)

# Desired output power in dBm
drive_power = -30 #change
# Get the full_scale_power_dBm and waveform amplitude corresponding to the desired powers
xy_full_scale, xy_amplitude = get_full_scale_power_dBm_and_amplitude(drive_power)


# NOTE: be aware of coupled ports for bands
for i, q in enumerate(machine.qubits):
    ## Update qubit rr freq and power
    machine.qubits[q].resonator.opx_output.full_scale_power_dbm = rr_full_scale 
    machine.qubits[q].resonator.opx_output.upconverter_frequency = rr_LO
    machine.qubits[q].resonator.opx_input.downconverter_frequency = rr_LO
    machine.qubits[q].resonator.opx_input.band = get_band(rr_LO)
    machine.qubits[q].resonator.opx_output.band = get_band(rr_LO)
    machine.qubits[q].resonator.intermediate_frequency = rr_if[i]

    ## Update qubit xy freq and power
    machine.qubits[q].xy.opx_output.full_scale_power_dbm = xy_full_scale
    machine.qubits[q].xy.opx_output.upconverter_frequency = xy_LO[i]
    machine.qubits[q].xy.opx_output.band = get_band(xy_LO[i])
    machine.qubits[q].xy.intermediate_frequency = xy_if[i]

    # Update flux channels 
    machine.qubits[q].z.opx_output.output_mode = "amplified"
    machine.qubits[q].z.opx_output.upsampling_mode = "pulse"
    machine.qubits[q].z.operations["const"].amplitude = 1.25

    ## Update pulses
    # Readout
    machine.qubits[q].resonator.operations["readout"].length = 1 * u.us #change
    machine.qubits[q].resonator.operations["readout"].amplitude = rr_amplitude
    # Qubit saturation
    machine.qubits[q].xy.operations["saturation"].length = 20 * u.us #change
    machine.qubits[q].xy.operations["saturation"].amplitude = 0.5
    # Single qubit gates - DragCosine
    machine.qubits[q].xy.operations["x180_DragCosine"].length = 16 #change #16
    machine.qubits[q].xy.operations["x180_DragCosine"].amplitude = xy_amplitude 
    machine.qubits[q].xy.operations["x90_DragCosine"].amplitude = (
        machine.qubits[q].xy.operations["x180_DragCosine"].amplitude / 2
    )
    # Single qubit gates - Square
    machine.qubits[q].xy.operations["x180_Square"].length = 16 #change
    machine.qubits[q].xy.operations["x180_Square"].amplitude = xy_amplitude
    machine.qubits[q].xy.operations["x90_Square"].amplitude = (
        machine.qubits[q].xy.operations["x180_Square"].amplitude / 2
    )
    # set joint_offset and min_offest the same as the independent_offest # for AS
    machine.qubits[q].z.independent_offset = 0.0
    machine.qubits[q].z.joint_offset = f"#./independent_offset"
    machine.qubits[q].z.min_offset = f"#./independent_offset"

# golden state
# %%
for qp in machine.qubit_pairs.values():
    qp.coupler.opx_output.output_mode = "amplified"
    qp.coupler.opx_output.upsampling_mode = "pulse"
    qp.coupler.operations["const"].amplitude = 1.25
    pair_qubits = qp.name.split("_")[1:]
    qp.extras["T1"] = 30e-6
    qp.extras["T2"] = 5e-6

# %%

# Default: 
machine.qubits["q1"].xy.core = "a"
machine.qubits["q2"].xy.core = "b"
machine.qubits["q3"].xy.core = "c"
machine.qubits["q4"].xy.core = "d"
machine.qubits["q5"].xy.core = "e"

machine.qubits["q1"].resonator.core = "a"
machine.qubits["q2"].resonator.core = "b"
machine.qubits["q3"].resonator.core = "c"
machine.qubits["q4"].resonator.core = "d"
machine.qubits["q5"].resonator.core = "e"

# %%
# save into state.json
save_machine(machine, path)



