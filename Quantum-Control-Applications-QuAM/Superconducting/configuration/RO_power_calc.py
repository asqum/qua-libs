import numpy as np
from quam_libs.components import QuAM

# %%
def closest_number(lst, target):
    return min(lst, key=lambda x: abs(x - target))


def get_full_scale_power_dBm_and_amplitude(desired_power: float, max_amplitude: float = 0.5) -> tuple[int, float]:
    """Get the full_scale_power_dbm and waveform amplitude for the MW FEM to output the specified desired power.

    The keyword `full_scale_power_dbm` is the maximum power of normalized pulse waveforms in [-1,1].
    To convert to voltage:
        power_mw = 10**(full_scale_power_dbm / 10)
        max_voltage_amp = np.sqrt(2 * power_mw * 50 / 1000)
        amp_in_volts = waveform * max_voltage_amp
        ^ equivalent to OPX+ amp
    Its range is -11dBm to +16dBm with 3dBm steps.

    Args:
        desired_power (float): Desired output power in dBm.
        max_amplitude (float, optional): Maximum allowed waveform amplitude in V. Default is 0.5V.

    Returns:
        tuple[float, float]: The full_scale_power_dBm and waveform amplitude realizing the desired power.
    """
    allowed_powers = [-11, -8, -5, -2, 1, 4, 7, 10, 13, 16]
    resulting_power = desired_power - 20 * np.log10(max_amplitude)
    if resulting_power < 0:
        full_scale_power_dBm = closest_number(allowed_powers, max(resulting_power + 3, -11))
    else:
        full_scale_power_dBm = closest_number(allowed_powers, min(resulting_power + 3, 16))
    amplitude = 10 ** ((desired_power - full_scale_power_dBm) / 20)
    if -11 <= full_scale_power_dBm <= 16 and -1 <= amplitude <= 1:
        return full_scale_power_dBm, amplitude
    else:
        raise ValueError(
            f"The desired power is outside the specifications ([-11; +16]dBm, [-1; +1]), got ({full_scale_power_dBm}; {amplitude})"
        )
    


# %%
""" Input Zone """
state_PATH:str|None = None
desired_power = -40

""" Input Zone end """

if state_PATH is not None:
    machine = QuAM.load(state_PATH)
else:
    machine = QuAM.load()


qubits = machine.active_qubits
rr_full_scale, rr_amplitude = get_full_scale_power_dBm_and_amplitude(    desired_power, max_amplitude= 0.5 / len(qubits)  )

print(f" Amplitude= {rr_amplitude},\n full scale power = {rr_full_scale} dBm")


for q in qubits:
    q.resonator.operations['readout'].amplitude = rr_amplitude
    q.resonator.opx_output.full_scale_power_dbm = rr_full_scale

# %%
machine.save()
# %%