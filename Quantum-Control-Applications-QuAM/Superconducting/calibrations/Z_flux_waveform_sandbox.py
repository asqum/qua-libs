#%%
import numpy as np
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.units import unit
from quam.components.pulses import WaveformPulse
from quam.components.pulses import SquarePulse
from quam_libs.components import QuAM

# %% {Initialize_QuAM_and_QOP}
pulse_length = 32

u = unit(coerce_to_integer=True)
machine = QuAM.load()

## For an arbitrarily-shaped pulse
some_waveform = np.random.rand(pulse_length) / 2  # an array of random numbers
machine.qubit_pairs["coupler_q1_q2"].gates["Cz_unipolar"].coupler_flux_pulse = WaveformPulse(waveform_I=some_waveform)

## For a square pulse:
# machine.qubit_pairs["coupler_q1_q2"].gates["Cz_unipolar"].coupler_flux_pulse = SquarePulse(
#     length=pulse_length,
#     amplitude=0.1
#
# )

config = machine.generate_config()
qmm = machine.connect()

with program() as prog:
    machine.qubit_pairs["coupler_q1_q2"].gates["Cz_unipolar"].execute()

qm = qmm.open_qm(config)

job = qmm.simulate(config, prog, SimulationConfig(duration=500))
samples = job.get_simulated_samples()
waveform_report = job.get_simulated_waveform_report()

waveform_report.create_plot(samples, plot=True, save_path="./")

# Save the QuAM-state
machine.save(content_mapping={"wiring.json": {"wiring", "network"}})
