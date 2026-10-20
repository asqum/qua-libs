"""
Add the macro information into target_qubit_pair's extras. Add a CZaux flux pulse into the participant's z operation. Aim for adding a flux tuning during the CZ gate.
- The length is default to be the target_qubit_pair's CZ length, but can be set to a specific value.
- The wavefrom can be set by refering to quam_libs.lib.pulses.

* Products:
    - in target_qubit_pair's extras, there will be a dict with a name 'CZ_aux_macro' which record the participant's name and the corresponding CZaux operation name to indicate. For eaxample, ''
    - in each participant's z operation, there will be a new flux pulse added to the operation's pulse list. The name of the flux pulse is the same as which is recorded in the target_qubit_pair's extras 'CZ_aux_macro'.

"""


from quam_libs.components import QuAM
from quam_libs.components import Transmon, TransmonPair
from quam_libs.lib.pulses import *


# %%
## Settings
target_qubit_pair: str = 'coupler_q4_q5'
length:int|None = None  # If None, length will be set to target_qubit_pair's CZ length. Otherwaise, length will be set to the provided value.


# Wavefrom setting, refer: quam_libs.lib.pulses 
participants_and_CZaux = {
    'coupler_q3_q4': CosineFlatTopPulse(length=length, amplitude=0.0, flat_length=10, axis_angle=None),
    'coupler_q5_q6': CosineFlatTopPulse(length=length, amplitude=0.0, flat_length=10, axis_angle=None),
    'q3': CosineFlatTopPulse(length=length, amplitude=0.0, flat_length=10, axis_angle=None)
}



# load
machine = QuAM.load()

# An initial value







machine.save()