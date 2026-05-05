# %%
"""
The transition frequency flux spectrum for the target coupler.

Prerequisites:
    - the driving frequency for the target coupler.

Status:
    - All good
    - Load_data is working
"""


# %% {Imports}
from quam_libs.experiments.coupler_RD_related.CouplerFluxSpectroscopy import CP_FS_EXP, node


# %% {Node_parameters}
node.parameters.coupler = ['coupler_q6_q7']
node.parameters.flux_point_joint_or_independent = 'independent'
node.parameters.operation_amplitude_factor = 0.05
node.parameters.num_averages = 350
node.parameters.Driving_LO_GHz = 3.9
node.parameters.frequency_span_in_mhz = 100
node.parameters.frequency_step_in_mhz = 1
node.parameters.min_flux_offset_in_v = 0.1
node.parameters.max_flux_offset_in_v = 0.45
node.parameters.num_flux_points = 51
node.parameters.qubits_detune_flux_amp = 0.35
node.parameters.enforce_aswap_on_coupler = False

node.parameters.load_data_id = None
node.parameters.simulate = False





# %% {Initialize}
EXP = CP_FS_EXP(node)
if node.parameters.load_data_id is None:
    EXP.easy_preparation()


# %% {Run and Fetch}
if node.parameters.load_data_id is None:
    EXP.iteratively_run(1)
    EXP.dataset_post_proccess()


# %% {Data analyze}
signal_threshold:float = 95 # (95 means the top 5% strong signal)
gaussian_filter_sigma:float = 0.5
EXP.analyze(gaussian_filter_sigma, signal_threshold)


# %% {Visualization}
EXP.visualize()


# %% {Update state}
node.parameters.UPDATE_STATE = False
EXP.state_management(node.parameters.UPDATE_STATE)


# %% {Run all above}
