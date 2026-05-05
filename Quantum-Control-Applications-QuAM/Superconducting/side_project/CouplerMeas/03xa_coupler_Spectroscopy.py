# %%
"""
    Coupler spectroscopy to find the resonant frequency of the coupler.

    Prerequisites:
        - Having a aSWAP operation for you detector_qb in state.json with amplitude is a half flux period, length is about 400 ns which can be manually extended in the future if needed.

    Status:
        - All good
        - Load_data is working
        
"""


# %% {Imports}
from quam_libs.experiments.coupler_RD_related.CouplerSpectroscopy import CP_Spectrum_EXP, node


# %% {Node_parameters}
node.parameters.coupler = ['coupler_q4_q5']
node.parameters.flux_point_joint_or_independent = 'independent'
node.parameters.operation_amplitude_factor = 0.02
node.parameters.num_averages = 350
node.parameters.Driving_LO_GHz = None
node.parameters.frequency_span_in_mhz = 300
node.parameters.frequency_step_in_mhz = 1
node.parameters.load_data_id = None
node.parameters.simulate = False
node.parameters.readme_password = ''




# %% {Initialize}
EXP = CP_Spectrum_EXP(node)
if node.parameters.load_data_id is None:
    EXP.easy_preparation()


# %% {Run and Fetch}
if node.parameters.load_data_id is None:
    EXP.iteratively_run(1)
    EXP.dataset_post_proccess()


# %% {Data analyze}
EXP.analyze()


# %% {Visualization}
EXP.visualize()


# %% {Update state}
node.parameters.UPDATE_STATE = False
EXP.state_management(node.parameters.UPDATE_STATE)


# %% {Run all above}