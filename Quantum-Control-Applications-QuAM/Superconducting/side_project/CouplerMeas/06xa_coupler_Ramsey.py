"""
Ramsey experiment for a coupler
Prerequisites:
    - pi_pulse for the target qubit and the coupler. Note that the duration of the pi pulse for the coupler and the qubit should be same so that the sequence can be well aligned.
Updates:
    - Driving IF for this coupler.
"""

# %%
from quam_libs.experiments.coupler_RD_related.Ramsey import node, CP_Ramsey_EXP


# %% {Node_parameters}
node.parameters.coupler = ['coupler_q6_q7']
node.parameters.num_averages = 450
node.parameters.frequency_detuning_in_mhz = 2
node.parameters.min_wait_time_in_ns = 16
node.parameters.max_wait_time_in_ns = 2016
node.parameters.wait_time_step_in_ns = 12
node.parameters.reset_type = 'active'

node.parameters.flux_point_joint_or_independent_or_arbitrary = 'independent'
node.parameters.load_data_id = 2144
node.parameters.simulate = False


# %% {Initialize}
EXP = CP_Ramsey_EXP(node)
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