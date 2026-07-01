"""
The T2 echo for the target coupler.

Prerequisites:
    - pi_pulse for the coupler.

Status:
    - All good
    - Load_data is working
"""

# %%
from quam_libs.experiments.coupler_RD_related.T2 import CP_T2_EXP, node


# %% {Node_parameters}
node.parameters.coupler = ['coupler_q6_q7']
node.parameters.num_averages = 300
node.parameters.flux_point_joint_or_independent_or_arbitrary = 'independent'
node.parameters.min_wait_time_in_ns = 16
node.parameters.max_wait_time_in_ns = 6008
node.parameters.wait_time_step_in_ns = 120
node.parameters.reset_type = 'active'
node.parameters.load_data_id = None
node.parameters.simulate = False
node.parameters.histo_num = 1



# %% {Initialize}
EXP = CP_T2_EXP(node)
if node.parameters.load_data_id is None:
    EXP.easy_preparation()


# %% {Run and Fetch}
if node.parameters.load_data_id is None:
    EXP.iteratively_run(node.parameters.histo_num)
    EXP.dataset_post_proccess()


# %% {Data analyze}
EXP.analyze()


# %% {Visualization}
EXP.visualize()


# %% {Update state}
node.parameters.UPDATE_STATE = True
EXP.state_management(node.parameters.UPDATE_STATE)


# %% {Run all above}