"""
The T1 for the target coupler.

Prerequisites:
    - pi_pulse for the coupler.
    - node 08x to check the robustness for aSWAP method active reset if you want to use active reset.
Status:
    - All good
    - Load_data is working
"""
# %% {Imports}
from quam_libs.experiments.coupler_RD_related.T1 import CP_T1_EXP, node

# %% {Node_parameters}
node.parameters.coupler = ['coupler_q6_q7']
node.parameters.num_averages = 300
node.parameters.flux_point_joint_or_independent_or_arbitrary = 'independent'
node.parameters.min_wait_time_in_ns = 16
node.parameters.max_wait_time_in_ns = 180016
node.parameters.wait_time_step_in_ns = 1800
node.parameters.reset_type = 'active'
node.parameters.load_data_id = None
node.parameters.simulate = False
node.parameters.histo_num = 1



# %% {Initialize}
EXP = CP_T1_EXP(node)
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