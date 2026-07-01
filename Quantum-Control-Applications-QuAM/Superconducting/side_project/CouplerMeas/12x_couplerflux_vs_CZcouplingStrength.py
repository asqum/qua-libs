# %%
"""
Status:
    - TESTING
    
"""

# %% {Imports}
from quam_libs.experiments.coupler_RD_related.CZoffset_positioning import CP_CZposition_EXP, node


# %% {Node_parameters}
node.parameters.coupler = ['coupler_q6_q7']
node.parameters.num_averages = 300

node.parameters.coupler_flux_min = -0.1
node.parameters.coupler_flux_max = 0.1
node.parameters.coupler_flux_step = 0.005

node.parameters.min_wait_time_in_ns = 16
node.parameters.max_wait_time_in_ns = 416
node.parameters.wait_time_step_in_ns = 4

node.parameters.reset_type = 'active'
node.parameters.load_data_id = None
node.parameters.simulate = False
node.parameters.operation = 'Cz'




# %% {Initialize}
EXP = CP_CZposition_EXP(node)
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


# %%
