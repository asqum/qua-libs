# %%
"""
Coupler to qubit JAZZ
Measures the ZZ from coupler to its readout_q.

Prerequisites:
- π-pulse calibrated for the coupler.

Outcomes:
- ZZ strength statistics

Status:
    - All good
    - Load_data is working
"""

# %% {Imports}
from quam_libs.experiments.coupler_RD_related.C2Q_ZZ import CP_c2qJAZZ_EXP, node


# %% {Node_parameters}
node.parameters.coupler = ['coupler_q6_q7']
node.parameters.num_averages = 300
node.parameters.flux_point_joint_or_independent_or_arbitrary = 'independent'
node.parameters.frequency_detuning_in_mhz = 2
node.parameters.min_wait_time_in_ns = 16
node.parameters.max_wait_time_in_ns = 3016
node.parameters.wait_time_step_in_ns = 28
node.parameters.reset_type = 'active'
node.parameters.load_data_id = None
node.parameters.simulate = False
node.parameters.histo_num = 11



# %% {Initialize}
EXP = CP_c2qJAZZ_EXP(node)
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
EXP.state_management()


# %%
