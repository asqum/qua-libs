"""
    2Q Standard RB and Interleaved RB.

Prerequisites:
    - Bell State fidelity good (> 0.75)

Warning:
    - If it's your first time to run it, you may try 30 random circuits with depth_rank = 'short'. This combination will take only 150 seconds. You can increase the randomness and depth_rank after you are experienced.
"""

from typing import Literal
from quam_libs.experiments.batched_SI2QRB import run_batched_rb
from numpy import mean, std
# %%
# # 20 random circuits takes 10 mins
couplers = ['coupler_q3_q4']
target_operation:Literal['cz', 'idle_2q'] = 'cz'
random_gates_per_depth:int = 30                  # If CZ is good, you may increase it to 198 with depth_rank = 'long' and give it like 1 hrs to run.
repeat_num:int = 1
depth_rank:Literal["short", "standard", "long"] = "short"
job_id:int|None = None
remove_zero_to_analyze:bool = False # Always False

F = []
for _ in range(repeat_num):
    fidelities, names = run_batched_rb(couplers, depth_rank, target_operation, random_gates_per_depth, job_id, time_mark=True, zero_removal_plot=remove_zero_to_analyze)
    F.append(fidelities)

# %%
for i, name in enumerate(names):
    f_list = [f[i] for f in F]
    print(f"{name}: {mean(f_list):.4%} ± {std(f_list):.4%}")

# %%