from typing import Literal
from quam_libs.experiments.batched_SI2QRB import run_batched_rb
from numpy import mean, std
# %%
# # 20 random circuits takes 10 mins
couplers = ['coupler_q3_q4']
target_operation:Literal['cz', 'idle_2q'] = 'cz'
random_gates_per_depth:int = 30
repeat_num:int = 1
depth_rank:Literal["short", "standard", "long"] = "long"
job_id:int|None = None
remove_zero_to_analyze:bool = False

F = []
for _ in range(repeat_num):
    fidelities, names = run_batched_rb(couplers, depth_rank, target_operation, random_gates_per_depth, job_id, time_mark=True, zero_removal_plot=remove_zero_to_analyze)
    F.append(fidelities)

# %%
for i, name in enumerate(names):
    f_list = [f[i] for f in F]
    print(f"{name}: {mean(f_list):.4%} ± {std(f_list):.4%}")

# %%