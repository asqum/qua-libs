# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Optional, List, Literal, Dict, Union
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
import random


from quam_libs.macros import qua_declaration, active_reset, readout_state

from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components.quam_root import BatchableList
from qm.qua import *


names:List[str] = ["Cafish", "Ratis", "WeiEnChiu", "013", "KaoHY", "HCC", "KKC", "WYS", "PoAn"]
need_number:int = 2

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    shots: int = 4096
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    reset_type: Literal["active", "thermal"] = "active"
    timeout: int = 100
    multiplexed: bool = False

node = QualibrationNode(name="GodsDice", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load()
qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)
qubits = BatchableList(qubits, node.parameters.multiplexed)

config = machine.generate_config()

# %% {QUA_program}
n_runs = node.parameters.shots
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type


with program() as iq_blobs:
    reset_global_phase()
    _, _, _, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]

    for multiplexed_qubits in qubits.batch():
        machine.set_all_fluxes(flux_point=flux_point, target=list(multiplexed_qubits.values())[0])

        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)

            
            # measure ground-state IQ blob for all qubits
            for i, qubit in multiplexed_qubits.items():
                if reset_type == "active":
                    active_reset(qubit)
                    # active_reset_gef(qubit)
                elif reset_type == "thermal":
                    qubit.wait(2 * qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")

            align(*[q.xy.name for q in multiplexed_qubits.values()] +
                   [q.resonator.name for q in multiplexed_qubits.values()] +
                   [q.z.name for q in multiplexed_qubits.values()])


            for i, qubit in multiplexed_qubits.items():
                qubit.xy.play("x90")
                readout_state(qubit, state[i])
                save(state[i], state_st[i])

                
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            state_st[i].save_all(f"state{i + 1}")

# %% {Execute}
with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    job = qm.execute(iq_blobs)
    for i in range(num_qubits):
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_runs, start_time=results.start_time)

    ds = fetch_results_as_xarray(
        job.result_handles, qubits,
        measurement_axis={"N": np.linspace(1, node.parameters.shots, node.parameters.shots)}
    )


# %%
print(ds)

# %%

def get_random_binary_string(name_list)->Dict:
    encoded = {}
    n = len(name_list)
    # 計算需要的最小位元長度 (例如 8人需 3 bits)
    bit_length:int = (n - 1).bit_length() if n > 1 else 1
    
    # 1. 建立一個 0 到 n-1 的索引清單
    indices = list(range(n))
    
    # 2. 隨機打亂這些索引 (Shuffle)
    random.shuffle(indices)
    
    # 3. 轉換為補零後的二進位字串
    binary_results = [bin(i)[2:].zfill(bit_length) for i in indices]
    print(binary_results)

    for idx, name in enumerate(name_list):
        encoded[binary_results[idx]] = name

    # 4. 輸出成單一字串
    return encoded

def pick_full_binary_range(data_array, bit_length):
    i_size, j_size = data_array.shape
    total_elements = i_size * j_size
    
    # 安全檢查：確保陣列總點數夠抽
    if total_elements < bit_length:
        raise ValueError("陣列總元素量小於所需的 bit_length 數量")

    # 1. 在整個陣列的扁平化索引中，隨機抽取 bit_length 個「不重複」的位置
    # 例如：(3, 10) 陣列有 30 個點，我們從 0~29 中抽 4 個
    random_flat_indices = random.sample(range(total_elements), bit_length)
    
    # 2. 將扁平索引轉回二維座標 (row_idx, col_idx)
    # np.unravel_index 會把 15 轉成 (1, 5) 這樣的座標
    rows, cols = np.unravel_index(random_flat_indices, (i_size, j_size))
    
    # 3. 取得數值
    picked_values = data_array[rows, cols]

    lottery = ""
    for k in picked_values:
        lottery += str(k[0])
    
    return lottery


# %%
lucky_ppl = []
from time import sleep
while True:

    random_output = get_random_binary_string(names)
    bit_N = len(list(random_output.keys())[0])
    lottery = pick_full_binary_range(ds.state.values, bit_N)

    print(lottery)

    if len(lottery) != bit_N:
        print(f"{bit_N} bits mismatched with {lottery} !")
        break

    if lottery in random_output.keys():
        print("Y")
        lucky_ppl.append(random_output[lottery])
        names.remove(random_output[lottery])
    
    if len(lucky_ppl) == need_number:
        break

print(lucky_ppl)

        


# %%