import dataclasses
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

@dataclasses.dataclass
class RBResult:
    """
    Class for analyzing and visualizing the results of a Randomized Benchmarking (RB) experiment.

    Attributes:
        circuit_depths (list[int]): List of circuit depths used in the RB experiment.
        num_repeats (int): Number of repeated sequences at each circuit depth.
        num_averages (int): Number of averages for each sequence.
        state (np.ndarray): Measured states from the RB experiment.
    """

    circuit_depths: list[int]
    num_repeats: int
    num_averages: int
    state: np.ndarray

    def __post_init__(self):
        """
        Initializes the xarray Dataset to store the RB experiment data.
        """
        self.data = xr.Dataset(
            data_vars={"state": (["repeat", "circuit_depth", "average"], self.state)},
            coords={
                "repeat": range(self.num_repeats),
                "circuit_depth": self.circuit_depths,
                "average": range(self.num_averages),
            },
        )

    def plot_hist(self, n_cols=3):
        """
        Plots histograms of the N-qubit state distribution at each circuit depth.
        """
        if len(self.circuit_depths) < n_cols:
            n_cols = len(self.circuit_depths)
        n_rows = max(int(np.ceil(len(self.circuit_depths) / n_cols)), 1)
        plt.figure()
        for i, circuit_depth in enumerate(self.circuit_depths, start=1):
            ax = plt.subplot(n_rows, n_cols, i)
            self.data.state.sel(circuit_depth=circuit_depth).plot.hist(ax=ax, xticks=range(4))
        plt.tight_layout()

    def plot(self):
        """
        Plots the raw recovery probability decay curve as a function of circuit depth.
        """
        recovery_probability = (self.data.state == 0).sum(("repeat", "average")) / (
            self.num_repeats * self.num_averages
        )
        recovery_probability.rename("Recovery Probability").plot.line()

    def plot_with_fidelity(self, simultaneous_SQ_RB:bool=False):
        """
        Plots the RB fidelity as a function of circuit depth, including a fit to an exponential decay model.
        """
        sequence_means = (self.data.state == 0).mean(dim="average")
        error_bars = sequence_means.std(dim="repeat").data

        fig = plt.figure()
        plt.errorbar(
            self.circuit_depths,
            self.get_decay_curve(),
            yerr=error_bars,
            fmt=".",
            capsize=2,
            elinewidth=0.5,
            color="blue",
            label="Experimental Data",
        )

        circuit_depths_smooth_axis = np.linspace(self.circuit_depths[0], self.circuit_depths[-1], 100)
        plt.plot(
            circuit_depths_smooth_axis,
            rb_decay_curve(np.array(circuit_depths_smooth_axis), self.A, self.alpha, self.B),
            color="red",
            linestyle="--",
            label="Exponential Fit",
        )

        # 組合標題 (包含誤差)
        err_str = f" \u00B1 {self.fidelity_err * 100:.2f}%" if hasattr(self, 'fidelity_err') else ""
        
        if type(self).__name__ == "InterleavedRBResult": # 避免迴圈 import 問題的寫法
            title = f"target gate fidelity = {self.fidelity * 100:.2f}%{err_str}"
        else:
            if not simultaneous_SQ_RB:
                title = f"2Q average Clifford fidelity = {self.fidelity * 100:.2f}%{err_str}"
            else:
                title = rf"1Q$\otimes$1Q Clifford Fidelity = {self.fidelity * 100:.2f}%{err_str}"
            
        plt.text(
            0.5,
            0.95,
            title,
            horizontalalignment="center",
            verticalalignment="top",
            fontdict={"fontsize": "large", "fontweight": "bold"},
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
        )
        
        # Add average gate fidelity if it was calculated
        if hasattr(self, 'average_gate_fidelity'):
            plt.text(
                0.5,
                0.88,
                f"Average Gate Fidelity = {self.average_gate_fidelity * 100:.2f}%",
                horizontalalignment="center",
                verticalalignment="top",
                fontdict={"fontsize": "large", "fontweight": "bold"},
                transform=plt.gca().transAxes,
            )

        num_repeats = self.num_repeats
        text_info = f"Random circuits per depth: {num_repeats}"
        
        plt.text(                  # 如果是用 plt.text
            0.05, 0.05,            
            text_info, 
            transform=plt.gca().transAxes, # 修正這裡：先抓取當前 ax 再呼叫 transAxes
            fontsize=11,
            verticalalignment='bottom',
            bbox=dict(
                boxstyle='round,pad=0.5', 
                facecolor='white', 
                alpha=0.8, 
                edgecolor='gray'
            )
        )

        plt.xlabel("Circuit Depth")
        plt.ylabel(r"Probability to recover to $|00\rangle$")
        plt.legend(framealpha=0)
        
        return fig

    def plot_two_qubit_state_distribution(self):
        """
        Plot how the two-qubit state is distributed as a function of circuit-depth on average.
        """
        # (保持你原本的程式碼不變)
        plt.plot(
            self.circuit_depths,
            (self.data.state == 0).mean(dim="average").mean(dim="repeat").data,
            label=r"$|00\rangle$", marker=".", color="c", linewidth=3,
        )
        plt.plot(
            self.circuit_depths,
            (self.data.state == 1).mean(dim="average").mean(dim="repeat").data,
            label=r"$|01\rangle$", marker=".", color="b", linewidth=1,
        )
        plt.plot(
            self.circuit_depths,
            (self.data.state == 2).mean(dim="average").mean(dim="repeat").data,
            label=r"$|10\rangle$", marker=".", color="y", linewidth=1,
        )
        plt.plot(
            self.circuit_depths,
            (self.data.state == 3).mean(dim="average").mean(dim="repeat").data,
            label=r"$|11\rangle$", marker=".", color="r", linewidth=1,
        )
        plt.axhline(0.25, color="grey", linestyle="--", linewidth=2, label="2Q mixed-state")

        plt.xlabel("Circuit Depth")
        plt.ylabel(r"Probability to recover to a given state")
        plt.title("2Q State Distribution vs. Circuit Depth")
        plt.legend(framealpha=0, title=r"2Q State $\mathbf{|q_cq_t\rangle}$", title_fontproperties={"weight": "bold"})
        plt.show()

    def fit_exponential(self, use_weights=False):
        """
        Fits the decay curve of the RB data to an exponential model.
        """
        decay_curve = self.get_decay_curve()
        p0 = [0.75, 0.9, 0.25]

        if use_weights:
            # 計算出實驗資料的標準差作為權重
            sigma = (self.data.state == 0).mean(dim="average").std(dim="repeat").data
            # 防呆：避免標準差為0導致 curve_fit 發生除以零的錯誤
            sigma = np.where(sigma == 0, 1e-8, sigma)
            
            popt, pcov = curve_fit(
                rb_decay_curve, self.circuit_depths, decay_curve, 
                p0=p0, maxfev=10000, sigma=sigma, absolute_sigma=True
            )
        else:
            popt, pcov = curve_fit(
                rb_decay_curve, self.circuit_depths, decay_curve, 
                p0=p0, maxfev=10000
            )

        print("***** ", popt)
            
        A, alpha, B = popt

        
        # 儲存參數與計算出來的誤差 (對角線開根號)
        self.alpha = alpha
        self.fit_errors = np.sqrt(np.diag(pcov)) 

        # --- 新增：計算 R^2 ---
        # 1. 根據擬合出的參數，計算理論上的預測值
        fit_values = rb_decay_curve(self.circuit_depths, A, alpha, B)
        
        # 2. 計算殘差平方和 (Residual Sum of Squares)
        ss_res = np.sum((decay_curve - fit_values) ** 2)
        
        # 3. 計算總平方和 (Total Sum of Squares)
        ss_tot = np.sum((decay_curve - np.mean(decay_curve)) ** 2)
        
        # 4. 算出 R^2
        r_squared = 1 - (ss_res / ss_tot)
        self.r_squared = r_squared

        return A, alpha, B
    

    def fit(self, average_layers_per_clifford=None, average_gates_per_2q_layer=None, use_weights=False):
        """
        Fits the RB data and calculates all error and fidelity metrics.
        """
        # Fit exponential decay (傳入 use_weights 參數)
        A, alpha, B = self.fit_exponential(use_weights=use_weights)
        self.A = A
        self.alpha = alpha
        self.B = B
        
        # 提取 alpha 的擬合誤差
        self.alpha_err = self.fit_errors[1]
        
        # Calculate fidelity and error per Clifford
        fidelity = self.get_fidelity(alpha)
        self.fidelity = fidelity
        self.epc = 1 - self.fidelity
        
        # 誤差傳遞 (Error Propagation)：計算 Fidelity 的誤差
        # 公式: r = (1 - alpha) * (1 - 1/d), fidelity = 1 - r
        n_qubits = 2
        d = 2**n_qubits
        self.fidelity_err = self.alpha_err * (1 - 1/d)
        
        # Calculate additional metrics if constants are provided
        if average_layers_per_clifford is not None and average_gates_per_2q_layer is not None:
            self.error_per_2q_layer = (1 - fidelity) / average_layers_per_clifford
            self.error_per_gate = self.error_per_2q_layer / average_gates_per_2q_layer
            self.average_gate_fidelity = 1 - self.error_per_gate

    def get_fidelity(self, alpha):
        """
        Calculates the average fidelity per Clifford based on the decay constant.
        """
        n_qubits = 2  
        d = 2**n_qubits
        r = 1 - alpha - (1 - alpha) / d 
        fidelity = 1 - r

        return fidelity

    def get_decay_curve(self):
        return (self.data.state == 0).sum(("repeat", "average")) / (self.num_repeats * self.num_averages)
    
    def get_decay_curve_1q(self, qubit_index: int):
        if qubit_index == 0:
            return ((self.data.state == 0) | (self.data.state == 1)).sum(("repeat", "average")) / (self.num_repeats * self.num_averages)
        elif qubit_index == 1:
            return ((self.data.state == 0) | (self.data.state == 2)).sum(("repeat", "average")) / (self.num_repeats * self.num_averages)
        else:
            raise ValueError(f"Qubit index {qubit_index} not supported")
        


def rb_decay_curve(x, A, alpha, B):
    """
    Exponential decay model for RB fidelity.

    Args:
        x (array-like): Circuit depths.
        A (float): Amplitude of the decay.
        alpha (float): Decay constant.
        B (float): Offset of the curve.

    Returns:
        np.ndarray: Calculated decay curve.
    """
    return A * alpha**x + B


class InterleavedRBResult(RBResult):
    """
    Class for analyzing and visualizing the results of an Interleaved Randomized Benchmarking (IRB) experiment.
    """
    standard_rb_alpha: float = 1.0
    standard_rb_alpha_err: float = 0.0  # 新增：用來接收 SRB 衰減率的誤差
    
    def __init__(self, standard_rb_alpha: float, circuit_depths: list[int], num_repeats: int, num_averages: int, state: np.ndarray, standard_rb_alpha_err: float = 0.0):
        # 初始化父類別 (RBResult)
        super().__init__(circuit_depths, num_repeats, num_averages, state)
        self.standard_rb_alpha = standard_rb_alpha
        self.standard_rb_alpha_err = standard_rb_alpha_err

    def get_fidelity(self, alpha: float):
        """
        Calculates the interleaved gate fidelity using the formula from https://arxiv.org/pdf/1203.4550.
        """
        n_qubits = 2  # Assuming 2 qubits as per the context
        d = 2**n_qubits
        self.IRB_decayTau = 1 - (1 - alpha - (1 - alpha) / d) # error per clifford
        
        return 1 - ((d - 1) * (1 - alpha / self.standard_rb_alpha) / d)

    def fit(self, average_layers_per_clifford=None, average_gates_per_2q_layer=None, use_weights=False):
        # 1. 先呼叫父類別的 fit()
        super().fit(average_layers_per_clifford, average_gates_per_2q_layer, use_weights)
        
        d = 2**2  # 4
        
        # ==========================================
        # [新增] 儲存 IRB 曲線本身的衰減誤差 (藍線圖例要用的)
        # ==========================================
        self.IRB_decayTau_err = self.alpha_err * ((d - 1) / d)
        
        # 2. 計算 IRB 特有的 Target Gate Fidelity Error (結合 SRB 誤差傳遞)
        ratio = self.alpha / self.standard_rb_alpha
        
        if self.standard_rb_alpha_err > 0:
            ratio_err = ratio * np.sqrt(
                (self.alpha_err / self.alpha)**2 + 
                (self.standard_rb_alpha_err / self.standard_rb_alpha)**2
            )
        else:
            ratio_err = self.alpha_err / self.standard_rb_alpha
            
        # 這個是最後顯示在標題上的目標閘誤差
        self.fidelity_err = ((d - 1) / d) * ratio_err
    
def plot_combined_rb(qp_name, rb_result_SRB, rb_result_IRB, target_gate:str|None=None):

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ==========================================
    # 1. 處理 Standard RB (SRB) 數據與繪圖
    # ==========================================
    decay_curve_SRB = rb_result_SRB.get_decay_curve()
    sequence_means_SRB = (rb_result_SRB.data.state == 0).mean(dim="average")
    error_bars_SRB = sequence_means_SRB.std(dim="repeat").data

    irb_r_square = rb_result_IRB.r_squared.item()
    srb_r_square = rb_result_SRB.r_squared.item()
    
    averaged_r_squarre = np.mean([irb_r_square, srb_r_square])

    # 動態產生誤差字串 (如果有計算出誤差的話)
    srb_err_str = f" \u00B1 {rb_result_SRB.fidelity_err * 100:.2f}%" if hasattr(rb_result_SRB, 'fidelity_err') else ""

    ax.errorbar(
        rb_result_SRB.circuit_depths,
        decay_curve_SRB,
        yerr=error_bars_SRB,
        fmt="o", 
        capsize=3,
        elinewidth=1.0,
        color="red",
        label="SRB Experimental Data",
    )

    circuit_depths_smooth = np.linspace(rb_result_SRB.circuit_depths[0], rb_result_SRB.circuit_depths[-1], 100)
    ax.plot(
        circuit_depths_smooth,
        rb_decay_curve(np.array(circuit_depths_smooth), rb_result_SRB.A, rb_result_SRB.alpha, rb_result_SRB.B),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"SRB Fit (Clifford Fidelity = {rb_result_SRB.fidelity * 100:.2f}%)",
    )
    
    # ==========================================
    # 2. 處理 Interleaved RB (IRB) 數據與繪圖
    # ==========================================
    decay_curve_IRB = rb_result_IRB.get_decay_curve()
    sequence_means_IRB = (rb_result_IRB.data.state == 0).mean(dim="average")
    error_bars_IRB = sequence_means_IRB.std(dim="repeat").data
    
    _ = rb_result_IRB.get_fidelity(rb_result_IRB.alpha)
    
    # 動態產生 IRB 閘保真度的誤差字串與曲線本身衰減誤差
    irb_err_str = f" \u00B1 {rb_result_IRB.fidelity_err * 100:.2f}%" if hasattr(rb_result_IRB, 'fidelity_err') else ""
    irb_decay_err_str = f" \u00B1 {rb_result_IRB.IRB_decayTau_err * 100:.2f}%" if hasattr(rb_result_IRB, 'IRB_decayTau_err') else ""

    ax.errorbar(
        rb_result_IRB.circuit_depths,
        decay_curve_IRB,
        yerr=error_bars_IRB,
        fmt="s", 
        capsize=3,
        elinewidth=1.0,
        color="blue",
        label="IRB Experimental Data",
    )

    ax.plot(
        circuit_depths_smooth,
        rb_decay_curve(np.array(circuit_depths_smooth), rb_result_IRB.A, rb_result_IRB.alpha, rb_result_IRB.B),
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"IRB Fit, IRB Decay = {rb_result_IRB.IRB_decayTau * 100:.2f}%",
    )

    # ==========================================
    # 3. 設定圖表標題與格式
    # ==========================================
    gate_name = "Target" if target_gate is None else target_gate
    ax.set_title(f"{qp_name} {gate_name} Gate Fidelity = {rb_result_IRB.fidelity * 100:.2f}%", fontsize=16)
    
    ax.set_xlabel("Circuit Depth", fontsize=12)
    ax.set_ylabel(r"Probability to recover to $|00\rangle$", fontsize=12)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # ==========================================
    # 4. [新增] 在圖表左下角加入 Repeat 資訊文字方塊
    # ==========================================
    num_repeats = rb_result_SRB.num_repeats
    text_info = f"Random circuits per depth: {num_repeats}\n" + rf"$R^{2}$={averaged_r_squarre:.1%}"
    
    # 使用 ax.text，並設定 transform=ax.transAxes 讓座標比例以整張圖為基準 (0~1)
    ax.text(
        0.05, 0.05,            # X, Y 座標 (左下角)
        text_info, 
        transform=ax.transAxes, 
        fontsize=11,
        verticalalignment='bottom',
        bbox=dict(
            boxstyle='round,pad=0.5', 
            facecolor='white', 
            alpha=0.8, 
            edgecolor='gray'
        )
    )
    
    return fig