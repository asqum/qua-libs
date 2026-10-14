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
                title = rf"1Q$\otimes$1Q gate Fidelity = {(1 - (1-self.fidelity)/1.875) * 100:.2f}%{err_str}"
            
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


def _retention_decay(depths, amplitude, tau):
    """Computational population, linear in depth. amplitude absorbs depth-independent SPAM."""
    return amplitude * (1.0 - tau * depths)


def _postselected_decay(depths, amplitude, lam, offset):
    """Post-selected |00> survival. lam is the computational error per Clifford."""
    return amplitude * (1.0 - lam) ** depths + offset


def _subspace_population(depths, intercept, asymptote_frac, lam1):
    """Wood & Gambetta subspace population, p(m) = A + B λ1^m.

    intercept is p(0) = A + B. asymptote_frac in [0, 1] sets A = asymptote_frac * intercept,
    so the curve is monotonic from the intercept toward an asymptote at or below it.
    """
    asymptote = asymptote_frac * intercept
    return asymptote + (intercept - asymptote) * np.power(lam1, depths)


def _lrb_survival(depths, A0, B0, C0, lam2, lam1):
    """Unconditional |00> survival, p0(m) = A0 + B0 λ1^m + C0 λ2^m."""
    return A0 + B0 * np.power(lam1, depths) + C0 * np.power(lam2, depths)


def _lrb_survival_single(depths, A0, C0, lam2):
    """Weak-leakage |00> survival, p0(m) = A0 + C0 λ2^m."""
    return A0 + C0 * np.power(lam2, depths)


def _r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


@dataclasses.dataclass
class LeakageRBResult:
    """
    Two-qubit RB with leakage, using leakage post-selection in the computational-dominant regime.

    State codes 0, 1, 2, 3 are the computational basis. Any state >= 4 is leakage.
    Retention P(state < 4) = A (1 - τ ℓ) gives the leakage rate τ.
    Post-selected P(state == 0 | state < 4) = A (1 - λ)^ℓ + B gives the computational error λ.
    Average fidelity is F = 1 - ((d_C - 1) / d_C) λ - τ, with d_C = 4.

    This is the LPS analysis of Chen & Baldwin, arXiv:2502.00154, Sec. IV.2.3.
    The single-exponential fit of P(|00>) is kept as naive_fidelity for comparison.

    fit() also runs the Wood & Gambetta leakage RB model (arXiv:1704.03081):
    P(no leakage) = A + B λ1^m, with L1 = (1 - A)(1 - λ1) and L2 = A(1 - λ1).
    P(|00>) = A0 + B0 λ1^m + C0 λ2^m, and
    F = [(d_C - 1) λ2 + 1 - L1] / d_C.
    plot_lrb() draws that fit. L1 and L2 are separately determined only once the
    subspace population has approached its asymptote.
    """

    circuit_depths: list[int]
    num_repeats: int
    num_averages: int
    state: np.ndarray
    d_c: int = 4

    def __post_init__(self):
        self.state = np.asarray(self.state)
        expected = (self.num_repeats, len(self.circuit_depths), self.num_averages)
        if self.state.shape != expected:
            raise ValueError(
                f"state shape {self.state.shape} does not match "
                f"(num_repeats, num_depths, num_averages) = {expected}"
            )
        self.depths = np.asarray(self.circuit_depths, dtype=float)

    def _sequence_probabilities(self):
        computational = (self.state == 0).mean(axis=-1)
        retention = (self.state < 4).mean(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            postselected = np.where(retention > 0, computational / retention, np.nan)
        return computational, retention, postselected

    @staticmethod
    def _mean_and_sem(values):
        mean = np.nanmean(values, axis=0)
        count = np.sum(np.isfinite(values), axis=0)
        std = np.nanstd(values, axis=0, ddof=1)
        sem = np.full(mean.shape, np.nan, dtype=float)
        enough = count >= 2
        sem[enough] = std[enough] / np.sqrt(count[enough])
        finite_mean = np.isfinite(mean)
        zero_sem = finite_mean & (~np.isfinite(sem) | (sem == 0))
        sem[zero_sem] = 1e-8
        return mean, sem

    def _fit_curve(self, model, y, sigma, p0, bounds, return_cov=False):
        mask = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        if np.count_nonzero(mask) < len(p0):
            raise RuntimeError(
                f"Not enough finite points ({np.count_nonzero(mask)}) to fit {len(p0)} parameters."
            )
        popt, pcov = curve_fit(
            model,
            self.depths[mask],
            y[mask],
            p0=p0,
            bounds=bounds,
            sigma=sigma[mask],
            absolute_sigma=True,
            maxfev=20000,
        )
        errors = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        r_squared = _r_squared(y[mask], model(self.depths[mask], *popt))
        if return_cov:
            return popt, errors, r_squared, pcov
        return popt, errors, r_squared

    def fit(self):
        computational, retention, postselected = self._sequence_probabilities()
        self.p_comp, self.p_comp_err = self._mean_and_sem(computational)
        self.p_retention, self.p_retention_err = self._mean_and_sem(retention)
        self.p_post, self.p_post_err = self._mean_and_sem(postselected)

        max_depth = float(np.max(self.depths)) if len(self.depths) else 1.0
        tau_upper = 0.95 / max(max_depth, 1.0)
        amplitude0 = float(np.clip(self.p_retention[0], 0.05, 1.0))
        (self.A_ret, self.tau), (self.A_ret_err, self.tau_err), self.retention_r_squared = self._fit_curve(
            _retention_decay,
            self.p_retention,
            self.p_retention_err,
            p0=[amplitude0, 1e-3],
            bounds=([0.0, 0.0], [1.0, tau_upper]),
        )

        (self.A_post, self.lam, self.B_post), (
            self.A_post_err,
            self.lam_err,
            self.B_post_err,
        ), self.post_r_squared = self._fit_curve(
            _postselected_decay,
            self.p_post,
            self.p_post_err,
            p0=[(self.d_c - 1) / self.d_c, 0.01, 1.0 / self.d_c],
            bounds=([0.0, 0.0, 0.0], [1.0, 0.999, 1.0]),
        )

        weight = (self.d_c - 1) / self.d_c
        self.fidelity = 1.0 - weight * self.lam - self.tau
        self.fidelity_err = float(np.sqrt((weight * self.lam_err) ** 2 + self.tau_err ** 2))
        self.comp_dominant_ok = bool(self.tau * max_depth < 0.2)

        (naive_A, naive_alpha, naive_B), (
            self.naive_A_err,
            self.naive_alpha_err,
            self.naive_B_err,
        ), self.naive_r_squared = self._fit_curve(
            rb_decay_curve,
            self.p_comp,
            self.p_comp_err,
            p0=[0.75, 0.9, 0.25],
            bounds=([-1.0, 0.0, -1.0], [2.0, 1.0, 2.0]),
        )
        self.naive_A = naive_A
        self.naive_alpha = naive_alpha
        self.naive_B = naive_B
        self.naive_fidelity = 1.0 - weight * (1.0 - naive_alpha)
        self.naive_fidelity_err = float(weight * self.naive_alpha_err)
        self._fit_wood_gambetta()
        return self

    def _fit_wood_gambetta(self):
        """Leakage and seepage rates from Wood & Gambetta, arXiv:1704.03081."""
        self.lrb_ok = False
        self.L1 = np.nan
        self.L1_err = np.nan
        self.L2 = np.nan
        self.L2_err = np.nan
        self.lrb_A = np.nan
        self.lrb_B = np.nan
        self.lrb_lam1 = np.nan
        self.lrb_lam1_err = np.nan
        self.lrb_lam2 = np.nan
        self.lrb_lam2_err = np.nan
        self.lrb_A0 = np.nan
        self.lrb_B0 = np.nan
        self.lrb_C0 = np.nan
        self.lrb_fidelity = np.nan
        self.lrb_fidelity_err = np.nan
        self.lrb_subspace_r_squared = np.nan
        self.lrb_survival_r_squared = np.nan
        self.lrb_survival_model = "none"
        self.lrb_plateau_resolved = False
        try:
            intercept0 = float(np.clip(self.p_retention[0], 0.05, 1.0))
            lam1_guess = float(np.clip(1.0 - max(5.0 * self.tau, 1e-3), 0.5, 0.999))
            (intercept, asymptote_frac, lam1), errors, self.lrb_subspace_r_squared, pcov = self._fit_curve(
                _subspace_population,
                self.p_retention,
                self.p_retention_err,
                p0=[intercept0, 0.8, lam1_guess],
                bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                return_cov=True,
            )
            self.lrb_A = float(asymptote_frac * intercept)
            self.lrb_B = float(intercept - self.lrb_A)
            self.lrb_lam1 = float(lam1)
            self.lrb_lam1_err = float(errors[2])
            one_minus = 1.0 - self.lrb_lam1
            self.L1 = (1.0 - self.lrb_A) * one_minus
            self.L2 = self.lrb_A * one_minus
            if np.all(np.isfinite(pcov)):
                jacobian = np.array(
                    [
                        [-asymptote_frac * one_minus, -intercept * one_minus, -(1.0 - self.lrb_A)],
                        [asymptote_frac * one_minus, intercept * one_minus, -self.lrb_A],
                    ]
                )
                rate_cov = jacobian @ pcov @ jacobian.T
                self.L1_err = float(np.sqrt(max(rate_cov[0, 0], 0.0)))
                self.L2_err = float(np.sqrt(max(rate_cov[1, 1], 0.0)))
            max_depth = float(np.max(self.depths)) if len(self.depths) else 0.0
            self.lrb_plateau_resolved = bool(np.power(self.lrb_lam1, max_depth) < 0.2)
            self.lrb_ok = True
            try:
                self._fit_lrb_survival()
            except (RuntimeError, ValueError, FloatingPointError):
                self.lrb_survival_model = "none"
        except (RuntimeError, ValueError, FloatingPointError):
            return

    def _fit_lrb_survival(self):
        """Fit P(|00>) with λ1 held fixed. Fall back to one exponential when leakage is weak."""
        lam1 = self.lrb_lam1

        def two_exp(depths, A0, B0, C0, lam2):
            return _lrb_survival(depths, A0, B0, C0, lam2, lam1)

        A0_guess = 1.0 / self.d_c
        C0_guess = float(np.clip(self.p_comp[0] - A0_guess, 0.0, 1.0))
        lam2_guess = float(np.clip(self.naive_alpha, 0.5, 0.999))
        double_ok = False
        try:
            (A0, B0, C0, lam2), errors, r_squared = self._fit_curve(
                two_exp,
                self.p_comp,
                self.p_comp_err,
                p0=[A0_guess, 0.0, C0_guess, lam2_guess],
                bounds=([0.0, -1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
            )
            if np.isfinite(errors[3]) and errors[3] < 0.2:
                double_ok = True
        except (RuntimeError, ValueError, FloatingPointError):
            A0 = B0 = C0 = lam2 = np.nan
            errors = np.full(4, np.nan)

        if not double_ok:
            (A0, C0, lam2), errors_single, r_squared = self._fit_curve(
                _lrb_survival_single,
                self.p_comp,
                self.p_comp_err,
                p0=[A0_guess, C0_guess, lam2_guess],
                bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            )
            B0 = 0.0
            errors = np.array([errors_single[0], 0.0, errors_single[1], errors_single[2]])
            self.lrb_survival_model = "single"
        else:
            self.lrb_survival_model = "double"

        self.lrb_A0 = float(A0)
        self.lrb_B0 = float(B0)
        self.lrb_C0 = float(C0)
        self.lrb_lam2 = float(lam2)
        self.lrb_lam2_err = float(errors[3])
        self.lrb_survival_r_squared = float(r_squared)
        self.lrb_fidelity = ((self.d_c - 1) * self.lrb_lam2 + 1.0 - self.L1) / self.d_c
        lam2_weight = (self.d_c - 1) / self.d_c
        l1_term = 0.0 if not np.isfinite(self.L1_err) else (self.L1_err / self.d_c) ** 2
        self.lrb_fidelity_err = float(np.sqrt((lam2_weight * self.lrb_lam2_err) ** 2 + l1_term))

    def plot(self, title=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        depths = self.depths
        smooth = np.linspace(depths.min(), depths.max(), 200)

        ax.errorbar(
            depths, self.p_comp, yerr=self.p_comp_err, fmt="o", color="tab:blue",
            capsize=3, label=r"$P(|00\rangle)$",
        )
        ax.errorbar(
            depths, self.p_retention, yerr=self.p_retention_err, fmt="s", color="black",
            capsize=3, label="retention $P($no leakage$)$",
        )
        ax.errorbar(
            depths, self.p_post, yerr=self.p_post_err, fmt="^", color="tab:green",
            capsize=3, label=r"post-selected $P(|00\rangle)$",
        )
        ax.plot(smooth, _retention_decay(smooth, self.A_ret, self.tau), color="black", linestyle="--")
        ax.plot(
            smooth, _postselected_decay(smooth, self.A_post, self.lam, self.B_post),
            color="tab:green", linestyle="--",
        )
        ax.plot(
            smooth,
            _retention_decay(smooth, self.A_ret, self.tau)
            * _postselected_decay(smooth, self.A_post, self.lam, self.B_post),
            color="tab:blue", linestyle="--", label="model",
        )

        regime = "τℓ ≪ 1 holds" if self.comp_dominant_ok else "τℓ is not ≪ 1 at the longest depth"
        ax.set_title(
            f"F = {self.fidelity * 100:.3f}% ± {self.fidelity_err * 100:.3f}%\n"
            f"λ = {self.lam:.3e} ± {self.lam_err:.3e},  τ = {self.tau:.3e} ± {self.tau_err:.3e}\n"
            f"single-exponential $P(|00\\rangle)$ F = {self.naive_fidelity * 100:.3f}%"
        )
        ax.text(
            0.04, 0.04,
            f"circuits per depth: {self.num_repeats}\n"
            f"retention $R^2$ = {self.retention_r_squared:.3f},  "
            f"post-selected $R^2$ = {self.post_r_squared:.3f}\n"
            f"{regime}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
        )
        ax.set_xlabel("Circuit depth (Cliffords)")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(framealpha=0, loc="upper right")
        if title:
            fig.suptitle(title)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
        else:
            fig.tight_layout()
        return fig

    def plot_lrb(self, title=None):
        """Wood & Gambetta leakage and seepage fit, arXiv:1704.03081."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
        depths = self.depths
        smooth = np.linspace(depths.min(), depths.max(), 200)
        ax_pop, ax_surv = axes

        ax_pop.errorbar(
            depths, self.p_retention, yerr=self.p_retention_err, fmt="s", color="black",
            capsize=3, label="retention $P($no leakage$)$",
        )
        ax_surv.errorbar(
            depths, self.p_comp, yerr=self.p_comp_err, fmt="o", color="tab:blue",
            capsize=3, label=r"$P(|00\rangle)$",
        )

        if self.lrb_ok:
            ax_pop.plot(
                smooth,
                self.lrb_A + self.lrb_B * np.power(self.lrb_lam1, smooth),
                color="black", linestyle="--", label=r"$A + B\lambda_1^{m}$",
            )
            ax_pop.axhline(self.lrb_A, color="tab:orange", linestyle=":", label=f"asymptote $A$ = {self.lrb_A:.3f}")
            ax_pop.plot(
                smooth, _retention_decay(smooth, self.A_ret, self.tau),
                color="0.6", linestyle="--", label="linear retention",
            )
            regime = (
                r"asymptote resolved: $\lambda_1^{m_{\max}} < 0.2$"
                if self.lrb_plateau_resolved
                else r"asymptote not resolved; $L_1$ and $L_2$ are not separate"
            )
            ax_pop.set_title(
                f"$L_1$ = {self.L1:.3e} ± {self.L1_err:.3e}\n"
                f"$L_2$ = {self.L2:.3e} ± {self.L2_err:.3e}\n"
                f"$\\lambda_1$ = {self.lrb_lam1:.5f}"
            )
            ax_pop.text(
                0.04, 0.04,
                f"subspace $R^2$ = {self.lrb_subspace_r_squared:.3f}\n{regime}",
                transform=ax_pop.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
            )
        else:
            ax_pop.set_title("subspace fit failed")

        if self.lrb_survival_model == "double":
            ax_surv.plot(
                smooth,
                _lrb_survival(smooth, self.lrb_A0, self.lrb_B0, self.lrb_C0, self.lrb_lam2, self.lrb_lam1),
                color="tab:blue", linestyle="--", label="two-exponential",
            )
        elif self.lrb_survival_model == "single":
            ax_surv.plot(
                smooth,
                _lrb_survival_single(smooth, self.lrb_A0, self.lrb_C0, self.lrb_lam2),
                color="tab:blue", linestyle="--", label="single-exponential",
            )

        if self.lrb_survival_model in ("double", "single"):
            ax_surv.set_title(
                f"$F$ = {self.lrb_fidelity * 100:.3f}% ± {self.lrb_fidelity_err * 100:.3f}%\n"
                f"$\\lambda_2$ = {self.lrb_lam2:.5f} ± {self.lrb_lam2_err:.3e}\n"
                f"survival model: {self.lrb_survival_model}"
            )
            ax_surv.text(
                0.04, 0.04,
                f"survival $R^2$ = {self.lrb_survival_r_squared:.3f}",
                transform=ax_surv.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
            )
        else:
            ax_surv.set_title(r"$P(|00\rangle)$ fit failed")

        for ax in axes:
            ax.set_xlabel("Circuit depth (Cliffords)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle=":", alpha=0.7)
            ax.legend(framealpha=0, loc="upper right")
        ax_pop.set_ylabel("Probability")
        if title:
            fig.suptitle(title)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
        else:
            fig.tight_layout()
        return fig


def _avg_mb_decay(depths, amplitude, r, offset):
    """Averaged-basis survival. r = 1 - λ - τ is the depolarizing parameter."""
    return amplitude * r ** depths + offset


@dataclasses.dataclass
class AvgMBLeakageRBResult:
    """
    Two-qubit RB, average over measurement basis, computational-dominant leakage regime.

    Each random circuit is measured four times. A final II, X(target), X(control), or XX
    maps the ideal |00> to |00>, |01>, |10>, or |11>. p_avg is the average probability of
    the matching outcome. Retention P(state < 4) = A (1 - τ ℓ) gives the leakage rate.

    p_avg(ℓ) = A r^ℓ + B, with r = 1 - λ - τ.
    F = 1 - ((d_C - 1) / d_C) λ - τ.

    Chen & Baldwin, arXiv:2502.00154, Sec. IV.2.2. State layout is
    (num_random * 4, depth, shot), bases ordered 0, 1, 2, 3 within each random circuit.
    """

    circuit_depths: list[int]
    num_repeats: int
    num_averages: int
    state: np.ndarray
    n_bases: int = 4
    d_c: int = 4

    def __post_init__(self):
        self.state = np.asarray(self.state)
        expected = (self.num_repeats, len(self.circuit_depths), self.num_averages)
        if self.state.shape != expected:
            raise ValueError(
                f"state shape {self.state.shape} does not match "
                f"(num_repeats, num_depths, num_averages) = {expected}"
            )
        if self.num_repeats % self.n_bases != 0:
            raise ValueError(
                f"num_repeats ({self.num_repeats}) is not divisible by n_bases ({self.n_bases})"
            )
        self.depths = np.asarray(self.circuit_depths, dtype=float)
        self.num_random_circuits = self.num_repeats // self.n_bases

    def _basis_probabilities(self):
        state = self.state.reshape(
            self.num_random_circuits, self.n_bases, len(self.circuit_depths), self.num_averages
        )
        expected = np.arange(self.n_bases)
        success = state == expected[None, :, None, None]
        p_avg = success.mean(axis=(1, 3))
        p_retention = (state < 4).mean(axis=(1, 3))
        return p_avg, p_retention

    def _fit_curve(self, model, y, sigma, p0, bounds):
        mask = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        if np.count_nonzero(mask) < len(p0):
            raise RuntimeError(
                f"Not enough finite points ({np.count_nonzero(mask)}) to fit {len(p0)} parameters."
            )
        popt, pcov = curve_fit(
            model,
            self.depths[mask],
            y[mask],
            p0=p0,
            bounds=bounds,
            sigma=sigma[mask],
            absolute_sigma=True,
            maxfev=20000,
        )
        errors = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        return popt, errors, _r_squared(y[mask], model(self.depths[mask], *popt))

    def fit(self):
        p_avg_rep, p_ret_rep = self._basis_probabilities()
        self.p_avg, self.p_avg_err = LeakageRBResult._mean_and_sem(p_avg_rep)
        self.p_retention, self.p_retention_err = LeakageRBResult._mean_and_sem(p_ret_rep)

        max_depth = float(np.max(self.depths)) if len(self.depths) else 1.0
        tau_upper = 0.95 / max(max_depth, 1.0)
        amplitude0 = float(np.clip(self.p_retention[0], 0.05, 1.0))
        (self.A_ret, self.tau), (self.A_ret_err, self.tau_err), self.retention_r_squared = self._fit_curve(
            _retention_decay,
            self.p_retention,
            self.p_retention_err,
            p0=[amplitude0, 1e-3],
            bounds=([0.0, 0.0], [1.0, tau_upper]),
        )
        (self.A_avg, self.r, self.B_avg), (
            self.A_avg_err,
            self.r_err,
            self.B_avg_err,
        ), self.avg_r_squared = self._fit_curve(
            _avg_mb_decay,
            self.p_avg,
            self.p_avg_err,
            p0=[(self.d_c - 1) / self.d_c, 0.99, 1.0 / self.d_c],
            bounds=([0.0, 0.0, 0.0], [1.5, 1.0, 1.0]),
        )

        self.lam = 1.0 - self.r - self.tau
        self.lam_err = float(np.sqrt(self.r_err ** 2 + self.tau_err ** 2))
        weight = (self.d_c - 1) / self.d_c
        self.fidelity = 1.0 - weight * self.lam - self.tau
        self.fidelity_err = float(np.sqrt((weight * self.r_err) ** 2 + ((1.0 - weight) * self.tau_err) ** 2))
        self.bound_infidelity = weight * (1.0 - self.r)
        self.comp_dominant_ok = bool(self.tau * max_depth < 0.2)
        return self

    def plot(self, title=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        depths = self.depths
        smooth = np.linspace(depths.min(), depths.max(), 200)

        ax.errorbar(
            depths, self.p_avg, yerr=self.p_avg_err, fmt="o", color="tab:blue",
            capsize=3, label=r"$p_{\mathrm{avg}}$",
        )
        ax.errorbar(
            depths, self.p_retention, yerr=self.p_retention_err, fmt="s", color="black",
            capsize=3, label="retention $P($no leakage$)$",
        )
        ax.plot(smooth, _avg_mb_decay(smooth, self.A_avg, self.r, self.B_avg), color="tab:blue", linestyle="--")
        ax.plot(smooth, _retention_decay(smooth, self.A_ret, self.tau), color="black", linestyle="--")

        regime = "τℓ ≪ 1 holds" if self.comp_dominant_ok else "τℓ is not ≪ 1 at the longest depth"
        ax.set_title(
            f"F = {self.fidelity * 100:.3f}% ± {self.fidelity_err * 100:.3f}%\n"
            f"r = {self.r:.5f},  λ = {self.lam:.3e},  τ = {self.tau:.3e}\n"
            f"bound infidelity $(3/4)(1-r)$ = {self.bound_infidelity:.3e}"
        )
        ax.text(
            0.04, 0.04,
            f"random circuits per depth: {self.num_random_circuits},  4 final bases\n"
            f"avg $R^2$ = {self.avg_r_squared:.3f},  retention $R^2$ = {self.retention_r_squared:.3f}\n"
            f"{regime}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="gray"),
        )
        ax.set_xlabel("Circuit depth (Cliffords)")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(framealpha=0, loc="upper right")
        if title:
            fig.suptitle(title)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
        else:
            fig.tight_layout()
        return fig