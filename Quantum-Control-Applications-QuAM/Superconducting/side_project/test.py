# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from dataclasses import dataclass

# ==========================================
# 1. 這是剛剛定義好的 Pulse Class
# ==========================================
@dataclass
class RatioCosineBipolarPulse:
    length: int
    amplitude: float
    flip_point_ratio: float
    flat_length_ratio: float
    axis_angle: float = None

    def waveform_function(self):
        def halfcos_up(n: int):
            if n <= 0: return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 - np.cos(np.pi * t))

        def halfcos_down(n: int):
            if n <= 0: return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 + np.cos(np.pi * t))

        L = int(self.length)
        A = float(self.amplitude)
        
        # 計算長度
        pos_total_len = int(np.round(L * self.flip_point_ratio))
        neg_total_len = L - pos_total_len

        pos_flat_len = int(np.round(pos_total_len * self.flat_length_ratio))
        neg_flat_len = int(np.round(neg_total_len * self.flat_length_ratio))

        # Positive Lobe
        pos_remain = pos_total_len - pos_flat_len
        len_rise = pos_remain // 2
        len_fall_to_zero = pos_remain - len_rise

        # Negative Lobe
        neg_remain = neg_total_len - neg_flat_len
        len_fall_from_zero = neg_remain // 2
        len_return_zero = neg_remain - len_fall_from_zero

        # 拼接
        seg_rise = A * halfcos_up(len_rise)
        seg_flat_pos = A * np.ones(pos_flat_len)
        seg_switch_1 = A * halfcos_down(len_fall_to_zero)
        
        seg_switch_2 = -A * halfcos_up(len_fall_from_zero) 
        seg_flat_neg = -A * np.ones(neg_flat_len)
        seg_return = -A * halfcos_down(len_return_zero)

        p = np.concatenate([
            seg_rise, seg_flat_pos, seg_switch_1, 
            seg_switch_2, seg_flat_neg, seg_return
        ])

        # Padding/Trimming
        current_len = len(p)
        if current_len < L:
            pad_total = L - current_len
            pad_front = pad_total // 2
            p = np.concatenate([np.zeros(pad_front), p, np.zeros(pad_total - pad_front)])
        elif current_len > L:
            trim_total = current_len - L
            trim_front = trim_total // 2
            p = p[trim_front: current_len - (trim_total - trim_front)]

        if self.axis_angle is not None:
            p = p * np.exp(1j * self.axis_angle)

        return p

@dataclass
class CosineFlattopPulse:
    amplitude: float
    length:int
    flat_ratio: float  # default no flat top
    axis_angle: float = None

    def waveform_function(self):
        L = int(self.length)
        F = int(L*self.flat_ratio)
        if F > L:
            raise ValueError(
                f"CosineFlatTopPulse.flat_length ({F}) cannot exceed total length ({L})."
            )

        # Remaining samples divided into rise and fall
        remaining = L - F
        rise_len = remaining // 2
        fall_len = remaining - rise_len

        def halfcos_up(n: int):
            if n <= 0:
                return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 - np.cos(np.pi * t))  # 0 → 1

        def halfcos_down(n: int):
            if n <= 0:
                return np.array([])
            t = np.arange(n) / n
            return 0.5 * (1 + np.cos(np.pi * t))  # 1 → 0

        A = float(self.amplitude)
        seg_rise = A * halfcos_up(rise_len)
        seg_flat = A * np.ones(F)
        seg_fall = A * halfcos_down(fall_len)

        # Concatenate waveform
        p = np.concatenate([seg_rise, seg_flat, seg_fall])

        # Ensure exact total length (pad/trim if needed)
        current_len = len(p)
        if current_len < L:
            p = np.pad(p, (0, L - current_len))
        elif current_len > L:
            p = p[:L]

        # Apply axis angle for IQ output if provided
        if self.axis_angle is not None:
            p = p * np.exp(1j * self.axis_angle)

        return p

# ==========================================
# 2. 畫圖測試程式
# ==========================================

def plot_pulse_scenarios(scenarios:list, tot_len:int, flattop_composed:bool=False):
    # 設定三個不同的測試案例
    

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 12), sharex=True)

    amp = 1.0

    for axa, (flip_r, flat_r, title_sup) in zip(axes, scenarios):
        ax:Axes = axa
        # 產生波形
        if not flattop_composed:
            title = f"Positive={(flip_r)*100}%, Negative={(1-flip_r)*100}%, flat ratio = {flat_r*100}%, "
            pulse = RatioCosineBipolarPulse(tot_len, amp, flip_r, flat_r)
            wave = pulse.waveform_function()
            
            # 繪圖
            x_axis = np.arange(len(wave))
            ax.plot(x_axis, wave, linewidth=2.5, color='royalblue')
            
            # 畫出 Zero line
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            
            # 畫出 Flip Point 分界線 (理論值)
            flip_idx = tot_len * flip_r
            ax.axvline(flip_idx, color='red', linestyle=':', linewidth=2, label=f'Flip Point ({flip_idx:.0f})')
            
            # 填充顏色幫助視覺化
            ax.fill_between(x_axis, wave, 0, where=(wave>0), color='green', alpha=0.1, label='Positive Lobe')
            ax.fill_between(x_axis, wave, 0, where=(wave<0), color='red', alpha=0.1, label='Negative Lobe')

            # 標示與美化
            ax.set_title(title+title_sup, fontsize=14, fontweight='bold')
            ax.set_ylabel("Amplitude (V)", fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.2, 1.2)

            # 標示 Flat Length 比例
            pos_len = tot_len * flip_r
            neg_len = tot_len * (1-flip_r)
            ax.text(pos_len/2, 0.5, f"Pos Len: {pos_len:.0f}\nFlat: {flat_r*100:.0f}%", 
                    ha='center', va='center', color='darkgreen', fontweight='bold')
            ax.text(pos_len + neg_len/2, -0.5, f"Neg Len: {neg_len:.0f}\nFlat: {flat_r*100:.0f}%", 
                    ha='center', va='center', color='darkred', fontweight='bold')
        else:
            title = f'Negative amp = {flip_r} * positive amp, flat ratio = {flat_r*100}%, '
            pos_pole = CosineFlattopPulse(amp, tot_len//2, flat_r)
            neg_pole = CosineFlattopPulse(-1*amp*flip_r, tot_len//2, flat_r)
            wave = np.concatenate([pos_pole.waveform_function(), neg_pole.waveform_function()])

            # 繪圖
            x_axis = np.arange(len(wave))
            ax.plot(x_axis, wave, linewidth=2.5, color='royalblue')
            
            # 畫出 Zero line
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            
            # 畫出 amp modifications
            flip_idx = -1 * amp * flip_r
            ax.axhline(flip_idx, color='red', linestyle=':', linewidth=2, label=f'modified amplitude')
            
            # 填充顏色幫助視覺化
            ax.fill_between(x_axis, wave, 0, where=(wave>0), color='green', alpha=0.1, label='Positive Lobe')
            ax.fill_between(x_axis, wave, 0, where=(wave<0), color='red', alpha=0.1, label='Negative Lobe')

            # 標示與美化
            ax.set_title(title+title_sup, fontsize=14, fontweight='bold')
            ax.set_ylabel("Amplitude (V)", fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.2, 1.2)


    axes[-1].set_xlabel("Time (clicks)", fontsize=12)
    plt.tight_layout()
    plt.show()

# %%{plot}
tot_len_ns = 100
scenarios = [
        # (flip_ratio, flat_ratio, 標題補充說明)
        (0.5, 0.5, ""),
        (0.75, 0.85, ""),
        (0.9, 0.85, ""),
]
plot_pulse_scenarios(scenarios, tot_len=tot_len_ns, flattop_composed=True)
# %%
