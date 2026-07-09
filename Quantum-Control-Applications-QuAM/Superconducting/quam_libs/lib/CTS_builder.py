import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
from abc import ABC, abstractmethod
import math

class CTSmethod(ABC):
    def __init__(self):
        # Time domain sampling rate (samples/ns)
        self.fs = 1 
        self.res = None

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    @abstractmethod
    def plot_result(self, *args, **kwargs):
        pass

    @abstractmethod
    def output_waveform(self, *args, **kwargs):
        pass



class CTS_GRAPE(CTSmethod):
    def __init__(self, gate_time_ns:int, freqs_to_avoid:list, aiming_sup_power_dB:int):
        super().__init__()
        # target crosstalk suppression power in dB
        self.sup_power = aiming_sup_power_dB
        # gate time ns
        self.gate_time = gate_time_ns
        # avoid frequencies
        self.target_holes = freqs_to_avoid

        self.t = np.arange(0, self.gate_time + 0.5/self.fs, 1/self.fs)
        self.N = len(self.t)  

    
    def low_pass_filter(self):
        cutoff_mhz = np.max(np.abs(self.target_holes))*1.25
        nyquist_mhz = (self.fs * 1000) / 2  
        Wn = cutoff_mhz / nyquist_mhz  

        # 建立 4 階低通巴特沃斯濾波器係數
        return butter(4, Wn, btype='low')
    

    def reconstruct_and_filter(self, coeffs):
        # coeffs 是優化器試圖控制的「原始不平滑振幅」
        I_raw = coeffs[:self.N].copy()
        Q_raw = coeffs[self.N:].copy()
        
        # 使用 filtfilt 進行雙向濾波，確保「零相位延遲」（波形不會在時間軸上向後位移）
        # 這是逼迫演算法保持平滑的核心步驟
        b_filter, a_filter = self.low_pass_filter()
        I_smooth = filtfilt(b_filter, a_filter, I_raw)
        Q_smooth = filtfilt(b_filter, a_filter, Q_raw)
        
        # 強制頭尾歸零 (邊界條件)
        I_smooth[0] = I_smooth[-1] = 0.0
        Q_smooth[0] = Q_smooth[-1] = 0.0
        
        return I_raw, Q_raw, I_smooth, Q_smooth
    
    def grape_filtered_objective(self, coeffs):
        # 拿到燙平後的平滑波形
        _, _, I_smooth, Q_smooth = self.reconstruct_and_filter(coeffs)
        complex_signal = I_smooth + 1j * Q_smooth
        
        # 高倍率補零，精準對齊頻率格點
        N_padded = self.N * 50
        fft_y = np.fft.fftshift(np.fft.fft(complex_signal, n=N_padded))
        freqs = np.fft.fftshift(np.fft.fftfreq(N_padded, 1/(self.fs * 1000)))
        
        # 頻譜內部正規化
        psd_norm = np.abs(fft_y)**2 / (np.max(np.abs(fft_y)**2) + 1e-12)
        
        # 計算 9 個洞的懲罰值
        penalty_holes = 0.0
        for hole_f in self.target_holes:
            idx = np.argmin(np.abs(freqs - hole_f))
            db_val = 10 * np.log10(psd_norm[idx] + 1e-12)
            if db_val > self.sup_power:
                penalty_holes += (db_val - self.sup_power)**2 
                
        # 注意：這裡完全不需要加任何 smoothness penalty！
        return penalty_holes 
    

    def optimize(self, optimizer:Literal['L-BFGS-B','Nelder-Mead','Powell','CG','BFGS','Newton-CG','TNC','COBYLA','COBYQA','SLSQP','trust-constr','dogleg','trust-ncg','trust-exact','trust-krylov']='L-BFGS-B'):
        initial_coeffs = np.zeros(2 * self.N)
        initial_coeffs[:self.N] = np.sin(np.pi * self.t / self.gate_time)
        initial_coeffs[self.N:] = 0.1 * np.sin(np.pi * self.t / self.gate_time)
        box_bounds = [(-1.0, 1.0)] * (2 * self.N)

        self.res = minimize(self.grape_filtered_objective, initial_coeffs, method=optimizer, bounds=box_bounds, options={'maxiter': 60000})

        print(f"Results:\n Final Penalty={self.res.fun:.4f}")

    
    def plot_result(self):

        plt.rcParams.update({
            'font.size': 14, 'axes.linewidth': 2, 'lines.linewidth': 2.5,
            'xtick.major.width': 2, 'ytick.major.width': 2,
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'xtick.major.size': 8, 'ytick.major.size': 8,
        })

        opt_I_raw, opt_Q_raw, opt_I_plot, opt_Q_plot = self.reconstruct_and_filter(self.res.x)
    
        plt.figure(figsize=(14, 8))

        # --- 左圖：時域波形對比 ---
        plt.subplot(1, 2, 1)
        # 畫出最終要送進 Qubit 的平滑波形 (實線)
        plt.plot(self.t, opt_I_plot, 'b-', linewidth=2.5, label='Filtered I')
        plt.plot(self.t, opt_Q_plot, 'r-', linewidth=2.5, label='Filtered Q')
        # 隱約畫出優化器在背後操控的原始離散點 (虛線/點)，觀察濾波器如何馴服它
        plt.plot(self.t, opt_I_raw, 'b.', alpha=0.2, linestyle='--', label='GRAPE I')
        plt.plot(self.t, opt_Q_raw, 'r.', alpha=0.2, linestyle='--', label='GRAPE Q')

        plt.title('Time Domain')
        plt.ylabel('Amplitude [a.u.]')
        plt.xlabel('Time (ns)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # --- 右圖：頻域能量譜 ---
        plt.subplot(1, 2, 2)
        N_p = self.N * 50
        freqs_p = np.fft.fftshift(np.fft.fftfreq(N_p, 1/(self.fs * 1000)))
        fft_complex = np.fft.fftshift(np.fft.fft(opt_I_plot + 1j * opt_Q_plot, n=N_p))
        db_complex = 10 * np.log10(np.abs(fft_complex)**2 / np.max(np.abs(fft_complex)**2))

        plt.plot(freqs_p, db_complex, 'purple', linewidth=2, label='Filtered Spectrum')
        for i, hole_f in enumerate(self.target_holes):
            plt.axvline(hole_f, color='red', linestyle='--', alpha=0.6, label='Target Holes' if i == 0 else "")
        plt.axhline(self.sup_power, color='black', linestyle=':', linewidth=1.5, label=f'{self.sup_power} dB Line')

        plt.ylim(-100, 5)
        plt.xlim(-300, 300)
        plt.ylabel('Amplitude [dB]')
        plt.xlabel('Frequency (MHz)')
        plt.title('Frequency Domain')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.suptitle(f"Hardware-In-The-Loop GRAPE Optimization\nTarget avoid frequencies: {', '.join(map(str, self.target_holes))} MHz \nGate Time: {self.gate_time} ns", y=0.98)
        plt.tight_layout()
        plt.show()

    def output_waveform(self):
        """ Return dict with keyname 'waveform_I' and 'waveform_Q' and its value in list."""

        _, _, opt_I_plot, opt_Q_plot = self.reconstruct_and_filter(self.res.x)
        max_amp = np.max(np.abs(opt_I_plot + 1j * opt_Q_plot))
        final_I = np.array(opt_I_plot / max_amp)
        final_Q = np.array(opt_Q_plot / max_amp)

        return {"waveform_I":final_I.tolist(), "waveform_Q":final_Q.tolist()}



class CTS_dCRAB(CTSmethod):
    def __init__(self, gate_time_ns:int, freqs_to_avoid:list, aiming_sup_power_dB:int):
        super().__init__()
        # target crosstalk suppression power in dB
        self.sup_power = aiming_sup_power_dB
        # gate time ns
        self.gate_time = gate_time_ns
        # avoid frequencies
        self.target_holes = freqs_to_avoid

        self.t = np.arange(0, self.gate_time + 0.5/self.fs, 1/self.fs)
        self.N = len(self.t)  
        self.max_macro_iters = 10
        self.accumulated_freqs = []
        self.envelope = np.sin(np.pi * self.t / self.gate_time)

    def reconstruct_dcrab(self, coeffs, freqs_list):
        n_freqs = len(freqs_list)
        if n_freqs == 0:
            return np.zeros(self.N), np.zeros(self.N)
            
        # 前 n_freqs 個係數是 I 的振幅，後 n_freqs 個是 Q 的振幅
        a = coeffs[:n_freqs]
        b = coeffs[n_freqs:]
        
        I_core = np.zeros(self.N)
        Q_core = np.zeros(self.N)
        
        # 疊加隨機頻率成份 (使用 GHz 單位進行時間乘法)
        for i, f_mhz in enumerate(freqs_list):
            f_ghz = f_mhz / 1000.0
            I_core += a[i] * np.sin(2 * np.pi * f_ghz * self.t)
            Q_core += b[i] * np.cos(2 * np.pi * f_ghz * self.t)
            
        # 乘以包絡線強迫邊界歸零，並加上一個直流基本項確保主瓣存在
        I_ch = self.envelope * (1.0 + I_core)
        Q_ch = self.envelope * Q_core
            
        return I_ch, Q_ch 
    

    def dcrab_objective(self, coeffs):
        I_ch, Q_ch = self.reconstruct_dcrab(coeffs, self.accumulated_freqs)
        complex_signal = I_ch + 1j * Q_ch
        
        # 高解析度 FFT
        N_padded = self.N * 50
        fft_y = np.fft.fftshift(np.fft.fft(complex_signal, n=N_padded))
        freqs_fft = np.fft.fftshift(np.fft.fftfreq(N_padded, 1/(self.fs * 1000)))
        
        psd_norm = np.abs(fft_y)**2 / np.max(np.abs(fft_y)**2)
        
        worst_db = -100.0
        penalty = 0.0
        
        # 檢查 9 個目標點
        for hole_f in self.target_holes:
            idx = np.argmin(np.abs(freqs_fft - hole_f))
            db_val = 10 * np.log10(psd_norm[idx] + 1e-12)
            worst_db = max(worst_db, db_val)
            
            # 只要沒有低於 -60 dB，就給予平方懲罰
            if db_val > self.sup_power:
                penalty += (db_val - (self.sup_power))**2

          
        return penalty


    def optimize(self, optimizer:Literal['L-BFGS-B','Nelder-Mead','Powell','CG','BFGS','Newton-CG','TNC','COBYLA','COBYQA','SLSQP','trust-constr','dogleg','trust-ncg','trust-exact','trust-krylov']='L-BFGS-B'):
        np.random.seed(42)
        
        for macro_it in range(1, self.max_macro_iters + 1):

            new_freqs = np.random.uniform(30, 250, size=2)
            self.accumulated_freqs.extend(new_freqs)
            
            n_current = len(self.accumulated_freqs)
            print(f"\n[Iter: {macro_it}] Accumulated {n_current} random basis ...")
            print(f"-> New random basis: {new_freqs[0]:.1f} MHz, {new_freqs[1]:.1f} MHz")
            
            # 初始化目前所有頻率的係數 (2 * n_current 個變數)
            initial_coeffs = np.zeros(2 * n_current)

            # ✨ 修正：如果不是第一輪，就把上一輪優化好的成果繼承過來（Warm-start）
            if macro_it > 1 and self.res is not None:
                n_old = n_current - 2  # 上一輪的頻率數量
                # 繼承舊的 I 通道係數 (a)
                initial_coeffs[:n_old] = self.res.x[:n_old]
                # 繼承舊的 Q 通道係數 (b)
                initial_coeffs[n_current : n_current + n_old] = self.res.x[n_old:]
            box_bounds = [(-1.0, 1.0)] * (2 * n_current)
            
            # 執行優化
            self.res = minimize(self.dcrab_objective, initial_coeffs, method=optimizer, bounds=box_bounds, options={'maxiter': 60000})
            
            
            
            # 驗證這一輪優化完後，最爛的洞表現如何
            opt_I, opt_Q = self.reconstruct_dcrab(self.res.x, self.accumulated_freqs)
            fft_res = np.fft.fftshift(np.fft.fft(opt_I + 1j*opt_Q, n=self.N*50))
            freqs_fft = np.fft.fftshift(np.fft.fftfreq(self.N*50, 1/(self.fs * 1000)))
            psd_res = np.abs(fft_res)**2 / np.max(np.abs(fft_res)**2)
            
            all_holes_db = [10 * np.log10(psd_res[np.argmin(np.abs(freqs_fft - h))] + 1e-12) for h in self.target_holes]
            worst_hole_current = max(all_holes_db)
            
            print(f"The worst power: {worst_hole_current:.2f} dB")
            
            # 如果最差的洞都順利跌破 -60 dB，大功告成，提前提早結束！
            if worst_hole_current <= self.sup_power:
                print(f"All target avoid freqs under the aiming suppression power {self.sup_power} dB")
                break


    def plot_result(self):

        plt.rcParams.update({
            'font.size': 14, 'axes.linewidth': 2, 'lines.linewidth': 2.5,
            'xtick.major.width': 2, 'ytick.major.width': 2,
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'xtick.major.size': 8, 'ytick.major.size': 8,
        })
        waveform_dict = self.output_waveform()
        opt_I, opt_Q = np.array(waveform_dict['waveform_I']),  np.array(waveform_dict['waveform_Q'])
        N_p = self.N * 50
        freqs_plot = np.fft.fftshift(np.fft.fftfreq(N_p, 1/(self.fs * 1000)))
        fft_final = np.fft.fftshift(np.fft.fft(opt_I + 1j*opt_Q, n=N_p))
        db_final = 10 * np.log10(np.abs(fft_final)**2 / np.max(np.abs(fft_final)**2))

        plt.figure(figsize=(14, 8))

        # 左圖：時間域（自帶包絡線，絕對平滑且頭尾歸零）
        plt.subplot(1, 2, 1)
        plt.plot(self.t, opt_I, 'b.-', linewidth=2, label='I Channel')
        plt.plot(self.t, opt_Q, 'r.-', linewidth=2, label='Q Channel')
        plt.title(f'Time Domain')
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        

        plt.plot(freqs_plot, db_final, 'purple', linewidth=1.5, label='dCRAB Spectrum')
        for i, hole_f in enumerate(self.target_holes):
            plt.axvline(hole_f, color='red', linestyle='--', alpha=0.5, label='9 Target Holes' if i == 0 else "")
        plt.axhline(self.sup_power, color='black', linestyle=':', label=f'{self.sup_power} dB Line')

        plt.ylim(-85, 5)
        plt.xlim(-300, 300)
        plt.title(f'Frequency Domain')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Amplitude (dB)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(f"dCRAB Optimization Result\nTarget avoid frequencies: {', '.join(map(str, self.target_holes))} MHz \nGate Time: {self.gate_time} ns | Total Random Frequencies: {len(self.accumulated_freqs)}\n")
        plt.tight_layout()
        plt.show()


    def output_waveform(self):
        """ Return dict with keyname 'waveform_I' and 'waveform_Q' and its value in list."""

        opt_I, opt_Q = self.reconstruct_dcrab(self.res.x, self.accumulated_freqs)
        max_amp = np.max(np.abs(opt_I + 1j * opt_Q))
        final_I = np.array(opt_I / max_amp)
        final_Q = np.array(opt_Q / max_amp)

        return {"waveform_I":final_I.tolist(), "waveform_Q":final_Q.tolist()}



class CTS_CRAB(CTSmethod):
    def __init__(self, gate_time_ns:int, freqs_to_avoid:list, aiming_sup_power_dB:int):
        super().__init__()
        # target crosstalk suppression power in dB
        self.sup_power = aiming_sup_power_dB
        # gate time ns
        self.gate_time = gate_time_ns
        # avoid frequencies
        self.target_holes = freqs_to_avoid

        self.t = np.arange(0, self.gate_time + 0.5/self.fs, 1/self.fs)
        self.N = len(self.t)  
        self.best_M = None
        self.best_res = None

        MM = math.ceil(2 * self.gate_time * max(np.abs(self.target_holes))/1000)

        if MM < 5:
            self.Ms = np.arange(1, MM+5, 1).tolist()
        else:
            # 使用 max(1, MM-5) 確保起始值至少從 1 開始，避免出現 0
            self.Ms = np.arange(max(1, MM-5), 2*MM, 1).tolist()

    
    def reconstruct_raw(self, coeffs, M_val):
        a = coeffs[:M_val]
        b = coeffs[M_val:]
        I_ch = np.zeros(self.N)
        Q_ch = np.zeros(self.N)
        for k in range(1, M_val + 1):
            # 純 Sine 基底確保頭尾完美歸零
            I_ch += a[k-1] * np.sin(k * np.pi * self.t / self.gate_time)
            Q_ch += b[k-1] * np.sin(k * np.pi * self.t / self.gate_time)
        
        return I_ch, Q_ch
    

    def optimize(self, optimizer:Literal['L-BFGS-B','Nelder-Mead','Powell','CG','BFGS','Newton-CG','TNC','COBYLA','COBYQA','SLSQP','trust-constr','dogleg','trust-ncg','trust-exact','trust-krylov']='L-BFGS-B'):
        
        
        best_penalty = float('inf')

        for M in self.Ms:
            def basis_objective_log(coeffs):
                I_ch, Q_ch = self.reconstruct_raw(coeffs, M)
                complex_signal = I_ch + 1j * Q_ch
                
                N_padded = self.N * 50
                fft_y = np.fft.fftshift(np.fft.fft(complex_signal, n=N_padded))
                freqs = np.fft.fftshift(np.fft.fftfreq(N_padded, 1/(self.fs * 1000)))
                
                # 頻譜內部正規化
                psd_norm = np.abs(fft_y)**2 / np.max(np.abs(fft_y)**2)
                
                penalty_holes = 0.0
                for hole_f in self.target_holes:
                    idx = np.argmin(np.abs(freqs - hole_f))
                    db_val = 10 * np.log10(psd_norm[idx] + 1e-12)
                    if db_val > self.sup_power:
                        penalty_holes += (db_val - self.sup_power)**2 

               
                        
                return penalty_holes

            # 初始化係數
            initial_coeffs = np.zeros(2 * M)
            box_bounds = [(-1.0, 1.0)] * (2 * M)
            initial_coeffs[0] = 1.0  # I Channel 主頻成分
            initial_coeffs[M] = 0.2  # Q Channel 初始微調

            self.res = minimize(basis_objective_log, initial_coeffs, method=optimizer, bounds=box_bounds, options={'maxiter': 60000})
            
    
            # 紀錄最佳結果
            if self.res.fun < best_penalty:
                best_penalty = self.res.fun
                self.best_res = self.res
                self.best_M = M


        self.res = self.best_res


    def plot_result(self):
        plt.rcParams.update({
            'font.size': 14, 'axes.linewidth': 2, 'lines.linewidth': 2.5,
            'xtick.major.width': 2, 'ytick.major.width': 2,
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'xtick.major.size': 8, 'ytick.major.size': 8,
        })
        waveform_dict = self.output_waveform()
        opt_I, opt_Q = np.array(waveform_dict['waveform_I']), np.array(waveform_dict['waveform_Q'])

        plt.figure(figsize=(14, 8))

        # 時域圖
        plt.subplot(1, 2, 1)
        plt.plot(self.t, opt_I, 'b.-', label='I Channel')
        plt.plot(self.t, opt_Q, 'r.-', label='Q Channel')
        plt.title('Optimized Time Domain (Normalized)')
        plt.ylabel('Amplitude [a.u.]')
        plt.xlabel('Time (ns)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 頻域圖
        plt.subplot(1, 2, 2)
        N_p = self.N * 30
        freqs = np.fft.fftshift(np.fft.fftfreq(N_p, 1/(self.fs * 1000)))
        fft_complex = np.fft.fftshift(np.fft.fft(opt_I + 1j*opt_Q, n=N_p))
        db_complex = 10 * np.log10(np.abs(fft_complex)**2 / np.max(np.abs(fft_complex)**2))

        plt.plot(freqs, db_complex, 'purple', label='Optimized Spectrum')
        for i, hole_f in enumerate(self.target_holes):
            plt.axvline(hole_f, color='red', linestyle='--', alpha=0.5, label='Target Holes' if i == 0 else "")
        plt.axhline(self.sup_power, color='black', linestyle=':', label=f'{self.sup_power} dB Line')

        plt.ylim(-100, 5)
        plt.ylabel('Amplitude [dB]')
        plt.xlim(-300, 300)
        plt.legend()
        plt.title('Optimized Frequency Domain')
        plt.grid(True, alpha=0.3)

        plt.suptitle(f"L-BFGS-B Optimization Result\nTarget Avoid Frequencies: {', '.join(map(str, self.target_holes))} MHz\nGate Time: {int(self.gate_time)} ns | Optimal Sine Basis: M={self.best_M}")
        plt.tight_layout()
        plt.show()


    def output_waveform(self):
        """ Return dict with keyname 'waveform_I' and 'waveform_Q' and its value in list."""


        opt_I_raw, opt_Q_raw = self.reconstruct_raw(self.res.x, self.best_M)
        max_amp = np.max(np.abs(opt_I_raw + 1j * opt_Q_raw))
        opt_I = np.array(opt_I_raw / max_amp)
        opt_Q = np.array(opt_Q_raw / max_amp)

        return {"waveform_I":opt_I.tolist(), "waveform_Q":opt_Q.tolist()}



class WaveformEngineer:

    def __init__(self, gate_time_ns:int, freqs_to_avoid:list, aiming_sup_power_dB:int=-45):
        # target crosstalk suppression power in dB
        self.sup_power = aiming_sup_power_dB
        # gate time ns
        self.gate_time = gate_time_ns
        # avoid frequencies
        self.target_holes = freqs_to_avoid
        
        self.method_log = CTSmethod.__subclasses__()



    
    def cooking_serve(self, method:Literal['GRAPE', 'CRAB', 'dCRAB']='GRAPE', optimizer:Literal['L-BFGS-B','Nelder-Mead','Powell','CG','BFGS','Newton-CG','TNC','COBYLA','COBYQA','SLSQP','trust-constr','dogleg','trust-ncg','trust-exact','trust-krylov']='L-BFGS-B'):

        match method:
            case 'GRAPE':
                self.M = CTS_GRAPE(self.gate_time, self.target_holes, self.sup_power)
            case 'CRAB':
                self.M = CTS_CRAB(self.gate_time, self.target_holes, self.sup_power)
            case 'dCRAB':
                self.M = CTS_dCRAB(self.gate_time, self.target_holes, self.sup_power)
        

        self.M.optimize(optimizer)
        self.M.plot_result()


    def compare_all_methods(self, optimizer: str = 'L-BFGS-B'):
        """ Compare all the methods """
        engines = {}
        self.waveforms = {}
        # 1. 依序執行三種演算法的最佳化
        for cls in self.method_log:
            m = cls.__name__.replace('CTS_', '')
            print(f"\n{'='*20} Method: {m} {'='*20}")
            
            eng = cls(self.gate_time, self.target_holes, self.sup_power)
            
            eng.optimize(optimizer)
            engines[m] = eng

        # 2. 開始畫對比圖
        plt.rcParams.update({
            'font.size': 12, 'axes.linewidth': 2, 'lines.linewidth': 2,
            'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
            'xtick.direction': 'in', 'ytick.direction': 'in',
        })

        # 稍微加高底色畫布 (figsize y軸從 8 改到 8.5)，留空間給最下方的全域圖例
        fig = plt.figure(figsize=(16, 10))
        # 透過 gridspec_kw 留出下方 12% 的空間給全域圖例
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], wspace=0.25, hspace=0.35)

        cmap = plt.colormaps['tab10']

        ax_I = fig.add_subplot(gs[0, 0])     
        ax_Q = fig.add_subplot(gs[1, 0])     
        ax_freq = fig.add_subplot(gs[:, 1])  

        # 用來儲存主要線條的控制把手（Handles），方便後面建立全域圖例
        method_lines = []
        method_labels = []

        # 3. 遍歷所有執行完的方法，繪製數據
        for idx, (m_name, eng) in enumerate(engines.items()):
            current_color = cmap(idx % 10)

            wave = eng.output_waveform()
            I_arr, Q_arr = np.array(wave['waveform_I']), np.array(wave['waveform_Q'])
            self.waveforms[m_name] = {"I": wave['waveform_I'], "Q":wave['waveform_Q']}
            # --- 左上：I 通道 ---
            line_I, = ax_I.plot(eng.t, I_arr, color=current_color, linestyle='-',alpha=0.6)
            ax_I.scatter(eng.t, I_arr, color=current_color)
            # --- 左下：Q 通道 ---
            ax_Q.plot(eng.t, Q_arr, color=current_color, linestyle='-',alpha=0.6)
            ax_Q.scatter(eng.t, Q_arr, color=current_color)
            # 記錄第一次出現的線條與標籤（用來當全域圖例）
            method_lines.append(line_I)
            method_labels.append(m_name)

            # --- 右側：功率譜 ---
            N_p = eng.N * 50
            freqs_plot = np.fft.fftshift(np.fft.fftfreq(N_p, 1/(eng.fs * 1000)))
            fft_complex = np.fft.fftshift(np.fft.fft(I_arr + 1j * Q_arr, n=N_p))
            db_complex = 10 * np.log10(np.abs(fft_complex)**2 / np.max(np.abs(fft_complex)**2))
            ax_freq.plot(freqs_plot, db_complex, color=current_color, alpha=0.8)

        
        ax_I.set_title('★  Time Domain: I Channel', fontsize=13, fontweight='bold')
        ax_I.set_ylabel('Amplitude [a.u.]')
        # ax_I.set_ylim(-1.1, 1.1)
        ax_I.grid(True, alpha=0.3)

        ax_Q.set_title('★ Time Domain: Q Channel', fontsize=13, fontweight='bold')
        ax_Q.set_xlabel('Time (ns)')
        ax_Q.set_ylabel('Amplitude [a.u.]')
        # ax_Q.set_ylim(-1.1, 1.1)
        ax_Q.grid(True, alpha=0.3)

        # 5. 美化右側頻域圖
        for i, hole_f in enumerate(self.target_holes):
            line_hole = ax_freq.axvline(hole_f, color='red', linestyle='--', alpha=0.4)
        ax_freq.axhline(self.sup_power, color='black', linestyle=':', linewidth=2)
        
        ax_freq.set_title('★ Frequency Domain' , fontsize=15, fontweight='bold')
        ax_freq.set_xlabel('Frequency (MHz)')
        ax_freq.set_ylabel('Power [dB]')
        ax_freq.set_ylim(-90, 5)
        ax_freq.set_xlim(-300, 300)
        ax_freq.grid(True, alpha=0.3)
        

        # 6. 👑 【核心修改】：建立置底的全域橫向圖例 (Global Legend)
        # ncol=len(method_labels) 會讓所有方法橫向排成一列
        fig.legend(
            handles=method_lines, 
            labels=method_labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.02), 
            ncol=len(method_labels), 
            fontsize=13,
            frameon=True,
            facecolor='#f8f9fa', # 加上淡淡的灰底，看起來更有科技感
            edgecolor='#cfd8dc'
        )

        plt.suptitle(f"Optimal Qubit Control Benchmark\nTarget Avoid Freqs: {', '.join(map(str, self.target_holes))} MHz | Gate Time: {self.gate_time+1} ns\nAiming Suppression Power {self.sup_power} dB", y=0.98, fontsize=15, fontweight='bold')
        
        # 調整邊距，確保緊湊佈局時不會壓到最下方的全域圖例
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        plt.show()

if __name__ == '__main__':
    tg = 24
    aiming_sup_power_dB = -45
    # avoid_freqs = np.array([-90, 104, 97, -132, -116, -100, 122, 147, -153])
    # avoid_freqs =  np.array([-45, 104, 73, -132, -116, -100, 122, 147, -153])
    # avoid_freqs =  np.array([-40, 104, 73, -69, -116, -100, 122, 147, -183])
    # avoid_freqs = np.array([-180, 60, -120])
    avoid_freqs = np.array([50])
    method = 'GRAPE'
    optimizer = 'L-BFGS-B'


    WE = WaveformEngineer(tg, avoid_freqs, aiming_sup_power_dB)
    WE.cooking_serve(method, optimizer)


    WE.compare_all_methods(optimizer)
    print(WE.waveforms[method])