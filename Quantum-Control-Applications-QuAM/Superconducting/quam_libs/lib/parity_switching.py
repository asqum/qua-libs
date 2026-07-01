import xarray as xr
import numpy as np
from xarray import DataArray
from lmfit import Model, Parameter
from lmfit.model import ModelResult
from numpy import ndarray, fft
from numpy import asarray, linspace, mean, argmax
from numpy import cos, abs, exp, max, min, pi, nan
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from quam_libs.lib.qubit_thermometer import FunctionFitting
from matplotlib.axes import Axes

class FitDampingBeat(FunctionFitting):
    """
    Fit a damped beat model to data:
    a * exp(-x/tau) * (cos(2*pi*f_1*x + phi_1) + cos(2*pi*f_2*x + phi_2)) + c
    """
    def __init__(self, data:DataArray=None):
        self._data_parser(data)
        self.model = Model(self.model_function)
        self.params = None

    def _data_parser(self, data:DataArray):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")
        self.y = data.values
        self.x = data.coords["x"].values

    def model_function(self, x, a_1, kappa_1, f_1, phi_1, a_2, kappa_2, f_2, phi_2, c):
        return  a_1 *exp(-x*kappa_1) *cos(2*pi*f_1*x + phi_1) + a_2*exp(-x*kappa_2)*cos(2*pi*f_2*x + phi_2) + c

    def guess(self):
        y = self.y
        t = self.x
        dt = float(t[1] - t[0])
        max_val = float(max(y))
        min_val = float(min(y))
        # FFT for frequency guesses
        amp = fft.fft(y)[: len(y) // 2]
        freq = fft.fftfreq(len(y), dt)[: len(amp)]
        amp[0] = 0  # Remove DC part
        power = abs(amp)
        peak_indices = asarray(power).argsort()[::-1]
        freq = asarray(freq)
        f_1_idx = peak_indices[0]
        # Find second peak index with sufficient separation
        f_2_idx = None
        for idx in peak_indices[1:5]:
            if abs(idx - f_1_idx) >= 3 and power[idx]/power[f_1_idx]>0.5:
                f_2_idx = idx
                break
        f_1_guess = float(abs(freq[f_1_idx]))
        a_1_guess = float(abs(amp[f_1_idx]))
        a_1_guess_dict = dict(value=a_1_guess, min=0.0, max=a_1_guess*2)
        f_1_guess_dict = dict(value=f_1_guess, min=0.0, max=1.0/dt/2)
        phi_1_guess_dict = dict(value=0.0, min=-float(pi), max=float(pi))
        # kappa_1 guess: use 1/(t[-1]/2) as typical decay rate
        kappa_1_guess = 1.0 / abs(t[-1]/2) if abs(t[-1]/2) > 0 else 1.0
        kappa_1_guess_dict = dict(value=kappa_1_guess, min=0, max=10*kappa_1_guess)
        # If second frequency is not resolvable, fit single frequency only
        if f_2_idx is None:
            a_2_guess_dict = dict(value=0, vary=False)
            f_2_guess_dict = dict(value=0, vary=False)
            phi_2_guess_dict = dict(value=0, min=-float(pi), max=float(pi), vary=False)
            kappa_2_guess_dict = dict(value=0, min=0, max=kappa_1_guess, vary=False)
        else:
            f_2_guess = float(abs(freq[f_2_idx]))
            a_2_guess = float(abs(amp[f_2_idx]))
            a_2_guess_dict = dict(value=a_2_guess, min=0.0, max=a_2_guess*2)
            f_2_guess_dict = dict(value=f_2_guess, min=0.0, max=1.0/dt/2)
            phi_2_guess_dict = dict(value=0.0, min=-float(pi), max=float(pi))
            # kappa_2 guess: same as kappa_1
            kappa_2_guess_dict = dict(value=kappa_1_guess, min=0, max=10*kappa_1_guess)
        c_guess_dict = dict(value=float(mean(y)), min=min_val, max=max_val)
        self.params = self.model.make_params(
            a_1=a_1_guess_dict,
            kappa_1=kappa_1_guess_dict,
            a_2=a_2_guess_dict,
            kappa_2=kappa_2_guess_dict,
            f_1=f_1_guess_dict,
            phi_1=phi_1_guess_dict,
            f_2=f_2_guess_dict,
            phi_2=phi_2_guess_dict,
            c=c_guess_dict
        )
        return self.params

    def fit(self, data:DataArray=None) -> ModelResult:
        if data is not None:
            self._data_parser(data)
        if self.params is None:
            self.guess()
        result = self.model.fit(self.y, self.params, x=self.x)
        self.result = result
        return result

def plot_results(rawdata:xr.Dataset, analysis_result:dict=None):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    ax.scatter(rawdata.coords['idle_time'].values, rawdata, label='Raw Data')
    if analysis_result is not None:
        # Show best_fit curve if available
        if 'best_fit' in analysis_result:
            ax.plot(rawdata.coords['idle_time'].values, analysis_result['best_fit'], '-', label='Fit')
        ax.legend()
        # Add textbox with fit parameters
        params = analysis_result
        k1 = params.get('kappa_1', float('nan'))
        tau1 = 1/k1 if k1 != 0 else float('nan')
        f1 = params.get('f_1', float('nan'))
        a_2 = params.get('a_2', float('nan'))

        if a_2 != 0:
            k2 = params.get('kappa_2', float('nan'))
            f2 = params.get('f_2', float('nan'))
            tau2 = 1/k2 if k2 != 0 else float('nan')

            textstr = (
                f"κ₁ = {k1:.4g} (τ₁={tau1:.4g})\n"
                f"a₁ = {params.get('a_1', float('nan')):.4g}\n"
                f"f₁ = {f1:.4g}\n"
                f"ϕ₁ = {params.get('phi_1', float('nan')):.4g}\n"
                f"κ₂ = {k2:.4g} (τ₂={tau2:.4g})\n"
                f"a₂ = {params.get('a_2', float('nan')):.4g}\n"
                f"f₂ = {f2:.4g}\n"
                f"ϕ₂ = {params.get('phi_2', float('nan')):.4g}\n"
                f"f+/-df = {(f1+f2)/2:.4g} +/- {abs(f1-f2)/2:.4g}\n"
            )
        else:
            textstr = (
                f"κ₁ = {k1:.4g} (τ₁={tau1:.4g})\n"
                f"a₁ = {params.get('a_1', float('nan')):.4g}\n"
                f"f₁ = {f1:.4g}\n"
                f"ϕ₁ = {params.get('phi_1', float('nan')):.4g}\n"
            )
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel('idle time', fontsize=20)
    ax.set_ylabel('state', fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    fig.tight_layout()

    return fig

def plot_fft(freq, amp):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(freq, amp, label='FFT (positive freq)')
    ax.set_xlabel('Frequency', fontsize=20)
    ax.set_ylabel('Amplitude', fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.legend()
    fig.tight_layout()
    return fig

class RamseyAnalysis:

    def __init__(self, data: xr.DataArray):
        self.data = data
        self.fit_result = None
        self.fitter = None
        self._fit()


    
    def _fit(self):
        fit_data = self.data["signal"].rename({"idle_time": "x"}).squeeze()
        self.fitter = FitDampingBeat(fit_data)
        self.fit_result = self.fitter.fit()

    def get_fft_data(self):
        idle_times = self.data["signal"].coords["idle_time"].values
        y = self.data["signal"].values
        n = len(idle_times)
        dt = idle_times[1] - idle_times[0] if n > 1 else 1.0
        amp = np.fft.fft(y)[:n // 2]
        freq = np.fft.fftfreq(n, dt)[:len(amp)]
        amp[0] = 0  # Remove DC part
        return freq, np.abs(amp)
    
    def get_fit_report(self):
        return self.fit_result.fit_report()
    
    def _plot_results(self):
        freq, amp = self.get_fft_data()
        freq = freq*1e6 #GHz to kHz
        # Convert fit_result.params to a simple dictionary
        analysis_result = {k: v.value for k, v in self.fit_result.params.items()} if self.fit_result is not None else None
        plot_info = analysis_result 
        if analysis_result is not None:
            plot_info['f_1'] = analysis_result['f_1']*1e6 #GHz to kHz
            plot_info['kappa_1'] = analysis_result['kappa_1']*1e3 #GHz to MHz
            plot_info['f_2'] = analysis_result['f_2']*1e6 #GHz to kHz
            plot_info['kappa_2'] = analysis_result['kappa_2']*1e3 #GHz to MHz
            
            plot_info['best_fit'] = self.fit_result.best_fit
        spec_fig = plot_fft(freq, amp)
        time_fig = plot_results(self.data["signal"], plot_info)
        return {"time_fig":time_fig, "spec_fig":spec_fig}
    
    def get_fit_data(self):
        return self.fitter.x, self.fitter.y, self.fit_result.best_fit

    def plot(self, ax: Axes|None, qubit_name: str = "", use_state_discriminator:bool=True):
        """
        在指定的 ax 上繪製 Ramsey 擬合結果。
        
        :param ax: matplotlib 的 Axes 物件
        :param qubit_name: 用於標題的 Qubit 名稱
        繪製圓滑曲線，根據比例顯示 a1, a2，並回傳 ax, tau, tau_err。
        """
        raw_data = self.data["signal"]
        x_coord = "idle_time" if "idle_time" in raw_data.coords else "x"
        idle_times = raw_data.coords[x_coord].values
        
        # 初始化回傳值
        res_tau, res_tau_err = 0.0, 0.0

        if self.fit_result is not None:
            res = self.fit_result
            p = res.params
            
            # 產生平滑曲線
            x_smooth = np.linspace(np.min(idle_times), np.max(idle_times), 1000)
            y_smooth = res.model.eval(p, x=x_smooth)

            # 提取參數與誤差
            a1, k1 = p['a_1'].value, p['kappa_1'].value
            a2, k2 = p['a_2'].value, p['kappa_2'].value
            k1_e = p['kappa_1'].stderr if p['kappa_1'].stderr is not None else 0.0
            k2_e = p['kappa_2'].stderr if p['kappa_2'].stderr is not None else 0.0

            # 計算 tau 與其誤差 (delta_tau = delta_k / k^2)
            tau1 = 1/k1 if k1 != 0 else 0.0
            tau1_e = (k1_e / (k1**2)) if k1 != 0 else 0.0
            tau2 = 1/k2 if k2 != 0 else 0.0
            tau2_e = (k2_e / (k2**2)) if k2 != 0 else 0.0

            # 判定邏輯
            max_a = float(np.max([a1, a2]))
            min_a = float(np.min([a1, a2]))
            show_both = (min_a / max_a > 0.2) if max_a > 1e-9 else False

            if show_both:
                res_tau = round(float(np.average([tau1, tau2])), 1)
                res_tau_err = round(float(np.average([tau1_e, tau2_e])), 2)
                textstr = (
                    f"a1={a1:.2f}, a2={a2:.2f}\n"
                    f"$T_{{2}}^{{*}}$ ={res_tau}$\pm${res_tau_err} µs"
                )
                
            else:
                if a1 >= a2:
                    idx, amp, tau, tau_e = "1", a1, tau1, tau1_e
                else:
                    idx, amp, tau, tau_e = "2", a2, tau2, tau2_e
                
                res_tau_err = round(tau_e, 2)

                textstr = (
                    f"a{idx} = {amp:.2f}\n"
                    f"$T_{{2}}^{{*}}$ = {tau:.1f}$\pm${res_tau_err:.2f} µs"
                )
                res_tau = round(tau, 2)
                
            if ax is not None:
                # 繪圖
                ax.scatter(idle_times, raw_data.values, label='Raw Data', alpha=0.4, s=12)
                ax.plot(x_smooth, y_smooth, '-', color='red', label='Fit', lw=1.5)
                
                ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            if ax is not None:
                ax.scatter(idle_times, raw_data.values, label='Raw Data')
        if ax is not None:
            ax.set_title(f"{qubit_name}")
            ax.set_xlabel('Idle Time (µs)')
            if use_state_discriminator:
                ax.set_ylabel('State')
            else:
                ax.set_ylabel('I')
            ax.grid(True, linestyle=':', alpha=0.5)
        
        return ax, res_tau, res_tau_err