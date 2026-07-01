from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state_coupler, active_reset_coupler, active_reset, readout_state

import matplotlib.pyplot as plt
import numpy as np

import xarray as xr
from scipy.stats import norm

from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from quam_libs.components.macros.coupler_RD_decoding import CouplerReadoutDecoder
from quam_libs.experiments.coupler_RD_related import ExpTemplate
from qm.jobs.running_qm_job import RunningQmJob
from typing import Literal, Optional
u = unit(coerce_to_integer=True)

class Parameters(NodeParameters):
    coupler: list[str] = ['coupler_q7_q8']
    num_averages: int = 500
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 6008
    wait_time_step_in_ns: int = 120
    frequency_detuning_in_mhz: float = 2.0
    reset_type: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent_or_arbitrary: Literal['joint', 'independent'] = 'independent'   
    load_data_id: Optional[int] = None
    simulate: bool = False
    timeout: int = 100
    histo_num:int = 1
    UPDATE_STATE: bool = True

node = QualibrationNode( name="10x_CtoQ_JaZZ", parameters=Parameters())

class CP_c2qJAZZ_EXP(ExpTemplate):

    def __init__(self, node:QualibrationNode):
        super().__init__()
        self.node = node
        self._check_coupler_num_()
        self.machine = QuAM.load()


    # Checking all the required participants
    def participants_stand_by(self):
        self.CRD = CouplerReadoutDecoder(self.machine, self.node.parameters.coupler, coupler_arbi_LO_manual=None, simulate=self.node.parameters.simulate, load_data_id=self.node.parameters.load_data_id)
        self.config = self.machine.generate_config()
        self.participant_num = len(list(self.CRD.paired_elements.keys()))
        if self.node.parameters.load_data_id is None:
            self.qmm = self.machine.connect()

    # Counting variables 
    def exp_variable_arangement(self):
        # Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
        idle_times = np.arange(
            self.node.parameters.min_wait_time_in_ns // 4,
            self.node.parameters.max_wait_time_in_ns // 4,
            self.node.parameters.wait_time_step_in_ns // 4,
        )
        self.detuning = int(1e6 * self.node.parameters.frequency_detuning_in_mhz)
        self.variables['idle_time'] = idle_times

        self.variables['flux_coupler'] = np.arange(node.parameters.histo_num)


    # Composing QUA program
    def qua_composer(self):
        with program() as self.qua_prog:
            _, _, _, _, n, n_st = qua_declaration(num_qubits=self.participant_num)
            t = declare(int)  # QUA variable for the idle time
            flux_coupler = declare(int)
            t_half = declare(int)
            phi = declare(fixed)
            current_state = [declare(int) for _ in range(self.participant_num)]
            init_state = [declare(int) for _ in range(self.participant_num)]
            state_target = [declare(int) for _ in range(self.participant_num)]
            state = [declare(int) for _ in range(self.participant_num)]
            state_st = [declare_stream() for _ in range(self.participant_num)]
            
            if not self.node.parameters.simulate:
                self.machine.apply_all_couplers_to_min()
            
            for i, c_name in enumerate(self.CRD.paired_elements):

                # Bring the active qubits to the desired frequency point
                if not self.node.parameters.simulate:
                    for q_type in ["drive_q", "readout_q"]:
                        self.machine.set_all_fluxes(flux_point=self.node.parameters.flux_point_joint_or_independent_or_arbitrary, target=self.CRD.paired_elements[c_name][q_type])
                    
                    wait(400)


                with for_(n, 0, n < self.node.parameters.num_averages, n + 1):
                    save(n, n_st)
                    with for_(*from_array(flux_coupler, self.variables['flux_coupler'])):
                        with for_(*from_array(t, self.variables['idle_time'])):
                            self.CRD.get_obj_with_type(c_name,"drive_q").xy.update_frequency(self.CRD.get_obj_with_type(c_name,"drive_q").xy.intermediate_frequency)
                            wait(4)
                            assign(phi, Cast.mul_fixed_by_int(self.detuning * 1e-9, 4 * t))
                    
                            assign(t_half, t/2)
                            align()

                            if self.node.parameters.reset_type == "active":
                                active_reset(self.CRD.paired_elements[c_name]["readout_q"])
                                active_reset_coupler(self.CRD.paired_elements[c_name]["drive_q"], self.CRD.paired_elements[c_name]["readout_q"], f"x180_{c_name}", flux_applied_target=self.CRD.paired_elements[c_name]["aswap_supplier"], method='aswap')
                                active_reset(self.CRD.paired_elements[c_name]["drive_q"])

                            else:
                                
                                if not self.node.parameters.simulate:
                                    if max(self.CRD.get_obj_with_type(c_name,"drive_q").thermalization_time//5, self.CRD.get_obj_with_type(c_name,"readout_q").thermalization_time//5) < self.CRD.get_obj_with_type(c_name,"coupler").extras['T1']*1e9:
                                        wait(10*int(self.CRD.get_obj_with_type(c_name,"coupler").extras['T1']*1e9)//4)
                                    else:
                                        wait(max(self.CRD.get_obj_with_type(c_name,"drive_q").thermalization_time, self.CRD.get_obj_with_type(c_name,"readout_q").thermalization_time)//4)

                            align()
                            self.CRD.get_obj_with_type(c_name,"drive_q").xy.update_frequency(self.CRD.get_obj_with_type(c_name,"coupler").extras["RD"]["IF"])
                            wait(4)

                            align()
                            self.CRD.get_obj_with_type(c_name,"readout_q").xy.play(f"x90")
                    
                            wait(t_half)

                            align()
                            self.CRD.get_obj_with_type(c_name,"drive_q").xy.play(f"x180_{c_name}")
                            self.CRD.get_obj_with_type(c_name,"readout_q").xy.play(f"x180")
                            align()

                            wait(t_half)
                            self.CRD.get_obj_with_type(c_name,"readout_q").xy.frame_rotation_2pi(phi)
                            align()

                            self.CRD.get_obj_with_type(c_name,"readout_q").xy.play(f"x90")
                            align()

                            # Measure the state of the resonators
                            readout_state(self.CRD.get_obj_with_type(c_name,"readout_q"), current_state[i])
                            assign(state_target[i], init_state[i] ^ current_state[i])
                            save(state_target[i], state_st[i])
                            reset_frame(self.CRD.get_obj_with_type(c_name,"readout_q").xy.name)
                    


            with stream_processing():
                n_st.save("n")
                for i, c_name in enumerate(self.CRD.paired_elements):
                    state_st[i].buffer(len(self.variables['idle_time'])).buffer(len(self.variables['flux_coupler'])).average().save(f"state_target{i + 1}")

    # execute or simulate
    def qua_executor(self):

        if self.node.parameters.simulate:
            # Simulates the QUA program for the specified duration
            simulation_config = SimulationConfig(duration=self.node.parameters.simulation_duration_ns//4)  # In clock cycles = 4ns
            job = self.qmm.simulate(self.config, self.qua_prog, simulation_config)
            # Get the simulated samples and plot them for all controllers
            samples = job.get_simulated_samples()
            samples.con1.plot()
            self.node.results = {"figure": plt.gcf()}
            wf_report = job.get_simulated_waveform_report()
            wf_report.create_plot(samples, plot=True, save_path=None)
            self.node.save()

            return None

        else:
            with qm_session(self.qmm, self.config, timeout=self.node.parameters.timeout) as self.qm:
                job = self.qm.execute(self.qua_prog)
                results = fetching_tool(job, ["n"], mode="live")
                while results.is_processing():
                    # Fetch results
                    n = results.fetch_all()[0]
                    # Progress bar
                    if self.progress_bar_display:
                        progress_counter(n, self.node.parameters.num_averages, start_time=results.start_time)
                    
            ds = self.data_catcher(job)
            return ds


    # fetch data
    def data_catcher(self, job:RunningQmJob) -> xr.Dataset:
        
        ds = fetch_results_as_xarray(job.result_handles, self.CRD.get_all_TransmonPairs(), self._sort_variables_(['idle_time','flux_coupler']))
        
        return ds
    

    # assign new coordinates
    def dataset_post_proccess(self):
        ''' Once the dataset had been saved into node.results '''

        ds:xr.Dataset = self.node.results["ds"]
        
        flux_coupler_full = np.array([
            self.variables['flux_coupler'] for qp in self.CRD.get_all_TransmonPairs()
        ])
        ds = ds.assign_coords({
            "flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)
        })
        ds = ds.assign_coords(idle_time = 4 * self.variables['idle_time'] / 1e3)
        self.node.results["ds"] = ds


    # analyze data
    def analyze(self):
        from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
        if not self.node.parameters.simulate:
            if self.node.parameters.load_data_id is not None:
                self.load_data = True
                from quam_libs.experiments.coupler_RD_related import load_data_only
                self.node, self.machine, _, self.CRD = load_data_only(self.node)

            ds:xr.Dataset = self.node.results['ds']
            self.node.results["fit_results"] = {}   # optional if you also store raw fits
            self.node.results["analysis_success"] = {}
            try:
                for i, qp in enumerate(self.CRD.get_all_TransmonPairs()):
                    print(f"Analyzing qubit pair {i}: {qp.id if hasattr(qp, 'id') else qp}")

                    # ---------------- FIT ----------------
                    fit_data = fit_oscillation_decay_exp(ds.state_target, "idle_time")

                    # Save fit data for reference

                    # Determine if fit succeeded (e.g. presence of valid freq)
                    fit_ok = (
                        np.isfinite(fit_data.sel(fit_vals="f")).any()
                        and not np.isnan(fit_data.sel(fit_vals="f")).all()
                    )

                    # ---------------- χZZ COMPUTATION ----------------
                    fitted = oscillation_decay_exp(
                        ds.state_target.idle_time,
                        fit_data.sel(fit_vals="a"),
                        fit_data.sel(fit_vals="f"),
                        fit_data.sel(fit_vals="phi"),
                        fit_data.sel(fit_vals="offset"),
                        fit_data.sel(fit_vals="decay"),
                    )
                    ds["state_target_fit"] = fitted

                    chiZZ = 2 * (fit_data.sel(fit_vals="f") * 1e3 - self.node.parameters.frequency_detuning_in_mhz * 1e3)  # MHz → kHz
                    ds["chiZZ"] = chiZZ

                    chiZZ_std = 1e3 * np.sqrt(fit_data.sel(fit_vals="f_f"))
                    ds["chiZZ_std"] = chiZZ_std

                    chiZZ_std_filt = chiZZ_std.where(chiZZ_std < chiZZ_std.median() * 2)
                    chiZZ_filt = chiZZ.where(~np.isnan(chiZZ_std_filt))

                    max_idx = abs(chiZZ_filt).argmax(dim="flux_coupler").item()
                    min_idx = abs(chiZZ_filt).argmin(dim="flux_coupler").item()

                    flux_full = ds["flux_coupler_full"].isel(qubit=i)
                    flux_val_max = float(flux_full.isel(flux_coupler=max_idx))
                    flux_val_min = float(flux_full.isel(flux_coupler=min_idx))
                    chi_max = float(chiZZ.isel(flux_coupler=max_idx))
                    chi_min = float(chiZZ.isel(flux_coupler=min_idx))

                    # ---------------- SAVE PER-PAIR RESULTS ----------------
                    self.node.results["fit_results"][qp.name] = {
                        "chiZZ_max_flux": flux_val_max,
                        "chiZZ_min_flux": flux_val_min,
                        "chiZZ_max_value": chi_max,
                        "chiZZ_min_value": chi_min,
                    }

                    # ---------------- SAVE FIT SUCCESS FLAG ----------------
                    self.node.results["analysis_success"][qp.name] = "successful" if fit_ok else "failed"

                    print(f"  → max |χZZ|={chi_max:.2f} kHz @ {flux_val_max:.3f}")
                    print(f"  → χZZ≈0={chi_min:.2f} kHz @ {flux_val_min:.3f}")
                    print(f"  → fit status: {'success' if fit_ok else '❌ failed'}")

                print("Analysis complete ")
                self.node.results['ds'] = ds

            except Exception as e:
                node.results["analysis_error"] = str(e)
                print(f"Analysis failed : {e}")

    
    # plot
    def visualize(self, histogram_outlier_percentage:float=1):
        '''
        - histogram_outlier_percentage: A float in the range [0, 100).
        '''
        ds:xr.Dataset = self.node.results['ds']
        for i, qp in enumerate(self.CRD.get_all_TransmonPairs()):
            qubit_name = qp.name
            fit_status = self.node.results.get("analysis_success", {}).get(qubit_name, "failed")

            # =====================================================
            # --- Always plot raw Ramsey / state_target heatmaps ---
            # =====================================================
            print(f"Plotting raw data for {qubit_name}")
            ds_pair = ds.isel(qubit=i)
            flux_full = ds_pair["flux_coupler_full"]
            flux_mV = flux_full.squeeze()
            idle_time_us = ds_pair["idle_time"]

            fig, ax = plt.subplots(figsize=(6, 4))
            data = ds_pair["state_target"]
            im = ax.pcolormesh(
                flux_mV,
                idle_time_us,
                data.transpose(),
                shading="auto",
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            fig.colorbar(im, ax=ax, label="State probability")
            ax.set_xlabel("Index")
            ax.set_ylabel("Idle time [µs]")
            ax.set_title(f"{qubit_name} – Raw Ramsey", fontsize=10)
            plt.tight_layout()
            self.node.results[f"figure_raw_{qubit_name}"] = fig
            plt.show()

            if fit_status == "successful" and "chiZZ" in ds:
                print(f"Plotting χZZ analysis for {qubit_name}")
                ds_pair = ds.isel(qubit=i)
                chi = ds_pair["chiZZ"]
                chi_std = ds_pair["chiZZ_std"]
                flux_full = ds_pair["flux_coupler_full"]
                flux_mV = flux_full.squeeze()

                res = self.node.results["fit_results"][qubit_name]
                flux_val_max = res["chiZZ_max_flux"]
                flux_val_min = res["chiZZ_min_flux"]
                chi_max = res["chiZZ_max_value"]
                chi_min = res["chiZZ_min_value"]

                # --- χZZ vs flux plot ---
                # --- χZZ vs flux: 2 subplots (linear + log-log) ---
                fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False)

                # Common data
                chi_val = chi.squeeze()
                chi_std_val = chi_std.squeeze()
                chi_abs = np.abs(chi_val)

                # ---------------- (1) Linear plot ----------------
                ax = axes[0]
                ax.errorbar(
                    flux_mV,
                    chi_val,
                    yerr=chi_std_val,
                    fmt="-o",
                    color="tab:red",
                    lw=1.8,
                    elinewidth=1,
                    capsize=3,
                    markersize=4,
                    alpha=0.9,
                    label="χZZ ± fit std",
                )
                ax.axvline(flux_val_max, color="k", linestyle="--", lw=1.2, alpha=0.8, label="max |χZZ|")
                ax.axvline(flux_val_min, color="gray", linestyle="--", lw=1.2, alpha=0.6, label="min |χZZ|")
                # ax.axhline(0, color="k", lw=1, alpha=0.4)
                ax.set_title("Linear scale")
                ax.set_xlabel("Index")
                ax.set_ylabel("χZZ [kHz]")
                ax.grid(True)
                ax.legend(fontsize=8)

                # ---------------- (2) Log-log plot ----------------
                ax = axes[1]   # axes[1]
                ax.errorbar(
                    flux_mV,     # must be positive for log
                    chi_abs,
                    yerr=chi_std_val,
                    fmt="-o",
                    color="tab:blue",
                    lw=1.8,
                    elinewidth=1,
                    capsize=3,
                    markersize=4,
                    alpha=0.9,
                    label="|χZZ| ± std",
                )
                ax.set_yscale("log")
                ax.set_title("Log-log scale")
                ax.set_xlabel("Index")
                ax.set_ylabel("|χZZ| [kHz]")
                ax.grid(True, which="both", ls="--", alpha=0.5)
                ax.legend(fontsize=8)

                fig.suptitle(f"{qubit_name}: χZZ vs Coupler Flux", fontsize=12)

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                self.node.results[f"figure_zz_curve_{qubit_name}"] = fig
                plt.show()


                # --- Time-domain fits at χZZ max and min ---
                for label, flux_val, chi_val in [
                    ("max", flux_val_max, chi_max),
                    ("min", flux_val_min, chi_min),
                ]:
                    flux_idx = int((abs(flux_full - flux_val)).argmin(dim="flux_coupler").item())
                    meas = ds_pair["state_target"].isel(flux_coupler=flux_idx)
                    fit = ds_pair["state_target_fit"].isel(flux_coupler=flux_idx)
                    idle_time = ds_pair["idle_time"]

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.scatter(
                        idle_time,
                        meas,
                        alpha=0.6,
                        label='meas'
                    )
                    ax.plot(
                        idle_time,
                        fit,
                        "-",
                        lw=2,
                        color='red',
                        label='fit'
                    )
                    
                    ax.legend()
                    ax.set_xlabel("Idle time [µs]")
                    ax.set_ylabel("State")
                    fig.suptitle(
                        f"{qubit_name} ({label} |χZZ|): Index={int(flux_val)}, χZZ={chi_val:.2f} kHz",
                        fontsize=12,
                    )
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    self.node.results[f"figure_fit_{label}_chiZZ_{qubit_name}"] = fig
                    plt.show()

                # --- 繪製 χZZ 直方圖 (Histogram) ---
                # 1. 建立新的圖表視窗
                fig_hist, ax_h = plt.subplots(figsize=(6, 4))

                # 2. 準備資料（排除 NaN 值以確保繪圖正常）
                chi_raw = ds.isel(qubit=i)["chiZZ"].squeeze().values.flatten()
                chi_raw = chi_raw[~np.isnan(chi_raw)]

                lower_bound = np.percentile(chi_raw, histogram_outlier_percentage)   # 下界
                upper_bound = np.percentile(chi_raw, 100-histogram_outlier_percentage)  # 上界
                chi_data = chi_raw[(chi_raw >= lower_bound) & (chi_raw <= upper_bound)]

                # 2. 繪製直方圖 (維持 Counts，不使用 density=True)
                
                n, bins, patches = ax_h.hist(
                    chi_data, 
                    bins='auto', 
                    color="#B19CD9", # 淡淡的紫色
                    edgecolor="#00F8D3", 
                    alpha=0.99, 
                    label="χZZ distribution"
                )

                # 3. 計算統計參數
                mu, std = norm.fit(chi_data)

                # 4. 生成高斯曲線並「縮放」以匹配次數
                # xmin, xmax = ax_h.get_xlim()
                xmin, xmax = chi_data.min(), chi_data.max()
                x = np.linspace(xmin, xmax, 10*chi_data.shape[0])
                p = norm.pdf(x, mu, std)

                # 重點：將機率密度縮放成次數
                # 縮放因子 = 資料總筆數 * 組距寬度
                bin_width = bins[1] - bins[0]
                p_scaled = p * len(chi_data) * bin_width

                # 5. 繪製縮放後的高斯曲線
                ax_h.plot(x, p_scaled, 'r', linewidth=2, label=f'$\mu$={mu:.1f}, $\sigma$={std:.1f}')

                # 6. 加入平均值虛線
                ax_h.axvline(mu, color="red", linestyle="dashed", linewidth=1.5)

                # 7. 圖表美化
                ax_h.set_title(f"#={self.node.parameters.histo_num}", fontsize=12)
                ax_h.set_xlabel("χZZ [kHz]")
                ax_h.set_ylabel("Counts") # 回到次數單位
                ax_h.grid(axis='y', alpha=0.3)
                ax_h.legend()
                plt.suptitle(f"{qubit_name} to {self.CRD.get_obj_with_type(qubit_name,'readout_q').name} χZZ Histogram")
                plt.tight_layout()
                self.node.results[f"figure_zz_histogram_{qubit_name}"] = fig_hist
                
                plt.show()
        

            else:
                print(f"No χZZ analysis or failed fit for {qubit_name} — only raw data plotted.")

    
    # Save state and results
    def state_management(self, update_state_also:bool=False):
        if not self.node.parameters.simulate:
            if not self.load_data:
                for q in self.CRD.get_all_Transmons('drive_q'):
                    q.xy.opx_output.upconverter_frequency = self.CRD.original_lo[q.name] # revert the driving LO
                for c_name in self.CRD.paired_elements:
                    if self.CRD.paired_elements[c_name]["aswap_supplier"] is None:
                        self.CRD.get_obj_with_type(c_name, 'readout_q').z.operations['aSWAP'].slope_direction = -1 
                    else:
                        self.CRD.get_obj_with_type(c_name, 'coupler').coupler.operations['aSWAP'].slope_direction = -1
            self.node.results["initial_parameters"] = self.node.parameters.model_dump()
            self.node.machine = self.machine
            self.node.save()


