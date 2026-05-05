from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state_coupler
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from quam_libs.components.macros.coupler_RD_decoding import CouplerReadoutDecoder
from quam_libs.experiments.coupler_RD_related import ExpTemplate
from qm.jobs.running_qm_job import RunningQmJob
from typing import Literal, Optional, List
from scipy.ndimage import gaussian_filter


u = unit(coerce_to_integer=True)

class Parameters(NodeParameters):
    coupler: list[str] = ['coupler_q6_q7']
    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.1 #0.004, 0.02 # q6:3e-3, q7:1e-2, q8:3e-3, q9:***,
    operation_len_in_ns: Optional[int] = None
    Driving_LO_GHz: float|None = 3.9 # 3.18
    frequency_span_in_mhz: float = 300 #12, 120
    frequency_step_in_mhz: float = 3 #0.1, 1
    frequency_shift_in_mhz: float = 0 #0  
    min_flux_offset_in_v: float = 0.1 ##-0.042
    max_flux_offset_in_v: float = 0.45 #0.042
    num_flux_points: int = 75
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    qubits_detune_flux_amp: float = 0.35 # once you see the spectrum split at sweet spot due to q-c ZZ, you can try this to detune its all neighboring qubits. 0.3 is recommanded. 
    enforce_aswap_on_coupler:bool = False # Use it once you see the coupler's sweet spot freq is higher than your detector qubit. Make sure you have 'aSWAP' in your coupler.coupler.operations !!!
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    UPDATE_STATE:bool = True


node = QualibrationNode(name="03x_coupler_Spectroscopy_vs_Flux", parameters=Parameters())


class CP_FS_EXP(ExpTemplate):

    def __init__(self, node:QualibrationNode):
        super().__init__()
        self.node = node
        self._check_coupler_num_()
        self.machine = QuAM.load()
        

    # Checking all the required participants
    def participants_stand_by(self):
        self.CRD = CouplerReadoutDecoder(self.machine, self.node.parameters.coupler, coupler_arbi_LO_manual=self.node.parameters.Driving_LO_GHz, simulate=self.node.parameters.simulate, load_data_id=self.node.parameters.load_data_id)
        self.config = self.machine.generate_config()
        self.participant_num = len(list(self.CRD.paired_elements.keys()))
        if self.node.parameters.load_data_id is None:
            self.qmm = self.machine.connect()

    # Counting variables 
    def exp_variable_arangement(self):
        self.operation_len = self.node.parameters.operation_len_in_ns
        self.shift = int(self.node.parameters.frequency_shift_in_mhz * 1e6)
        if self.node.parameters.operation_amplitude_factor:
            self.operation_amp = self.node.parameters.operation_amplitude_factor
        else:
            self.operation_amp = 1.0

        dfs = np.arange(-self.node.parameters.frequency_span_in_mhz * 1e6 // 2, +self.node.parameters.frequency_span_in_mhz * 1e6 // 2, self.node.parameters.frequency_step_in_mhz*1e6, dtype=np.int32)
        self.variables['freq'] = dfs
        
        dcs = np.linspace(
            self.node.parameters.min_flux_offset_in_v,
            self.node.parameters.max_flux_offset_in_v,
            self.node.parameters.num_flux_points,
        )
        self.variables['flux'] = dcs
        

    # Composing QUA program
    def qua_composer(self):
        with program() as self.qua_prog:
            _, _, _, _, n, n_st = qua_declaration(num_qubits=self.participant_num)
            
            state = [declare(int) for _ in range(self.participant_num)]
            state_st = [declare_stream() for _ in range(self.participant_num)]
            df = declare(int)
            dc = declare(fixed)
            
            if not self.node.parameters.simulate:
                self.machine.apply_all_couplers_to_min()
            
            for i, c_name in enumerate(self.CRD.paired_elements):

                max_freq = self.variables['freq'][-1] + self.CRD.get_obj_with_type(c_name,'coupler').extras["RD"]["IF"]
                min_freq = self.variables['freq'][0] + self.CRD.get_obj_with_type(c_name,'coupler').extras["RD"]["IF"]

                assert max_freq <= 400e6 and min_freq >= -400e6, (
                    f"{c_name} IF span out of range: min={min_freq/1e6:.2f} MHz, "
                    f"max={max_freq/1e6:.2f} MHz (limit ±400 MHz), please adjust the frequency span.")

                # Bring the active qubits to the desired frequency point
                if not self.node.parameters.simulate:
                    for q_type in ["drive_q", "readout_q"]:
                        self.machine.set_all_fluxes(flux_point=self.node.parameters.flux_point_joint_or_independent, target=self.CRD.paired_elements[c_name][q_type])
                    
                    wait(400)


                with for_(n, 0, n < self.node.parameters.num_averages, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, self.variables["freq"])):
                        with for_(*from_array(dc, self.variables["flux"])):
                        
                            self.CRD.get_obj_with_type(c_name,"drive_q").xy.update_frequency(df + self.CRD.get_obj_with_type(c_name,"coupler").extras["RD"]["IF"])
                            wait(4)
                            align()
                            if not node.parameters.simulate:
                                duration = self.operation_len //4 if self.operation_len is not None else self.CRD.get_obj_with_type(c_name,'drive_q').xy.operations[self.node.parameters.operation].length//4
                            else:
                                duration = 100 //4


                            self.CRD.get_obj_with_type(c_name,'drive_q').z.play("const", amplitude_scale= self.node.parameters.qubits_detune_flux_amp /  self.CRD.get_obj_with_type(c_name,'drive_q').z.operations["const"].amplitude, duration=duration)
                            self.CRD.get_obj_with_type(c_name,'readout_q').z.play("const", amplitude_scale= self.node.parameters.qubits_detune_flux_amp / self.CRD.get_obj_with_type(c_name,'readout_q').z.operations["const"].amplitude, duration=duration)
                            self.CRD.get_obj_with_type(c_name,'coupler').coupler.play("const", amplitude_scale=dc / self.CRD.get_obj_with_type(c_name,'coupler').coupler.operations["const"].amplitude, duration=duration)
                    
                            
                            self.CRD.get_obj_with_type(c_name,'drive_q').xy.play(
                                self.node.parameters.operation,
                                amplitude_scale=self.operation_amp,
                                duration=duration,
                            )
                            align()

                            # Measure the state of the resonators
                            readout_state_coupler(self.CRD.paired_elements[c_name]["readout_q"], state[i], flux_applied_target=self.CRD.paired_elements[c_name]["aswap_supplier"], method=self.CRD.paired_elements[c_name]["readout_method"], active_reset_readout_q=True)
                            save(state[i], state_st[i])


            with stream_processing():
                n_st.save("n")
                for i, c_name in enumerate(self.CRD.paired_elements):
                    state_st[i].buffer(len(self.variables["flux"])).buffer(len(self.variables["freq"])).average().save(f"state{i + 1}")

    # execute or simulate
    def qua_executor(self):

        if self.node.parameters.simulate:
            # Simulates the QUA program for the specified duration
            simulation_config = SimulationConfig(duration=self.node.parameters.simulation_duration_ns // 4)  # In clock cycles = 4ns
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
        
        ds = fetch_results_as_xarray(job.result_handles, self.CRD.get_all_TransmonPairs(), self._sort_variables_(["flux", "freq"]))
        
        return ds
    

    # assign new coordinates
    def dataset_post_proccess(self):
        ''' Once the dataset had been saved into node.results '''
        ds:xr.Dataset = self.node.results['ds']
        
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([self.shift + self.variables['freq'] + c.extras["RD"]["IF"] + (self.node.parameters.Driving_LO_GHz*1e9 if self.node.parameters.Driving_LO_GHz is not None else c.extras["RD"]["LO"]) for c in self.CRD.get_all_TransmonPairs()]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
        self.node.results["ds"] = ds



    # analyze data
    def analyze(self, gaussian_filter_sigma:float=1.5, signal_threshold_percentage:float=95):
        """
        - peak_prominence_factor: the arg used in `quam_libs.lib.fit.peaks_dips()`
        """
        if not self.node.parameters.simulate:
            if self.node.parameters.load_data_id is not None:
                self.load_data = True
                from quam_libs.experiments.coupler_RD_related import load_data_only
                self.node, self.machine, _, self.CRD = load_data_only(self.node)
                
            ds:xr.Dataset = self.node.results['ds']
            self.node.results["fit_results"] = {}
            for q in self.CRD.get_all_TransmonPairs():
                q_name = q.name
                da = ds.state.sel(qubit=q_name)
                
                # --- 自適應 ROI 計算 ---
                flux_values = da.flux.values
                flux_span = np.ptp(flux_values) # 計算目前 Flux 的總掃描寬度
                adaptive_width = flux_span * 0.35 # 動態取總寬度的 70% 作為 ROI
                
                # 影像平滑以精確定位中心
                smoothed = gaussian_filter(da.values, sigma=gaussian_filter_sigma)
                max_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
                center_flux = flux_values[max_idx[1]]
                
                # 動態切片：不受固定數值限制
                roi_mask = (da.flux >= center_flux - adaptive_width) & (da.flux <= center_flux + adaptive_width)
                roi_da = da.where(roi_mask, drop=True)
                
                # --- 加權擬合邏輯 ---
                # 提高門檻，確保只抓取黃色條紋最亮的部分
                threshold = np.percentile(roi_da.values, signal_threshold_percentage)
                f_idx, x_idx = np.where(roi_da.values > threshold)
                
                fit_x = roi_da.flux.values[x_idx]
                fit_y = roi_da.freq_full.values[f_idx]
                weights = roi_da.values[f_idx, x_idx] # 使用數值強度作為權重

                if len(fit_x) >= 6:
                    # 加權二次擬合：y = ax^2 + bx + c
                    p = np.polyfit(fit_x, fit_y, 2, w=weights)
                    
                    # 物理限制：Coupler 頂點向上，a (二次項) 必須小於 0
                    if p[0] > 0: 
                        # 如果算出來開口向上，代表被雜訊干擾，嘗試排除邊緣點再算一次
                        p = np.polyfit(fit_x, fit_y, 2, w=weights**2)

                    f_shift = -p[1] / (2 * p[0])
                    d_freq = p[0] * f_shift**2 + p[1] * f_shift + p[2]
                    
                    self.node.results["fit_results"][q_name] = {
                        "flux_shift": float(f_shift),
                        "drive_freq": float(d_freq),
                        "quad_term": float(p[0]),
                        "coeff":[p[2], p[1], p[0]],
                        "center_detected": float(center_flux)
                    }
                else:
                    print(f"❌ {q_name} Fitting Failed !")
                    

    
    # plot
    def visualize(self):
        
        if not self.node.parameters.simulate:
            ds:xr.Dataset = self.node.results['ds']
            grid_names, qubit_pair_names = grid_pair_names(self.CRD.get_all_TransmonPairs())
            grid = QubitPairGrid(grid_names, qubit_pair_names)
            
            for ax, qubit in grid_iter(grid):
                self.analyzed_items[qubit['qubit']] = {}
                qubit_ds = ds.loc[qubit].state
                q_fit = node.results.get("fit_results", {}).get(qubit['qubit'])
                
                # 繪圖
                qubit_ds.assign_coords(freq_GHz=qubit_ds.freq_full / 1e9).plot(
                    ax=ax, 
                    add_colorbar=True, 
                    x="flux", 
                    y="freq_GHz", 
                    robust=True,
                    cmap="viridis"
                )
                
                # 疊加擬合曲線與點 (維持原樣)
                if "coeff" in q_fit:
                    self.analyzed_items[qubit['qubit']]["sweet_freq"] = q_fit["drive_freq"]    
                    self.analyzed_items[qubit['qubit']]["quad_term"] = q_fit["quad_term"]
                    self.analyzed_items[qubit['qubit']]["bias_to_sweet"] = q_fit["flux_shift"]
                    self.analyzed_items[qubit['qubit']]["neighboring_qubit_detune_flux_amp"] = self.node.parameters.qubits_detune_flux_amp

                    c = q_fit["coeff"]
                    # 直接使用絕對頻率計算
                    f_vals = (c[2]*ds.flux**2 + c[1]*ds.flux + c[0])
                    
                    # 繪圖時直接除以 1e9 對齊座標系
                    (f_vals / 1e9).plot(ax=ax, ls="--", color="r")
                    ax.axhline(q_fit["drive_freq"] / 1e9, color="cyan", ls=":")

                else:
                    print(f"Warning: No valid fit found for {qubit['qubit']}")
                
                # 標籤與格式
                ax.set_ylabel("Freq (GHz)")
                ax.set_xlabel("Flux (V)")
                ax.set_title(f"{qubit['qubit']}") # 明確標示當前 Qubit
                ax.grid(True, alpha=0.3)
            
            grid.fig.suptitle("Coupler spectroscopy VS Flux")
            plt.tight_layout()
            plt.show()
            self.node.results["figure"] = grid.fig
            


    
    # Save state and results
    def state_management(self, update_state_also:bool):
        if not self.node.parameters.simulate:
            if update_state_also:
                with self.node.record_state_updates():
                    for c_name in self.analyzed_items:
                        if "Fx" not in self.CRD.get_obj_with_type(c_name,"coupler").extras:
                            self.CRD.get_obj_with_type(c_name,"coupler").extras["Fx"] = self.analyzed_items[c_name]
                        else:
                            for item in self.analyzed_items[c_name]:
                                self.CRD.get_obj_with_type(c_name,"coupler").extras["Fx"][item] = self.analyzed_items[c_name][item]
                            

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

