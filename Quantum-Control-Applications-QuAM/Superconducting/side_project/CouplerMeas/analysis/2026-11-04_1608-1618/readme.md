# Coupler Flux Spectroscopy：2026-11-04，#1608–#1618

本目錄整理 `as-qpu-10qV2` 的 `coupler_q4_q5` 分段頻譜，包含原始整合、Flux crosstalk 修正、連續曲線拼接，以及加入雙躍遷頻率約束的物理模型擬合。

## 主要結果與圖檔

優先查看下列最新版；圖的右側列出峰值擷取、Flux 修正、局部拼接、完整 spline、物理模型及躍遷約束的函數與參數。

- [完整圖：含所有 fitting functions（PNG）](transmon_model_anharmonicity_constrained/coupler_flux_spectroscopy_all_fitting_functions.png)
- [完整圖：含所有 fitting functions（PDF）](transmon_model_anharmonicity_constrained/coupler_flux_spectroscopy_all_fitting_functions.pdf)
- [物理模型與殘差檢查](transmon_model_anharmonicity_constrained/transmon_model_diagnostics.png)
- [完整擬合結果與函數資料（JSON）](transmon_model_anharmonicity_constrained/constrained_transmon_fit_results.json)

| 項目 | 結果 | 解讀 |
| --- | --- | --- |
| 原始量測 | 11 筆，56,100 個訊號點 | 每筆 100 個頻率 × 51 個 Flux 點 |
| 全段曲線使用的共振峰 | 458 點 | 原始量測訊號均保留；曲線另排除未辨識、截斷與偏離主支線的峰 |
| #1615 橫軸修正 | −16.21188 mV | 僅平移該段 Flux 座標，保留量測頻率 |
| 合計等效 voltage crosstalk | −5.40396% | q4、q5 各加 +0.30 V 時的合計係數 |
| 經驗 sweet spot | −0.25294569 V，3.96177352 GHz | #1615 的二次擬合頂點，使用修正後座標 |
| 提供的參考點 anharmonicity | −176 MHz | 由同一參考工作點的兩個躍遷計算 |
| 完整能階模型的 $E_C/h$ | 約 152.26 MHz | 雙躍遷約束下的數值解 |
| 全頻譜模型的 $E_{J\Sigma}/h$ | 約 13.94 GHz | 單顆 isolated-transmon 模型下的條件估計 |
| Junction asymmetry $d$ | 落在 0 的下界 | 尚未可靠辨識，不能據此確認兩接面完全對稱 |
| 受約束物理模型 RMSE | 約 4.25 MHz | 仍存在系統性殘差 |

## 資料來源與掃描範圍

原始資料目錄：

```text
/home/ratiswu/Qualibrate_data/as-qpu-10qV2/2026-11-04
```

量測入口為 [03xb_coupler_FluxSpectroscopy.py](../../03xb_coupler_FluxSpectroscopy.py)。每筆資料夾包含 `ds.h5`、`node.json`、`data.json`、`figure.png` 與量測時儲存的 `quam_state/`。實際資料夾路徑與時間列於 [scan_summary.csv](scan_summary.csv)。

| Run | 原始 Flux 範圍（V） | 實際頻率點範圍（GHz） | Averages | Drive amplitude factor | 每條 qubit 的額外 detuning（V） |
| --- | --- | --- | ---: | ---: | ---: |
| #1608 | −0.010 ～ +0.010 | 2.710584 ～ 2.809584 | 350 | 0.05 | 0 |
| #1609 | −0.015 ～ −0.005 | 2.810584 ～ 2.909584 | 350 | 0.05 | 0 |
| #1610 | −0.025 ～ −0.015 | 2.910584 ～ 3.207584 | 350 | 0.05 | 0 |
| #1611 | −0.055 ～ −0.025 | 2.910584 ～ 3.207584 | 350 | 0.05 | 0 |
| #1612 | −0.075 ～ −0.055 | 3.110584 ～ 3.407584 | 350 | 0.05 | 0 |
| #1613 | −0.105 ～ −0.075 | 3.310584 ～ 3.607584 | 350 | 0.05 | 0 |
| #1614 | −0.145 ～ −0.105 | 3.510584 ～ 3.807584 | 350 | 0.10 | 0 |
| #1615 | −0.315 ～ −0.145 | 3.710584 ～ 4.007584 | 1400 | 0.18 | +0.30 |
| #1616 | 0 ～ +0.020 | 2.510584 ～ 2.807584 | 350 | 0.10 | 0 |
| #1617 | +0.020 ～ +0.040 | 2.310584 ～ 2.607584 | 350 | 0.10 | 0 |
| #1618 | +0.040 ～ +0.070 | 2.010584 ～ 2.307584 | 350 | 0.10 | 0 |

原始掃描合計涵蓋 Flux −0.315 ～ +0.070 V、頻率約 2.010584 ～ 4.007584 GHz；這是所有掃描視窗的外框，各視窗之外仍有未量測區域。

橫軸是 coupler flux pulse amplitude 的電壓座標。修正後以 coupler-equivalent voltage 表示；物理模型再透過 $V_{\Phi_0}$ 將電壓換成規一化磁通。各段以原生取樣間距繪製，共用全體訊號 2–98 百分位的色階；重疊區由較晚的 run 顯示於上層。

讀取原始檔時須注意兩個實際格式細節：`ds.h5` 的內容是 NetCDF3，可使用 `xarray` 的 `engine="scipy"`；`freq_full` 的數值單位是 Hz，原存的 `units="GHz"` 標籤與數值不一致，繪圖時需除以 $10^9$。頻率座標已與各 run 儲存的 LO、IF 核對。

## Flux crosstalk 修正與連續拼接

#1615 對 drive qubit q5 與 readout qubit q4 **各施加 +0.30 V**。依照這個變化造成水平 Flux 位移的假設，以 #1614、#1615 的 98 個共振峰共同擬合局部曲線，估計得到：

$$
V_c=V_{\mathrm{raw}}+\Delta V\,I_{1615},\qquad
\Delta V=-0.01621188060019709\ \mathrm{V},
$$

其中 $I_{1615}$ 在 #1615 為 1，其他 run 為 0。原始頻率、原始訊號與原始 Flux 座標均保留於資料中，另加入修正座標。

等效係數的定義為：

$$
V_{\mathrm{eff}}=V_{\mathrm{coupler}}+\kappa_{q4}V_{q4}+\kappa_{q5}V_{q5},\qquad
\kappa_{q4}+\kappa_{q5}=\frac{\Delta V}{0.30\ \mathrm{V}}=-0.054039602.
$$

因此合計 crosstalk 的大小約 **5.4%**，帶符號為 **−5.4%**。兩條線同時各加相同電壓，這組量測只能決定兩係數的和；個別係數需要分別激發兩條線才能區分。此處係數是以 coupler 電壓等效位移定義。

局部四次式條件下，500 次分層 bootstrap 的 95% 區間為位移約 −16.7172 ～ −15.8102 mV、合計係數約 −5.5724% ～ −5.2701%。改用不同局部階數與範圍，係數大小約為 5.0831% ～ 5.8531%；後者是模型敏感度範圍，與 bootstrap 信賴區間不同。

修正後 #1615 的右端為 −0.16121188 V，#1614 的左端為 −0.145 V，中間約 16.21 mV 的區域仍未量測。背景保留白色，連續擬合線跨越此區；跨越接縫的曲線屬插值。詳見 [修正前後比較](flux_crosstalk_corrected/flux_stitch_diagnostics.png) 與 [拼接參數](flux_crosstalk_corrected/flux_stitch_results.json)。

## Fitting functions

### 共振峰擷取

每個 Flux 點的頻率截面，以正 Lorentzian 加線性背景擬合：

$$
P(f)=b_0+b_1(f-f_s)+\frac{A}{1+[(f-f_r)/\gamma]^2}.
$$

$f_r(V)$ 是後續曲線使用的共振中心，FWHM 為 $2\gamma$。完整 561 個截面的狀態與峰值結果列於 [registered_peak_fits.csv](flux_crosstalk_corrected/registered_peak_fits.csv)；458 個主峰用於全段曲線。

### 局部拼接與全段經驗曲線

#1614／#1615 的共同局部四次式為：

$$
\frac{f_{\mathrm{join}}}{\mathrm{GHz}}=3.961871679826394-0.419400078801085z^2+0.021771851859480z^4,
\qquad z=\frac{V_c+0.253018787249107\ \mathrm{V}}{0.15\ \mathrm{V}}.
$$

局部擬合 RMSE 約 1.12 MHz。此式用於估計接縫附近的位移。

全段橘色虛線使用三次平滑 B-spline：

$$
f_{\mathrm{spline}}(V_c)=\sum_j c_j B_{j,3}(V_c;t).
$$

其 458 個主峰的 RMSE 約 1.51 MHz，支援的主峰座標範圍為 −0.33121188 ～ +0.0556 V。[empirical_fitting_functions.json](transmon_model/empirical_fitting_functions.json) 儲存完整 knot vector、係數與局部函數，可直接用 `scipy.interpolate.BSpline` 重建。Spline 的係數描述經驗曲線，沒有直接的 $E_J$、$E_C$、$d$ 解讀。

### SQUID／transmon 物理模型

定義 $e_C=E_C/h$、$e_{J\Sigma}=(E_{J1}+E_{J2})/h$，以 GHz 為單位，使用普通頻率而非角頻率。電荷基底模型為：

$$
H/h=4e_C(\hat n-n_g)^2-e_{J,\mathrm{eff}}(V_c)\cos\hat\varphi,
$$

$$
e_{J,\mathrm{eff}}=e_{J\Sigma}\sqrt{\cos^2\theta+d^2\sin^2\theta},\qquad
\theta=\pi\frac{V_c-V_s}{V_{\Phi_0}},\qquad
d=\frac{|E_{J1}-E_{J2}|}{E_{J1}+E_{J2}}.
$$

這裡 $V_s$ 為模型 sweet spot，$V_{\Phi_0}$ 為模型中的電壓週期，$n_g$ 為 offset charge。若 $\lambda_i$ 是 $H/h$ 的能階，則：

$$
f_{01}=\lambda_1-\lambda_0,\qquad f_{02}/2=(\lambda_2-\lambda_0)/2.
$$

實作採用數值對角化的能階差，並以經驗證的能階查表加速擬合。模型定義可參考 [scqubits TunableTransmon](https://scqubits.readthedocs.io/en/v4.1/guide/qubits/tunable_transmon.html)。

圖中亦列出 $E_J/E_C\gg1$ 時的解析近似：

$$
f_{01}\simeq\sqrt{8e_Ce_{J\Sigma}}\left(\cos^2\theta+d^2\sin^2\theta\right)^{1/4}-e_C.
$$

同階近似給出 $f_{12}-f_{01}\simeq-e_C$；完整數值能階還包含高階修正。[理論來源：Koch et al.](https://arxiv.org/html/cond-mat/0703002v2)

## 雙躍遷約束與參數解讀

使用者提供同一參考工作點的：

$$
f_{01}=2.760584\ \mathrm{GHz},\qquad f_{02}/2=2.672584\ \mathrm{GHz}.
$$

因此 $f_{02}=5.345168$ GHz、$f_{12}=2.584584$ GHz，且：

$$
\alpha=f_{12}-f_{01}=f_{02}-2f_{01}=-0.176\ \mathrm{GHz}=-176\ \mathrm{MHz}.
$$

一階近似給出 $E_C/h\approx176$ MHz。最新版以完整模型同時匹配兩個躍遷，先推估參考點的 $E_{J,\mathrm{eff}}$ 與 $E_C$，再擬合全段 Flux 曲線；這個步驟不需先指定參考點的 Flux 電壓。

| 完整模型參數 | 目前數值 | 備註 |
| --- | ---: | --- |
| $E_C/h$ | 0.15225565 GHz | 掃描 $n_g=0$～0.5 時，參考頻率約束給出 151.982～152.256 MHz |
| 參考點 $E_{J,\mathrm{eff}}/h$ | 7.01029711 GHz | 與整段模型的 $E_{J\Sigma}$ 分別記錄 |
| $E_{J\Sigma}/h$ | 13.93695159 GHz | 全段頻譜的條件估計 |
| $V_{\Phi_0}$ | 0.75130452 V | 單顆 transmon 模型擬合的週期 |
| 物理模型 $V_s$ | −0.24983949 V | 與經驗二次擬合頂點分別記錄 |
| $n_g$ | 約 0.5 | 模型條件下的最佳值 |
| $d$ | 趨近 0 的下界 | 尚未可靠辨識 |
| 推得 $C_\Sigma=e^2/(2E_C)$ | 約 127.22 fF | 同樣依賴單顆 transmon 假設 |
| 全段物理模型 RMSE | 4.2491 MHz | 系統性殘差仍存在 |

目前提供的雙躍遷頻率已取代早期僅由 $f_{01}$ 曲線自由估計 $E_C$ 的方式。原 `transmon_model/` 中約 0.409 GHz 的 $E_C/h$ 是先前未加入這個約束的條件解；本表使用新版結果。

經驗 spline 適合描述與拼接本次資料，物理模型用來檢查元件常數；兩者的擬合目標與限制不同。目前 $E_J$、週期、$d$ 的解讀仍受接縫位移假設與全頻譜系統性殘差影響。`d` 的下界解不等同於獨立確認接面對稱。數值查表與直接對角化的最大差約 0.0036 Hz，這是數值實作檢查，與參數或量測不確定度不同；目前未提供統計參數信賴區間。

## 目錄與資料欄位

| 路徑 | 內容與用途 |
| --- | --- |
| [combined_native_data.nc](combined_native_data.nc)、[scan_summary.csv](scan_summary.csv) | 原生分段資料與各段量測設定；根目錄 PNG／PDF 是原始座標整合圖 |
| [fit/](fit/) | 早期以 #1615 頻率 offset 約 +51.84 MHz 做比較的結果；正式拼接使用後續 Flux 修正 |
| [flux_crosstalk_corrected/](flux_crosstalk_corrected/) | Flux 軸修正、連續經驗曲線、修正前後比較及原始資料的修正版座標 |
| [transmon_model/](transmon_model/) | 僅用頻譜 $f_{01}$ 的初步物理模型、能階查表及可重建的經驗函數係數 |
| [transmon_model_anharmonicity_constrained/](transmon_model_anharmonicity_constrained/) | 加入提供的 $f_{01}$、$f_{02}/2$ 約束後的主要物理結果，以及包含所有函數的整合圖 |

[combined_flux_corrected.nc](flux_crosstalk_corrected/combined_flux_corrected.nc) 保留 `run × frequency_point × flux_point` 的原生資料結構：

- `state`：原始平均 readout state。
- `frequency_hz`：每個 run 的實際頻率座標，單位 Hz。
- `flux_v`：原始 Flux 座標，單位 V。
- `flux_axis_correction_v`：每個 run 的橫軸修正；只有 #1615 非零。
- `flux_corrected_v`：`flux_v + flux_axis_correction_v`。

在本 README 所在目錄讀取其中一段：

```python
import xarray as xr

with xr.open_dataset(
    "flux_crosstalk_corrected/combined_flux_corrected.nc", engine="scipy"
) as ds:
    scan = ds.sel(run=1615)
    voltage = scan.flux_corrected_v.values
    frequency_ghz = scan.frequency_hz.values / 1e9
    signal = scan.state.values
```

## 重跑方式

需要 Python、NumPy、SciPy、Matplotlib、xarray；分析以 Python 3.12.7、SciPy 1.13.1 執行。中文圖說使用 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`。這些分析腳本讀取已儲存資料，無須連線 QPU。

以下依序重建原始整合、Flux 修正、物理查表／經驗函數，以及最新版含雙躍遷約束的圖。預設使用 #1608–#1618，各腳本將結果寫入本目錄對應子目錄；也可使用 `--output-dir` 指定輸出位置，但後續步驟的來源路徑需保持一致。

```bash
cd /home/ratiswu/Documents/GitHub/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/side_project/CouplerMeas

export MPLCONFIGDIR=/tmp/coupler-flux-matplotlib
export XDG_CACHE_HOME=/tmp/coupler-flux-cache
export PYTHONDONTWRITEBYTECODE=1

python merge_coupler_flux_spectroscopy.py
python stitch_coupler_flux_crosstalk.py
python fit_coupler_transmon_model.py
python annotate_coupler_fits_with_anharmonicity.py \
    --f01-ghz 2.760584 \
    --half-f02-ghz 2.672584
```

如需重現 `fit/` 的早期比較，可另執行 `python fit_coupler_flux_spectroscopy.py`。

相關程式：[原始整合](../../merge_coupler_flux_spectroscopy.py)、[Flux 拼接](../../stitch_coupler_flux_crosstalk.py)、[物理模型](../../fit_coupler_transmon_model.py)、[雙躍遷約束與完整函數圖](../../annotate_coupler_fits_with_anharmonicity.py)。

## 已完成的檢查

- 匯出資料的 56,100 個原始訊號點、原始頻率與原始 Flux 座標，已逐 run 與來源檔案核對一致。
- 新增的 Flux 修正座標僅對 #1615 加上固定水平位移。
- 物理模型的主峰／預測輸出共 458 組，其輸入頻率保留原值。
- 完整數值模型同時匹配提供的 $f_{01}$ 與 $f_{02}/2$，並核對 lookup 能階差與直接對角化結果。
- 匯出的 B-spline knots／係數可重建原有經驗曲線。
