# ===========================================================================
# 自動校準實驗序列 — 直接執行各實驗 .py 檔
#
# 使用方式（於 Superconducting 目錄下）:
#   powershell -ExecutionPolicy Bypass -File Script/run_calibration_sequence.ps1
#
# 修改實驗:
#   1. 調整下方「全域參數」
#   2. 在「實驗順序」增刪或註解掉 Add-Experiment 行
#   3. 單一實驗額外參數寫在 Add-Experiment 最後一個參數
# ===========================================================================

$ErrorActionPreference = "Stop"

# --- Python 直譯器（請改成你的 qualibrate_env）---
$Python = "C:\Users\richa\anaconda3\envs\qualibrate_env\python.exe"

# --- 路徑 ---
$Root = Split-Path -Parent $PSScriptRoot
$RunFile = Join-Path $PSScriptRoot "_run_file.py"

Set-Location $Root

# ===========================================================================
# 全域參數 — 修改這裡即可
# ===========================================================================
$Qubits           = "None"      # None = 所有 active qubits
$Multiplexed      = "True"
$FluxPoint        = "joint"
$ResetActive      = "active"
$HistoNum         = "100"
$SkipFailed       = $true       # 單一實驗失敗時是否繼續

# $UseGlobalParams = $false  → 直接 python 執行 .py（用各檔案內建 Parameters，零掃描）
# $UseGlobalParams = $true   → 透過 _run_file.py 套用下方全域參數（只 inspect 該檔案）
$UseGlobalParams  = $true

# 常用參數組（依各 node 參數命名）
$CommonCalib = @(
    "qubits=$Qubits"
    "flux_point_joint_or_independent=$FluxPoint"
    "multiplexed=$Multiplexed"
    "reset_type_thermal_or_active=$ResetActive"
)
$CommonRamsey = @(
    "qubits=$Qubits"
    "flux_point_joint_or_independent=$FluxPoint"
    "multiplexed=$Multiplexed"
    "reset_type=$ResetActive"
)
$CommonRamseyFlux = @(
    "qubits=$Qubits"
    "flux_point_joint_or_independent=$FluxPoint"
    "multiplexed=$Multiplexed"
)
$CommonPowerRabi = @(
    "qubits=$Qubits"
    "flux_point_joint_or_independent=$FluxPoint"
    "reset_type_thermal_or_active=$ResetActive"
)
$CommonStats = @(
    "qubits=$Qubits"
    "flux_point_joint_or_independent_or_arbitrary=$FluxPoint"
    "multiplexed=$Multiplexed"
    "reset_type=$ResetActive"
    "histo_num=$HistoNum"
)

# 批次佇列（整段序列只啟動一次 Python）
$ExperimentQueue = @()

function Add-Experiment {
    param(
        [string]$Label,
        [string]$ScriptPath,
        [string[]]$Params = @()
    )
    $entry = [ordered]@{
        label  = $Label
        path   = $ScriptPath
        params = @($Params)
    }
    if (-not $UseGlobalParams) {
        $entry.params = @()
    }
    $script:ExperimentQueue += ,$entry
}

Write-Host "Automatic Calibration Sequence"
Write-Host "  Root        : $Root"
Write-Host "  qubits      : $Qubits"
Write-Host "  multiplexed : $Multiplexed"
Write-Host "  flux_point  : $FluxPoint"
Write-Host "  reset       : $ResetActive"
Write-Host "  histo_num   : $HistoNum"
Write-Host "  Parameter Mode    : $(if ($UseGlobalParams) { 'Single file inspect + parameter overwrite' } else { 'Direct python .py (zero scan)' })"

# ===========================================================================
# 實驗順序 — 增刪或註解掉 Add-Experiment 即可
# ===========================================================================

Add-Experiment "06a Ramsey vs Flux" `
    "calibration_graph\06a_Ramsey_vs_Flux_Calibration.py" `
    $CommonRamseyFlux

Add-Experiment "06 Ramsey" `
    "calibration_graph\06_Ramsey.py" `
    $CommonRamsey

Add-Experiment "09 Power Rabi x180" `
    "calibration_graph\09_Power_Rabi_State.py" `
    ($CommonPowerRabi + "operation_x180_or_any_90=x180")

Add-Experiment "09 Power Rabi x90" `
    "calibration_graph\09_Power_Rabi_State.py" `
    ($CommonPowerRabi + "operation_x180_or_any_90=x90")

Add-Experiment "07b IQ Blobs" `
    "calibration_graph\07b_IQ_Blobs.py" `
    $CommonCalib

Add-Experiment "09b DRAG x180" `
    "calibration_graph\09b_DRAG_Calibration_180_minus_180.py" `
    ($CommonCalib + "operation=x180")

Add-Experiment "09b DRAG x90" `
    "calibration_graph\09b_DRAG_Calibration_180_minus_180.py" `
    ($CommonCalib + "operation=x90")

Add-Experiment "09a Stark Detuning x180" `
    "calibration_graph\09a_Stark_Detuning.py" `
    ($CommonCalib + "operation=x180")

Add-Experiment "09a Stark Detuning x90" `
    "calibration_graph\09a_Stark_Detuning.py" `
    ($CommonCalib + "operation=x90")

Add-Experiment "10x Single-Qubit RB" `
    "calibration_graph\10x_Single_Qubit_Randomized_Benchmarking.py" `
    $CommonCalib

# # --- 統計實驗（不需要時註解掉以下三行）---
# Add-Experiment "05st T1 statistics" `
#     "side_project\StatisticsMustDo\05st_T1_statics.py" `
#     $CommonStats

# Add-Experiment "06st T2e statistics" `
#     "side_project\StatisticsMustDo\06st_T2e_statics.py" `
#     $CommonStats

# Add-Experiment "06st T2* statistics" `
#     "side_project\StatisticsMustDo\06st_T2star_statics.py" `
#     $CommonStats

# ===========================================================================
# 執行（整段序列只啟動一次 Python 行程）
# ===========================================================================

Write-Host "  Experiment Number    : $($ExperimentQueue.Count)"
Write-Host ""

$sw = [System.Diagnostics.Stopwatch]::StartNew()

$batchArgs = @($RunFile, "--batch")
if ($SkipFailed) { $batchArgs += "--skip-failed" }

$json = $ExperimentQueue | ConvertTo-Json -Depth 5 -Compress
$json | & $Python @batchArgs

$sw.Stop()
Write-Host ""
Write-Host ("Completed. Total time: {0:N1} minutes" -f ($sw.Elapsed.TotalMinutes))
