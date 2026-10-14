# JY_agent workflow / 工作流程

## 中文

### 0. 一次性安裝

1. 確認 Qualibrate、`~/.qualibrate/config.toml`、QuAM state/wiring、Data 與
   calibration library 可正常使用。
2. 在 repository 最上層執行 `powershell -ExecutionPolicy Bypass -File
   .\jy_agent.ps1 -Action Install`。
   也可要求一般 coding agent 執行同一指令；這一步不需要 JY MCP，也不開始量測。
3. 用 Codex、Claude Code 或 Cursor 開啟 repository 最上層為 project/workspace，
   接受一次性的 MCP trust 提示。最外層資料夾名稱可變，且不需另開 `JY_agent`。

### 1. Client bootstrap

Client 讀取 repository-root project config，啟動 `jy_agent.ps1 -Action Stdio`；
root shim 再以相對路徑定位並啟動 `JY_agent/start_mcp_stdio.ps1`。此流程：

1. 尋找已安裝的 Qualibrate Python。
2. 啟動 client-owned STDIO MCP protocol，讓 client 取得 `jy_*` tools。
3. STDIO 的 stdout 只傳 MCP frame；不啟動網站、tunnel、port，也不依賴網路。

沒有這一步，agent 尚未有 `jy_*` tools，因此自然語言本身無法從不存在的 MCP
啟動 MCP。這就是 bootstrap 的雞生蛋邊界。開啟 project 只建立工具，不會建立
run、啟動公開網站、接觸儀器或修改 state。

收到任一進入語句後，agent 才以一般 shell 能力執行 idempotent
`jy_agent.ps1 -Action Ensure`：啟動／沿用 loopback MCP HTTP service
(`127.0.0.1:8765`) 與 Approval service (`127.0.0.1:8766`)，建立或重建只 forward
Dashboard port 的 Cloudflare quick tunnel 和內部 256-bit access token（瀏覽器登入
則使用 `runtime/dashboard-password.txt` 的固定密碼），並將 public URL、
PID、port 與 Python 路徑記錄到 `runtime/server-bootstrap.json`。長駐 STDIO service
會動態讀取這份 metadata，因此不必為了取得新 Dashboard URL 重啟 MCP。最後 agent
才呼叫對應的 `jy_enter_*` tool。

### 2. 進入模式

完整中英文自然語言與本機維護指令見 [Command.md](Command.md)。英文入口為
`Enter JY measurement mode` 與 `Enter JY automatic measurement mode`。

- `進入 JY 量測模式`：agent 先檢查服務健康度，再呼叫
  `jy_enter_measurement_mode`（不必傳 targets／multiplexed；新 workflow 使用
  `state.json` `active_qubit_names` 且預設 multiplexed=true）。使用者取得一個共用
  Dashboard URL；每次透過
  `jy_request_conversational_run` 建立單一提案、網頁核准、執行、snapshot 分析、
  Decision/Reason/Next action，state commit 另行核准。
- `進入 JY 自動量測模式`：agent 呼叫 `jy_enter_autonomy_mode`（同樣省略
  targets／multiplexed，除非操作者明確覆寫），使用者在 Dashboard
  的核准分頁選擇起點、選用節點、量測方式與 flux 校正，預覽後核准一次 bounded
  lease。之後只用 `jy_autonomy_*` tools；超出 targets、nodes、授權時數（預設 8
  小時，進入時可指定）、20 次／node／qubit 或參數安全範圍必須重新授權。Codex 若提供 durable
  goal，會建立 session-scoped goal，並用 `jy_wait_for_autonomy_status` 等待事件；
  Claude/Cursor 的 host 若結束 turn，需以 Dashboard 的 `已核准` 提示續跑。
- 對使用者只呈現 entry response 最外層的 `browser_url`。這個 `/` 是共用入口，
  會連到同一 session 的首頁、核准、實驗結果分頁；舊 `/approve/...`、
  `/autonomy/...` 與 `/session/...` 路徑只做 303 redirect。
  直接呼叫 lease request 或替換停止過的 lease 時，service 仍必須重用並重新綁定
  同一個 workflow session。

### 3. 每次實驗的資料流

1. `PolicyEngine` 將 defaults 與提案參數合併並驗證 hard policy。
2. Worker 在 lock 與 subprocess 邊界執行已登錄 node；agent 不直接跑 script。
   停止時 runner 寫入含 run/process token 的 request，worker 協作式停止並寫 exit
   receipt；不再依賴 Windows console `CTRL_BREAK_EVENT`。
3. 完成後保存 snapshot、完整參數、開始／結束／耗時及狀態。
4. `SnapshotAnalyzer` 讀 `node.json`、`data.json`、`ds.h5` 與結果圖，獨立驗證
   fitting、SNR、coverage 與 node-specific 規則；node outcome 只是 advisory。
5. 決策與原因存入 SQLite，Dashboard 收到 SSE event 後更新；舊結果保留。
6. 自動流程只 commit 與 decision patch 完全一致的通過證據；`needs_review` 的 run
   只 commit 其 `passing_targets`。只有 `snapshot_missing`、`state_hash_conflict`、
   `hardware_lock_anomaly` 會結束 lease。接觸硬體前被拒絕的請求（超出 policy 的
   參數、超出範圍的 targets）只被拒絕並記錄；已登錄的 worker crash 保留 lease 並
   提示修正，其他 worker crash 暫停 lease，以 `恢復量測` 續跑。

儀器 transport 例外是獨立的 pause 分支：run 會記錄
`instrument_unreachable`，一般 workflow 或 bounded lease 轉為 `paused`，Dashboard
顯示恢復方式。可證明連線從未建立且 state 已恢復時釋放 lock；執行中失聯則保留
local quarantine，但不阻擋完整關機關閉服務。

預設順序是 `config/agent.yaml` 的 `workflow.sequence`：
`02 → 02x → 02c → 02a → 02b → 03a → 03b → 04 → 07d → 07b → 05 → 06 → 06a → 06b → 10a → 05st → 06st_t2star → 06st_t2e`。
`02`、`06a` 與統計節點為選用；`02b`、`03b`、`06a` 只在選擇 flux 校正時執行；未
勾選的節點直接跳過。

每個 calibration node 的第一次 run 必須用同一組參數 multiplex 全部 active
targets；分析後排除已通過目標，只對未通過子群改參數重測。誤排已通過目標會被
server 拒絕，但不會 hard-stop lease。例外：`04` 每次最多 5 顆、`07d` 最多 4 顆；
07b 之後 active／thermal reset 兩組各自第一批；submission timeout 時群組依序減半
到單顆；per-qubit 模式或共用 XY output 時，從 `03a` 起一次一顆 qubit 跑完整個
後段。per-qubit pass 中，統計節點只在其來源節點（05st←05、06st_t2star←06、
06st_t2e←06b）通過時執行；07b 未通過或 fidelity 低於 0.70 的 qubit 只能以
`use_state_discrimination=false` 跑統計。

`06b` 緊接在 `06` 後執行。所有有 reset 參數的實驗預設使用 thermal；thermal 07b
fidelity 高於 0.80 的 qubit 再跑一次 active-reset 07b，fidelity `>= 0.80` 才取得
active 資格，之後有 reset 參數的節點分成 active／thermal 兩組。若統計實驗要用
active，必須先用 active 重跑並接受 05、06、06b。三個基準實驗均須至少涵蓋 3.5
lifetimes，統計時間窗約為對應 lifetime 的 3.5–5.5 倍（建議 4.5 倍）。前置條件
通過後，05st 與兩個 06st 各跑一次（至少完成 90／100 次重複才算數）；JY 逐次
重新 fitting、剔除不良 fit 後記錄平均與分散，沒有 100/100 否決門檻。

### 4. Dashboard 與停止

Dashboard 右上角可選 `繁體中文` 或 `English`。Prediction、Parameters、Analysis、
Next action 都是摺疊面板；Decision 與 Reason 直接顯示。只有新結果或控制事件才
更新，不做固定秒數刷新。共有五個分頁。首頁顯示工作階段狀態與四種 operator
狀態：暫停目前實驗後的排程、繼續同一份未過期 lease、結束自動授權但保留網站、
緊急停止 worker 但保留網站，另有完整關機。核准頁只在 proposal 事件更新。實驗
結果頁最上方是此晶片跨 workflow 的成功結果（晶片歷史），其下保留本次所有實驗。
Token 用量頁只統計本機 Claude Code 對話紀錄；使用說明頁附各節點範例圖。

`結束量測`、`停止量測`、`退出 JY 量測模式` 等設定語句，或 Dashboard 的
首頁的「完整結束量測」按鈕（需輸入 `SHUTDOWN <session-id>`），會先持久化每個
session 的 shutdown request，再安全停止／等待 active run，接著永久停止 workflow，
再由本機 coordinator 執行 `stop_server.ps1`。這個按鈕與 authorization 的停止
不同，而且 lease 已 `halted` 時仍會顯示。若 helper 啟動失敗，頁面會顯示原因並
保留重試按鈕。

AI App／外部網路斷線不會讓 Dashboard 控制失效。網路恢復後可輸入
`結束量測`／`End measurement`；若下一次 entry 被殘留 process、workflow 或
lifecycle state 阻擋，輸入 `恢復`／`Recover`。Agent 會執行 root
`jy_agent.ps1 -Action Recover` 收斂到 closed state；這條路徑不需 MCP attachment，
且不會自動刪除 retained lock 或終止未驗證的 process。

若失敗 run 刻意保留 hardware lock，coordinator 會先確認沒有任何 JY worker 活著，
然後把 workflow 標成 `recovery_required`、保留 lock 與 marker，但仍關閉 MCP、
Dashboard 與 tunnel。下次 `Ensure` 強制為 loopback recovery-only；現場 operator
查看 QOP/OPX 與相連儀器並確認沒有 active job/output 後，再於量測電腦開啟
`http://127.0.0.1:8765/operator`。頁面會顯示所有檢查、實體確認 checkbox 與完整
確認句；成功後會封存 lock、留下 receipt，之後才可重新進入量測模式。
本機 terminal 仍保留為 fallback：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\jy_agent.ps1 `
  -Action RecoverLock -RunId <run-id>
```

此流程會重新檢查 worker、request identity 與 state/recovery hashes；只有全數通過，
且 operator 輸入畫面顯示的完整確認句後，才會封存 lock 並留下 audit receipt。不得
手動刪除 `runtime\hardware.lock`。

手機或第二個瀏覽器直接開啟同一個 Dashboard 網址，輸入量測電腦
`runtime/dashboard-password.txt` 的固定密碼即可，不需配對連結。每個瀏覽器取得
各自獨立的 12 小時 secure cookie。

## English

### 0. One-time installation

1. Verify that Qualibrate, `~/.qualibrate/config.toml`, QuAM state/wiring,
   Data, and the calibration library work.
2. From the repository root, run `powershell -ExecutionPolicy Bypass -File
   .\jy_agent.ps1 -Action Install`.
   An ordinary coding agent can run the same command; JY MCP is not needed and
   no measurement starts.
3. Open the repository root as the Codex, Claude Code, or Cursor workspace and
   accept the one-time project MCP trust prompt. The outer folder may be renamed;
   opening the nested `JY_agent` folder is not required.

### 1. Client bootstrap

The client reads its repository-root project config and launches
`jy_agent.ps1 -Action Stdio`. That name-independent root shim resolves the nested
`JY_agent/start_mcp_stdio.ps1`. The launcher discovers Qualibrate Python and
starts only the client-owned STDIO MCP protocol. Its stdout carries MCP frames
only; website, tunnel, port, and network startup are deliberately excluded.
Opening the project creates tools only; it does not start a public site, create a
run, touch instruments, or mutate state.

After an entry phrase, the agent runs the idempotent root
`jy_agent.ps1 -Action Ensure` using ordinary shell access. Ensure starts or
reuses the loopback MCP HTTP and Approval services, forwards only the Approval
port through a Cloudflare quick tunnel, creates an internal 256-bit access token
(browsers log in with the fixed password in `runtime/dashboard-password.txt`), and
records verified URL/process/port metadata. The already-running STDIO service
reads that metadata dynamically, so it need not restart before the selected
`jy_enter_*` tool is called. A website or tunnel failure therefore cannot close
the MCP connection before its `initialize` response.

### 2. Enter a mode

- `進入 JY 量測模式` calls `jy_enter_measurement_mode` without requiring chat-side
  targets or multiplexed; a new workflow uses `state.json` `active_qubit_names`
  and defaults `multiplexed=true`. One shared Dashboard
  handles per-run and per-state-change approval.
- `進入 JY 自動量測模式` calls `jy_enter_autonomy_mode` the same way. On the
  Dashboard Approval page the operator chooses the start node, optional nodes,
  measurement mode, and flux calibration, previews, and approves one bounded
  lease; later actions must remain within targets, nodes,
  a lease budget (eight hours by default, specifiable at entry), 20 attempts per
  node/qubit, and hard parameter policy. Codex uses
  a session-scoped durable goal when available and waits with
  `jy_wait_for_autonomy_status`; Claude/Cursor require a host resume if their turn ends.
- Surface only the entry response's top-level `browser_url`. The common `/`
  entry links to Home, Approval, and Experiment results for one session; legacy
  approval/control/session paths issue a 303 redirect. Direct and
  replacement lease requests must rebind the workflow's stable session.

### 3. Per-experiment data flow

The policy engine validates merged parameters. A lock-protected worker runs only
a registered node. Stop delivery uses an authenticated request file and a worker
exit receipt rather than Windows console signaling. Snapshot artifacts, parameters, timing, and statuses are
recorded. The independent analyzer rechecks raw data, fitting, SNR, coverage, and
node rules. Decision, Reason, and Next action are appended to SQLite and the
Dashboard updates on an SSE event. Only passing evidence identical to the
decision patch is committed automatically; a `needs_review` run commits only its
`passing_targets`. Only `snapshot_missing`, `state_hash_conflict`, and
`hardware_lock_anomaly` end the lease. A request refused before the hardware
(out-of-policy parameters, out-of-scope targets) is refused and recorded; a
registered worker crash keeps the lease with its remedy, and any other worker
crash pauses the lease until `Resume measurement` (`恢復量測`).

The default sequence is `config/agent.yaml` `workflow.sequence`:
`02 → 02x → 02c → 02a → 02b → 03a → 03b → 04 → 07d → 07b → 05 → 06 → 06a → 06b → 10a → 05st → 06st_t2star → 06st_t2e`.
`02`, `06a`, and the statistics nodes are optional; `02b`, `03b`, and `06a` run
only with flux calibration; unticked nodes are skipped. The first run of every
node must multiplex every active target with the same parameters. Later retries
may omit resolved qubits, but unresolved qubits that can share a parameter
change must be retried together. Exceptions: `04` runs at most 5 targets and
`07d` at most 4; after 07b the active- and thermal-reset groups each get their
own first batch; a submission timeout halves the group down to one target; in
per-qubit mode or with a shared XY output, one qubit at a time goes through the
whole tail from `03a`. On a per-qubit pass a statistics node runs only if its
source node passed (05st after 05, 06st_t2star after 06, 06st_t2e after 06b), and
a qubit whose 07b did not pass or is below 0.70 fidelity runs statistics only
with `use_state_discrimination=false`.

06b runs immediately after 06. Every reset-capable node defaults to thermal
reset. A qubit whose thermal 07b fidelity is above 0.80 earns one active-reset
07b repeat and qualifies for active reset when that repeat reaches at least
0.80; later reset-capable nodes then split into active and thermal groups.
Active-reset statistics require accepted active-reset repeats of 05, 06, and 06b
first. Those baselines must cover at least 3.5 lifetimes, and the statistics
window remains 3.5–5.5 times the corresponding lifetime (4.5x is recommended).
After the prerequisites pass, each statistics node runs once and counts only
when at least 90 of 100 repetitions completed; JY refits every repetition, drops
poor fits, and records the filtered mean and spread, with no 100/100 rejection
gate.

### 4. Dashboard and shutdown

The language selector switches the entire UI. Prediction, Parameters, Analysis,
and Next action are collapsed; Decision and Reason stay visible. Updates occur
only on relevant events. There are five tabs. Home contains status, Pause,
Resume, End automation while keeping the site, Emergency-stop worker while
keeping the site, and full shutdown; Approval contains pending authorization;
Experiment results opens with the chip's successful results across workflows
(chip history) and retains every experiment of the session; Token usage counts
only local Claude Code transcripts; Guide shows an example plot per node. Full
shutdown requires the exact
`SHUTDOWN <session-id>` confirmation. Each session persists its own shutdown
request before the worker is touched. Configured exit phrases safely stop or
poll an active run, leave/quarantine the workflow, and run `stop_server.ps1`.
A retained hardware lock is preserved but, after every JY worker is verified
dead, no longer keeps the services or tunnel running. The next Ensure is
loopback recovery-only. The same full-shutdown action is present
on a halted Dashboard. A verified recovery can then be completed from the lab PC's
loopback-only `/operator` page; the terminal `RecoverLock` action remains a
fallback. Phones and secondary browsers open the same Dashboard URL and enter
the fixed password from `runtime/dashboard-password.txt`; no pairing link is
needed, and each browser receives its own 12-hour secure cookie.
