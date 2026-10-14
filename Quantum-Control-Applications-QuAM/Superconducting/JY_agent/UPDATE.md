# Update log / 更新紀錄

## 2026-10-14

- 04 拉長 gate 的上限：共用 XY 線 1000 → 400 ns；私有線維持 100 ns。
- 03a 的 `operation_amplitude_factor` 上限依 XY 接線決定：共用線最大 1.0，第一次
  800 MHz coarse 用 1.0；私有線（可 multiplex）最大 0.05，第一次 coarse 用 0.01。
  03a 預設 saturation 長度 50000 → 20000 ns。
- 從頭量測：目前晶片沒有任何 JY 量測紀錄的 qubit（例如從別的晶片複製的 state），
  第一次量測前須先套用 `jy_request_fresh_qubit_defaults`：清空 extras；T1、
  T2ramsey、T2echo = 100 µs；anharmonicity = 200 MHz；saturation 0.5／20000 ns；
  readout 長度與 depletion 2000 ns；readout 判別值歸零、confusion matrix 設為單位
  矩陣；z offset 與 quad term 歸零；x180 = 0.05、x90 = 0.025、DRAG 歸零；gate 長度
  共用線 400 ns、私有線 40 ns。有紀錄的 qubit 不受影響。

- 04 gate-lengthening cap: shared XY line 1000 -> 400 ns; private stays 100 ns.
- The 03a `operation_amplitude_factor` cap follows the XY wiring: a shared line
  may go to 1.0 and its first 800 MHz coarse run uses 1.0; a private
  (multiplexable) line stays at or below 0.05 and its first coarse run uses
  0.01. The 03a default saturation length is 20000 ns (was 50000).
- Measuring from scratch: a qubit with no JY run on the current chip (for
  example a state copied from another chip) needs
  `jy_request_fresh_qubit_defaults` before its first run. The commit empties
  extras and sets T1/T2ramsey/T2echo to 100 us, anharmonicity to 200 MHz,
  saturation to 0.5 / 20000 ns, readout length and depletion to 2000 ns, the
  readout discrimination to zero with an identity confusion matrix, the z
  offsets and quad term to 0, x180 to 0.05 and x90 to 0.025 with DRAG terms 0,
  and the gate to 400 ns (shared line) or 40 ns (private line). Qubits with a
  record keep their state.

- Token 用量頁改為統計整顆晶片：最上方是晶片開始時間（第一個 workflow 建立時）
  與所有 workflow 的總用量，分成實驗、量測前準備、程式開發、對話。可選整個晶片、
  本 session、今天、近 7 天、本月或自訂起迄（台北時間）；另列此時段全部用量
  （含其他晶片與 workflow 之間的工作）。實驗依日期分組，每列為開始時間、節點、
  qubits、模型、總 token（點開看拆分）、output 與估計費用；繼承的 run 不列出。

- Token usage now covers the whole chip: its start time (first workflow) and the
  total across all its workflows at the top, split into experiments, before-run,
  development, and conversation. Pick the whole chip, this session, today, the
  last 7 days, this month, or a custom from/to (Taipei time); All usage in range
  also counts other chips and work between workflows. Experiments are grouped by
  date with start time, node, qubits, model, total tokens (click for the split),
  output, and estimated cost; inherited runs are not listed.

## 2026-10-13

- 校正順序改為 `02 → 02x → 02c → 02a → 02b → 03a → 03b → 04 → 07d → 07b → 05 →
  06 → 06a → 06b → 10a → 05st → 06st_t2star → 06st_t2e`。新增選用的 `02`
  寬頻共振腔掃描（02x 之前）；`07d` readout 頻率／長度／功率優化緊接在 `07b`
  之前；`05` 移到 `07b` 之後，以優化後的 readout 量 T1。
- 新增 flux 校正 `02b`（02a 後）、`03b`（03a 後）與選用的 `06a`（06 後），只在
  核准頁選「是」時執行；有 qubit 共用 Z output 時無法選。
- 共用 XY drive output 的晶片從 `03a` 起一次只量一顆 qubit，跑完整個後段再換下
  一顆，且不能選 multiplex；只有 03a 找不到線或 04 沒有 pi pulse 才提早結束該
  qubit。每個節點前以 `jy_request_xy_lo_plan` 規劃 drive LO（可行時共用一個
  100 MHz 格點 LO）。
- `04` 若在最長 gate 與滿幅仍看不到 Rabi，可把 qubit 送回 `03a` 找另一條線，
  每顆最多 3 次。`04` 每次最多 5 顆、`07d` 最多 4 顆。
- 自動量測核准頁新增選項：起點（依此晶片最近 workflow 的晶片歷史）、選用節點
  （02、06a、05st、06st_t2star、06st_t2e）、從 03a 起 per-qubit 或 multiplex、
  是否做 flux 校正；先按預覽，再勾選同意並輸入核准碼。
- Dashboard 新增 Token 用量與使用說明分頁（共五個分頁）。Token 用量只讀本機
  Claude Code 對話紀錄；使用說明附各節點範例圖。實驗結果頁最上方新增晶片歷史，
  列出此晶片跨 workflow 的成功結果，並在 `runtime/chip_history/` 保留關機後仍可
  開啟的靜態副本。
- Lease 只在 `snapshot_missing`、`state_hash_conflict`、`hardware_lock_anomaly`
  時結束。接觸硬體前被拒絕的請求只拒絕並記錄；已登錄的 worker crash 保留 lease
  並提示修正，其他 crash 暫停 lease，以 `恢復量測` 續跑。`needs_review` 的 run
  仍 commit 其 `passing_targets`。第一次 run 因程式過大 submission timeout 時，
  群組依序減半到單顆。
- 新增 setup 工具：`jy_request_02x_probe_setup`（02x 探測功率餘裕、每顆 qubit
  獨立的圖格與 QUA thread），以及 readout 長度正規化與 readout 振幅預算。
- Active reset 資格門檻為 07b fidelity 0.80；07b 之後 active／thermal 兩組分開
  量測。

- The sequence is now `02 → 02x → 02c → 02a → 02b → 03a → 03b → 04 → 07d → 07b →
  05 → 06 → 06a → 06b → 10a → 05st → 06st_t2star → 06st_t2e`. An optional `02`
  broadband resonator survey runs before 02x; `07d` readout
  frequency/duration/power optimization runs right before `07b`; `05` moved after
  `07b` so T1 is measured on the optimized readout.
- Flux calibration: `02b` (after 02a), `03b` (after 03a), and the optional `06a`
  (after 06) run only when the lease page says yes; a Z output shared by two
  qubits disables it.
- On a chip with a shared XY drive output, every node from `03a` on measures one
  qubit at a time through the whole tail before the next qubit, and multiplex is
  not offered. A pass ends early only when 03a finds no line or 04 finds no pi
  pulse. Before each node `jy_request_xy_lo_plan` places the drive LO (one shared
  100 MHz-grid LO when possible).
- `04` may send a qubit with no Rabi at full amplitude and the longest gate back
  to `03a` to look for another line, at most 3 times per qubit. `04` runs at most
  5 targets per run and `07d` at most 4.
- The lease approval page now asks for the start node (from this chip's history),
  optional nodes (02, 06a, 05st, 06st_t2star, 06st_t2e), per-qubit or multiplex
  from 03a, and flux calibration; approval needs Preview, then the agree box and
  the approval code.
- The Dashboard gains Token usage and Guide tabs (five tabs in all). Token usage
  reads only local Claude Code transcripts; Guide shows an example plot per node.
  Experiment results now opens with the chip history -- successful results for
  this chip across workflows -- with a static copy under `runtime/chip_history/`
  that survives shutdown.
- Only `snapshot_missing`, `state_hash_conflict`, and `hardware_lock_anomaly` end
  the lease. Requests refused before the hardware are refused and recorded; a
  registered worker crash keeps the lease with its remedy, and any other crash
  pauses it until `Resume measurement`. A `needs_review` run still commits its
  `passing_targets`. A first run that times out on submission because the
  program is too large halves its group down to one target.
- New setup tools: `jy_request_02x_probe_setup` (02x probe headroom, a plot cell
  and QUA thread per qubit), plus readout-length normalization and the readout
  amplitude budget.
- Active-reset qualification uses 07b fidelity 0.80; after 07b the active and
  thermal groups are measured separately.

## 2026-09-20

- Dashboard 版面調整：暫停排程、繼續排程、結束自動授權與緊急停止 worker 從
  「結果與控制」頁移到首頁，排在狀態卡下方、完整關機卡上方；第三個分頁因此
  改名為「實驗結果」。結果頁最上方新增「結果摘要」：依序列出本次服務的
  `實驗N: <node>`，只有在 JY 判定該 run 通過、且節點回報該 qubit 為 successful
  時，才在右側寫出成功的 qubit 並附上該次實驗的結果圖縮圖（最多 6 張，其餘
  以數量標註）。未成功或進行中的實驗仍會列出，但右側不放 qubit 與圖。點任一
  列或縮圖會跳到下方該次實驗的完整面板。
- Dashboard layout: pause, resume, end-automation, and emergency-stop-worker
  moved from the results page to Home, between the status card and full
  shutdown, so the third tab is now named Experiment results. The results page
  opens with a new result summary that lists `Experiment N: <node>` in order.
  Successful qubits and result-plot thumbnails (capped at six, with the
  remainder noted) appear only when JY passed the run and the node also
  reported that qubit as successful. Unsuccessful and in-progress experiments
  keep their row without qubits or plots. Selecting a row or thumbnail opens
  that experiment's full panel below.

## 2026-09-06

- 取消 `02x`/`02a` fixed full-batch retry：與其他節點相同，第一次全目標 multiplex，
  之後只重測 unresolved 子群。誤排已通過目標改為 soft-reject，不再 hard-stop lease。
- Removed fixed full-batch retries for `02x`/`02a`. Every node now follows
  first-batch-all-targets then unresolved-subgroup retries. Re-including
  resolved targets is rejected without halting the autonomy lease.

## 2026-09-04

- 所有節點改為 02x 式 multiplex-first：第一次 run 必須用同一組參數量測全部
  active targets。只有部分 qubit 失敗時，才把那些未通過的 qubit 成組改參數
  重測；不可一開始就逐顆孤立量測。`02c`/`04`/`05`/`07b`/`06`/`06b`/`10a`
  由 server 強制第一輪全顆 multiplex。
- All nodes now follow the 02x multiplex-first schedule. The first run must
  measure every active target with one shared parameter set. Retries may omit
  resolved qubits, but unresolved qubits that can share a parameter change are
  grouped. Sequential isolated one-qubit starts are no longer the default.

## 2026-09-02

- Resonator 策略改為：操作者先把足夠正確的 resonator RF 寫入 `state.json`；
  `02x`/`02c`/`02a` 只做數 MHz 級微調。`frequency_span_in_mhz` 上限 60 MHz，
  且掃描窗口不可包含 readout `upconverter_frequency`，因為原始 fitting 會把
  LO（IF=0）誤認為共振。預設順序改為 `02x → 02c → 02a → …`；`02a` 只做
  dressed frequency 小範圍掃描。
- Resonator strategy: operator-supplied RF in `state.json` is already close.
  `02x`/`02c`/`02a` only fine-tune a few MHz, cap span at 60 MHz, and reject
  windows that include the readout upconverter. Sequence is now
  `02x → 02c → 02a → …`; `02a` is a simple dressed-frequency scan.

- 進入語句不必再寫 `target:` 或 `multiplex:`。新 workflow 從 `state.json` 的
  `active_qubit_names` 取 targets，`multiplexed` 預設為 true；明確覆寫仍可用。
  既有非 multiplex workflow 以 phrase-only 恢復時，不會被預設值改成 multiplex。
- Phrase-only entry no longer requires chat-side `target` or `multiplex`. New
  workflows use `state.json` `active_qubit_names` and default `multiplexed=true`.
  Resuming a non-multiplexed workflow with an omitted multiplexed flag keeps it
  unchanged.

## 2026-08-29

- 修正完整關機順序：先寫 SQLite 與
  `runtime/shutdown_requests/<session-id>.json`，再向 worker 寫 authenticated stop
  request；worker 留下 exit receipt 與 termination cause，不再使用 Windows
  `CTRL_BREAK_EVENT`。Grace 逾時後仍只可終止驗證過的 worker PID。
- retained hardware lock 改為 recovery quarantine：所有 worker 死亡後仍保留 lock
  與 audit marker，但 `stop_server.ps1` 會關閉 Dashboard、MCP、Cloudflare tunnel
  並撤銷 secret；下次 Ensure 僅啟動 loopback recovery-only 服務。
- 核准成功訊息依模式分流：一般模式顯示回對話輸入 `已核准`；自動模式直接續跑，
  並把同一句保留為 host 暫停時的喚醒 fallback。新增
  `jy_wait_for_autonomy_status` 與 Codex session-scoped durable-goal contract。
- 公開 `/` 改為手機與電腦的共同 Dashboard 入口，並將同一 session 分為首頁、
  核准、結果與控制三頁；舊 `/session/...`、`/approve/...`、`/autonomy/...` 保留
  303 相容導向。
- SSE 更新依頁面分流：核准頁只追 proposal，結果頁只追 run／decision／control，
  首頁追工作階段狀態；結果下拉歷史仍保留到完整關機。
- 結果頁補齊 Resume，並清楚區分 Pause、End automation（保留網站）、Emergency
  stop worker（保留網站）。首頁完整關機新增 `SHUTDOWN <session-id>` 二次確認。
- system-level 單一控制 Agent ownership lease 列入 TODO；本次未宣稱現有 DB／lock
  已能防止兩個不同 AI controller 同時發出不同有效動作。

## 2026-08-30

- 新增儀器連線例外分類：QOP/OPX/Octave/transport 無法連線時不再當成一般 crash，
  而是暫停 workflow／bounded lease；Dashboard 顯示雙語原因與恢復步驟。只有可證明
  連線從未建立、state hash 已恢復的情況才自動釋放 lock，中途失聯仍保留本機隔離。
- 新增自然語言 `恢復`／`Recover` 與 root `jy_agent.ps1 -Action Recover`。它不依賴
  MCP attachment，可協作停止 verified worker、收斂 stale workflow/lease/session，
  並關閉 verified Dashboard/MCP/tunnel；不刪 retained lock、不終止未知程序。
- 新增 [Command.md](Command.md)，統一 Codex、Claude、Cursor 的中英文進入、核准
  喚醒、暫停、繼續、自動授權停止、緊急停止、完整關機與 recovery 指令。
- Dashboard Home 加入 AI／網路斷線自助處置；網路恢復後可輸入 `結束量測`／
  `End measurement`，下次 entry 被 stale state 阻擋時可輸入 `恢復`／`Recover`。

- Worker 現在會在載入或執行量測節點之前，先把 active-state pre-run SHA-256 與
  recovery copy 路徑寫入 SQLite/audit；即使程序之後被作業系統直接終止，正式
  hardware-lock recovery 也不會因缺少 DB checkpoint 而成為無法完成的死路。
- 為既有舊版 abrupt-exit lock 加入嚴格相容復原：只有 run/request/lock identity、
  terminal/worker 檢查都通過，且 configured active state、run active state 與受保護
  recovery backup 完全同 hash 時，才顯示現場 recovery form。仍需實體檢查、操作人
  身分與完整確認句；不會自動刪除 lock。
- 修正 recovery-only 的暫時 `remote_provider=local` 被下一次正常 Ensure 當成永久
  偏好重用。Retained lock 存在時仍只允許 loopback recovery；正式復原後的正常進入
  預設恢復 token-protected Cloudflare Dashboard，且 recovery mode 改變會強制重啟
  MCP/Dashboard process 以載入正確環境。

## 2026-08-29 (English)

- Persisted each full-shutdown request before worker stop delivery. Workers now
  consume authenticated stop files and write exit receipts/termination causes;
  Windows `CTRL_BREAK_EVENT` is no longer used.
- Retained locks now enter recovery quarantine after every worker exits. Services,
  tunnel, and secrets still close; the next Ensure is loopback recovery-only.
- Added mode-aware post-approval text, `jy_wait_for_autonomy_status`, and the
  Codex session-scoped durable-goal contract with `已核准` as a host-resume fallback.
- Made `/` the common phone/desktop entry and split one session into Home,
  Approval, and Results & controls. Legacy routes remain compatibility redirects.
- Added page-specific event refresh, browser Resume, explicit keep-site semantics
  for automation stop/emergency stop, and an exact second confirmation for full
  shutdown. Added system-level single-controller ownership to TODO only.

## 2026-08-28

- 修正 02a edge-feature retry 與 fixed multiplex workflow 衝突：完整 bounded lease
  會把單一 qubit retry 正規化成完整 workflow targets；子群 lease 在硬體啟動前
  拒絕，不再把既有 lease 誤標成 halted。
- 公開 Dashboard 新增一次性 10 分鐘裝置配對與每裝置 12 小時 secure cookie；
  桌面、手機憑證彼此獨立，可從本機 operator console 個別撤銷。
- halted Dashboard 新增獨立「完整結束量測」控制；安全等待 active run 後停止
  workflow，延遲呼叫受驗證的 `stop_server.ps1`，並在 helper 失敗時顯示可重試錯誤。
- 新增 `http://127.0.0.1:8765/operator` 本機操作頁，整合手機配對、device 管理與
  正式 hardware-lock inspection/recovery。Recovery 仍要求所有軟體檢查、實體
  attestation、精確確認句，並留下 archive、receipt 與 audit；queued shutdown
  會在 recovery 成功後自動繼續。
- 新增針對 fixed-target autonomy、裝置配對／replay／撤銷、active-run shutdown、
  retained-lock gate、dispatch retry 與網頁 recovery 的離線 regression。

## 2026-08-28 (English)

- Normalized target-local 02a retries to the full fixed multiplex target set
  under a full bounded lease; subgroup leases fail before hardware without
  falsely halting the lease.
- Added ten-minute one-time device pairing, independent 12-hour browser sessions,
  and per-device revocation for the public Dashboard.
- Added a full-shutdown control that remains available after autonomy is halted,
  waits for active work, reports dispatch failures, and invokes the verified
  stop helper after the response is delivered.
- Added a loopback-only operator console for device management and formal retained
  hardware-lock recovery with physical attestation, exact confirmation, archive,
  receipt, audit events, and automatic continuation of queued shutdown.

## 2026-08-25

- 統一人機介面為單一 `/session/...` Dashboard：核准、三層控制與所有結果下拉
  紀錄位於同頁；舊 `/approve/...`、`/autonomy/...` 頁面改為相容性 redirect。
- 修正直接 MCP lease request／replacement lease 未重新綁定 session 的問題；所有
  entry path 現在都回傳同一 canonical `browser_url`，避免舊 lease 控制頁與新結果
  分裂。
- 修正 Codex `connection closed: initialize response`：STDIO launcher 改為
  protocol-only，不再於 MCP initialize 前同步依賴 Dashboard、Cloudflare、port
  或網路；網站 bootstrap 改由進入語句觸發的 `Ensure` 處理。
- 修正 root `jy_agent.ps1` 在 `JY_QUALIBRATE_PYTHON` 已設定時的 PowerShell named
  parameter 轉送，並將 Codex startup timeout 提高為 90 秒。
- 新增 root `jy_agent.ps1 -Action Doctor`：以真正 MCP client 執行 initialize／
  tools/list、驗證三種 project config，並將 STDIO、localhost HTTP、Dashboard／
  tunnel 問題分層。新增 [debugQA.md](debugQA.md) 的 Codex、Cursor、Claude 問答式
  runbook。
- STDIO service 會動態讀取後建立的 Dashboard bootstrap metadata；client 不必為
  每次新 tunnel URL 重啟 MCP。
- 完整安全停止會刪除 Dashboard access token 並清空公開 URL metadata，避免曾貼
  入聊天或 log 的 URL 在下一個 service lifetime 重新有效。
- 新增 repository-root `jy_agent.ps1` 入口，使用 `$PSScriptRoot` 與 repo 內部
  相對路徑定位 `JY_agent`，不依賴 `ASqum_QM_lab` 等最外層資料夾名稱。
- 在 repository root 加入 Codex、Claude Code、Cursor 的 MCP config、入口語句
  路由與 Codex skill；使用者不必再把 `JY_agent` 子資料夾開成 workspace。
- `setup.ps1` 現在同時驗證 root 與 nested compatibility config；根目錄
  `jy_agent.ps1 -Action Install/Ensure/Stop/Stdio` 統一委派生命週期操作。
- 更新 README、WORKFLOW 與 CLIENTS 文件，並加入 root discovery／不同 working
  directory 的離線 regression。

## 2026-08-25 (English)

- Unified the human surface on one canonical `/session/...` Dashboard containing
  approvals, all three controls, and the complete experiment dropdown. Legacy
  approval/control pages now redirect there.
- Fixed direct and replacement lease requests so they rebind the stable workflow
  session and return the same canonical `browser_url`.
- Fixed the Codex initialize-response disconnect by making STDIO startup
  protocol-only and moving Dashboard/tunnel startup to entry-time Ensure.
- Fixed named Python forwarding in the root PowerShell shim, increased the Codex
  cold-start timeout, and added a real initialize/tools-list Doctor command.
- Added dynamic Dashboard metadata discovery plus a Q&A runbook covering Codex,
  Cursor, and Claude discovery, trust, reload, HTTP, and tunnel failures.
- Full safe shutdown now removes the Dashboard token and public URL metadata so
  a previously shared link cannot become valid again.
- Added a name-independent repository-root `jy_agent.ps1` shim plus root project
  configs and phrase routing for Codex, Claude Code, and Cursor.
- Opening the repository root is now the recommended workflow; the nested
  `JY_agent` configs remain compatible.
- Extended setup validation, documentation, and offline regressions to cover
  root discovery and launcher resolution outside the current working directory.

## 2026-08-24

- 確認並以 regression 鎖定 `06 → 06b` 的預設順序。
- 所有 reset-capable node 的預設值統一為 thermal；10a、05st 與兩個 06st 不再
  預設 active。
- 新增 active-reset 證據閘門：必須是 active 07b、雙雲團 morphology 通過且
  readout fidelity `>= 0.85`，才可讓該 qubit 的後續實驗使用 active。
- active-reset 統計前必須用 active 複驗並接受 05、06、06b；服務允許這三個
  受限的 off-sequence prerequisite verification，而不改變目前下游節點。
- 依操作者更正移除 05st/06st 的逐筆 100-trace refitting、100/100 否決、重算
  mean/std 與額外分布門檻。保留 `histo_num=100`、同 reset 的基準品質及約四倍
  lifetime 時間窗；前置通過後每個統計節點只跑一次。

## 2026-08-24 (English)

- Locked the default `06 → 06b` ordering with regression coverage.
- Made thermal the default for every reset-capable node and added per-qubit
  active-reset qualification through an active 07b result with compact clouds
  and fidelity of at least 0.85.
- Active-reset statistics now require accepted active-reset verification runs of
  05, 06, and 06b.
- Superseded the 2026-08-23 all-100-refit rule: statistics retain exactly 100
  requested repetitions but run once after validated baselines, with no secondary
  per-iteration refit or all-or-nothing rejection gate.

## 2026-08-23

- Dashboard 新增繁體中文／English 全頁切換，語言偏好保留於瀏覽器。
- Prediction、Parameters、Analysis、Next action 統一改成摺疊面板；Decision 與
  Reason 保持直接顯示。
- 修正副檔名為 `.h5` 的 NetCDF snapshot 載入：依序嘗試 SciPy、h5netcdf 與
  automatic backend。
- 05/06/06b 的完成門檻提高為統計前置品質：relative uncertainty `< 0.25`、
  至少 `3.5` lifetimes；05/06b 另要求 R² `>= 0.90`。
- 05st、06st_t2star、06st_t2e 改為重新 fitting 100 個 raw iterations；必須
  100/100 通過，並核對原始 mean/std、約四倍時間窗與分布穩定性。
- 統計 state candidate 使用 JY 對全部 100 筆的重算結果，不採 percentile-trimmed
  summary；任何統計驗證失敗都抑制 state patch。
- 新增 Cursor project MCP 與 always-on project rule；Codex JY skill 改為可由
  固定進入語句自動觸發。
- `setup.ps1` 可自動尋找 `qualibrate_env` 並驗證三種 client project config。
- policy、playbook、experience 集中到 `rules/`；詳細文件集中到 `docs/`。
- 移除 WireGuard 程式分支與文件，公開手機流程統一為 token-protected Cloudflare
  Dashboard。
- Claude project MCP launcher 改用 `CLAUDE_PROJECT_DIR`，避免啟動目錄不同造成
  相對路徑失效；README 補上可由 coding agent 執行的一次性安裝方式。

## 2026-08-23 (English)

- Added whole-page Traditional Chinese/English selection and persisted browser
  preference.
- Collapsed Prediction, Parameters, Analysis, and Next action while keeping
  Decision and Reason visible.
- Added deterministic NetCDF backend fallback for `.h5` snapshots.
- Strengthened 05/06/06b statistics prerequisites and added all-100-iteration
  validation for 05st/06st nodes.
- Added Cursor project integration, automatic phrase-triggered JY skill routing,
  simpler installation, consolidated rules/docs, and removed WireGuard support.
