# Update log / 更新紀錄

## 2026-10-22

- 靜態量測歷史網站：`runtime/history_site/index.html` 列出所有晶片與每次量測工作階段；
  每個工作階段有一份實驗結果頁的靜態副本（圖、Decision、Reason、Next action），每顆
  晶片有它的歷史頁。每次分析、記錄判斷、工作階段開始或結束時更新；關閉 JY 服務後仍可
  在量測電腦開啟；有工作階段進行中時，首頁連到它的即時 Dashboard。
  `jy-agent history-site` 重建全部頁面（也會補上這個功能之前的工作階段）。
- 全自動判斷模式：hardware lock 被保留（存在但沒有 run 在跑）時不再代為決定，lease
  照舊暫停等人（之前會記成重跑，下一次啟動被 lock 拒絕而 halt）。
- 進入時的 `duration_hours` 必須是核准頁的選項（8／16／24），其他值在進入時就拒絕
  （之前會靜默變成 8）。續期沿用原本的時數。
- 首頁新增「全自動判斷模式」卡片，顯示開／關與代為決定的紀錄（核准分頁的授權卡也有）。
- 單顆 submission timeout 的提示說明全自動判斷時會自動繼續。
- 移除 `rules/policies.yaml` 中被覆蓋的重複 `analysis.statistics` 區塊（行為不變），並修正
  與數值不符的註解。
- 文件同步：README、WORKFLOW、AUTONOMY、CLIENTS、PUBLIC_DASHBOARD、debugQA、AGENTS、
  Command、PLAYBOOK、experiences 與根目錄 CLAUDE.md／AGENTS.md／Cursor 規則／skill，
  改為目前的授權時長、核准卡、全自動判斷、讀取門檻、首頁五種控制與五個分頁、密碼登入、
  強制停止後自動解鎖。TODO_LIST 依程式現況重寫。
- 補記先前未寫入的設定：06 頻率校正窗 816 ns（101 點、8 ns）；05／06／06b 最多 1000
  averages 與 101 點；active reset 資格 0.75（blob 外 ≤15%，2026-10-19）；03a 往上檢查
  f02/2（2026-10-19）。
- 更正 2026-10-20：首頁暫停後的簡易卡沒有核准碼（只選時長與判斷模式按「核准並繼續」，
  「進階」重新規劃才要核准碼）；續期卡也有時長與判斷模式選項。

- Static measurement history: `runtime/history_site/index.html` lists every chip
  and every measurement session; each session has a static copy of its
  Experiment results page (plots, Decision, Reason, Next action) and each chip
  its history. Pages are rewritten after each analysis and decision and when a
  session starts or ends; the folder opens on the lab PC with no JY service
  running, and while a session runs the homepage links to its live Dashboard.
  `jy-agent history-site` rebuilds everything (including sessions from before
  this feature).
- Full-auto judgement does not decide while a hardware lock is retained (it
  exists and no run is active); the lease pauses for the operator as before
  (it used to record a retry whose start was refused, halting the lease).
- An entry `duration_hours` must be an Approval page choice (8/16/24); any
  other value is refused at entry (it used to show as 8). Renewals keep their
  hours.
- Home has a full-auto judgement card (on/off and its log), as does the
  Approval tab's authorization card.
- The single-target submission-timeout message says full-auto resumes on its
  own.
- Removed the shadowed duplicate `analysis.statistics` block from
  `rules/policies.yaml` (no behaviour change) and fixed stale comments.
- Documentation brought up to date (README, WORKFLOW, AUTONOMY, CLIENTS,
  PUBLIC_DASHBOARD, debugQA, AGENTS, Command, PLAYBOOK, experiences, and the
  root CLAUDE.md/AGENTS.md/Cursor rule/skill): lease hours, approval cards,
  full-auto judgement, readout thresholds, five Home controls and five tabs,
  password login, forced-stop lock release. TODO_LIST rewritten against the
  code.
- Previously unlisted settings: the 06 frequency-calibration window is 816 ns
  (101 points at 8 ns); 05/06/06b take at most 1000 averages and 101 points;
  active reset qualifies at 0.75 with ≤15% off-blob shots (2026-10-19); 03a
  checks upward for f02/2 (2026-10-19).
- Correction to 2026-10-20: the simple card after a Home pause has no approval
  code (hours and judgement, then "Approve and continue"; only the folded
  re-plan needs the code); the renewal card also has the hours and judgement
  choices.

## 2026-10-21

- 06st_t2e 預設 num_averages 改為 500（原 1000），T2e 統計時間約減半。
  06st_t2e default num_averages is now 500 (was 1000), roughly halving the T2e statistics time.
- 晶片歷史合併：核准頁的表格改為 node x qubit，合併這顆 chip 所有 workflow，每格是該
  qubit 在該實驗最近一次成功的結果（workflow 編號與時間）；各 workflow 原本的表收在下方。
  每顆 qubit 的接續選項與沿用的結果改由「最近一次結果所在的 workflow」提供（不同 qubit
  可來自不同 workflow，含 02x/02c/02a 等共用實驗），記錄在 `options.qubit_sources`，
  `source_workflow_id` 為其中最近的一個。state.json 檢查改為逐 qubit：該 qubit 寫入過
  的值已不同時，只限制它最晚從第一個不符的實驗開始。已有 run 的 workflow（續期、首頁）
  也顯示同一張表並包含自己目前的結果（標 *），但不提供起點選擇；首頁新增此表。
  Chip history merged: the Approval table is node x qubit across every workflow of
  the chip (each cell the qubit's most recent pass, with workflow and time), each
  workflow's own table folded below. Each qubit resumes from, and inherits from, the
  workflow holding its most recent result (`options.qubit_sources`;
  `source_workflow_id` is the most recent of them). The state check is per qubit.
  Workflows with runs (renewal, Home) show the table with their own passes (*),
  without start choices; Dashboard Home shows it too.
- `jy_get_next_action` 對已完成（無目前 node）的 workflow 回傳 `action: "complete"`，
  不再出現 `'None' is not in list`。
  `jy_get_next_action` returns `action: "complete"` for a finished workflow instead
  of failing with `'None' is not in list`.
- 07d 選點權重改為 5:1（兩球分開程度 : off-blob 比例；原 3:1）。
  07d selection weight is now 5:1 (blob separation : off-blob share; was 3:1).

## 2026-10-20

- 07d 讀取點改由 JY 自己的 raw-shot 分析決定（不再用 node 的 fidelity 選點）：在
  off-blob ≤ 5%（同 07b）的點中，取 3 × 兩球重疊誤差 + 1 × off-blob 比例最小者
  （`analysis."07d".separation_weight` / `outlier_weight`）；fidelity 只作報告。
  07d now picks the readout point from JY's raw-shot score: lowest
  3 x blob-overlap error + 1 x off-blob share among points within 5% off-blob.
- 06 Ramsey 的 max wait 改為 16 + 400n ns（原為 16 + 500n）；816 ns 頻率校正窗因此
  落在格點上，不再需要例外。06 預設 max wait 改為 816 ns。
  06 Ramsey max wait is now 16 + 400n ns (was 16 + 500n), so the 816 ns
  calibration window is on the grid; the 06 default max wait is 816 ns.
- 03a window shift：粗掃的候選線在已決策的低功率 refinement／fine 掃描中消失
  （robust SNR 低於門檻）時視為多光子線，自動解除候選，可立即換窗。
  A coarse 03a candidate whose line vanishes (SNR below the floor) in a decided
  low-power scan is dismissed, so the qubit may shift window at once.
- 核准時長與全自動判斷：每份 lease 核准都選擇 8／16／24 小時（從核准起算）與是否
  開啟「全自動判斷模式」。首頁新增 `停止核准`：目前實驗跑完後結束本授權，Approval
  頁重新出現簡易核准，等候中的 agent 會跟到新的 lease（`superseded_lease_id`）。
- 簡易卡與完整卡：只有全新開始（上次是結束量測或已完成）才顯示完整選項卡；續期、
  首頁暫停後繼續（重新規劃收在「進階」）、Recover 之後的接續都只問時長、判斷模式與
  核准碼。Recover 關閉的 workflow 在 `進入JY自動量測模式` 或 `繼續JY自動量測模式` 時
  由新的接續 workflow 從各 qubit 停下處繼續（沿用之前的結果與選項）；
  `active_qubit_names` 改變時改回完整卡。workflow 新增 `end_kind` 記錄由誰結束。
- 全自動判斷模式：不明的 worker／分析錯誤自動重跑，同一 node/qubit 重跑 2 次後標為
  incomplete 並繼續；單一 qubit 仍 submission timeout 時標為 incomplete 並繼續；儀器
  斷線在 QOP 恢復回應後自動繼續（每 60 秒探測一次）；每份 lease 最多 10 次，用完則
  照舊暫停等人。三個 halt 條件與到期仍需人工。紀錄顯示在首頁授權卡。
- 關閉殘留的 quantum machine：`jy_agent/qm_cleanup.py`（由
  `00x_close_specific_qm.ipynb` 改寫）只關閉所有 port 都屬於目前 state.json 的 QM，
  先 cancel 其 running／queued job；只部分重疊的 QM 只回報不關。全自動判斷在重跑或
  自動繼續前會先執行；MCP 工具 `jy_close_stale_qm`（可 `dry_run`）。執行中的 run
  或 hardware lock 存在時拒絕。
- 流水線影子模式：每次啟動 run 與記錄決策時，記下確定性規劃器會怎麼做；
  `jy_get_autopilot_shadow_report` 依 node 回報吻合率、可完全決定的比例、儀器使用
  比例與 run 之間的空檔。尚未由 server 自動執行。
- Watchdog 對同一個失敗 run 只記一次 submission-timeout／分析失敗／可恢復例外事件
  （之前每 2 秒重記一次，累積 17117 筆）。
- 每顆 qubit 起點（2026-10-15 功能）修正：一顆一顆量（per-qubit pass）時，qubit 的
  可選起點包含它在自己 pass 中已量到的實驗（例如 workflow 已回到 03a 量 q2 時，q1
  仍可從 05 繼續）；之前只算到 workflow 當時所在的節點。讀取 IF 未設定的 qubit 設為
  「不量」時，02 不再被強制執行；此時 02 是預設勾選的一般選項，只要量測其中任一顆，
  server 仍會要求 02。
- 是否為「由使用者結束」也看 audit log：有 Dashboard 完整關機或結束指令紀錄的
  workflow，即使之後被 Recover 關閉（例如保留了 hardware lock），下次進入仍顯示完整
  選項卡，不會被當成接續。

- Lease hours and full-auto judgement: every lease approval chooses 8/16/24 h
  (from the approval) and whether full-auto judgement is on. Home gains
  `Stop authorization`: the current run finishes, the lease ends and the
  Approval page asks again; the waiting agent follows the new lease
  (`superseded_lease_id`).
- Simple and full cards: only a fresh start (the last workflow was ended or
  completed) shows the full lease card; a renewal, a continue after a Home
  pause (re-plan folded under "Advanced") and a continuation after a Recover
  ask only hours, judgement and the code. A workflow closed by Recover is
  continued by a new workflow on entry or the continue phrase, each qubit
  where it stopped; changed `active_qubit_names` bring back the full card.
  Workflows record who ended them (`end_kind`).
- Full-auto judgement: an unknown worker or analyzer error is retried; a
  node/qubit that failed twice more is marked incomplete; a single target that
  still times out on submission is marked incomplete; an instrument outage
  resumes once the QOP answers (probed every 60 s); at most 10 decisions per
  lease, then the old pause. The halt conditions and the expiry stay human.
  The log is on the Home authorization card.
- Stale quantum machine cleanup: `jy_agent/qm_cleanup.py` (from
  `00x_close_specific_qm.ipynb`) closes only a QM whose every port belongs to
  the active state.json, cancelling its running/queued job first; an
  overlapping QM is reported, not closed. Full-auto judgement runs it before a
  retry or an automatic resume; MCP tool `jy_close_stale_qm` (with `dry_run`).
  Refused while a run is active or the hardware lock exists.
- Pipeline shadow mode: each run start and decision records what the
  deterministic planner would have done; `jy_get_autopilot_shadow_report`
  gives the match rate per node, the share it could fully determine, the
  instrument share and the gaps between runs. The server does not act on it
  yet.
- The watchdog logs a failed run's submission-timeout, analysis-failure and
  recoverable-exception event once (it re-logged every 2 s: 17 117 events).
- Per-qubit plan fixes (the 2026-10-15 feature): in a one-qubit-at-a-time
  pass, a qubit's choices include the nodes it reached in its own pass (q1 can
  resume at 05 while the workflow is back at 03a for q2); before, they stopped
  at the workflow's node. A qubit with an unset readout IF that is set to "not
  measured" no longer forces 02; 02 is then an ordinary box, ticked by
  default, and the server still requires it while any such qubit is measured.
- Who ended a workflow is also read from the audit log: one with a Dashboard
  full shutdown or a shutdown phrase stays an operator end even when a later
  Recover closed it (for example over a retained hardware lock), so the next
  entry shows the full lease page, not a continuation.

- 強制停止後的硬體鎖：worker 在寬限期後被終止（`force_stopped`／
  `forced_after_grace`）時，server 確認 worker 已結束後，用該 run 的
  state.json／wiring.json 關閉本晶片殘留的 QM，記錄收據
  `runtime/recovery/<run>-qm-cleanup.json`（10 分鐘內有效），state 未變時補記
  `active_state_hash_after`，並沿用 recovery 的流程（封存＋收據）自動釋放鎖。本晶片
  QM 仍開著、有重疊或無法讀取的 QM、清理失敗或 state 改變時，維持原本的鎖與隔離。
  新證據模式 `forced_stop_checkpoint`；本機 operator console 新增「Close this chip's
  stale quantum machine」，並列出 ours／overlapping／unrelated。OPX1000 為共用：
  與本晶片無共用 port 的 QM（其他使用者）一律忽略，不關閉、不阻擋。

- Hardware lock after a forced stop: when a worker is terminated after the
  grace period (`force_stopped` / `forced_after_grace`), the server, once the
  worker is verified dead, closes this chip's stale QM using the run's
  state.json/wiring.json, records a receipt
  (`runtime/recovery/<run>-qm-cleanup.json`, valid 10 minutes), records
  `active_state_hash_after` when the state is unchanged, and releases the lock
  through the recovery code path (archive + receipt). A machine of ours still
  open, an overlapping or unreadable machine, a failed cleanup or a changed
  state keeps the lock and quarantine as before. New evidence mode
  `forced_stop_checkpoint`; the loopback operator console gains "Close this
  chip's stale quantum machine" and lists ours/overlapping/unrelated. The
  OPX1000 is shared: a machine with no port on this chip (another user's) is
  ignored, never closed and never a blocker.

- 新指令 `繼續JY自動量測模式`（`繼續 JY 自動量測模式`、`Continue JY automatic
  measurement mode`，MCP 工具 `jy_continue_autonomy_mode`）：自動授權到期（或已
  結束）而 workflow 仍開著時建立續期，沿用上一份的 targets、節點、上限、自動寫入
  狀態、中止條件、選項與時數；Approval 頁只顯示「續期 N 小時，從 <qubit> / <node>
  繼續」、沿用範圍與核准碼，核准後同一個 workflow 立即繼續，不回到起點。lease 暫停
  時直接恢復；仍有效時只回報。lease 到期時 handoff 改為回傳此指令。
- `已重新連線` / `Reconnected`：在 `/mcp` → `jy_bringup` → Reconnect 之後輸入，agent
  確認新的 MCP process 後繼續。

- New phrase `繼續JY自動量測模式` (aliases `繼續 JY 自動量測模式`, `Continue JY
  automatic measurement mode`; MCP tool `jy_continue_autonomy_mode`): an expired
  or ended lease of the open workflow is renewed with its targets, nodes, limits,
  auto-commit statuses, halt conditions, options and hours. The Approval page
  shows only "renew N h, continue from <qubit> / <node>", the copied scope and
  the approval code; approval continues the same workflow at once. A paused
  lease resumes; an active one is only reported. An expired lease's handoff now
  asks for this phrase.
- `已重新連線` / `Reconnected`: sent after `/mcp` -> `jy_bringup` -> Reconnect;
  the agent verifies the new MCP process and continues.

## 2026-10-15

- 06 Ramsey：detuning [MHz] × (max wait − min wait) [ns] 必須剛好等於 4000
  （20000 ns → 0.2 MHz），不再容許 25% 誤差；min wait 固定 16 ns，max wait 必須是
  16 + 500 的倍數（例如 20016）。不給 detuning 時由 policy 自動算出。預設 max wait
  40000 → 40016。
- 06st_t2star 的 JY refit 改用帶振盪的衰減模型（damped cosine）擬合 Ramsey 條紋。
- 02c 讀取功率改為 dressed 上限 −5 dB（原 −3 dB）；07d amplitude 掃描改為
  0.2–1.778（+5 dB 回到 dressed 上限），27 點，07d 仍為 single pass。
- Readout length：03a 之前所有目標 qubit 設為 10000 ns（fresh defaults 也改為
  10000），qubit 第一次進 03a 時改為 4000 ns；上限截斷改為 10000 ns，depletion
  維持 2000 ns。由 `jy_request_readout_length_normalize` 執行，server 會在 run 前要求。
- 06 Ramsey 分段：每顆 qubit 第一次 06 一律用 max wait 1016 ns 只做頻率校正（不 commit
  T2*）；接著在 1016 ns 重複到兩個 fringe 都對上 detuning 才加長；加長時依可信的
  T2* 估計取約 4 倍，否則依 1016 → 4016 → 16016 → 40016 → 100016 ns 階梯。上限
  100016 ns。前一次的頻率修正沒 commit 前不能跑下一次 06。04 → 03a 回頭後重新開始。
- 自動量測核准頁面改為每顆 qubit 各自選擇：不量、從頭，或從上次量到的實驗繼續
  （已通過的 node 或第一個沒通過的 node）；最上面「全部設為」一次套用。整體從最早的
  起點開始，較晚的 qubit 只沿用自己之前的結果，走到自己的起點才加入。不量的 qubit
  從 workflow 與 lease 排除。進入自動模式時不再指定 targets。

- 06 Ramsey: detuning [MHz] x (max wait - min wait) [ns] must equal 4000 exactly
  (20000 ns -> 0.2 MHz), no longer within 25%; min wait is fixed at 16 ns and max
  wait is 16 + a multiple of 500 (e.g. 20016). An omitted detuning is derived.
  Default max wait 40000 -> 40016.
- The 06st_t2star JY refit fits a damped cosine to the Ramsey fringes.
- 02c readout power is the dressed limit minus 5 dB (was 3 dB); the 07d
  amplitude sweep is 0.2-1.778 (+5 dB back to the limit), 27 points, still
  single pass.
- Readout length: 10000 ns for every target before 03a (fresh defaults too),
  4000 ns from a qubit's first 03a; the cap is 10000 ns and depletion stays at
  2000 ns. jy_request_readout_length_normalize applies it; the server requires
  it before the run.
- 06 Ramsey stages: a qubit's first 06 runs at 1016 ns and only corrects the
  frequency (no T2*); it stays at 1016 ns until both fringes sit on the
  detuning, then lengthens to about 4 x a reliable T2* or along
  1016 -> 4016 -> 16016 -> 40016 -> 100016 ns (the cap). The next 06 waits for
  the previous frequency correction; a 04 -> 03a return starts over.
- The automatic-measurement approval page picks each qubit: not measured,
  from the beginning, or resume where it stopped (a cleared node or the first
  one it did not clear); a Set-all row applies one choice. The workflow starts
  at the earliest choice, a later qubit carries only its own earlier results
  and joins at its node, and a skipped qubit leaves the workflow and lease.
  Automatic mode is entered without targets.

- Token 用量永久保存：每筆 Claude Code API 呼叫的數字（時間、模型、各類 token、
  工具類別；不含對話內容）存進 JY 資料庫的 `usage_calls`，Claude Code 刪掉舊對話
  紀錄後仍可查詢。開啟 Token 用量頁與進入量測模式（背景執行）時更新。
- Token 用量頁最上方新增「各晶片總用量」：每顆晶片的第一次／最近一次日期、
  workflow 數、實驗數、總 token 與估計費用，目前 session 的晶片以底色標示。
- 實驗結果頁的「晶片歷史結果」整張卡片與每個實驗節點預設收起；瀏覽器會記住展開
  了哪些，自動刷新後維持。

- Token usage is kept for good: each Claude Code API call's numbers (time,
  model, token kinds, tool kinds; no conversation text) are saved in the JY
  database table `usage_calls` and stay after Claude Code deletes old
  transcripts. They update when the Token usage page opens and, in the
  background, on measurement-mode entry.
- The Token usage page starts with Usage by chip: first and latest date,
  workflows, runs, total tokens and estimated cost per chip; this session's chip
  is highlighted.
- The Chip history card on the results page and each of its nodes start folded;
  the browser remembers what was opened across automatic refreshes.

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
