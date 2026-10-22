# JY commands / JY 指令

這些是 Codex、Claude Code 與 Cursor 在 repository root 共用的自然語言指令。
英文大小寫以表格中的寫法為準；中文可使用列出的無空格別名。所有量測、停止與
復原仍受 server policy、worker identity 與 hardware-lock 檢查約束。

These natural-language commands are shared by Codex, Claude Code, and Cursor
when the repository root is open. Measurement, stop, and recovery actions remain
subject to server policy, verified worker identity, and hardware-lock checks.

## 對話指令 / Conversation commands

| 用途 / Purpose | 中文 / Chinese | English | 行為 / Behavior |
|---|---|---|---|
| 一般量測 / Conversational measurement | `進入 JY 量測模式`、`進入JY量測模式` | `Enter JY measurement mode` | 啟動或重用服務；每次 run/state change 個別核准。新 workflow 的 targets 取自 `state.json` `active_qubit_names`，`multiplexed` 預設為 true。不必在對話寫 `target` 或 `multiplex`。 / Starts or reuses services; every run/state change is approved separately. New workflows use `state.json` `active_qubit_names` and default `multiplexed=true`; do not type `target` or `multiplex` in chat. |
| 有限自動量測 / Bounded automatic measurement | `進入 JY 自動量測模式`、`進入JY自動量測模式` | `Enter JY automatic measurement mode` | 一次核准有限 lease：在 Dashboard 核准時選 8／16／24 小時（預設 8，上限 24，從核准起算）與是否開啟「全自動判斷模式」，再於核准範圍內自動推進。進入時的 `duration_hours` 只能是 8、16 或 24，僅作預選，以頁面選擇為準。新 workflow 同樣使用 `active_qubit_names` 且 `multiplexed` 預設 true。 / Approves one bounded lease: the Dashboard approval chooses 8, 16 or 24 hours (default 8, maximum 24, counted from the approval) and whether full-auto judgement is on, then JY advances within scope. An entry `duration_hours` may only be 8, 16 or 24 and merely preselects the page; the page decides. New workflows also use `active_qubit_names` and default `multiplexed=true`. |
| 續期／繼續自動量測 / Continue automatic measurement | `繼續JY自動量測模式`、`繼續 JY 自動量測模式` | `Continue JY automatic measurement mode` | 接著同一個 workflow：lease 到期（或已結束）時建立續期，沿用上一份的 targets、節點、上限、自動寫入狀態、中止條件、選項與時數；Approval 頁只顯示「續期授權，從 <qubit> / <node> 繼續」、時長與判斷模式選項與核准碼，核准後立即生效，不會重新開 workflow 或回到起點。lease 暫停中則直接恢復；仍有效則只回報狀態。沒有開著的 workflow 時：若上一個 workflow 是被 Recover 關閉的，會建立接續 workflow（簡易卡）；否則請改用 `進入JY自動量測模式`。與進入相同：Ensure、呼叫 `jy_continue_autonomy_mode`，回傳裸 `browser_url` 與 `operator_handoff.chat` 後結束回合。 / Continues the same workflow: an expired (or ended) lease gets a renewal copying the previous targets, nodes, limits, auto-commit statuses, halt conditions, options and hours; the Approval page shows only "renew, continue from <qubit> / <node>", the hours and judgement choices and the approval code, and approving activates it at once without a new workflow or a restart. A paused lease resumes; an active one is only reported. With no open workflow: after a Recover it starts a continuation workflow (simple card); otherwise use `Enter JY automatic measurement mode`. Same fast entry: Ensure, call `jy_continue_autonomy_mode`, reply with the bare `browser_url` and `operator_handoff.chat`, end the turn. |
| 喚醒已核准工作 / Wake an approved session | `已核准` | `Approved` | 只在 agent host 暫停回合時使用；它不代替網頁核准。 / Used only when the agent host paused; it never substitutes for Dashboard approval. |
| 暫停自動排程 / Pause automatic scheduling | `暫停 JY 自動量測` | `Pause JY automatic measurement` | 目前 run 完成後不再排新實驗，保留 lease 與網站。 / Stops new scheduling after the current run; keeps lease and site. |
| 繼續自動排程 / Resume automatic scheduling | `繼續 JY 自動量測` | `Resume JY automatic measurement` | 恢復同一份未過期、paused lease（`繼續JY自動量測模式`、`恢復量測` 也可）。 / Resumes the same unexpired paused lease (the continue phrase and `恢復量測` also do). |
| MCP 重新連線後 / After an MCP reconnect | `已重新連線` | `Reconnected` | 在 `/mcp` → `jy_bringup` → Reconnect 之後輸入；agent 先確認新的 MCP process（例如新工具已出現、`jy_get_status` 可用），再接著原本的工作。 / Sent after `/mcp` → `jy_bringup` → Reconnect; the agent verifies the new MCP process (new tools listed, `jy_get_status` answers) and continues. |
| 儀器檢查後恢復量測 / Resume after instrument inspection | `恢復量測`、`恢复量测` | `Resume measurement` | 恢復同一 workflow、lease 與 Dashboard session；目前 node 以新 run-id 從頭執行，不會接續已中斷的 Python call stack。lease 開啟全自動判斷模式時，QOP 探測（每 60 秒）成功後 server 會自動恢復，不必輸入。 / Restores the same workflow, lease, and Dashboard session; the current node restarts as a new run rather than resuming a Python stack. Under full-auto judgement the server resumes by itself after a successful QOP probe (every 60 s). |
| 結束自動授權 / End automation only | `結束 JY 自動授權` | `End JY automatic authorization` | 撤銷 lease 並安全停止 active worker，但保留 workflow 與網站。 / Revokes the lease and safely stops an active worker while keeping the workflow and site. |
| 緊急停止 worker / Emergency-stop worker | `緊急停止 JY worker` | `Emergency stop JY worker` | 先協作停止，逾時後只終止驗證過的 worker；網站保留，lock 可能隔離。 / Requests cooperative stop, then terminates only the verified worker after grace; keeps the site and may quarantine the lock. |
| 完整結束 / Full shutdown | `結束量測`、`停止量測`、`退出 JY 量測模式`（以及設定內的 JY 空格別名） | `End measurement`、`Stop measurement`、`Exit JY measurement mode` | 停止 run/workflow，關閉 Dashboard、MCP HTTP service 與 tunnel。 / Stops run/workflow and closes Dashboard, HTTP MCP service, and tunnel. |
| 安全恢復到關閉狀態 / Recover to a closed state | `恢復`、`恢复` | `Recover` | 不依賴已掛載的 JY MCP；收尾 stale workflow/process，再關閉可驗證的 JY 服務。絕不自動刪除 retained hardware lock。 / Does not require the attached JY MCP; converges stale workflow/process state and closes verified JY services. It never auto-deletes a retained lock. |

`停止 JY 自動量測` 屬於完整結束語句；若只想保留網站，請使用「結束 JY 自動授權」
或首頁的對應按鈕。

`停止 JY 自動量測` is a full-shutdown phrase. To keep the site, use
`End JY automatic authorization` or the matching Home button.

## 對話回傳契約 / Operator reply contract

指令成功後，對話回覆必須固定，不可改寫成其他說明。量測模式仍開著時，給使用
者的回覆最後都要有結束量測提示；若本回合要暫停，還要有「需要使用者做什麼」
與「完成後回傳」。優先貼上 MCP 回傳的 `operator_handoff.chat`，必須維持三行
markdown list，不可收成同一段。

After a recognized command succeeds, the chat reply is fixed. While measurement
mode is still open, every operator-facing reply ends with the shutdown hint. A
paused turn also includes what to do next and which phrase to send afterwards.
Copy `operator_handoff.chat` as a three-item markdown list; never join it into
one paragraph.

| 時機 / When | 對話固定回覆 / Fixed chat reply |
|---|---|
| 成功進入一般或自動量測模式 / Successful entry | 只回傳 entry 最外層 `browser_url`（Cloudflare 裸網址）。若本回合要暫停，接著貼 `operator_handoff.chat`。 / Return only the top-level `browser_url`. If the turn must pause, append `operator_handoff.chat`. |
| 使用者輸入 `已核准` / `Approved` | 下一則回覆第一行必須是 `已核准`，然後繼續作業。 / The next reply’s first line is exactly `已核准`, then continue. |
| 等待 Dashboard 核准（run、state、lease）而對話暫停 / Waiting for Dashboard approval | 三行 list：`需要使用者做什麼`、`完成後回傳: 已核准`、結束量測提示 / Three-line list ending with `已核准` |
| 儀器連線失敗、排程已暫停 / Instrument connectivity pause | 三行 list：`完成後回傳: 恢復量測`。全自動判斷模式開啟時不暫停對話：繼續用 `jy_wait_for_autonomy_status` 等待，QOP 探測成功後 server 自動恢復。 / Under full-auto judgement do not pause the turn; keep waiting, the server resumes after a successful QOP probe. |
| 未登錄的 worker crash 或 JY 分析例外暫停 lease（全自動判斷關閉、本 lease 10 次自動判斷已用完，或保留 hardware lock） / Unrecognized worker crash or analyzer exception paused the lease (full-auto off, its 10 decisions used up, or a hardware lock retained) | 三行 list：`完成後回傳: 繼續 JY 自動量測`（`恢復量測` 也可；同一 lease 續跑，不需重新核准） / Three-line list ending with `繼續 JY 自動量測` (`恢復量測` is also accepted); the same lease resumes without a new approval |
| 殘留流程／服務，無法進入 / Stale lifecycle blocks entry | 三行 list：`完成後回傳: 恢復` |
| Ensure 回 `recovery_only`（硬體鎖隔離） / recovery_only lock quarantine | 三行 list：`完成後回傳: 恢復` |
| 自動量測已被暫停 / Autonomy paused | 三行 list：`完成後回傳: 繼續 JY 自動量測` |
| 自動授權已到期（lease expired） / Lease expired | 三行 list：`完成後回傳: 繼續JY自動量測模式`（續期同一 workflow；簡易卡只選時長、全自動判斷模式並輸入核准碼） / Three-line list ending with `繼續JY自動量測模式` (renews the same workflow; the simple card asks only the hours, full-auto judgement and the approval code) |
| Host 暫停但仍在授權內續跑 / Host paused while authorized work continues | 三行 list：`完成後回傳: 已核准` |
| 完整結束成功 / Full shutdown succeeded | 說明量測模式已關閉。不要再要使用者回傳量測指令，除非他們要重新進入。 / Report that measurement mode is closed. Do not request a continue phrase unless they want to re-enter. |

量測模式仍開著、對話要暫停時，固定貼成三行 list，例如：

```text
- 需要使用者做什麼: 開啟對話中的 Dashboard 網址並登入，到 Approval 頁核准。
- 完成後回傳: 已核准
- 如需結束量測，請於網頁首頁結束量測後再對話輸入「結束量測」。
```

不可把這三行收成同一段。每一則給使用者的回覆最後一項都是結束量測提示。

`已核准` 只喚醒已暫停的 agent 回合；它不能代替 Dashboard 上的核准動作。

`已核准` only wakes a paused agent turn. It never substitutes for Dashboard approval.

## 完整結束量測流程 / Full shutdown flow

完整結束只有一種結果：安全停止 active run、永久結束 workflow，並關閉
Dashboard、MCP HTTP service 與 tunnel。流程不因一般／自動模式而改變。改變的是
**入口**：agent 還在線時，對話裡的 `結束量測` 本身就會走完整關機；對話已暫停、
AI 斷線、或人在手機上時，必須先用網頁，因為當下沒有 agent 可執行 Stop。

Full shutdown has one outcome: stop the active run, permanently end the
workflow, and close the Dashboard, MCP HTTP service, and tunnel. Conversational
and automatic mode share that outcome. Only the entry point changes: a live
agent turn can execute `結束量測` itself; a paused or disconnected turn needs
the Dashboard first.

| 情景 / Situation | 正確做法 / Correct action | 不是這個 / Not this |
|---|---|---|
| 對話仍在線，要全部關掉 / Agent turn is live | 直接輸入 `結束量測`；agent 會停 run、關 workflow 與服務。 / Send `結束量測`; the agent stops the run and services. | 不必先上網頁。 / Web-first is not required. |
| 對話已暫停、AI 斷線、或只用手機 / Paused turn, disconnected AI, or phone-only | 先到 Dashboard 首頁輸入 `SHUTDOWN <session-id>` 結束量測；對話恢復後再輸入 `結束量測` 確認。 / Home page full shutdown first; then send `結束量測` to verify. | 只在暫停的對話打 `結束量測` 不會立刻關機。 / Typing it into a paused chat does not shut services down by itself. |
| 只想停自動排程、保留網站 / Pause automation, keep site | `暫停 JY 自動量測` 或首頁 Pause / Home Pause | 不是 `結束量測` |
| 只想撤銷自動授權、保留網站 / End lease, keep site | `結束 JY 自動授權` 或首頁 End automation / Home End automation | 不是 `結束量測` |
| 只要立刻停 worker、保留網站 / Emergency-stop worker, keep site | `緊急停止 JY worker` 或首頁 Emergency stop / Home Emergency stop | 不是 `結束量測` |
| 進不去、殘留服務或 workflow / Stale state blocks entry | 輸入 `恢復` | 這是收尾殘留狀態，不是正常量測結束。 / Recovery, not a normal end. |
| 關機後留下硬體鎖隔離 / Retained lock after shutdown | 量測電腦本機 operator console 現場確認；必要時再 `恢復` | 不能從手機 Dashboard 解鎖。grace 後的強制停止，server 會先自動關閉本晶片殘留的 QM，清理乾淨且 state 未變時自動釋放 lock；若 lock 仍在且本晶片的 QM 仍開著，先按「Close this chip's stale quantum machine」（只關本晶片的 QM，其他使用者的 QM 一律忽略），之後才會出現復原表單。 / Phone Dashboard cannot recover the lock. After a forced stop past the grace period the server first closes this chip's stale QM itself and releases the lock when the cleanup is clean and the state is unchanged; if the lock remains and this chip's QM is still open, press "Close this chip's stale quantum machine" (closes only this chip's QM; other users' machines are ignored), then recovery is offered. |

`停止 JY 自動量測` 也是完整結束語句，效果與 `結束量測` 相同。

`停止 JY 自動量測` is also a full-shutdown phrase and matches `結束量測`.

## Dashboard 控制 / Dashboard controls

- Home：`結束量測並關閉所有 JY 服務`，需輸入頁面顯示的
  `SHUTDOWN <session-id>`，以及 Pause、Resume、`停止核准`（目前實驗跑完後結束
  本授權，Approval 頁重新出現簡易核准以重設時長與判斷模式）、End automation
  （保留網站）、Emergency stop worker（保留網站）。首頁也顯示全自動判斷模式
  開／關與「全自動判斷紀錄」（JY 代為決定的事項）；沒有待核准項目時，Approval 頁
  的授權卡片也顯示同樣內容。
- Approval：核准待處理的 run、state change 或 bounded lease。每份 lease 核准都要
  選擇自動核准時長（8／16／24 小時，從核准起算）與是否開啟「全自動判斷模式」。
  只有全新開始（上次是結束量測或完成）才顯示完整選項卡（各 qubit 計畫、選擇性
  節點、量測方式、flux、預覽、時長與判斷模式、同意與核准碼）；續期與 Recover
  之後的接續只顯示簡易卡（時長、判斷模式、核准碼）。首頁暫停後的卡只選時長與判斷
  模式，按「核准並繼續」即可，不需核准碼，時長從現在起算；收在「進階」裡的「重新
  規劃各 qubit 起點」才需要核准碼。接續卡可按「改用完整選項重新設定」。若
  `active_qubit_names` 已改變，Recover 之後也會顯示完整選項卡。
- 實驗結果：最上方是此晶片跨 workflow 的成功結果（晶片歷史，關機後仍保留靜態
  副本），其下是結果摘要（每次實驗的成功 qubit 與結果圖縮圖）與該次服務的完整
  實驗下拉歷史。
- 歷史網站：`JY_agent/runtime/history_site/index.html` 是靜態網頁，列出所有晶片與
  量測 session 及其結果，完整關機後仍保留；session 進行中時可從它連到 live
  Dashboard。
- Token 用量：最上方是此晶片的開始時間與所有 workflow 的總用量；可選整個晶片、
  本 session、今天、近 7 天、本月或自訂起迄時段計算費用；實驗依日期分組，每列
  有開始時間、token、模型與估計費用。只讀取本機 Claude Code 對話紀錄；
  Codex／Cursor 用量不計入。
- 使用說明：操作方式、量測順序與各節點的成功範例圖。

- Home: full shutdown, protected by the displayed `SHUTDOWN <session-id>` text,
  plus Pause, Resume, `Stop authorization` (the current run finishes, the lease
  ends and the Approval page asks again for the hours and the judgement mode),
  End automation (keep site), and Emergency stop worker (keep site). Home also
  shows whether full-auto judgement is on and its log of what JY decided on the
  operator's behalf; with nothing pending, the Approval page's authorization
  card shows the same.
- Approval: approves a pending run, state change, or bounded lease. Every lease
  approval chooses the hours (8/16/24, counted from the approval) and whether
  full-auto judgement is on. Only a fresh start (the last workflow was ended or
  completed) shows the full card (per-qubit plan, optional nodes, measurement
  mode, flux, preview, hours and judgement, agreement and approval code); a
  renewal and a continuation after a Recover show the simple card (hours,
  judgement, approval code). After a Home pause the card asks only the hours
  and the judgement, with "Approve and continue" and no approval code (the
  hours count from then); only the re-plan form folded under "Advanced" needs
  the code. The continuation card can switch to the full options. Changed
  `active_qubit_names` also bring back the full card after a Recover.
- Experiment results: the chip history (successful results for this chip across
  workflows; a static copy survives shutdown), then the result summary
  (successful qubits and plot thumbnails per experiment) and the
  service-lifetime experiment history.
- History site: `JY_agent/runtime/history_site/index.html` is a static page that
  lists every chip and measurement session with its results; it survives full
  shutdown and links to the live Dashboard while a session runs.
- Token usage: the chip's start time and its total across all workflows at the
  top; a time range (whole chip, this session, today, last 7 days, this month, or
  a custom from/to) for the cost of any period; experiments grouped by date, each
  with start time, tokens, model, and estimated cost. Read only from local
  Claude Code transcripts; Codex and Cursor usage is not counted.
- Guide: how to operate JY, the measurement order, and an example plot per node.

## 一次性維護指令 / One-line maintenance commands

正常使用不必開終端機；下列是本機安裝或排錯 fallback，均從 repository root 執行。
Normal use does not require a terminal. These are local installation/debug
fallbacks, run from the repository root.

```powershell
# 一次安裝 / one-time install
.\jy_agent.ps1 -Action Install

# MCP/client 設定與真實 handshake 診斷 / diagnose client config and handshake
.\jy_agent.ps1 -Action Doctor

# 啟動或重用本機服務（通常由 agent 自動執行） / ensure services (normally automatic)
.\jy_agent.ps1 -Action Ensure

# 完整停止服務 / stop services
.\jy_agent.ps1 -Action Stop

# 與自然語言「恢復 / Recover」相同的安全收尾 / safe close-state recovery
.\jy_agent.ps1 -Action Recover

# 僅限量測電腦、實體確認後的 retained-lock recovery
# retained-lock recovery only after local physical inspection
.\jy_agent.ps1 -Action RecoverLock -RunId <run-id>
```

MCP 工具 `jy_close_stale_qm` 只關閉本晶片殘留的 QM 與其 job（run 中斷後下一個 run
開不了 port 時使用）；有 active run 或 hardware lock 存在時會拒絕，`dry_run` 只列出
開著的 QM。其他使用者的 QM 一律不動。lock 保留時改用本機 operator console 的
「Close this chip's stale quantum machine」。

The MCP tool `jy_close_stale_qm` closes only this chip's leftover QM and its job
(use it when a broken-off run leaves the next run unable to open the ports). It
is refused while a run is active or the hardware lock exists; `dry_run` only
lists the open machines. Other users' machines are never touched. With a
retained lock, use "Close this chip's stale quantum machine" on the local
operator console instead.

若 AI App 斷線但 Dashboard 仍可用，優先用 Dashboard 暫停或完整關機。網路恢復後
可在原對話輸入 `結束量測`／`End measurement` 再確認一次。若重新進入因殘留服務、
workflow 或設定失敗，輸入 `恢復`／`Recover`；只有軟體無法證明硬體 safe state 的
情況才會留下 local recovery quarantine。

If the AI app disconnects while the Dashboard remains reachable, use Dashboard
pause or full shutdown. After connectivity returns, send `End measurement` in
the original conversation to verify closure. If a later entry fails because of
stale services, workflow state, or configuration, send `Recover`. Local recovery
quarantine remains only when software cannot prove the hardware safe state.

量測程式中的已辨識儀器連線錯誤會先停止可辨識的 active node 並驗證 state 回復；
證據完整時釋放該次 Hardware 鎖、保留網站並暫停排程，現場確認後輸入
`恢復量測`（全自動判斷模式下 QOP 探測成功後會自動恢復）。grace 後由 JY 強制停止
的 worker，server 會關閉本晶片殘留的 QM，清理乾淨且 state 未變時自動釋放 Hardware
鎖。Windows 關機、Task Manager／作業系統直接終止、native crash、斷電、worker 無退出
證據、active node 無法確認停止、QM 清理失敗或 state 回復失敗，仍會保留 Hardware 鎖
並要求本機 recovery。

A recognized instrument-connectivity error first stops the identifiable active
node and verifies the state restore; with complete evidence it releases that
run's hardware lock, keeps the site, and pauses scheduling; send `恢復量測`
after on-site inspection (under full-auto judgement the server resumes after a
successful QOP probe). After JY force-stops a worker past the grace period, the
server closes this chip's stale QM and releases the lock when the cleanup is
clean and the state is unchanged. Windows shutdown, a Task Manager or OS kill, a
native crash, power loss, missing worker exit evidence, an active node that
cannot be confirmed stopped, a failed QM cleanup, or a failed state restore
still retains the hardware lock and requires local recovery.
