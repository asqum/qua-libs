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
| 有限自動量測 / Bounded automatic measurement | `進入 JY 自動量測模式`、`進入JY自動量測模式` | `Enter JY automatic measurement mode` | 一次核准有限 lease（預設 8 小時，進入時可指定，上限見 `policies.yaml` 的 `autonomy.max_duration_hours`），再於核准範圍內自動推進。新 workflow 同樣使用 `active_qubit_names` 且 `multiplexed` 預設 true。 / Approves one bounded lease (eight hours by default, specifiable at entry up to `autonomy.max_duration_hours`), then advances within scope. New workflows also use `active_qubit_names` and default `multiplexed=true`. |
| 喚醒已核准工作 / Wake an approved session | `已核准` | `Approved` | 只在 agent host 暫停回合時使用；它不代替網頁核准。 / Used only when the agent host paused; it never substitutes for Dashboard approval. |
| 暫停自動排程 / Pause automatic scheduling | `暫停 JY 自動量測` | `Pause JY automatic measurement` | 目前 run 完成後不再排新實驗，保留 lease 與網站。 / Stops new scheduling after the current run; keeps lease and site. |
| 繼續自動排程 / Resume automatic scheduling | `繼續 JY 自動量測` | `Resume JY automatic measurement` | 恢復同一份未過期、paused lease。 / Resumes the same unexpired paused lease. |
| 儀器檢查後恢復量測 / Resume after instrument inspection | `恢復量測`、`恢复量测` | `Resume measurement` | 恢復同一 workflow、lease 與 Dashboard session；目前 node 以新 run-id 從頭執行，不會接續已中斷的 Python call stack。 / Restores the same workflow, lease, and Dashboard session; the current node restarts as a new run rather than resuming a Python stack. |
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
| 儀器連線失敗、排程已暫停 / Instrument connectivity pause | 三行 list：`完成後回傳: 恢復量測` |
| 未登錄的 worker crash 暫停 lease / Unrecognized worker crash paused the lease | 三行 list：`完成後回傳: 恢復量測`（同一 lease 續跑，不需重新核准） / Three-line list ending with `恢復量測`; the same lease resumes without a new approval |
| 殘留流程／服務，無法進入 / Stale lifecycle blocks entry | 三行 list：`完成後回傳: 恢復` |
| Ensure 回 `recovery_only`（硬體鎖隔離） / recovery_only lock quarantine | 三行 list：`完成後回傳: 恢復` |
| 自動量測已被暫停 / Autonomy paused | 三行 list：`完成後回傳: 繼續 JY 自動量測` |
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
| 關機後留下硬體鎖隔離 / Retained lock after shutdown | 量測電腦本機 operator console 現場確認；必要時再 `恢復` | 不能從手機 Dashboard 解鎖。 / Phone Dashboard cannot recover the lock. |

`停止 JY 自動量測` 也是完整結束語句，效果與 `結束量測` 相同。

`停止 JY 自動量測` is also a full-shutdown phrase and matches `結束量測`.

## Dashboard 控制 / Dashboard controls

- Home：`結束量測並關閉所有 JY 服務`，需輸入頁面顯示的
  `SHUTDOWN <session-id>`，以及 Pause、Resume、End automation（保留網站）、
  Emergency stop worker（保留網站）。
- Approval：核准待處理的 run、state change 或初始 bounded lease（lease 需先選擇
  起點與選項並預覽）。
- 實驗結果：最上方是此晶片跨 workflow 的成功結果（晶片歷史，關機後仍保留靜態
  副本），其下是結果摘要（每次實驗的成功 qubit 與結果圖縮圖）與該次服務的完整
  實驗下拉歷史。
- Token 用量：最上方是此晶片的開始時間與所有 workflow 的總用量；可選整個晶片、
  本 session、今天、近 7 天、本月或自訂起迄時段計算費用；實驗依日期分組，每列
  有開始時間、token、模型與估計費用。只讀取本機 Claude Code 對話紀錄；
  Codex／Cursor 用量不計入。
- 使用說明：操作方式、量測順序與各節點的成功範例圖。

- Home: full shutdown, protected by the displayed `SHUTDOWN <session-id>` text,
  plus Pause, Resume, End automation (keep site), and Emergency stop worker
  (keep site).
- Approval: approves a pending run, state change, or initial bounded lease (a
  lease needs its start node and options chosen and previewed first).
- Experiment results: the chip history (successful results for this chip across
  workflows; a static copy survives shutdown), then the result summary
  (successful qubits and plot thumbnails per experiment) and the
  service-lifetime experiment history.
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
`恢復量測`。Windows 關機、Task Manager／緊急 force-stop、worker 無退出證據、
active node 無法確認停止或 state 回復失敗，仍會保留 Hardware 鎖並要求本機 recovery。
