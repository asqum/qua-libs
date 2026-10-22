# JY_agent

JY_agent 是綁定此 Qualibrate repository 的安全量測代理層。它讓 Codex、
Claude Code 或 Cursor Agent 透過相同的 MCP 工具、政策、Dashboard 與稽核紀錄
執行量測；AI 不會直接執行 calibration script、寫入 `state.json`，也不會代替
人員核准。

JY_agent is the policy-gated measurement-agent layer for this Qualibrate
repository. Codex, Claude Code, and Cursor Agent share the same MCP tools,
Dashboard, policies, and audit records. The AI never runs calibration scripts
directly, writes `state.json` directly, or approves on behalf of an operator.

## 安裝 / Installation

前提是 Qualibrate 與本 repository 已可正常使用。第一次只需在 repository
最上層執行一行：

```powershell
powershell -ExecutionPolicy Bypass -File .\jy_agent.ps1 -Action Install
```

`setup.ps1` 會優先自動找到 `qualibrate_env`，以 editable mode 安裝 JY_agent，
初始化資料庫並檢查 Codex、Claude、Cursor 的 project config。自動搜尋失敗時才需：

```powershell
powershell -ExecutionPolicy Bypass -File .\jy_agent.ps1 -Action Install -Python C:\path\to\qualibrate_env\python.exe
```

也可以直接在已開啟 repository 最上層的 coding agent 對話輸入：

```text
請執行 .\jy_agent.ps1 -Action Install 完成 JY_agent 的一次性離線安裝與設定驗證，不要開始量測。
```

Agent 只需使用一般 shell 能力完成這次安裝；此時不依賴 JY MCP，也不會接觸儀器。

Assuming Qualibrate and this repository already work, run the first command
once from the repository root. It discovers `qualibrate_env`, installs the package in
editable mode, initializes storage, and validates all three client configs. Use
`-Python` only when automatic discovery cannot find the environment. You may
instead ask the coding agent to run `jy_agent.ps1 -Action Install` for one-time offline installation
and configuration validation. This step needs ordinary shell access, not the JY
MCP, and does not operate instruments.

## 什麼是「載入 JY_agent」？ / What does “load JY_agent” mean?

它現在是指用 AI coding client 把整個 repository 最上層開成 workspace/project；
不必再把 `JY_agent` 子資料夾另外開啟：

- Codex 讀取根目錄由 Install 依 `.codex/config.toml.example` 產生的 host-local
  `.codex/config.toml`、`AGENTS.md` 與 `.agents/skills/`。
- Claude Code 讀取根目錄 `.mcp.json` 與 `CLAUDE.md`。
- Cursor 讀取根目錄 `.cursor/mcp.json` 與 `.cursor/rules/jy-agent.mdc`。

第一次開啟時，各 client 可能顯示一次 project MCP trust 提示；這是唯一必要的
client 安全確認。接受後，client 會在背景啟動根目錄 `jy_agent.ps1`，再由它定位
並啟動 `JY_agent/start_mcp_stdio.ps1`；這一層只建立 STDIO MCP tools，不啟動
網站或 tunnel。收到進入語句後，agent 才在背景執行 `Ensure` 建立網站服務，
使用者不需開著終端機。定位只依 repository 內部相對路徑，因此
`ASqum_QM_lab` 或其他最外層資料夾名稱都可以。

It now means opening the repository root as the AI client's workspace or
project; opening the nested `JY_agent` folder is no longer required. Each
supported client discovers its root project configuration. Codex's active
`.codex/config.toml` is host-local and generated from the checked-in example.
The first open may show a one-time MCP trust prompt. After that, the client
launches the root `jy_agent.ps1`, which resolves and starts the nested
`start_mcp_stdio.ps1`. This phase creates only the STDIO MCP tools; the agent runs
Ensure after an entry phrase to establish the website/tunnel. No user-operated
terminal is needed, and the outer repository folder can have any name.

純網頁版 chatgpt.com 本身不能啟動本機 MCP。ChatGPT/Codex 的本機桌面工作階段
可以；手機 remote 是回到這台量測電腦上既有的對話，而不是讓手機直接連儀器。

The browser-only chatgpt.com product cannot launch a local project MCP by
itself. A local Codex/ChatGPT desktop agent can. Mobile remote control continues
the desktop conversation; the phone never connects directly to instruments.

## 執行量測 / Run an agentic experiment

在已載入專案的對話輸入以下其中一句：

```text
進入 JY 量測模式
```

不必再附加 `target:` 或 `multiplex:`。新 workflow 的 qubit 清單來自
`state.json` 的 `active_qubit_names`，`multiplexed` 預設為 true。若要量不同
qubit，請先改 `active_qubit_names`，不要在對話裡重寫一份 target 清單。

空格可省略為 `進入JY量測模式`；自動模式同樣可輸入
`進入JY自動量測模式`。英文指令分別是 `Enter JY measurement mode` 與
`Enter JY automatic measurement mode`；完整中英文指令表請見
[Command.md](Command.md)。

一般模式一次討論、核准並執行一個實驗，每次 run/state change 都需在同一個
Dashboard 頁面核准。

```text
進入 JY 自動量測模式
```

自動模式只先核准一次有限授權：每次核准時在 Dashboard 選 8、16 或 24 小時
（`autonomy.duration_choices_hours`，預設 8 小時，從核准起算），並可勾選「全自動
判斷模式」；進入時傳入的 `duration_hours` 只能是 8／16／24，僅作預選，由頁面決定。其後只要 targets、nodes、參數與
20 次／node／qubit 上限均在授權內，就會繼續安全量測；硬停損仍立即停止。
Codex 支援 durable `/goal` 時，進入自動模式會建立一個只屬於該 session 的持續
目標，並以 `jy_wait_for_autonomy_status` 等待核准或新結果，不會只因暫時沒有工具
可呼叫就結束。Terra 與 Sol 服從相同 goal/turn 控制；模型本身不決定無限執行，
真正的終止條件是 workflow 完成、使用者停止、lease 到期、硬停損，或 host 結束
該 turn。Claude/Cursor 沒有由本 repo 強制保持無限 turn 的通用 API，因此會盡量
維持當前 agent run；若 host 暫停，Dashboard 會提示回對話輸入 `已核准` 續跑。

Enter either exact phrase in a project-loaded conversation. Do not append
`target:` or `multiplex:`; a new workflow uses `state.json` `active_qubit_names`
and defaults `multiplexed=true`. Conversational mode
requires approval for every run or state change. Automatic mode requests one
bounded authorization -- 8, 16, or 24 hours chosen on the Dashboard at every
approval (`autonomy.duration_choices_hours`, default 8, counted from the
approval), plus an optional full-auto judgement tick; an entry `duration_hours`
may only be 8/16/24 and merely preselects it -- and then continues only inside its targets,
nodes, safe parameters, and 20-attempt-per-node/qubit limit.
The exact English entry commands are `Enter JY measurement mode` and
`Enter JY automatic measurement mode`; see [Command.md](Command.md) for all
bilingual controls.
When Codex durable goals are available, automatic entry creates one
session-scoped goal and waits through `jy_wait_for_autonomy_status`; Terra and
Sol follow the same goal/turn lifecycle. Completion, operator stop, lease expiry,
and hard stops are terminal. Repository instructions cannot guarantee an
indefinite Claude/Cursor turn, so `已核准` remains the host-resume fallback.

預設校正順序是 `config/agent.yaml` 的 `workflow.sequence`：
`02 → 02x → 02c → 02a → 02b → 03a → 03b → 04 → 07d → 07b → 05 → 06 → 06a → 06b → 10a → 05st → 06st_t2star → 06st_t2e`。
`02`、`06a` 與三個統計節點是選用節點，在自動量測核准頁勾選；`02b`、`03b`
（與 `06a`）只在選擇 flux 校正時執行，qubit 共用 Z output 時無法選。`07d` 在
`07b` 前優化 readout，`05` 在 `07b` 之後以優化後的 readout 量測。所有 reset
預設為 thermal；thermal 07b fidelity 高於 0.80 的 qubit 會再跑一次 active-reset
07b，該次 fidelity 至少 0.75（`reset_policy.active_min_readout_fidelity`）且不在
兩個 blob 上的 shot 不超過 15% 才取得 active 資格。07b 之後有 reset 參數的節點把合格與其他 qubit 分成 active／thermal
兩組，各有自己的第一批 run，不混在同一 run。若統計節點採 active，會先以 active
複驗 05、06、06b。統計節點各執行一次（100 次重複），至少完成 90 次才算數；JY
逐次重新 fitting、剔除不良 fit 後記錄平均與分散，沒有 100/100 否決門檻。

The default calibration sequence is `config/agent.yaml` `workflow.sequence`:
`02 → 02x → 02c → 02a → 02b → 03a → 03b → 04 → 07d → 07b → 05 → 06 → 06a → 06b → 10a → 05st → 06st_t2star → 06st_t2e`.
`02`, `06a`, and the three statistics nodes are optional and ticked on the lease
approval page; `02b`, `03b` (and `06a`) run only when flux calibration is chosen,
which is disabled when qubits share a Z output. `07d` optimizes the readout
before `07b`, and `05` runs after `07b` on that optimized readout. Reset defaults
to thermal. A qubit whose thermal 07b fidelity is above 0.80 earns one
active-reset 07b repeat, and qualifies for active reset when that repeat reaches
at least 0.75 (`reset_policy.active_min_readout_fidelity`) with at most 15% of
its shots on neither blob. After 07b,
reset-capable nodes split qualified and other qubits into active and thermal
groups, each with its own first batch and never sharing a run. Active-reset
statistics first revalidate 05, 06, and 06b under active reset. Each statistics
node runs once (100 repetitions) and counts only when at least 90 completed; JY
refits every repetition, drops poor fits, and records the filtered mean and
spread, with no 100/100 rejection gate.

`start_mcp_stdio.ps1` 必須先讓 client 擁有 MCP 工具，這是「雞生蛋」邊界：完全
沒有 MCP 的 agent 無法只靠一句自然語言建立 MCP。專案載入負責這個一次性的
protocol bootstrap；它不依賴 Dashboard、Cloudflare、port 或網路。輸入進入語句
後，agent 才會執行 `ensure_server.ps1` 並建立或恢復量測 workflow，絕不會因為
開啟資料夾就自動操作硬體。這個兩階段設計也確保 tunnel 失敗不會讓 MCP 在
`initialize` response 前斷線。

The stdio MCP must exist before an agent can interpret a phrase as an MCP
action—this is the bootstrap boundary. Project loading supplies that lightweight
protocol bootstrap without depending on the Dashboard, Cloudflare, ports, or
network. The entry phrase then runs Ensure and creates or resumes the measurement
workflow. Merely opening the folder never operates hardware, and a tunnel failure
cannot break the MCP initialize response.

### 自動量測核准頁 / Lease approval page

自動量測只在 Dashboard 的核准頁核准，由 operator 選擇，agent 不代選：

- 各 qubit 計畫：頁面顯示此晶片（`config.toml` 指定的 QuAM state）的量測紀錄。
  每顆 qubit 選「不量」、「從頭」或「從上次量到的實驗繼續」（之前的 run 作為繼承
  證據帶入），也可用「全部設為」一次設定。workflow 從最早的選擇開始，較晚的
  qubit 走到自己的節點才加入。`state.json` 之後若有變動，頁面會警告。
- 選用節點：`02`、`06a`、`05st`、`06st_t2star`、`06st_t2e` 預設勾選；取消勾選即
  從此 workflow 與 lease 移除。只要有量測的 qubit 的 readout IF 為 0 或重複，
  `02` 為必選；把這些 qubit 設為「不量」後才可取消勾選。
- 量測方式：從 `03a` 起選 per-qubit（一顆 qubit 跑完整個後段再換下一顆）或
  multiplex。wiring 中有 qubit 共用 XY output 時只能 per-qubit；選定後此
  workflow 固定不變。
- Flux 校正：選「是」才執行 `02b`（`02a` 後）與 `03b`（`03a` 後），`06a` 另需
  勾選；有 qubit 共用 Z output 時無法選「是」。
- 選完按「預覽」查看會執行的實驗，再選授權時數（8／16／24 小時）與是否「全自動
  判斷模式」，勾選同意並輸入核准碼；預覽後更改選項需重新預覽。
- 只有全新開始（上一個 workflow 由 operator 結束或已完成）才顯示上述完整卡片。
  續期（到期後輸入 `繼續JY自動量測模式`）與 Recover 後的接續只顯示簡易卡片：
  時數、全自動判斷與核准碼。首頁暫停後的卡片只問時數與全自動判斷，按「核准並
  繼續」即可（不需核准碼，時數從現在起算）；重新規劃收在「進階」內且需核准碼。

Automatic measurement is approved only on the Dashboard Approval page; the
operator chooses, never the agent:

- Qubit plan: the page shows the chip's (the QuAM state named in `config.toml`)
  measured history. Each qubit is not measured, starts from the beginning, or
  continues from the last experiment it reached (earlier runs are carried in as
  inherited evidence); a "Set all to" row sets every qubit at once. The
  workflow starts at the earliest choice and later qubits join at their own
  node. The page warns if `state.json` changed since then.
- Optional nodes: `02`, `06a`, `05st`, `06st_t2star`, and `06st_t2e` are ticked
  by default; an unticked node is removed from that workflow and lease. `02` is
  required while any measured qubit has a zero or duplicated readout IF; it can
  be unticked once those qubits are set to not measured.
- Measurement mode from `03a` on: per-qubit (one qubit through the whole tail,
  then the next) or multiplex. A shared XY output in the wiring allows only
  per-qubit. The workflow keeps its mode.
- Flux calibration: yes runs `02b` (after `02a`) and `03b` (after `03a`); `06a`
  also needs its tick. A Z output shared by two qubits disables yes.
- Press Preview to see which experiments will run, then choose the hours
  (8/16/24) and whether to use full-auto judgement, tick the agree box, and
  enter the approval code. Changing a choice after the preview needs a new
  preview.
- Only a fresh start (the last workflow was ended by the operator or completed)
  shows this full card. A renewal (`繼續JY自動量測模式` after expiry) and a
  continuation after a Recover show a simple card: hours, full-auto judgement,
  and the approval code. After a Home pause the card asks only the hours and
  full-auto judgement and has "Approve and continue" (no approval code; the
  hours count from now); re-planning is folded under "Advanced" and needs the
  code.

## Dashboard

進入語句觸發的 Ensure 只把 MCP HTTP service 綁在 `127.0.0.1:8765`，並把
review-only Dashboard 透過 Cloudflare HTTPS quick tunnel 公開。Cloudflare 裸 URL
不帶任何 token；所有電腦與手機直接開啟同一網址，輸入
`runtime/dashboard-password.txt` 的固定密碼後取得 secure cookie。此密碼首次
正常啟動時產生一次並固定沿用，不會 commit 或寫入網址。公開頁面不能呼叫 MCP
或任意執行程式。

每次進入模式只對使用者顯示 entry result 最外層的 `browser_url`；這是手機與電腦
共用的 `/` 首頁入口。首頁會解析目前 session，並提供五個分頁：首頁（狀態、
量測控制與完整關機）、核准、實驗結果、Token 用量、使用說明。舊
`/approve/...`、`/autonomy/...` 與 `/session/...` 連結只會導向對應的新分頁，不會
建立另一個網站。直接建立或替換 lease 也會重新綁定同一個 workflow session。

實驗結果頁最上方是晶片歷史：此晶片在所有 workflow 中成功（JY `passing_targets`）
的結果，依節點與 qubit 排列，最新一次在前；每次分析後也會寫一份靜態副本到
`runtime/chip_history/<chip>/index.html`，關機後仍可開啟。其下是本次結果摘要與
下拉選單，保留本次 server 生命週期內所有實驗，且只在新結果、判斷或控制狀態
出現時更新；核准頁只在提案改變時更新。Token 用量頁最上方是此晶片的開始時間與總用量，
可選時段計算費用，並依日期列出每個實驗的模型、token 數與估計費用，只讀取本機 Claude Code 對話紀錄中的數字；Codex／Cursor 的用量不
計入。使用說明頁介紹操作方式與量測順序，並以 `docs/guide_assets/` 的範例圖說明
各節點成功時的樣子。右上角可選繁體中文或 English；時間預測、參數、分析與下一步
預設為摺疊面板。首頁提供五個控制：目前實驗完成後暫停、繼續、停止核准（目前
實驗跑完後授權結束，核准頁出現續期卡片重新設定時數與判斷模式）、結束自動授權
（保留網站）、緊急停止 worker（保留網站），另有完整關機；完整關機需輸入
`SHUTDOWN <session-id>` 防止誤觸。首頁也顯示全自動判斷是否開啟與其紀錄（沒有待
核准項目時，核准頁的授權卡片同樣顯示）。另外有一份靜態歷史網站
`runtime/history_site/index.html`，收錄所有晶片與所有量測工作階段的結果，關機後
仍保留，在量測電腦上開啟；量測進行中時它會連到即時 Dashboard。

手機或第二個瀏覽器直接使用同一裸 URL 與固定密碼，不再需要配對連結；每個瀏覽器
仍取得各自 12 小時 cookie。本機 `http://127.0.0.1:8765/operator` 只提供
retained hardware lock 的正式檢查／recovery，仍要求現場實體確認與完整確認句，
且不會經 public tunnel。本機 Dashboard 是 `http://127.0.0.1:8766/`；Cloudflare
quick-tunnel hostname 在建立新 tunnel 後可能改變，以當次 entry 回傳的網址為準。

The MCP remains loopback-only at `127.0.0.1:8765`. Only the review Dashboard is
forwarded through a password-protected Cloudflare HTTPS quick tunnel. The bare
URL contains no secret; every browser enters the fixed ignored runtime password
and receives an independent 12-hour secure cookie. The public surface cannot
invoke MCP or arbitrary code. Agents show only the entry result's
top-level `browser_url`, a common `/` entry for phones and computers. It resolves
the active session and links to five tabs: Home, Approval, Experiment results,
Token usage, and Guide. Legacy approval, autonomy, and session URLs redirect to
the appropriate view. Experiment results opens with the chip history -- every
successful result (JY `passing_targets`) for this chip across its workflows, by
node and qubit, newest first -- and a static copy at
`runtime/chip_history/<chip>/index.html` is rewritten after each analysis and
survives shutdown. Below it, the summary and selector retain every run for that
server session; updates are event-driven. Token usage opens with the chip's start
time and total, computes the cost of any time range, and lists model, tokens,
and estimated cost per experiment grouped by date, read only as numbers from the local Claude Code
transcripts; Codex and Cursor usage is not counted. Guide explains operation and
the sequence, with an example plot per node from `docs/guide_assets/`. Language
is selectable at the top right, and Prediction, Parameters, Analysis, and Next
action are collapsed by default. Home provides five controls -- pause after
the current run, Resume, Stop authorization (the current run finishes, the
lease ends, and the Approval page shows a renewal card to set the hours and
judgement again), End automation (keep site), and Emergency-stop worker (keep
site) -- plus full shutdown, which is guarded by the exact
`SHUTDOWN <session-id>` confirmation. Home also shows whether full-auto
judgement is on and its log (as does the Approval tab's authorization card when
nothing is pending). A static history site at
`runtime/history_site/index.html` collects every chip and every measurement
session with its results; it survives shutdown, is opened on the lab PC, and
links to the live Dashboard while a session runs. Direct or replacement lease requests also
rebind the workflow's stable session to the current lease.
Phones and secondary browsers use the same URL and password; no pairing link is
required. The loopback-only `/operator` page provides formal retained-lock
inspection/recovery with physical attestation and is never tunneled.

離開或停止語句（例如 `結束量測`、`停止量測`、`退出 JY 量測模式`）會先安全
終止／等待 active run，再關閉 workflow、Dashboard、MCP HTTP service 與 tunnel。
完整停止也會刪除內部 Dashboard access token；下一次進入會建立新的 quick tunnel，
hostname 可能改變，但登入密碼不變。
Dashboard 的「完整結束量測」按鈕在 lease halted 後仍可使用；它會等待 active run，
若 worker 無法證明 cleanup 而留下 lock，coordinator 會確認所有 JY worker 都已死亡，
保留 lock 與 recovery marker，但仍關閉 Dashboard、MCP 與 tunnel。下一次 Ensure
只會啟動 loopback recovery-only 服務，不會重新公開網站；現場 operator 復原 lock
後，才能再次進入量測模式。JY 在寬限期後強制停止 worker 時，server 會先關閉本晶片
殘留的 QM；清理乾淨且 state 檔未變時由 server 自行釋放 lock。只有工作管理員／OS
終止、native crash、斷電、缺少 exit 證據、無法證明 node 已停止、QM 清理失敗或
state 回復失敗時，lock 才保留並需現場復原。Worker 會在接觸硬體前先持久化 pre-run state hash，
避免 abrupt exit 讓正式 recovery 缺少必要證據；既有舊 lock 只在 active/configured/
protected-backup state 完全相符時使用相容復原。不得手動刪除 lock。

Exit or stop phrases safely stop or poll an active run, close the workflow, and
then shut down the Dashboard, HTTP MCP service, and public tunnel. Full shutdown
also deletes the internal Dashboard access token; the next entry opens a new
quick tunnel whose hostname may differ, while the login password stays the same.
The full-shutdown button remains available after a halted lease and waits for an
active run. If cleanup leaves a lock, JY verifies that every worker has exited,
keeps the lock/receipt for recovery, and still closes Dashboard, MCP, and tunnel.
The next Ensure is loopback recovery-only until an on-site operator formally
recovers the lock. After a JY forced stop at the end of the grace period, the
server first closes this chip's stale QM and releases the lock itself when the
cleanup is clean and the state file is unchanged; the lock is kept for local
recovery only after a Task Manager/OS kill, native crash, power loss, missing
exit evidence, an unverifiable node stop, a failed QM cleanup, or a failed
state restore. Before hardware access, workers persist a pre-run state hash
so an abrupt exit cannot make formal recovery evidence impossible. Legacy locks
use the compatibility path only when active, configured, and protected-backup
state hashes are identical. Never delete a lock manually.

若 QOP/OPX 或儀器無法連線，worker 會暫停本次實驗與後續排程，Dashboard 直接顯示
檢查與恢復方式。可證明連線未建立，或硬體執行中仍能找到 active node、成功 stop，
且 state 已回復時，會自動釋放本次 lock 並保留網站；現場確認後輸入
`恢復量測`，同一 workflow/session 會恢復，目前 node 以新 run-id 從頭執行。
若 worker 被強制關閉、active node 無法證明停止或 state 回復失敗，lock 留在本機
safety quarantine；完整關機仍會關閉網站、MCP 與 tunnel。勾選全自動判斷模式時，
JY 每 60 秒探測 QOP，一旦回應就自行恢復，不需輸入 `恢復量測`（lock 被保留時仍
等 operator）。

AI App 或網路斷線時，Dashboard 的暫停、停止與完整關機仍可獨立使用。網路恢復後
可輸入 `結束量測`／`End measurement` 確認收尾。若下次進入因殘留 process、workflow
或 lifecycle 設定失敗，輸入 `恢復`／`Recover`；agent 會執行 root
`jy_agent.ps1 -Action Recover`，不依賴 JY MCP，也不會強制刪除 hardware lock。
Recover 之後（不是 operator 結束），下一次自動量測進入語句或
`繼續JY自動量測模式` 會開始接續 workflow，只顯示簡易卡片，各 qubit 從原本停下處
繼續；以 `結束量測` 或 Dashboard 完整關機結束的 workflow，下次一律顯示完整卡片。

If QOP/OPX or an instrument cannot be reached, JY classifies the outage and
pauses the run and new scheduling; the Dashboard shows the recovery steps. A
provable no-connect failure releases its lock after state restoration. A
mid-execution disconnect also releases only when the recorded active node stops
successfully and state restoration verifies; otherwise it remains in local
quarantine. After inspection, `Resume measurement` restores the same session and
restarts the current node as a new run. With full-auto judgement ticked, JY
probes the QOP every 60 s and resumes on its own once it answers (a retained
lock still waits for the operator). Full shutdown still closes the site, MCP
service, and tunnel. If the AI app or network
disconnects, Dashboard controls remain usable. After reconnection, use `End
measurement`; if a later entry is blocked by stale state, use `Recover`. After a
Recover (not an operator end), the next automatic entry or
`繼續JY自動量測模式` starts a continuation workflow with the simple card, and
each qubit resumes where it stood; a workflow ended by `結束量測` or Dashboard
full shutdown always brings back the full card.

## 主要檔案架構 / Main file structure

```text
repository root/
├─ jy_agent.ps1                       root install/lifecycle/MCP shim
├─ .mcp.json / .codex/ / .cursor/     root client discovery
├─ AGENTS.md / CLAUDE.md / .agents/   root phrase routing
└─ Quantum-Control-Applications-QuAM/Superconducting/JY_agent/
   ├─ README.md / Command.md / WORKFLOW.md / UPDATE.md / TODO_LIST.md
   ├─ AGENTS.md                           detailed agent instructions
   ├─ setup.ps1                          installation implementation
   ├─ start_mcp_stdio.ps1                client-owned MCP launcher
   ├─ diagnose_mcp.ps1                   non-measurement MCP self-test
   ├─ ensure_server.ps1 / stop_server.ps1 server and tunnel lifecycle
   ├─ recover_closed_state.ps1           safe stale-state close coordinator
   ├─ config/agent.yaml                  non-policy service configuration
   ├─ rules/
   │  ├─ PLAYBOOK.md                     scientific decision rules
   │  ├─ policies.yaml                   server-enforced hard limits
   │  └─ experiences/                    accepted settings and lessons
   ├─ docs/                              detailed operating documentation
   │  └─ guide_assets/                   example plots for the Dashboard Guide
   ├─ src/jy_agent/                      MCP, policy, worker, Dashboard
   ├─ tests/                             offline regression tests
   └─ runtime/                           ignored local audit/runtime data
```

關機／停止的可稽核資料位於 `runtime/`：SQLite 的 `runs` 與
`shutdown_requests` 保存狀態；`requests/<run-id>.stop.json` 是帶 process token
驗證的協作停止請求；`requests/<run-id>.exit.json` 是 worker/OS 退出證據；
`shutdown_requests/<session-id>.json` 讓不同 Dashboard session 不會互相覆蓋；
`recovery-required.json` 與 `hardware.lock` 則保留現場復原邊界。這些都是 ignored
local runtime data，不應 commit、分享 token，或手動修改。

Shutdown/stop evidence lives under `runtime/`: SQLite records run and shutdown
state; per-run stop/exit JSON files authenticate delivery and record process exit;
per-session shutdown JSON prevents stale-session collisions; the recovery marker
and hardware lock preserve the on-site recovery boundary. These local files are
ignored, secret-bearing runtime state and must not be edited or committed.

詳細連結 / Detailed links:

- [WORKFLOW.md](WORKFLOW.md): 從 0 開始的程式與指令流程 / from-zero program flow.
- [Command.md](Command.md): 中英文自然語言與維護指令 / bilingual conversation and maintenance commands.
- [rules/PLAYBOOK.md](rules/PLAYBOOK.md): 科學判讀與節點規則 / scientific node rules.
- [rules/policies.yaml](rules/policies.yaml): 硬性安全限制 / enforced safety limits.
- [docs/CLIENTS.md](docs/CLIENTS.md): Codex、Claude、Cursor 設定 / client setup.
- [debugQA.md](debugQA.md): MCP、Dashboard 與 client 問答式排錯 / Q&A troubleshooting.
- [docs/PUBLIC_DASHBOARD.md](docs/PUBLIC_DASHBOARD.md): 公開 Dashboard 安全模型 / public Dashboard security.
- [docs/AUTONOMY.md](docs/AUTONOMY.md): 有限自動量測 / bounded autonomy.
- [UPDATE.md](UPDATE.md) and [TODO_LIST.md](TODO_LIST.md): 更新與待辦 / changes and remaining work.
