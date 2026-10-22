# Bounded autonomy / 有限自動量測

`進入 JY 自動量測模式` 會建立一份人工核准的 bounded lease。每次核准時在
Dashboard 選 8、16 或 24 小時（`autonomy.duration_choices_hours`，預設 8，從核准
起算）並決定是否勾選「全自動判斷模式」；進入時的 `duration_hours` 只能是
8／16／24，僅作預選。整批 run 數不設上限、每個 `(node, qubit)` 最多 20 次。某個
qubit 達上限時只把它標成該 node 未完成；其他已有證據的 qubit 繼續往後。全新開始
時，各 qubit 計畫（不量／從頭／從上次量到的實驗繼續）、選用節點、量測方式與
flux 校正都在核准頁選擇並預覽；續期與 Recover 後的接續只問時數與全自動判斷（見
[README](../README.md)）。

`進入 JY 自動量測模式` creates a human-approved bounded lease. Every approval
chooses 8, 16, or 24 hours on the Dashboard (`autonomy.duration_choices_hours`,
default 8, counted from the approval) and whether to tick full-auto judgement;
an entry `duration_hours` may only be 8/16/24 and merely preselects it. The
lease has no batch-wide run cap and allows at most 20 attempts for each
`(node, qubit)`. Exhausting one target marks only that target incomplete; other
resolved targets continue. On a fresh start the per-qubit plan (not measured /
from the beginning / continue from the last experiment reached), optional nodes,
measurement mode, and flux calibration are chosen and previewed on the approval
page; a renewal or a continuation after a Recover asks only the hours and
full-auto judgement (see [README](../README.md)).

`jy_get_next_action` 回傳確定性的下一步：目前節點、各 target 的
resolved／incomplete／unresolved、剩餘次數、需要的 setup 工具，以及要改的參數或
「沒有登錄規則涵蓋」的說明；未勾選的節點會直接跳過。成功只在 state commit 後
計入，且每個節點每個 qubit 只計一次。

`jy_get_next_action` returns the deterministic next step: current node,
per-target resolved/incomplete/unresolved sets, attempts left, required setup
tools, and either the parameters to change or a note that no registered rule
covers the situation. Unticked nodes are skipped. A success counts once per node
and qubit, and only when its values were committed.

Codex 支援 durable goal 時，這個進入語句會建立一個 session-scoped goal；Terra
與 Sol 使用相同生命週期，並以 `jy_wait_for_autonomy_status` 等待核准、worker 或
控制事件。模型不是常駐 daemon，終止由 goal 完成、operator stop、lease expiry、
hard stop 或 host 終止 turn 決定。Claude/Cursor 沒有跨平台的強制常駐 API；若 host
暫停，回原對話輸入 `已核准` 即可從 server-side SQLite/lease 狀態續跑。

When Codex durable goals are available, this phrase creates a session-scoped
goal; Terra and Sol share the same lifecycle and wait through
`jy_wait_for_autonomy_status`. A model is not a resident daemon: completion,
operator stop, lease expiry, hard stop, or host turn termination ends execution.
Claude/Cursor have no universal persistence API; `已核准` resumes from the
server-side SQLite/lease state if their host pauses.

`snapshot_missing`、`state_hash_conflict`、`hardware_lock_anomaly` 三個條件會結束
lease（`autonomy.halt_conditions`）；無法回復的啟動錯誤同樣結束 lease，例如
service 拒絕啟動（如 "record a decision first"）、worker 啟動失敗，或 worker 結束
時沒有留下完成紀錄。在接觸硬體前就被拒絕的 scope／quota／policy 請求（超出
policy 的參數、超出範圍的 targets、過早的 analyze）只會被拒絕並記錄，lease 繼續。
符合 `autonomy.recoverable_worker_exceptions` 的 worker crash 保留 lease 並提示修正
方式。未勾選全自動判斷時，其他 worker crash 與儀器連線失敗會暫停 lease，operator
確認後以 `恢復量測`（或 `繼續 JY 自動量測`）續跑。勾選全自動判斷時，不明的
worker crash（JY 先關閉本晶片殘留的 quantum machine 與 job）或 JY analyzer 例外會
自動重跑（agent 記錄 `repeat`）；同一 node／qubit 自動重跑 2 次後仍失敗，就把該
target 標為此 node 的 incomplete；儀器斷線則在 QOP 回應探測（每 60 秒）後自行恢復。
每份 lease 最多 10 次自動判斷，用完後回到原本的暫停；hardware lock 被保留時全自動
不作判斷，lease 照舊暫停等 operator。三個 halt 條件與 lease 到期一律交給 operator。
低信心與 `manual_review` 不會暫停；`needs_review` 的 run 仍會
commit 其 `passing_targets`，其餘 target 不 commit。

The three halt conditions `snapshot_missing`, `state_hash_conflict`, and
`hardware_lock_anomaly` end the lease (`autonomy.halt_conditions`); so does a
non-recoverable start error, such as a service refusal ("record a decision
first"), a worker launch failure, or a worker that exits without a completion
record. A scope, quota, or policy request refused before the hardware --
out-of-policy parameters, out-of-scope targets, a premature analyze -- is
refused and recorded, and the lease continues. A worker crash matching
`autonomy.recoverable_worker_exceptions` keeps the lease and reports the remedy.
Without full-auto judgement, any other worker crash and an
instrument-connectivity failure pause the lease until the operator sends
`Resume measurement` (`恢復量測`, or `繼續 JY 自動量測`). With full-auto judgement
ticked, an unrecognized worker crash (after JY closes this chip's stale quantum
machine and its job) or a JY analyzer exception is retried (the agent records
`repeat`); after 2 automatic retries of the same node/qubit the target is marked
incomplete at that node; an instrument outage resumes on its own once the QOP
answers a probe (every 60 s). A lease allows at most 10 automatic decisions,
after which the old pause returns; while a hardware lock is retained, full-auto
does not decide and the lease pauses for the operator. The three halt
conditions and the expiry always stay with the operator. Low confidence and
`manual_review` do not pause; a `needs_review` run still commits its
`passing_targets` and nothing else.

Dashboard 首頁提供五個 authorization 控制：本次完成後暫停、繼續、停止核准（目前
實驗跑完後 lease 結束，核准頁出現續期卡片重新設定時數與判斷模式）、結束授權
但保留網站、緊急停止 worker 但保留網站；首頁也顯示全自動判斷狀態與紀錄。
首頁另有需精確二次確認的「完整結束量測」按鈕，即使 lease 已 `halted` 仍可使用。完整結束會
先保存 session-specific shutdown request，以 authenticated stop file 要求 worker
協作停止，等待 terminal、永久停止 workflow，再關閉 MCP、Dashboard 與 tunnel。
若 retained hardware lock 仍存在，確認所有 worker 已死亡後會轉為 quarantine；
lock 不刪除，但服務照常關閉。下次 Ensure 僅在 loopback 啟動 recovery-only 頁面。

Home exposes five authorization controls: pause after the current run, resume,
stop authorization (the current run finishes, the lease ends, and the Approval
page shows a renewal card to set the hours and judgement again), end/revoke
automation while keeping the site, and emergency-stop the worker while keeping
the site; Home also shows the full-auto judgement state and its log. A separate
confirmation-gated full-shutdown button
remains on Home after a lease is halted. It first persists a session
shutdown request, delivers an authenticated cooperative stop, waits for a
terminal run, stops the workflow, then closes MCP, Dashboard, and tunnel. If a
lock remains after every worker is verified dead, the lock is quarantined (not
deleted) while services still close. The next Ensure starts loopback
recovery-only service for the on-site operator.

每個 calibration node 的第一次 run 必須用同一組參數 multiplex 全部 active
targets。分析後已通過的 qubit 要排除；未通過的再成組改參數重測。誤把已通過目標
排進 retry 是 recoverable planning error：server 拒絕請求但不 hard-stop lease。
例外：`04` 每次最多 5 顆、`07d` 最多 4 顆，第一次 run 是第一組；07b 之後
active／thermal reset 兩組各有自己的第一批；第一次 run 若因程式過大而
submission timeout，群組依序減半（全部 → 一半 → 再一半 → 單顆），單顆仍逾時才交
給 operator（勾選全自動判斷時，JY 把該 target 標為此 node 的 incomplete 並讓 lease
繼續）；per-qubit 模式或共用 XY output 時，從 `03a` 起一次只量一顆 qubit。

The first run of every calibration node must multiplex every active target with
the same parameters. After analysis, omit already-resolved qubits and retry only
unresolved subgroups with modified parameters. Re-including resolved targets is
a recoverable planning error: the server rejects the request without hard-stopping
the lease. Exceptions: `04` runs at most 5 targets and `07d` at most 4, so the
first run is the first group; after 07b the active- and thermal-reset groups
each get their own first batch; a submission timeout from an oversized program
halves the group (all -> half -> half again -> single), and only a single-target
timeout goes to the operator (with full-auto judgement, JY marks that target
incomplete at the node and the lease resumes); in per-qubit mode or with a shared XY output,
every node from `03a` on measures one qubit at a time.

完整限制以 [rules/policies.yaml](../rules/policies.yaml) 為準，科學流程以
[rules/PLAYBOOK.md](../rules/PLAYBOOK.md) 為準。

The enforced limits live in [rules/policies.yaml](../rules/policies.yaml); the
scientific workflow lives in [rules/PLAYBOOK.md](../rules/PLAYBOOK.md).
