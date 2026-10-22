# TODO list / 待辦事項

Rewritten 2026-10-22 against the code (an audit of every earlier item). Items
that the code already does, or that conflict with the current design, were
removed or rewritten; their history stays in the dated sections below.

## 下一階段 / Next stages (operator decisions 2026-10-22)

The current branch keeps the usable version (stage 0). Stages 1–5 are
development work in a separate repository made from this folder's history.

1. **JY's own QM nodes and a per-workflow chip context.** Port the ~18 nodes
   JY runs (calibration_graph + side_project statistics) into JY_agent, load
   the QuAM state from an explicit path (no `~/.qualibrate/config.toml`), emit
   one dataset format, and replace the process-global `Settings` chip with a
   per-workflow chip context. JY runs no longer go through Qualibrate's global
   configuration.
2. **Several chips at once.** Choose the chip on the Approval page; per-chip
   workflows and leases; locks per instrument resource (cluster + port set);
   concurrent chips allowed when their ports do not overlap, refused
   otherwise, also when a non-JY QM with a running job holds overlapping ports;
   Recover/end per workflow (services stop with the last one); one Dashboard
   whose homepage lists chips and live sessions and stays available as a
   permanent history homepage; data folders and usage per chip/session.
3. **One state/wiring format (jy_state).** chip.yaml, wiring.yaml
   (instrument/slot/port per role), calibration.json as the master; validators
   per instrument; QuAM state generated from it; existing chips migrated with
   round-trip tests.
4. **Qblox backend.** Port the nodes from asqum/qblox-measurement-nodes
   (`new-version`, reference only), generate its hw/dut/flux configs, convert
   its datasets, add locking/busy checks, add the missing 02x and statistics
   nodes, validate on hardware (the cluster is reached through port
   forwarding on 10.21.19.201).
5. **Standalone repository.** Declared dependencies with extras ([qm],
   [qblox], [dev]) and pins, a cross-platform Python CLI (ps1 as thin Windows
   wrappers), the root integration files moved in, CI without hardware, a
   license, and history kept with `git subtree split`. Prepared, not pushed.

## 尚未完成 / Still open

### Behaviour

- **Server-side executor beyond shadow mode.** Shadow mode
  (`autopilot_shadow`, `jy_get_autopilot_shadow_report`) only records what
  the deterministic planner would have done; the watchdog never starts a run,
  so full-auto "retry" still needs a live agent, and `jy_wait_for_autonomy_status`
  is still capped at 55 s per call. Open nodes to the executor one at a time
  from the shadow match rates; PLAYBOOK rules 6, 7 and 9 then relax for them.
- **Planner halving.** After a submission timeout (or a DataFetchingError)
  `jy_get_next_action` proposes the full-width group again: the first-batch
  test counts only completed runs, and `_node_full_width_submission_failed` is
  used only in validation. Implement the 10 → 5 → 3/2 → 1 ladder in the planner.
- **A lock retained for any uncategorized worker exception.** The worker tries
  the node stop only for instrument_unreachable, program_submission_timeout and
  analysis_failure, so any other exception (including DataFetchingError, which
  is a registered recoverable signature) keeps the lock until local recovery,
  which makes the registered remedy and full-auto retries dead ends. Try the
  node stop for every hardware-phase failure and let the existing cleanup
  evidence decide.
- **One per-target verdict.** `workflow_subgroups.resolve_node_targets`
  re-derives pass/fail from fit metrics and never reads the analyzer's
  `passing_targets`; the 02x/02a readout-upconverter exclusion only exists in
  the analyzer, so a target can be resolved yet not committable. Make one the
  source of truth.
- **Resolved but uncommitted targets.** `_withheld_targets` reopens them for
  03a/03b/04/05/06/06a/06b/10a; 02x/02a/02b/02c/07b are still excluded, runs
  analyzed before `passing_targets` existed are not covered, and "committed"
  is judged from the decision's patch rather than from an applied commit.
- **Database overhead (measured 2026-10-22).** Every query opens a new SQLite
  connection (three PRAGMAs, then a WAL checkpoint when the last connection
  closes): about 30 ms per query on the live database. A 114-run Results page
  takes ~13 s, and every tool call pays the same per query, so this is
  probably part of the gap between runs. Measure, then keep one connection per
  thread for the long-running processes.
- **03a amplitude steps for shared XY lines** (0.10 → 0.075 → 0.05): not in
  the planner (optional per PLAYBOOK).
- **07b tail-power tool.** `jy_request_07b_tail_power_reduction` needs
  non-passing 07b evidence, but 07b runs once and always passes, so the tool
  can never be used, and PLAYBOOK's ×√0.5 recovery contradicts "07b runs once".
  Decide whether to reopen 07b or remove the tool and the text.
- **Hardware verification still missing:** active-reset 05/06b and the
  active-reset statistics gate (active 07b #351 and the active-reset 06 group
  #352/#353 are verified).

### Hardening

- **Policy keys that do nothing.** `protected_paths`, `version`,
  `autonomy.pause_on_analysis_statuses` (copied into the lease only),
  `state_commit.always_require_human_approval` (contradicted by lease commits),
  `state_commit.require_base_hash_match` (the check is unconditional),
  `autonomy.multiplex_submission_timeout.*`, and `autonomy.halt_conditions`
  (displayed; the halts are hard-coded). Enforce or remove each.
- **Custom ports.** Ensure and Stop take ports, but the root `jy_agent.ps1`
  forwards none, Doctor takes none and Recover calls Stop without them; there
  is no non-default-port lifecycle test. (Stage 2 keeps one shared service,
  so this matters only for a second checkout.)
- **Web server.** Replace the deprecated `@app.middleware("http")` decorator
  with an ASGI middleware class; add SSE keepalive and slow/disconnected
  client tests.
- **Retention.** Events are never pruned and per-run logs are never rotated
  (17 117 duplicate events accumulated before the 2026-10-20 dedupe).
- **cloudflared.** Pin the version, verify SHA-256, detect the CPU
  architecture; evaluate a named tunnel / fixed hostname (also what a
  permanent remote history homepage needs).
- **Size.** service.py is ~14 900 lines, approval_web.py ~3 900,
  mcp_server.py ~1 700; split them (stage 1 does this for the chip context).
  Add directory fsync after `os.replace` and fault-injection tests.
- **Client rules from one source.** test_remote_mobile.py checks phrase parity
  for the Cursor rule, root AGENTS.md, SKILL.md and Command.md; generate them
  from one source and include the root CLAUDE.md.
- **Leftovers.** `approve.ps1` (terminal approval: no hours or full-auto
  choice; it approves the entry hours with full-auto off) and
  `start_server.ps1` (no bootstrap record, so `stop_server.ps1` refuses to stop
  it) are unused; the device-pairing code and table in dashboard_access.py
  have no callers since the password login. Remove or keep deliberately.

### Ideas

- Prepare the next run while the previous one is analyzed (the shadow report
  now measures the gaps between runs).
- An AI conversation inside the Dashboard.

## 已完成 / Completed

### 2026-10-20 – 2026-10-22

- [x] Lease hours chosen on the Approval page (8/16/24, from the approval);
  entry hours must be one of them.
- [x] Home 停止核准 (re-authorization), simple vs full approval cards,
  continuation after a Recover, `workflows.end_kind` plus audit-log evidence.
- [x] Full-auto judgement (retry, incomplete after 2, outage auto-resume, 10 per
  lease, no decision while a lock is retained) with its log on Home.
- [x] `qm_cleanup.py` / `jy_close_stale_qm`; forced-stop lock release after a
  clean QM cleanup.
- [x] Shadow mode and `jy_get_autopilot_shadow_report`.
- [x] Static history site `runtime/history_site/` (`jy-agent history-site`).
- [x] The watchdog logs a failed run's event once.
- [x] Documentation synchronised with the code.

### Earlier (kept as history; corrections noted)

- [x] 儀器連線失敗改為可辨識的 pause 狀態；Dashboard 顯示錯誤與恢復步驟。
  （更新：狀態檔未變時 connectivity failure 也會釋放 lock。）
- [x] 跨 client 的 `恢復`／`Recover` close-state recovery 與雙語 [Command.md](Command.md)。
- [x] Dashboard 加入 AI App／外部網路中斷時的獨立 pause、stop、full-shutdown 指引。
- [x] 全節點首次全目標 multiplex，之後只重測 unresolved 子群；誤排已通過目標
  soft-reject。（更新：04 最多 5、07d 最多 4、共用 XY 一次 1；`FIXED_TARGET_NODES`
  已清空。）
- [x] ~~公開 Dashboard 每裝置 session、10 分鐘配對連結與裝置撤銷~~ → 已改為固定密碼
  登入與每瀏覽器 12 小時 cookie。
- [x] halted Dashboard 保留完整結束按鈕；retained lock 轉為 recovery quarantine，
  不再讓網站／MCP／tunnel 持續運作。
- [x] 本機 operator console 的 hardware-lock inspection／recovery（實體確認、
  receipt、audit）。
- [x] Worker stop 改為 authenticated cooperative request + exit receipt；grace 後只
  force-stop 經驗證的 PID。
- [x] `jy_wait_for_autonomy_status`、Codex durable goal、`已核准` 喚醒。
- [x] Repository root 成為唯一 client integration source。
- [x] P0：Dashboard 不信任 Host header（現為固定密碼與 secure cookie）；state commit
  與 hardware run 互斥；lifecycle mutex、instance nonce、PID 驗證。
- [x] P1：SQLite immediate transaction、唯一索引、schema migration、mutation
  idempotency；Codex MCP `required=false`；Doctor fail-closed。
- [x] 公開 `/` 為共同入口；首頁、核准、實驗結果、Token 用量、使用說明五個分頁。
- [x] Codex、Claude Code、Cursor 從 repository root 以入口語句啟動 JY。
- [x] Dashboard 雙語與摺疊面板；06b 固定接在 06 後（flux 校正且勾選 06a 時 06a 在中間）。
- [x] thermal 預設；active reset 資格（現為 active-reset 07b ≥ 0.75、blob 外 ≤15%，
  觸發條件 thermal 07b > 0.80；原文 ≥0.85 已過時）與 active 05/06/06b 複驗閘門。
- [x] 05st/06st 一次 100-repetition run（更新：2026-10-13 起 JY 逐次重擬合並過濾，
  原「移除逐次 fitting」已過時）。
- [x] 05/06/06b 至少 3.5 lifetimes 的統計前置門檻與 4.5x 建議時間窗。
- [x] README／WORKFLOW／UPDATE／TODO 與規則目錄整理；WireGuard 移除。

## Automatic-mode continuity — closed 2026-09-20

目標：自動模式在量出最終 T1/T2/fidelity 之前，只因儀器斷線無法處理或總時長到期
而停止。

- [x] **崩潰分類處理**。`autonomy.recoverable_worker_exceptions` 是 operator 可編輯
  的簽章表（node + exception type + message marker + 涉及的參數）。命中者保留
  lease 並記錄 `autonomy_recoverable_worker_exception` 與 remedy；未命中的 worker
  例外改為**暫停** lease（`autonomy_paused_for_worker_failure`），以 `恢復量測`
  復原，不再需要重新核准。目前登錄兩條：02c 小數 dBm、03a 在
  `freq_vs_flux_01_quad_term` 為零時的 `arbitrary_qubit_frequency_in_ghz`。
  （更新 2026-10-22：現登錄三條，含 DataFetchingError；`analysis_failure` 不暫停；
  全自動判斷模式下未命中者改為重跑，重跑 2 次後標為 incomplete。）
- [x] **這兩條同時做了事前預防**：02c 的 `min_power_dbm`/`max_power_dbm` 政策型別
  改為 `int`；03a 在 quad term 為零時直接拒絕 `arbitrary_qubit_frequency_in_ghz`。
- [x] **碰硬體前的拒絕一律不 halt**。policy violation、scope violation、state patch
  驗證失敗、deterministic setup 驗證失敗、過早 analyze，全部改為記錄
  `autonomy_action_refused` 後拒絕。`halt_conditions` 縮減為 `snapshot_missing`、
  `state_hash_conflict`、`hardware_lock_anomaly`。
- [x] `analyze_authorized_run` 先檢查 run 是否 completed，再跑 watchdog；過早呼叫
  不再因 watchdog 的 hardware-lock 檢查而中止 lease。
- [x] `_is_recoverable_scheduling_block` 改用 exception 型別
  （`AutonomyScopeError`／`AutonomyQuotaError`／`PolicyError`）而非字串比對。
- [x] lease 時長可在 `jy_enter_autonomy_mode(duration_hours=...)` 指定，上限為
  `autonomy.max_duration_hours`（預設 24），核准頁顯示實際時數。（更新 2026-10-20／22：
  每次核准在核准頁選 8／16／24 小時，進入時只能預選這三個值之一。）

Tests: `tests/test_autonomy_continuity.py`.

## 輕量模型決策支援 — 2026-09-20

- [x] `jy_get_next_action(workflow_id)`：確定性回報 current node、每顆 qubit 的
  resolved/incomplete/unresolved、剩餘 attempts、必要的 deterministic setup 工具，
  以及 `wait`/`analyze`/`record_decision`/`setup`/`run`/`advance`/`blocked`。
  `run.parameters` 只給與節點 defaults 不同的值，並附上套用的規則。沒有登錄規則
  時明說並要求讀規則書，不臆測參數。
- [x] `jy_get_decision_experience` 預設 limit 50→10，且改回傳 `evidence` 摘要
  （每顆 qubit 只留被 gate 的數值）而非完整 analysis 文件；需要時用
  `include_analysis=True`。
- [x] `snr_retry_recommendation` 從 04/05 擴及 03a（三者共用同一組
  `noise_confirmation_num_averages` 梯度）。

Tests: `tests/test_next_action_planner.py`.

## 06 Ramsey would have crashed on its first run — fixed 2026-09-20

`quam_libs/experiments/ramsey/analysis/fetch_dataset.py` called
`convert_IQ_to_V(ds, qubits)` unconditionally, and that helper does `ds["I"]`.
With `use_state_discrimination: true` — JY's configured 06 default — the node
only saves `state{i}` streams, so the first 06 run would have raised
`KeyError: 'I'` with no snapshot and no plot. The repository already contained
`tests/test_ramsey_state_dataset.py` asserting the guard, and that test had been
failing; the guard itself was never in the file's history. The conversion is now
skipped unless both `I` and `Q` are present.

This is a change outside the JY_agent package, in the shared calibration
library. `quam_libs` is not in `protected_paths`, but the node scripts under
`calibration_graph/` were not touched.

## Partial-pass state-commit gap — closed 2026-09-20

- [x] Direction 1 implemented. The analyzer now attributes every failure to the
  qubit that it names, keeps unattributable failures as
  `run_level_failure_reasons` that still block everything, and publishes
  `passing_targets`, `failure_reasons_by_target`, and
  `suppressed_patch_targets`. `candidate_state_patch` is filtered to the
  passing qubits instead of being emptied, and
  `_assert_partial_pass_state_commit` lets an autonomy lease commit a
  `needs_review` run only for those targets (policy key
  `autonomy.partial_pass_state_commit`). A non-passing target is refused
  without halting the lease. Tests: `tests/test_partial_pass_state_commit.py`.

Remaining from the same investigation:

- Direction 2 was not implemented: a target that was resolved *before* this fix
  and never committed still has no way to obtain a pass-backed commit, because
  the node refuses a further run for a resolved target. The operator accepted
  the stored 04 `q1`/`q2` values for that bring-up. Decide later whether to
  allow one extra run for a resolved-but-uncommitted target.
- `workflow_subgroups.resolve_node_targets` and the analyzer's per-target
  verdict are two separate implementations of "did this qubit pass". They agree
  today for the cases we have seen, but a check that exists only in the
  analyzer — the 02x/02a readout-upconverter exclusion, for example — can mark
  a target resolved while `passing_targets` excludes it, which recreates a
  narrow version of this gap. Make one of them the single source of truth.

Original report, for the record:

A run whose overall `analysis_status` is `needs_review` suppresses the state
patch for EVERY qubit in it, including the ones that passed. Combined with the
rule that refuses a further run once all targets are resolved, calibrated
values can end up permanently missing from `state.json` even though the node
reports complete.

Observed three times in workflow `0892399eba47422b96c9203e8ead1523`:

- 02c: nine targets produced passing dressed-plateau evidence across snapshots
  `#135`–`#137`, none committed; recovered only by re-running all ten once the
  gates were relaxed.
- 03a: snapshot `#152` resolved seven targets and committed none; recovered by
  re-running in batches expected to pass together, costing five extra runs.
- 04: `q1` (fitted x180 `0.0367134569` vs stored `0.0364876450`, 0.6 percent)
  and `q2` (`0.020853312` vs `0.020871838`, 0.09 percent) are still
  uncommitted, and no further 04 run is allowed because every target is
  resolved. The operator accepted the stored values for this bring-up and
  asked for the underlying gap to be fixed afterwards.

Directions considered, in `service.py` / the node analyzers:

1. Commit the per-target subset that passed inside a `needs_review` run,
   instead of suppressing the whole patch. — implemented.
2. Allow one more run for a resolved target whose accepted value was never
   committed, so a pass-backed commit can be obtained. — not implemented.

Either way, keep the rule that only passing evidence may be committed.

## Submission timeout is classified as an instrument disconnection — closed 2026-09-20

The operator's splitting rule says a multiplexed run that times out *on
submission* means too many qubits in one program: halve the group, keep
halving, and only treat it as a real disconnection when a single qubit still
times out. That rule cannot currently run autonomously.

`qm.execute` raising `QMTimeoutError` at the 100 s `QuantumMachinesManager`
deadline is classified as `instrument_unreachable`, which pauses the lease and
requires the operator's `恢復量測` phrase before anything can be scheduled
again. So every full-width first attempt at a large node stops the automatic
run and asks a human to inspect hardware that is demonstrably fine — the state
file is unchanged, the hardware lock is correctly not engaged, and the next
half-width run succeeds immediately.

Observed in workflow `5a1839c824e3432c971c925a133fb02a`, run
`b90300ae0a8c451aaa5aaaec267b017d`: 04 Power Rabi, ten qubits, 87 amplitude
points, failed at `qm.execute` with no snapshot; `hardware_lock_recovery_required`
was `false` and `termination_cause` was `instrument_unreachable`.

Distinguishing signal: the timeout happens in `execution_phase`
`hardware_execution` but before any snapshot exists, at the *submission* call,
and the group has more than one target. A genuine disconnection also fails with
a single target, and typically fails at `machine.connect()` rather than at
`add_to_queue`.

Direction: give the classifier a separate `program_submission_timeout` category
for a `QMTimeoutError` raised from `qm.execute`/`add_to_queue` with more than one
target in `parameters.qubits`. That category should not pause the lease; it
should let the agent halve the group and continue, exactly as the B' subgroup
allowance in `_validate_shared_parameter_first_batch` already anticipates. A
single-target submission timeout keeps the current pause-and-hand-off
behaviour.

**Closed.** `failures.classify_failure` now reports
`program_submission_timeout` when a `QMTimeoutError` carries a gRPC
`DEADLINE_EXCEEDED` status, no transport-failure marker, and a `qm` package
frame that submits a program (`execute` / `add_to_queue` / `_add_program` /
`compile`). Because that evidence still cannot separate an oversized program
from a wedged QOP, `instrument_probe.probe_qop` asks the instrument directly
right after the failure — resolve, connect, then ask the QOP to identify
itself — and the worker reclassifies back to `instrument_unreachable` when the
probe does not answer, recording `instrument_probe.cause` as `network` or
`instrument` so the operator message names the right thing to check.

A probe-confirmed submission timeout does not pause: `worker.py` leaves
`measurement_paused` false and `service._enforce_run_hard_stops` records
`autonomy_program_submission_timeout` instead of pausing for review, so the
agent halves the group on its own. A single target that still times out has no
room left to halve, so it pauses with a message naming the QOP job queue rather
than the network (with full-auto judgement, since 2026-10-20, that target is
marked incomplete and the lease resumes). `_node_full_width_submission_failed` accepts both the new and
the old termination cause, keeping the B' subgroup waiver working for runs
recorded either way. Tests:
`tests/test_submission_timeout_classification.py`, plus two cases in
`tests/test_workflow_subgroups.py`.

## A fetch-stage failure always retains the hardware lock — open 2026-09-20 (tracked above under Still open)

`worker.py` attempts `_stop_active_node_with_evidence()` only for
`instrument_unreachable` and `program_submission_timeout`. Every other
exception raised during `execution_phase: hardware_execution` therefore
retains the hardware lock without JY ever trying to stop the node, which puts
the whole server into recovery-only quarantine and needs an on-site operator at
the loopback console before any measurement can continue.

That is the right default when the instrument is unreachable — there is nothing
to ask. It is too strict when the instrument is provably healthy. Observed in
workflow `2dba0368b00c4b06aeed1a1ceabbb3d9`, run
`450064c6223f4f16bb1f71d6dbb15efc`: the 03a coarse program was submitted and
executed, then `results.fetch_all()` raised
`qm.exceptions.DataFetchingError: UNKNOWN: Unexpected error in RPC handling`.
A reachability probe immediately afterwards completed a cluster health check
against the QOP, so `QualibrationNode.active_node.stop()` would very likely
have succeeded and released the lock on verified evidence.

Direction: attempt the node stop for any failure in the hardware-execution
phase, not just the two recognized categories, and let the existing
`hardware_cleanup_verified` evidence decide whether the lock releases. Keep
retaining the lock whenever that stop cannot be confirmed. This does not weaken
the rule that an unverified process is never killed and a retained lock is never
deleted; it only stops JY from skipping the one check that could produce the
evidence.

Related: the operator decided 2026-09-20 that a repeated `DataFetchingError` at
the fetch stage should be treated like an oversized program and handled by the
halving ladder. Registering it in `autonomy.recoverable_worker_exceptions` would
also keep the lease alive so the agent can halve without an operator.
