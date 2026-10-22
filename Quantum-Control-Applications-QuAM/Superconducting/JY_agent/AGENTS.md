# JY_agent instructions

Use the MCP tools exposed by the persistent JY server for superconducting-qubit
bring-up. Read `rules/PLAYBOOK.md` and `rules/policies.yaml` before proposing an
experiment, but not before entry.

Fast entry (operator 2026-10-19): on an entry phrase, run only Ensure and the
matching `jy_enter_*` tool, then reply at once with the bare `browser_url` and
`operator_handoff.chat` as a three-item markdown list, and end the turn. Do not
read files first and do not poll `jy_wait_for_autonomy_status` for the lease
approval; the operator approves on the Dashboard and sends `已核准`, and the
waiting/advancing loop below starts from that reply. Conversational mode may
continue the entry turn to prepare its first proposal and end on its handoff.
The continue phrase `繼續JY自動量測模式` (aliases `繼續 JY 自動量測模式`,
`Continue JY automatic measurement mode`; operator 2026-10-20) follows the same
fast entry with `jy_continue_autonomy_mode`: an expired or otherwise ended lease
of the open workflow gets a renewal proposal that copies its scope (the Approval
page shows only "續期授權，從 <qubit> / <node> 繼續", the hours (8/16/24), the
full-auto judgement choice and the approval code;
the workflow, current node and per-qubit pass are kept), a paused lease resumes,
an active lease is only reported. When a lease expires, the handoff's reply
phrase is that continue phrase; do not send the operator back to the full entry
form. After `/mcp` → `jy_bringup` → Reconnect the operator sends `已重新連線` /
`Reconnected`: verify the new MCP process (the expected tools are listed and
`jy_get_status` answers) and continue. Also read `rules/experiences/README.md` and the relevant
`rules/experiences/<node>.md` file when it exists; use accepted entries as starting
parameters only, never as a substitute for validating the current qubit.

On the exact phrase `進入 JY 量測模式`, first ensure the local infrastructure by
running `powershell -NoProfile -ExecutionPolicy Bypass -File
.\ensure_server.ps1` from this directory, or repository-root
`.\jy_agent.ps1 -Action Ensure`, when it is not already healthy. This is
an idempotent bootstrap, not a measurement: it starts or reuses the loopback MCP,
Approval HTTP service, and password-protected Cloudflare quick tunnel that exposes
only the Approval/Dashboard port. Never expose MCP or a non-loopback listener.
If the tunnel or local services cannot be established safely, report it and stop.
If Ensure reports `recovery_only`, do not enter either measurement mode or create
a public tunnel; use only the lab PC's loopback operator console until the lock
is formally recovered.

Treat `進入JY量測模式` as an exact whitespace-only alias, and likewise treat
`進入JY自動量測模式` as an alias of `進入 JY 自動量測模式`. Also accept
`Enter JY measurement mode` and `Enter JY automatic measurement mode` as exact
English aliases. The bilingual user command list is in `Command.md`.

After bootstrap, keep the two entry phrases distinct but use their returned shared
session dashboard URL for human review. `進入 JY 量測模式` calls
`jy_enter_measurement_mode` and starts conversational mode: discuss and propose
exactly one policy-registered experiment at a time, use `jy_list_experiments` and
`jy_request_conversational_run`, preserve and show the entry response
`browser_url`, wait for each approval on its Approval page, then execute/analyze it;
any state commit also appears there. Omit `targets` and `multiplexed` unless the
operator explicitly overrides them; a new workflow uses `state.json`
`active_qubit_names` and defaults `multiplexed=true`. `進入 JY 自動量測模式` calls
`jy_enter_autonomy_mode`, shows its session `browser_url` with the handoff and
ends the turn; after `已核准` it confirms the one bounded-lease activation with
`jy_wait_for_autonomy_status`, and then uses only
`jy_autonomy_*` execution tools. On that approval page the operator chooses the
lease hours (8, 16 or 24, default 8, counted from the approval) and whether
"全自動判斷模式" / full-auto judgement is on; an entry tool may pass
`duration_hours` only as 8, 16 or 24, which merely preselects the page. The
operator also ticks the optional nodes (`02`, `06a`, `05st`, `06st_t2star`, `06st_t2e`), chooses
one-qubit-at-a-time or multiplexed measurement from 03a on (a shared XY output
allows only one at a time), and chooses whether to calibrate flux (02b/03b,
plus the ticked 06a; a shared Z output disables it), and picks for every qubit
whether it is measured and where it starts (not measured, from the beginning,
or resume where it stopped; enter without `targets`). Tell them so with the
link, and follow `jy_get_next_action`, which already skips unticked nodes. The
Dashboard's Guide tab explains the operation and shows a successful example
per node. On hosts with durable goals, the `已核准` after automatic-mode
entry creates one session goal and keeps waiting/advancing until completion,
operator stop, lease expiry, or hard stop. On other hosts, keep the current turn
alive while possible but do not promise indefinite model lifetime; `已核准` is
the Dashboard wake-up fallback. Never run both modes against one live lease.
Outside those modes, do not call experiment or state-changing JY tools. Treat
unrelated requests as general conversation, require an exact entry phrase again,
and never silently abort an experiment.

Treat configured exit/stop wording such as `退出 JY 量測模式`, `結束量測`,
`停止量測`, and their spacing variants as a full service shutdown. Persist the
session shutdown intent first. If a run is active, submit its authenticated
cooperative stop request and poll it to a terminal state; after the configured
grace period, force-stop only the verified worker PID. Then call
`jy_leave_measurement_mode` (or `jy_stop_workflow`) and run `powershell -NoProfile
-ExecutionPolicy Bypass -File .\stop_server.ps1` from this directory. The script
must close the verified MCP service, Approval HTTP service, and public Cloudflare
tunnel. This is risk-reducing and needs no new approval. A retained hardware lock
is never removed automatically: after all JY workers are verified dead, the
script closes the services and leaves the lock in local recovery quarantine.
The next Ensure is local recovery-only until formal operator recovery succeeds.
Exception (2026-10-20): after a forced stop the server closes this chip's stale
quantum machine and releases the lock itself when no machine of this chip (or
sharing its ports) remains and the state hashes are unchanged; otherwise the
operator console offers "Close this chip's stale quantum machine" and then
recovery (`forced_stop_checkpoint`). Other users' machines on the shared OPX
are ignored.

If entry fails because stale JY services, a workflow, or lifecycle metadata was
left open, instruct the user to issue `恢復` or `Recover`. This close-state
recovery does not depend on an attached MCP: run repository-root
`jy_agent.ps1 -Action Recover`. It cooperatively stops a verified worker, closes
open workflow/lease/session state, then stops verified services and the tunnel.
It never deletes a retained lock or kills an unidentified process. Report
`closed_recovery_required` as services-closed/local-quarantine, not a crash or a
successful new measurement entry.

Classified QOP/OPX/instrument connectivity failures are a pause condition, not a
generic crash. Do not schedule a new experiment until the operator repairs
connectivity and resumes the same mode. Exception: under full-auto judgement the
server probes the QOP every 60 s and, once it answers, closes this chip's stale
QM and resumes the same lease itself; keep waiting with
`jy_wait_for_autonomy_status` instead of asking the operator. Dashboard controls remain usable if the
AI app disconnects; after reconnecting, `結束量測` / `End measurement` verifies
shutdown. Map `暫停 JY 自動量測` / `Pause JY automatic measurement`, `繼續 JY
自動量測` / `Resume JY automatic measurement`, `結束 JY 自動授權` / `End JY
automatic authorization`, and `緊急停止 JY worker` / `Emergency stop JY worker`
to their corresponding server controls.

The Dashboard may be local or a password-protected public HTTPS URL (each
browser signs in with the fixed lab password for 12 hours). Its stable `/`
entry resolves the active session and links to five views: Home (status,
measurement controls, the full-auto judgement state and log, full shutdown, and
how to sign in another device with the shared lab password), Approval (pending
approvals, or the authorization card when nothing is pending), Experiment
results, Token usage, and Guide. Every run remains
in the Results experiment-number dropdown until full service shutdown. Wait
for the human to review scope, limits, prior result plots, Decision, Reason, next
action, and payload, then type the exact confirmation on that page. The
interactive terminal command is fallback only. Never substitute or fabricate a
URL. Show the user only the entry result's top-level `browser_url`; nested page
URLs are internal navigation on the same Dashboard service, not additional
websites. Legacy `/approve/...`, `/autonomy/...`, and `/session/...` links redirect
to the appropriate current view. Successful entry replies with that bare URL.
After the operator sends `已核准` / `Approved`, the next chat reply starts with
exactly `已核准`. While measurement mode is still open, any paused turn must
copy `operator_handoff.chat` as a three-item markdown list, never as one
paragraph, so the operator always sees what to do, which exact phrase to send
afterwards, and the shutdown hint. A live turn may execute
`結束量測` directly; a paused or disconnected turn needs Dashboard Home
shutdown first, then `結束量測` to verify. See `Command.md`. Never use
the lease outside its targets/nodes or after pause, expiry, revocation, or halt.
Low confidence, `needs_review`, and `manual_review` do not themselves pause the
lease, but only a completed `pass` run with an identical recorded decision patch
may use unattended scientific state commit, except that a completed
`needs_review` run commits its `passing_targets` and nothing else
(`autonomy.partial_pass_state_commit`). Wait on bounded status/events between actions.
In automatic mode, each `(node, qubit)` has an independent 20-attempt budget. When one qubit reaches
that limit without usable evidence, treat it as incomplete for that node, do not
schedule it again under the lease, and continue resolved qubits into later nodes.
Quota exhaustion is not a lease-wide halt. Only `snapshot_missing`,
`state_hash_conflict`, and `hardware_lock_anomaly` (`autonomy.halt_conditions`)
end the lease. An out-of-policy parameter, out-of-scope target set, or premature
analyze is refused and recorded, and the lease continues; fix the request. A
worker crash matching `autonomy.recoverable_worker_exceptions` keeps the lease
and returns its remedy. With full-auto judgement on (`autonomy.auto_judgement`),
an unrecognized worker crash or a JY analyzer exception is not a reason to stop
and ask the operator: record `repeat` and rerun. The server keeps the lease and
first closes this chip's stale QM and its job; while it does,
`jy_autonomy_start_run` is refused with "new actions are blocked", so wait about
a minute and call again. After 2 automatic retries of the same node/qubit the
server marks that target incomplete at the node; record `advance` once every
target is resolved or incomplete. A single target that still times out on
submission is marked incomplete and the lease resumes; an instrument outage
resumes by itself after a successful QOP probe, so keep waiting. At most 10
automatic decisions are made per lease. Once they are used up, while a hardware
lock is retained, or with full-auto judgement off, an unrecognized worker crash
or analyzer exception, and a recognized instrument-connectivity failure, pause
the lease until the operator resumes it with `繼續 JY 自動量測` or `恢復量測`
(both are accepted for a paused lease). Halt conditions and lease expiry always stay with
the operator. Exception enforced by the server today: a `jy_autonomy_start_run`
refused with a plain service error, such as a node that is not yet
`current_node` ("record a decision first") or a later-node target set that does
not exactly match the resolved carried targets, still halts the lease. Check
`jy_get_next_action` first and never send the next node's run in parallel with
the decision that advances to it.
While the lease remains active, keep executing safe in-scope retries and later
nodes until the workflow completes. Stop scheduling a target only when evidence
shows it is scientifically unusable, its node/qubit quota is exhausted, lease
time expires, or a configured hard stop/operator control occurs. For 04 and 05,
use `jy_recommend_0405_snr_retry`: completion requires robust SNR >= 7 and >= 15
respectively (`analysis."04"/"05".min_robust_snr`). Only 03a and 04 climb the
`num_averages` ladder for low-SNR subgroups (04: 500 -> 4000); 05 runs at most
1000 averages (`nodes."05".max_num_averages`), and a 05 run still below SNR 15
at 1000 counts as failed rather than earning more averages.
When every safe authorized adjustment for an unresolved qubit is exhausted,
leave it incomplete, continue other resolvable qubits, and report the attempted
parameters plus observed evidence to the user. The Home page exposes five
operator controls: pause after the current run, resume the same unexpired lease,
"停止核准" / Stop authorization, end/revoke automation while keeping the site,
and emergency-stop the worker while keeping the site. "停止核准" lets the
current run finish, ends the lease, and puts a renewal card on the Approval
page; `jy_wait_for_autonomy_status` follows the new lease and reports
`superseded_lease_id`, so keep waiting and use the new `authorization.id`.
After a Home pause the Approval card asks only the hours and full-auto
judgement ("核准並繼續" / Approve and continue, no approval code; the hours
count from then). Home alone exposes full shutdown, protected by an exact second
confirmation; it closes the workflow, Dashboard, HTTP services, and tunnel.
When the user supplies a new
testing strategy, record it in the relevant experience/rule file before applying
it to later work. Treat `暫停 JY 自動量測` as `jy_pause_autonomy`. Dialogue stop,
exit, and shutdown wording performs the full workflow/service shutdown; an
emergency-stop request first uses `jy_emergency_stop_autonomy` and then closes
services after the run is terminal. These controls are risk-reducing and need no
new approval proposal.

This repository may be used by different internal users and MCP-capable agents.
From the repository root, Codex, Claude, and Cursor use `jy_agent.ps1 -Action
Stdio`, which resolves this directory without depending on the outer folder name;
opening this `JY_agent` directory directly remains compatible. Both paths reach
`start_mcp_stdio.ps1`, which initializes only the client-owned STDIO protocol.
Dashboard/network bootstrap stays outside MCP initialization and runs
idempotently only after an entry phrase, so a tunnel problem cannot corrupt the
MCP `initialize` response.
Use a stable, user-specific `client_id` for the audit log; always call status
before starting or resuming because workflows and the hardware lock are shared.

Never edit or generate files in `../calibration_graph/` or `../Script/`. Never
run those Python files directly, write active `state.json` directly, remove the
hardware lock, or approve a proposal on the user's behalf.

Treat node outcomes as advisory. Analyze the saved snapshot, state the Decision
and Reason, and give the next node plus new parameters. Keep user-facing run
reports to result plot(s) and two or three short sentences. Embed every returned
result plot directly in the conversation; a local file path or link alone is not
a completed report. Show `Decision` and `Reason` before the next required approval
appears on the preserved session dashboard (state commit first when required,
otherwise the next run). If a run
fails before producing a snapshot, explicitly state that no result plot exists
and explain the failure instead of omitting the plot silently.

Use thermal reset by default for every reset-capable experiment. Only an accepted
active-reset 07b result with normalized readout fidelity at least 0.75
(`reset_policy.active_min_readout_fidelity`; at most 15% of shots off both blobs,
`analysis."07b".active_reset_max_outlier_fraction`; two Gaussian blobs are also
required only while `analysis."07b".active_reset_requires_morphology` is true)
qualifies that qubit for explicit active reset downstream. The thermal 07b
result that earns the active-reset repeat still needs fidelity above 0.80
(`analysis."07b".active_reset_trigger_fidelity`), and state discrimination needs
07b fidelity at least 0.80 (`analysis."07b".min_readout_fidelity`) with two
Gaussian blobs. Before
any active-reset 05st/06st run, use the conditional prerequisite-verification
path to repeat and accept 05, 06, and 06b with active reset. After same-reset
baselines and their lifetime windows pass, run each statistics node once with
`histo_num=100`. JY itself refits every repetition, drops poor fits, and
records the filtered mean and spread (the result counts only with at least 90
completed repetitions, `analysis.statistics.min_completed_iterations`); do not
refit the raw iterations yourself or add a 100/100 secondary acceptance gate.

After an experiment is accepted, append a traceable entry to
`rules/experiences/<node>.md` using the schema in `rules/experiences/README.md`. Record useful
failed/recovery observations only as explicitly labeled lessons; do not mix them
with accepted settings. Never promote a parameter to experience without a saved
snapshot and the node-specific acceptance evidence.

Use `~/.qualibrate/config.toml` as the sole source for QuAM state, wiring, Data,
project, and calibration-library paths. Do not duplicate those paths in JY config.
