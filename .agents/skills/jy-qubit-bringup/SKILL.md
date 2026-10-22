---
name: jy-qubit-bringup
description: Operate JY superconducting-qubit measurement from the repository-root workspace when the user uses a Chinese or English JY measurement-mode entry phrase. Use the policy-gated JY MCP for conversational or bounded automatic calibration, approvals, snapshots, decisions, and safe shutdown.
---

# JY repository-root entry

Fast entry (operator 2026-10-19): on an entry phrase, run only Ensure and the
matching `jy_enter_*` tool, then reply at once with the bare `browser_url` and
`operator_handoff.chat` as a three-item markdown list, and end the turn without
polling for the lease approval. Read the nested instructions below after that
reply (automatic mode: after the operator sends `已核准`). The continue phrase
`繼續JY自動量測模式` (aliases `繼續 JY 自動量測模式`, `Continue JY automatic measurement mode`) uses the same fast entry with
`jy_continue_autonomy_mode`: it renews an expired or ended lease on the same
workflow (the Approval page shows only the renewal card: hours, full-auto
judgement and code), resumes a paused lease, or reports an active one; after a
Recover it starts a continuation workflow with the same simple card. Home
"停止核准" ends the lease after the current run, and
`jy_wait_for_autonomy_status` follows the renewal (`superseded_lease_id`); keep
waiting and use the new `authorization.id`. After an MCP reconnect, `已重新連線` /
`Reconnected` means: verify the new MCP process, then continue.

With full-auto judgement on, do not stop to ask the operator about a failed run:
record `repeat` and rerun; the server keeps the lease, closes this chip's stale
QM first and marks the node/qubit incomplete after two more failures; outages
resume by themselves. After 10 automatic decisions per lease, or while a
hardware lock is retained, the old pause returns; halts and expiry still go to
the operator.

Read the complete nested instructions before taking any other JY action:

- `../../../Quantum-Control-Applications-QuAM/Superconducting/JY_agent/AGENTS.md`
- `../../../Quantum-Control-Applications-QuAM/Superconducting/JY_agent/rules/PLAYBOOK.md`
- `../../../Quantum-Control-Applications-QuAM/Superconducting/JY_agent/rules/policies.yaml`

The project MCP launches through `../../../jy_agent.ps1 -Action Stdio`. Its path
is relative to the repository root, so never depend on the outer folder name.
STDIO startup is protocol-only. On either entry phrase, first run repository-root
`jy_agent.ps1 -Action Ensure` to establish the loopback services and Dashboard;
then call the selected entry tool. If Ensure returns `recovery_only`, do not call
an entry tool or expose a public URL; direct the on-site operator to the returned
loopback operator console and wait for formal lock recovery.

For `進入 JY 量測模式`, call `jy_enter_measurement_mode` with `client_id` and
`activation_phrase` only unless the operator explicitly overrides them. Omit
`targets` and `multiplexed`; a new workflow uses `state.json` `active_qubit_names`
and defaults `multiplexed=true`. Then propose exactly
one registered run or state change at a time and wait for each Dashboard approval.
For `進入 JY 自動量測模式`, call `jy_enter_autonomy_mode` the same way (omit
`targets` and `multiplexed` unless overridden), reply with the Dashboard and
handoff, and end the turn. After the operator sends `已核准`, confirm the lease
with `jy_wait_for_autonomy_status`, then use only `jy_autonomy_*`
execution/state tools within scope. That `已核准` for the automatic-mode lease
also authorizes a host that supports durable goals (for example Codex `/goal`)
to create one goal for this measurement session. Its objective is to keep
advancing the authorized JY workflow until one of these terminal conditions:
the workflow completes, the operator stops it, the lease expires, or a
server-enforced hard stop occurs. Do not create a durable goal for conversational
mode. While the automatic goal is active, use the bounded wait tool whenever no
action is ready instead of ending the task merely because approval or a worker
result is pending. Mark the goal complete only at a real terminal condition.
If the client has no durable-goal facility, keep the current turn alive while it
can, state that host continuity is not guaranteed, and rely on the Dashboard's
`已核准` wake-up instruction if the host pauses the conversation.
`進入JY量測模式` and `進入JY自動量測模式` select the same respective modes.
`Enter JY measurement mode` and `Enter JY automatic measurement mode` are the
exact English aliases.
For either mode, show only the entry result's top-level `browser_url`; approval
and control fields are internal navigation metadata for the same Dashboard and
must not be presented as extra websites. The public `/` entry links to Home,
Approval, Experiment results, Token usage, and Guide pages for the active
session.
Return the bare URL without a bootstrap query. Every browser uses the fixed
ignored `JY_agent/runtime/dashboard-password.txt`; do not create pairing links.
Never approve for the operator or bypass policy, snapshot, state-hash, lock, or
quota gates.
While measurement mode is open, any paused turn must copy
`operator_handoff.chat` verbatim as a three-item markdown list, never as one
paragraph: what to do, which phrase to send afterwards, and
`如需結束量測，請於網頁首頁結束量測後再對話輸入「結束量測」。`
Successful entry shows only `browser_url`. After `已核准` / `Approved`, the
next reply starts with `已核准`. A live turn may execute `結束量測` directly; a
paused or disconnected turn needs Dashboard Home shutdown first, then
`結束量測` to verify.

For configured stop/exit wording, request full shutdown and poll any active run.
The worker first receives a cooperative authenticated stop request; after the
grace period only its verified PID may be force-stopped. Then leave/quarantine
the workflow and run `powershell -NoProfile -ExecutionPolicy Bypass -File
.\jy_agent.ps1 -Action Stop`. A retained lock is never deleted: once no JY
worker is alive, it becomes a local recovery quarantine and does not keep the
Dashboard, MCP service, or public tunnel running. If `jy_bringup` is unavailable, do not claim entry
succeeded or retry Ensure as an attachment fix; run `jy_agent.ps1 -Action Doctor`,
then enable/trust and reload the repository-root project MCP.

If entry fails because stale JY process/workflow/lifecycle state remains, route
the JY-context command `恢復` / `Recover` to repository-root
`jy_agent.ps1 -Action Recover`. This path does not need the MCP, never removes a
retained hardware lock, and never kills an unverified process. Retry entry only
after a `closed` result; `closed_recovery_required` means services are closed but
local safety recovery is still required.

Treat recognized instrument-connectivity failures as paused measurement state,
not a generic crash. Do not schedule another experiment until connectivity is
repaired and the operator issues `恢復量測` / `Resume measurement`; then call
`jy_resume_measurement_mode` for the same workflow/session. Exception: under
full-auto judgement the server resumes the same lease by itself after a
successful QOP probe; keep waiting with `jy_wait_for_autonomy_status`. Restart the current
node as a new run rather than claiming to continue the interrupted Python stack.
Do not resume through a retained hardware lock; use local recovery. If the AI app disconnects, Dashboard controls
remain available; after reconnection, `結束量測` / `End measurement` verifies
full shutdown. Read
`../../../Quantum-Control-Applications-QuAM/Superconducting/JY_agent/Command.md`
when a bilingual control alias is needed.
