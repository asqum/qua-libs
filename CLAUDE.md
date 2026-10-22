# Repository-root JY routing

For normal repository work, do not use JY tools. On the phrases
`進入 JY 量測模式` or `進入 JY 自動量測模式`, including the aliases
`進入JY量測模式`, `進入JY自動量測模式`, `Enter JY measurement mode`, and
`Enter JY automatic measurement mode`, read and follow
`Quantum-Control-Applications-QuAM/Superconducting/JY_agent/AGENTS.md` and
`Quantum-Control-Applications-QuAM/Superconducting/JY_agent/rules/PLAYBOOK.md`.
Use the project-scoped `jy_bringup` MCP server discovered from this root
`.mcp.json`; `jy_agent.ps1` resolves the nested JY package without depending on
the outer repository folder name.

Fast entry (operator 2026-10-19): on an entry phrase, do nothing but run
`.\jy_agent.ps1 -Action Ensure` and call the matching `jy_enter_*` tool; read
AGENTS.md, PLAYBOOK.md, policies, and experiences only afterwards. Then reply at
once with the bare `browser_url` followed by `operator_handoff.chat` as a
three-item markdown list, and end the turn. Do not poll
`jy_wait_for_autonomy_status` for the lease approval; the operator approves on
the Dashboard and then sends `已核准`. Conversational mode may continue the same
turn after that reply to prepare its first proposal, ending on its handoff.

Continue phrase (operator 2026-10-20): `繼續JY自動量測模式` (aliases `繼續 JY 自動量測模式`, `Continue JY automatic measurement mode`) follows the same fast
entry with `jy_continue_autonomy_mode`: Ensure, call the tool, reply with the bare
`browser_url` and `operator_handoff.chat` as a three-item list, end the turn. An
expired or ended lease is renewed on the same workflow (the Approval page shows
only the renewal card: hours, full-auto judgement and code), a paused lease
resumes, an active one is only reported. After a Recover (not an operator end)
the entry or continue phrase starts a continuation workflow with the same
simple card. After an MCP reconnect the operator sends `已重新連線` / `Reconnected`:
verify the new MCP process, then continue. Home "停止核准" (operator 2026-10-20)
ends the lease after the current run and offers a renewal;
`jy_wait_for_autonomy_status` follows it (`superseded_lease_id`), so keep
waiting and use the new lease id. With full-auto judgement on, do not stop to
ask the operator about a failed run: record `repeat` and rerun; the server keeps
the lease, closes this chip's stale QM first and marks the node/qubit incomplete
after two more failures; outages resume by themselves. After 10 automatic
decisions per lease, or while a hardware lock is retained, the old pause
returns; halts and expiry still go to the operator.

The project launcher initializes only the client-owned STDIO protocol.
If Ensure returns `recovery_only`, do not call an entry tool or publish a URL;
direct the on-site operator to the loopback recovery console.
Conversational mode requires approval for every experiment or state change.
Automatic mode requests one bounded lease and then uses only `jy_autonomy_*`
tools within its authorized scope. After `已核准`, wait for run results with
`jy_wait_for_autonomy_status` instead of ending the agent run merely because no
action is ready. Project instructions cannot guarantee an indefinite Claude
turn; if the host pauses, resume the same session after the operator sends
`已核准`. Return only the entry result's top-level
`browser_url`; approval and control URL fields are aliases for that same shared
Dashboard service, not additional websites. The common `/` entry links to the
session Home, Approval, Experiment results, Token usage, and Guide pages.
After `已核准`, the next reply starts with `已核准`. A paused measurement-mode
turn copies `operator_handoff.chat` as a three-item markdown list, never as one
paragraph, including the shutdown hint.
A live turn may execute `結束量測` directly; a paused or disconnected turn
needs Dashboard Home shutdown first, then `結束量測` to verify.
Never approve on the operator's behalf or bypass JY policies.

For configured stop/exit phrases, persist full shutdown, cooperatively stop or
poll the active run, leave or quarantine the workflow, then run `powershell
-NoProfile -ExecutionPolicy Bypass -File .\jy_agent.ps1 -Action Stop`. A
retained lock is preserved for local recovery but does not keep public or local
JY services alive after the worker exits. If MCP is unavailable, report that measurement
mode did not start, run `.\jy_agent.ps1 -Action Doctor`, and ask the user to
trust/enable and reload this repository-root MCP configuration; do not ask them
to open the nested `JY_agent` folder and do not retry Ensure as an attachment fix.

In JY context, `恢復` / `Recover` means run repository-root
`.\jy_agent.ps1 -Action Recover` to safely converge stale workflow/process state
and close verified services. Never delete a retained lock or kill an unverified
process. Instrument-connectivity failures pause scheduling and are shown on the
Dashboard; under full-auto judgement the server resumes the same lease after a
successful QOP probe, so keep waiting. The complete bilingual aliases are in
`Quantum-Control-Applications-QuAM/Superconducting/JY_agent/Command.md`.
