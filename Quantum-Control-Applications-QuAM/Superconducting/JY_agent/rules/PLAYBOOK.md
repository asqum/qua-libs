# JY superconducting-qubit bring-up playbook

## Non-negotiable rules

1. Do not start or resume a workflow until the user says exactly either
   `進入 JY 量測模式` (conversational) or `進入 JY 自動量測模式` (bounded
   automation). Configured exit/stop wording such as `退出 JY 量測模式`,
   `結束量測`, or `停止量測` always means full shutdown: persist the request,
   cooperatively stop/poll any active run (force only the verified worker after
   its grace period), stop or quarantine the workflow, and run `stop_server.ps1`
   to close MCP, Approval HTTP, and the public tunnel. A retained lock is kept
   for local recovery but does not keep those services alive. Outside measurement mode, do not call
   experiment or state-changing JY tools.
2. Treat `calibration_graph/` and `Script/` as read-only. Call only the six
   allowlisted calibration files through JY tools.
3. Read the active workflow and deterministic analysis before choosing an action.
   A node's own `successful` outcome is advisory, not proof.
4. In conversational mode call `jy_enter_measurement_mode`, preserve its session
   `browser_url`, then create exactly one `jy_request_conversational_run` proposal
   at a time. Every run/state approval appears on the Dashboard Approval page;
   results remain on the Results page and measurement controls on Home. The node may be
   any policy-registered experiment. In automatic mode call
   `jy_enter_autonomy_mode`, give the human its session `browser_url`, wait with
   `jy_wait_for_autonomy_status`, and use only `jy_autonomy_*` tools after the one
   lease activation. When the host supports a durable goal, keep one automatic
   session goal alive until completion, operator stop, lease expiry, or hard stop.
   A URL may be local or a
   token-protected public HTTPS URL; never replace it with a guessed URL.
   Preserve the same top-level `/` entry for the workflow/service lifetime. It
   links to Home, Approval, and Experiment results; every run remains in the
   numbered dropdown until shutdown. When ending a turn in measurement mode,
   copy `operator_handoff.chat` as a three-item markdown list, never as one
   paragraph, so the operator always sees what to do next, which exact phrase
   to send afterwards, and the shutdown hint. Successful
   entry shows only `browser_url`. After the operator sends `已核准`, the first
   line of the next reply is exactly `已核准`. A live turn may execute
   `結束量測` directly; a paused or disconnected turn needs Dashboard Home
   shutdown first, then `結束量測` to verify.
5. Never write `state.json` directly. Derive a JSON patch, explain each change,
   and apply it only through the lease's passing-evidence state-commit tool.
6. After every terminal run and deterministic analysis, directly embed every
   returned result plot in the conversation before continuing. A local file path
   or link alone does not satisfy this requirement.
7. Follow the plots with `Decision`, `Reason`, and `Next action` (`next node`,
   `new parameters`) in only two or three short sentences. The existing control
   page appends them automatically; do not create another approval page.
8. If a failed or stopped run produced no snapshot or plot, explicitly report
   that no result plot exists and state the failure reason before proposing any
   recovery action.
9. Before proposing a node, read its accepted and lesson entries under
   `rules/experiences/`. After accepting a run, append the complete reproducible
   setting and evidence to `rules/experiences/<node>.md`. Experience selects a useful
   starting point; it never bypasses the current node's acceptance checks.

## Workflow

The default sequence is:

`02 → 02x → 02c → 02a → 02b → 03a → 03b → 04 → 07d → 07b → 05 → 06 → 06a → 06b → 10a → 05st → 06st_t2star → 06st_t2e`

Operator instruction 2026-10-12: 05 (T1) runs after 07b, so every qubit's T1
is measured on the optimised readout.

`02`, `06a`, `05st`, `06st_t2star` and `06st_t2e` are optional (operator request
2026-10-04; `workflow_options.optional_nodes`). `02b`, `03b` and the ticked
`06a` run only when the same page also says yes to flux calibration
(`workflow_options.flux_calibration`; see "Flux calibration (02b, 03b, 06a)"). The operator ticks them on the lease approval page; an unticked
node is removed from that workflow's sequence and from the lease, so `advance`
names the next ticked node and `jy_get_next_action` reports it. Do not request
an unticked node and do not ask the operator to approve it separately. The same
page sets the measurement mode from 03a on (see "A shared XY drive output runs
one qubit at a time"); automatic measurement starts only after both choices.

The operator places already-correct resonator frequencies in `state.json`
before entry. Resonator spectroscopy (`02x`, `02c`, `02a`) only fine-tunes
those values; it is not a wide search. Keep `frequency_span_in_mhz` at most
60 MHz. The protected-node fitter will treat the readout
`upconverter_frequency` (IF = 0, equivalently detuning `−IF` from the stored
resonator RF) as a false resonance if that frequency falls inside the sweep,
so never widen a window until it includes the LO.

`06b` is mandatory and runs immediately after `06` in the default workflow.

The sequence is dynamic rather than a pre-approved batch. After every run:

1. Wait on bounded status/events until the worker reaches a terminal state.
2. Call `jy_analyze_run`.
3. Inspect the result plots, fit evidence, edge distance, robust SNR, candidate
   state patch, and policy warnings.
4. Directly embed every returned plot in the conversation.
5. Choose `advance`, `repeat`, `manual_review`, or `stop`, then report the
   `Decision` and `Reason`.
6. Record the decision and next parameters. For an accepted run, record the
   successful settings and evidence using `rules/experiences/README.md`; record a
   useful failed/recovery observation only as a labeled lesson. Commit any state
   patch only through the active lease and identical passing evidence.
7. Continue within scope after the result report. Explicitly say when the
   workflow is complete, stopped, expired, or halted.

Every run retains its complete parameters, start/finish timestamps, elapsed time,
snapshot, analysis, Decision, Reason, and Next action in the SQLite audit database.
Use `jy_get_run_telemetry` for duration forecasts and
`jy_get_decision_experience` before choosing parameters for a similar run. The
approved autonomy URL becomes an event-driven, append-only dashboard. It reloads
only for a newly analyzed result/decision or a manual/automatic control-state
change, and it keeps every run, plot, parameter set, elapsed time, Decision,
Reason, and Next action from that workflow/service lifetime. A plot can appear only after the worker
has produced a snapshot; intra-acquisition partial plots remain a future feature.

Manual retry proposals have no lifetime total limit. A single run approval may
cover at most the policy-configured number of unattended retries and expires.

In bounded autonomy, every `(node, qubit)` has its own 20-attempt counter. If a
qubit reaches 20 attempts without usable evidence, report it as incomplete for
that node and never request attempt 21 under the same lease. Quota exhaustion is
target-local: finish the remaining qubits, record `advance` when every active
target is either resolved or incomplete at quota, and carry only resolved qubits
into the next node. Do not treat this as a lease-wide halt. Only the configured
hard stops `snapshot_missing`, `state_hash_conflict`, and `hardware_lock_anomaly`
(`autonomy.halt_conditions`) halt the entire lease. Parameter-policy and scope
refusals are recorded and the lease continues (see "A refusal before the
hardware never ends the lease", including its `jy_autonomy_start_run`
exception); worker crashes are classified (see "A run crash is classified");
instrument-connectivity failures pause (see "Instrument-connectivity failures
never engage the hardware lock").
While the lease remains active, continue scheduling safe in-scope work until the
workflow finishes. Do not stop merely because a run is `needs_review`: retry an
unresolved subgroup, increase evidence quality within policy, or advance the
already resolved qubits after the other targets become scientifically unusable,
reach their local quota, or run out of lease time.
If every safe authorized adjustment for a qubit has been exhausted, leave it
incomplete, continue the remaining resolvable qubits, and give the user a compact
summary of attempted parameters and observed evidence. When the user explains a
new strategy, add that instruction to the relevant experience/rule file before
using it in later runs.
Before a new autonomy lease is created, the server revalidates previously
advanced 04/05 evidence against the current SNR floors. If an older advancement
no longer qualifies, the workflow deterministically reopens the earliest affected
node and records `workflow_evidence_revalidation`; the new lease must start there.

`暫停 JY 自動量測` pauses only the lease. Dialogue exit/stop/shutdown wording
permanently stops the workflow and closes MCP, Approval HTTP, and the public
tunnel after the active run is safely terminal. If cleanup cannot be verified,
the lock survives in recovery quarantine and the next Ensure is local-only.

## Parameter handling

- Always pass explicit `qubits`; multi-qubit runs are allowed.
- `02x` and `02a` retain their hardcoded minimum-flux behavior.
- `02c`, `03a`, `04`, and `05` must use `joint`.
- Stay within the server-returned policy limits. For 03a, the full sweep must
  stay inside the ±400 MHz qubit IF range. For `02x`, `02c`, and `02a`,
  `frequency_span_in_mhz` must stay at most 60 MHz and the IF window must not
  include the readout `upconverter_frequency`.
- If x180 is missing or zero before 03a/04/05, call `jy_request_bootstrap`.
  Bootstrap x180 is `0.5 × max_x180_wf_amplitude`; a missing/zero x90 is set
  to `x180 / 2`. Never overwrite an existing nonzero x180 or x90.
- A qubit with no JY run on the chip that `config.toml` names is measured from
  scratch (operator 2026-10-14). This is usually a state folder copied from
  another chip, whose old values would mislead every node. Before that
  qubit's first run, call `jy_request_fresh_qubit_defaults`; the server
  refuses any run of it until that commit is applied. The commit:
  - empties `extras`;
  - sets T1, T2ramsey and T2echo to 100 µs and anharmonicity to 200 MHz;
  - sets saturation to amplitude 0.5 and length 20000 ns;
  - sets the readout length and depletion time to 2000 ns;
  - resets threshold, rus_exit_threshold and integration_weights_angle to 0
    and the confusion matrix to identity;
  - sets the z offsets and `freq_vs_flux_01_quad_term` to 0;
  - sets x180 amplitude to 0.05, x90 to half of it, and the DRAG
    alpha/detuning to 0;
  - sets the gate length to 400 ns on a shared XY line or 40 ns on a private
    one.

  The values are in `fresh_qubit_defaults` in `rules/policies.yaml`. A qubit
  with any run on this chip keeps its state. This replaces bootstrap for
  fresh qubits.

### Multiplex workflows

Phrase-only entry (`進入 JY 量測模式` or `進入 JY 自動量測模式` with no extra
fields) creates a new workflow from `state.json` `active_qubit_names` and
defaults `multiplexed=true`. The server then forces `multiplexed=true` on every
run proposal and refuses attempts to disable it. Set the intended qubits in
`active_qubit_names` before entering; do not ask the operator to type `target`
or `multiplex` in chat. Explicit `targets` or `multiplexed=false` remain
optional overrides for tests and special cases.

For the q3–q8 profile, `active_qubit_names` should be
`["q3", "q4", "q5", "q6", "q7", "q8"]`.

Before creating the workflow or a run proposal, server policy validates target
count, resonator input/output wiring, same-line IF spacing, and aggregate readout
amplitude. Multiplexing applies to the qubits within one node; calibration nodes
still execute one at a time with a separate approval decision and analysis for
each, all displayed on the same session Dashboard.

Minimize the number of hardware runs. The first attempt of every node must
measure all currently-active workflow targets together with one shared parameter
set. Do not start a node by measuring one qubit at a time. If only some qubits
fail, retry those unresolved qubits together with the same modified parameters
and omit already-resolved targets. Split into separate multiplex subgroups only
when those qubits scientifically cannot share the sweep (different frequency or
power window, incompatible statistics wait time, or a 03a stage/LO constraint).
A singleton retry is a last resort after a shared-parameter multiplex retry, not
the default first action. None of this applies to a chip that shares an XY drive
output: there the workflow is a qubit-by-qubit pass from 03a onward, and the
section "A shared XY drive output runs one qubit at a time" replaces this one
for every node in that tail. Scheduling already-resolved targets again is a
recoverable planning error: the server rejects the request without hard-stopping
an active autonomy lease.

## Node decisions

### 02 — broadband resonator survey

`02_Broadband_Spectroscopy.py` sweeps every readout line once over its LO
±`max_if_in_mhz` (default ±400 MHz) at the 02x probe power, removes the
electrical delay and the slow baseline, and lists resonator candidates. Run it
once with every workflow target and the node defaults. It is single pass and
advisory: whatever it finds, `advance` to 02x next.

- Without `res_design_freq` (the default) it is a survey and proposes no state
  change. Check its per-line candidate count against `num_expected` and report
  a line with too few candidates, but 02x still runs.
- With `res_design_freq` (only when the operator supplies one: q1..qn design
  frequencies, length `res_num`), one global shift is fitted; only qubits whose
  candidate lies within `design_match_tolerance_in_mhz` (10 MHz) of the shifted
  design move `resonator/intermediate_frequency`. A `mismatch` or `not_found`
  qubit fails and keeps its stored frequency.
- Ticking 02 on the lease page requires the operator to enter `res_num` and
  `res_design_freq` (GHz). They are stored on the workflow, shown as
  `operator_parameters` by `jy_get_next_action`, and merged into every 02 run
  automatically; do not pass different values.
- Operator rule 2026-10-07: when any target's readout IF is 0 or equals a
  line-mate's, entry is not refused. 02 becomes required: the lease page shows
  it ticked and locked, and approval needs `res_num` and `res_design_freq`. Only
  02 may run with unset readout IFs (it skips the resonator-spacing check);
  every later node still enforces it, so 02 must commit distinct IFs first.
- Same day: a readout line whose amplitudes sum above
  `max_aggregate_readout_amplitude` (0.5) no longer blocks entry either. The
  planner lists `jy_request_readout_amplitude_budget` as required setup; call
  it with the lease id and apply it with `jy_autonomy_apply_setup_state`. It
  scales the line by one factor to `readout_amplitude_budget_target` (0.45).
  Every run still enforces 0.5, so apply it before the first run.
- Same day: when an operator design frequency lies outside a readout line's
  LO +- (`max_if_in_mhz` - margin), or inside the LO guard, the planner requires
  `jy_request_readout_lo_recenter` at 02. It moves that line's output and input
  LO together on the 100 MHz grid, keeps every assigned qubit's RF (its IF is
  recomputed), and leaves unassigned qubits at IF 0. 02 runs from before the
  move no longer count, so 02 runs once more. First case: CRSQ_26009_die(-1,3),
  LO 5.25 GHz covered only 4.85-5.65 GHz while the designs span 5.38-5.89 GHz.
- A target whose resonator 02 cannot find (no dip near its shifted design, no
  unassigned dip elsewhere) is marked scientifically unmeasurable at 02 and is
  dropped from every later node, so 02x never sweeps around an unset IF. First
  case: CRSQ_26009_die(-1,3) q7 (design 5.48 GHz), absent at both LOs.
- A new chip's `extras` may lack the keys the nodes record (02x crashed with
  `KeyError: 'bare_resonator_freq'` inside Qualibrate's state recorder). When
  `jy_get_next_action` notes missing extras keys, call
  `jy_request_node_extras_prerequisites` with the lease id and apply it before
  the next run. It adds every key listed under
  `autonomy.node_extras_prerequisites` for the remaining nodes. Frequency keys
  get the current RF; other keys get 0.0. Each node then overwrites its key.
- Long template readout pulses (CRSQ: 100 us) make every multiplexed point
  wait for the longest one. When the planner notes it, apply
  `jy_request_readout_length_normalize`, which caps length and depletion time
  at `autonomy.readout_length_normalize_ns` / `readout_depletion_normalize_ns`
  (2000 ns). 07d later picks each qubit's length.
- Operator choice 2026-10-07: line members marked scientifically unmeasurable
  no longer hold a readout line at its template full scale. Once the line's
  measurable qubits passed 02c, the planner notes it. Apply
  `jy_request_readout_full_scale_rescale`, which moves the line to the smallest
  integer full scale within the pulse and sum limits and keeps each qubit's
  delivered power. On CRSQ, con1/7/1 went from +10 to -11 dBm, with amplitudes
  0.017-0.073.
- The LO ±`lo_guard_in_mhz` region is excluded (LO leakage), and features
  narrower than `min_linewidth_in_mhz` are spurs, not resonators.

### 02x — bare resonator spectroscopy

`02x` only fine-tunes the operator-supplied resonator RF, typically by a few
MHz. Start from `state.json` and the 20 MHz default; do not treat 02x as a
search over tens of MHz. Advance only when every target has a visible resonator
feature, robust SNR of at least 3, and the feature is not at a sweep edge and is
not the readout `upconverter_frequency`. If a feature is clipped, recenter the
stored RF or take a small span increase still at most 60 MHz; never widen until
the LO/upconverter lies inside the window. Increase averages or adjust the step
when the trace is noisy. The protected node records
`extras/bare_resonator_freq`; if that key is absent, initialize it to the
qubit's current resonator RF frequency through a separately approved
state-commit proposal before running 02x. The first 02x run must multiplex every
active target. Passing qubits accumulate; retry only unresolved targets that can
share the same span/averages change. After every active target is resolved,
advance to `02c`.

### 02a — dressed resonator spectroscopy

`02a` runs after `02c`. It is only a simple dressed-frequency scan around the
already tracked resonator, not a power-dependent or wide search. Keep the span
at the 20 MHz default unless a few-MHz recenter is required, and never exceed
60 MHz. Advance when the dressed resonance is resolved for every target, the
feature is not the upconverter, robust SNR is at least 3, and the proposed
frequency change is finite and within IF limits. If a feature is at an edge,
recenter rather than opening a window that includes IF = 0. In bounded autonomy,
the first 02a run must multiplex every active target with the same sweep. Passing
qubits accumulate across runs; retry only unresolved targets that can share the
same span/averages change, and never remeasure an already-resolved target.
After a
passing 02a, advance to `03a`.

### 02c — resonator spectroscopy versus amplitude

`02c` runs immediately after `02x` and before the simple dressed scan `02a`.
Validate the tracked resonator frequency as power increases from low power
(more-negative dBm) to high power. The low power end shows the
`dressed frequency` and the high power end shows the `bare frequency`.

A run passes when those two frequencies can be told apart: they are separated by
at least the configured frequency distance, each is observed on at least
`min_classified_plateau_points` consecutive classifiable power points, and
neither sits within 5% of a frequency-sweep edge. The power segment in between,
where a point is neither clearly dressed nor clearly bare, is discarded rather
than judged. Plateau stability, plateau width, depletion-point count, and
transition monotonicity are reported as advisory notes and no longer reject a
run on their own; re-enable one through its `require_*` rule in `policies.yaml`
if a target needs the stricter gate.

The target readout power is the highest power that is still classified as
dressed, taken from the usable low-power part of the trace. It is deliberately
not pushed to the last point before punch-out. JY's deterministic analysis is
authoritative for the proposed frequency and power; the protected node's
derivative-threshold fit is advisory and must be ignored when it is empty or
disagrees with the dressed boundary.

Operator instruction, 2026-10-02: the committed readout power is that dressed
boundary minus `analysis."02c".readout_power_backoff_db` (3 dB since
2026-10-13; 10 dB before), so 07d and 07b start below the bare-resonator
transition. Each readout line's
`full_scale_power_dbm` (the MW-FEM port shared by its qubits) is then set to the
smallest integer in [-11, 16] dBm for which every pulse amplitude stays at or
below `max_readout_pulse_amplitude` (0.125, as in 07d) and the line's summed
amplitude stays at or below `instrument_limits.multiplex.max_aggregate_readout_amplitude`
(0.5). JY derives this; do not request it. The port changes only when every
qubit on that line has a new 02c point: a subgroup retry, or a `needs_review`
run where a line-mate failed, keeps the current full scale and expresses the
passing qubits' powers as amplitudes at it.

Operator choice 2026-10-14 (CRSQ_26009_die(5,-1) q3, `#507`): when a target's
02c point does not fit the current full scale (amplitude above 0.125 or line
sum above 0.5), its transition does not resolve the target -- the planner
keeps it unresolved instead of offering `advance`. Fix it by raising the line
full scale with an operator-approved `jy_request_state_commit` (no lease id,
`run_id` = that 02c run). The patch sets the port's `full_scale_power_dbm`,
the target's IF / amplitude / `extras.dressed_resonator_freq`, and rescales
the readout amplitude of every other qubit on the line by
10^((old - new) / 20) so its delivered power is unchanged. Non-target line
members are accepted only in that power-preserving form. Once committed, the
target counts as resolved.

Operator rules, 2026-10-05:

- **Window 2-5x the separation.** A target passes only when the frequency window
  is between `min_window_to_separation_ratio` (2) and
  `max_window_to_separation_ratio` (5) times its bare-dressed separation, so the
  shift is visible in the plot. The first 02c is one multiplexed 16 MHz run
  (the default) that finds each separation; a target outside the band fails
  with the recommended span (3.5x) in its failure text and in
  `recommended_frequency_span_mhz`. Before the narrow retry, call
  `jy_request_02c_window_recenter` (with the lease id, then
  `jy_autonomy_apply_setup_state`) to put each readout IF at the dressed/bare
  midpoint, then retry with that span. Targets with similar recommended spans
  share a run; others run separately. Pick the frequency step at about span/60.
- **Shift the power window, keep 40 dB.** A smeared or unstable low-power dressed
  plateau now blocks (`require_dressed_plateau_quality`), as does an obvious
  power break. Rescan with `min_power_dbm`/`max_power_dbm` shifted together; the
  policy requires `max - min = power_window_db` (40). Example: q7 on
  2026-10-05 showed a noise band below -44 dBm read as the dressed plateau;
  -42..-2 dBm excludes it.

02c fails only when dressed and bare still cannot be told apart after the
parameters have been driven to their policy limits — the frequency window up to
60 MHz (excluding the readout `upconverter_frequency`) and `num_averages` at
100. Missing, edge-clipped, or inseparable frequencies below those limits mean
widen the window or add averages and repeat. At the limits, treat the target as
manual review or scientifically unmeasurable instead of spending more attempts.
Never schedule `num_power_points` above 30 or `num_averages` above 100.

When an unresolved target's dressed/bare plateau transition is clipped by the
frequency-sweep boundary, spans most of the current sweep, or cannot be fully
contained in the current window, widen `frequency_span_in_mhz` before treating
the target as scientifically unusable or spending repeated attempts at the same
span, but stay at or below 60 MHz and never include the readout
`upconverter_frequency`. Keep qubits together when they can share the same
frequency and power window. Split only the unresolved qubits that actually need
a different window, preserve `multiplexed: true`, and reapply the normal
separability, edge-distance, and state-commit evidence gates to the wider scan.
Do not repeat an already resolved target merely because another target in the
same numeric range needs a wider frequency window.

The first 02c run must multiplex every active workflow target with the same
parameters. An 02c retry may then measure the unresolved subset while preserving
`multiplexed: true`. Group every qubit that needs the same frequency or power
change into one retry; do not walk the chip one isolated qubit at a time.
Do not advance to 02a until every original target has either a valid dressed
point or satisfies the strict bare-only absence rule.

A qubit may be marked `bare_only_absent` and skipped in every later node only
when all safeguards pass: the full power trace is a straight line matching the
previously measured bare frequency, minimum power reaches at least -50 dBm, the
configured frequency span is at most 10 MHz with step at most 0.1 MHz, at least
30 power points are present, the bare resonance remains resolved and away from
sweep edges, and either readout integration is at least 1000 ns or
`num_averages` is at least 200. Every state update still requires a separate
human-approved state-commit proposal.

### 03a — qubit spectroscopy

Require `fit_successful=true`, a finite drive frequency, adequate SNR, and a peak
away from sweep edges. If no peak is found, first adjust drive amplitude/averages,
then span and step while preserving the ±400 MHz IF limit. Do not accept a
clamped or non-finite x180 proposal.

Before the first 03a run, call `jy_request_initial_03a_zero_if` and
apply its separately human-approved state commit. It must preserve each active
target's current RF frequency by moving that target's private LO to the RF and
setting every XY IF to zero. The first coarse run must then measure all active
workflow targets together with `multiplexed: true`, an 800 MHz full span, the
configured stronger but policy-safe saturation factor, and enough coarse points
to avoid stepping over a resonance.

The 03a `operation_amplitude_factor` cap follows the XY wiring (operator
2026-10-14), because the wiring sets how much drive reaches the qubit. A qubit
on a shared XY line (one qubit per run, weak drive) may use up to 1.0, and its
first coarse run uses exactly 1.0. A qubit on a private XY line
(multiplexable) stays at or below 0.05, and its first coarse run uses exactly
0.01. On a private line every scan of at most 200 MHz therefore counts as a
fine scan.

Never accept the protected node's fit or success outcome by itself. A coarse peak
is a credible candidate only when the fit is finite and successful, the feature
is at least 5% away from either sweep edge, and robust SNR is at least 6. The
signal prominence must visibly exceed the surrounding noise; multiple noisy
local extrema are not a candidate even when the node reports a fit. High-power
coarse runs must not produce a state patch or count as final 03a evidence.

At higher drive power, explicitly check for a paired-transition signature before
accepting the fitted frequency. A narrow lower-frequency peak can be the
two-photon `|0> -> |2>` transition divided by two, and may be stronger than the
fundamental. When another credible, distinctly broader peak appears roughly
`|anharmonicity| / 2` above it (commonly about 100 MHz; the configured detection
window is 50-150 MHz), treat the lower narrow peak as the two-photon candidate
and select the higher-frequency peak as the `|0> -> |1>` candidate. Record both
peaks, their separation, widths, and SNR. The protected node's strongest-peak
fit must be overridden when it selected the lower member of such a pair. This
pair rule only chooses a coarse candidate; the higher-frequency result must
still pass a separately centered low-power fine scan.

For an unresolved target whose strongest feature is inside the sweep but robust
SNR is below 6, keep the same LO window and increase `num_averages` through
500, 1000, and at most 2000 as needed. Do not move its LO before this noise
confirmation reaches 2000 averages. If no credible peak remains at 2000
averages, it may move to another LO window. An edge-limited feature may shift
immediately because the current window does not cover it adequately.
When a high-SNR coarse candidate fails reproducibility at the same power and
2000 averages, record `manual_review` to explicitly reject that candidate.
The rejected target is no longer treated as found, becomes eligible for an LO
window shift, and may regain candidate status only from a later credible run.

Use `jy_request_03a_window_shift` only for targets that satisfy the preceding
shift rule. Move that unresolved subgroup to explicitly chosen 100 MHz-grid LO
centers with IF=0, then repeat the 800 MHz coarse search. Never shift an already
found target or a shared XY output automatically.

After each target has a credible candidate, lower the drive power and then run a
centered refinement/fine scan. Candidate targets may use an intermediate
refinement stage (for example amplitude factor 0.075) when a direct drop from
the coarse factor to 0.05 makes a previously reproducible peak disappear.
Refinement scans never produce a state patch and must still be followed by the
final low-power scan.

Do not chase ever-narrower frequency windows. A final 03a scan does not need a
40–50 MHz span; about 200 MHz is already fine enough once the LO/IF is centered
on a credible candidate. Prefer keeping the span near that scale and fixing
linewidth or morphology with drive amplitude instead. A final 03a result
requires a span no greater than 200 MHz, an operation amplitude factor no
greater than 0.05, a stable finite fit, robust SNR of at least 6, and a feature
at least 5% away from either sweep edge. Only this fine-scan evidence may
update idle frequency or resolve the target.

The final feature FWHM must also lie between 0.5 and 16 MHz. A still-broader
feature is power-broadened and must not resolve the target even when fit and SNR
pass; reduce `operation_amplitude_factor` and repeat at about the same ~200 MHz
window. If many peaks appear in one window (noisy multiplet or power-split
structure), lower the amplitude first rather than shrinking the span; weaker
drive usually collapses the clutter onto the fundamental. If the feature becomes
narrower than 0.5 MHz, slightly raise amplitude or increase resolution before
accepting it. Only commit `idle_freq` after the measured FWHM falls inside this
interval.

Before a candidate's refinement or fine scan, call
`jy_request_03a_candidate_center` when its stored RF center does not match the
credible coarse candidate. The server must derive the RF from the latest
non-rejected credible candidate, place the private XY LO on the nearest 100 MHz
grid, retain the residual IF, and validate the resulting state patch. Under an
active bounded-autonomy lease this deterministic setup may be delegated and
applied with `jy_autonomy_apply_setup_state`; it must never center a resolved,
non-candidate, shared-output, or out-of-policy target.

When `arbitrary_qubit_frequency_in_ghz` centers a fine or refinement scan, derive
RF as that explicit center plus the measured feature offset. Do not combine the
offset with the stored LO, and reject any protected-node frequency or state patch
that falls outside the actual requested sweep. A 03a scan that fails SNR, edge,
or fit checks must expose no candidate state patch.

An 03a run may measure any non-empty subset of the active workflow targets while
preserving `multiplexed: true`. Passing qubits accumulate only under these
rules. After the all-target coarse scan, retry unresolved qubits that can share
the same span, amplitude, and averages together; isolate a qubit only when its
LO/stage constraint differs. Do not advance to 04 until every active target has
final fine-scan evidence.

### 04 — power Rabi

Require a finite, nonzero fitted Pi amplitude and an oscillation that is not
edge-limited. Reduce `max_amp_factor` when the policy reports a waveform-limit
violation. Synchronizing x90 from an accepted x180 result is allowed, but its
state patch still requires human approval.

For 04 edge evidence, use the fitted `Pi_amplitude` relative to the recorded
amplitude-sweep minimum and maximum. A generic raw maximum/minimum can select a
small noisy local extremum even when the fitted Rabi extremum is visibly in the
interior; preserve that raw coordinate only as diagnostic evidence and do not
let it override a finite interior Pi fit. A fitted Pi amplitude within 5% of the
actual sweep boundary is still edge-limited and requires a repeat.

An 04 run may likewise measure any non-empty subset of the active workflow
targets while preserving `multiplexed: true`. The first 04 run must multiplex
every active target with the same sweep. Passing qubits accumulate across
runs when the Pi amplitude is finite and nonzero, robust SNR is at least 7
(`analysis."04".min_robust_snr`; operator instruction 2026-09-20 lowered it
from the 14 anchored to the q3/q4/q5 reference run, because a clean single
oscillation with a finite interior Pi fit is already correct below that), and
the feature is at least 5% away from either sweep edge. Retry unresolved qubits
that need the same averaging or amplitude-range change together. Do not advance
to 07d until every active target is resolved. If a
trace is below the floor, call `jy_recommend_0405_snr_retry` and increase only
that subgroup through 500, 1000, 2000, then 4000 averages.

Gate time (operator instruction 2026-10-12). When the last 04 already swept to
the amplitude limit (`max_amp_factor` x amplitude >= 90% of the waveform limit)
and the signal changes without turning over -- still rising or falling -- call
`jy_request_04_gate_time` to lengthen the XY gate, then run 04 again. The gate
may reach 400 ns on a shared XY line and 100 ns on a private one
(`xy_gate_length_ns` in `rules/policies.yaml`), in 4 ns steps.

No Rabi, back to 03a (operator instruction 2026-10-12). A qubit that never ran
03b (no flux evidence for its line) and shows no oscillation at all -- robust
SNR below 4 at the 4000-average step, or a failed run at the longest allowed
gate -- may return to 03a: record `advance` to 03a without a mark. That
evidence counts only from a run at full drive (operator instruction
2026-10-12, q3: a 32 ns gate swept to 4% of full scale): the amplitude swept to
>= 90% of the waveform limit (`analysis.04.return_min_amplitude_fraction`)
at the longest allowed gate. So for a flat 04, first raise `max_amp_factor`
to reach the limit, then `jy_request_04_gate_time` to the cap, then rerun. The
server
masks that line (+-max(5 MHz, its FWHM)) for the qubit, voids its earlier
03a/04 evidence and marks, and 03a starts again with the zero-IF setup. At most
three returns per qubit; after that, or when 03a finds no other line, the
qubit's pass ends at 04. A visible but noisy oscillation is optimised at 04
(averages, gate time), never sent back.

### 05 — T1

Require a finite positive exponential-decay lifetime, relative uncertainty below
0.25, R-squared of at least 0.90, at least two samples per lifetime, and a wait-time
span covering at least 3.5 fitted lifetimes. A clipped or under-covered decay
requires a repeat with a longer range or adjusted step; every T1 state update
still requires human approval.

An 05 run may measure any non-empty subset of the active targets while preserving
`multiplexed: true`. The first 05 run must multiplex every active target with
the same wait window and averages. If a later multiplexed protected fit aborts
the snapshot, retry the remaining unresolved qubits together before isolating a
single qubit. Passing T1 evidence accumulates by qubit; retry only unresolved
targets that can share the wait range. Because this workflow includes 100-run
statistics,
final 05 evidence must cover at least 3.5 fitted lifetimes so that the trace has
reached its equilibrium tail. Use the emitted
`recommended_statistics_max_wait_time_in_ns` (4.5x the fitted lifetime, rounded
to a 4 ns boundary) when forming compatible 05st target subgroups. It must also
have robust SNR of at least 15 (operator instruction 2026-10-12, lowered from
25: before 07d the CRSQ_26009_die(5,-1) readout gave clean decays at SNR 8-12).
If only some traces are below that floor, call
`jy_recommend_0405_snr_retry` and repeat only those targets through 2000, 4000,
then 8000 averages. Reaching the averaging ceiling without the SNR floor leaves
the target unresolved; it does not convert a noisy fit into a completed result.

05 runs after 07b (operator instruction 2026-10-12), on the optimised readout.
The earlier 05 -> 07d/07b -> 05 deferral is retired. When the ladder cannot
reach the floor, mark the qubit incomplete; its pass continues to 06.

### 07d and 07b run once on defaults

Operator instruction, 2026-09-21. 07d Readout Frequency/Duration/Power
Optimization is part of the sequence and runs immediately before 07b, so the
readout it optimises is the one 07b then characterises. Both nodes carry
`single_pass: true` in `rules/policies.yaml`.

A single-pass node runs exactly once, full width, on the node defaults, and the
workflow advances whatever the analysis says. Do not retry it, do not adjust its
parameters, and do not chase individual targets through it — a second run is
refused before the worker reaches hardware. Evidence gating is unchanged:
a target whose analysis passes still commits its state patch, because that is
the node's output, and a target that does not pass simply carries no update and
does not hold the workflow up.

Operator instruction, 2026-09-29: 07d runs in groups of at most four qubits
(`nodes."07d".max_multiplex_targets: 4`). The node's measurement batching in
`quam_libs/experiments/readout_optimization_3d/measurement_batching.py` drops
targets above four and then crashes with `KeyError`; four or fewer is a single
batch. Single pass then means one run per group, every group on the defaults.
07d stays out of `SUBGROUP_NODES`, so 07b still receives every target carried
into 07d, not only those with passing 07d evidence.

Operator choice, 2026-10-03: at 07b too, a target measured in the single pass
but not passing (for example q1/q2 failing the two-blob test) does not block
`advance` and stays active for 06/06b, which need no state discrimination. It
keeps its 07d readout and gets no 07b threshold/angle/confusion update. Only a
target explicitly marked scientifically unmeasurable at 07b is dropped.

Operator instruction, 2026-10-02 (updated 2026-10-13): with 02c committing the
dressed limit minus 3 dB, 07d sweeps the readout amplitude factor 0.2–1.41
(22 points), so its search reaches back up to the dressed limit and never past
it (q2's third IQ cloud at factor 2.03 under the old 3.16 sweep). Since 2026-10-13 the
readout duration sweeps up to 4000 ns (8 x 500 ns; the duration step must be a
multiple of 4 ns), and `readout_length_normalize_ns` is 4000 so 07b keeps it. A run made on an older value of a
`single_pass_default_keys` entry (here `max_amplitude_factor`) no longer counts
toward single pass, which is how the first group measured on the old 1.99
sweep earns one more run. The node's own power-setting step restores the
pre-run amplitude, so JY commits amplitude = pre-run amplitude x the factor at
the maximum of the node's smoothed `fidelity` map; an optimum at the top of the
sweep is reported as a warning.

Operator instruction, 2026-10-02 (later): 07d judges every swept point by how
close its g/e shots are to two blobs, not by fidelity alone. A shot on the
other blob (thermal population, decay) is a readout error that fidelity already
counts; a shot beyond `analysis."07d".outlier_sigma` (3 sigma) of *both* blob
centres is smear. The off-blob share is pooled over neighbouring frequency and
amplitude points, and JY commits the highest-fidelity point whose pooled share
is at most `max_outlier_fraction` (3%), rebuilding IF, amplitude and length from
that point. Operator choice 2026-10-03: no point above the qubit's 02c
dressed-limit power is ever chosen, and the two-blob point replaces the
best-fidelity point only when it costs at most `fidelity_tolerance_percent`
(2 points); otherwise the fidelity optimum is committed and 07b's two-blob test
judges it. `num_runs` is 100 so each
point has enough shots. 07b uses the same two-blob test on its 10000 shots
(`analysis."07b".morphology_mode: "two_blob"`, at most 5% of a cloud on neither
blob); the radius-quantile limits remain advisory notes. After a rule change,
`jy_reopen_07d_after_readout_rule_change` returns the workflow from 07b or 06
to 07d (lease stopped first). The reopen restarts the bring-up from 07d: runs
of 07d and every later node (07b, 06, 06b, 10a, statistics) made before it no
longer count, so 06 onward is re-measured with the reset groups below.

This replaces the tail-power ladder below as the normal path for 07b. Keep the
ladder documented: it is still the right tool if an operator deliberately
reopens 07b, and it records what the morphology rule is protecting against.

07b has one exception to the single run. Any qubit whose thermal result is
**above** `analysis."07b".active_reset_trigger_fidelity` (0.80) **with two
Gaussian blobs** earns exactly
one more 07b run, with `reset_type_thermal_or_active: "active"`. Run those
qubits together; that repeat is their accepted 07b result and it qualifies
them to use active reset in later nodes, while every other qubit stays
thermal. A second thermal run is still refused, and so is a second active run.
`jy_get_next_action` proposes the repeat itself, with the qualifying target
list, before it reports `advance`.

Operator decision 2026-10-14: morphology no longer gates the 07b pass, but it
does gate the readout's use. `active_reset_requires_morphology` is true, so
both the trigger and the active repeat need fidelity >= 0.80 with two Gaussian
blobs - the same rule that allows state discrimination. A thermal run never
qualifies a qubit for active reset, whatever its fidelity.

The statistics rule is unchanged: before any 05st or 06st run uses active
reset, 05, 06 and 06b must be repeated and accepted with active reset for
those qubits.

### Reset groups after 07b

Operator choice, 2026-10-04 (`reset_policy.group_by_reset_after_qualification`).
At every node after 07b that has a reset parameter (06, 06b, 10a, statistics),
the active targets split into two groups: the qubits qualified by an accepted
active-reset 07b run use active reset, the rest thermal. Each group gets its own
shared first batch, the two never share a run, and `jy_get_next_action`
proposes the active group first with the reset parameter filled in. After a
07d reopen only the new 07b runs qualify.

### 07b — IQ blobs

Operator decision 2026-10-14: 07b is measured once and **always passes** for a
qubit with a finite fidelity and rotation angle, so the integration-weight
angle, thresholds, confusion matrix and fidelity always commit and every later
node (05, 06, 06b, ...) sees its decay on I -- JY never has to pick Q.
Neither low fidelity (`min_readout_fidelity_blocks_pass: false`) nor failed
cloud shape (`morphology_blocks_pass: false`) fails the run any more.

They decide the readout's *use* instead. A qubit qualifies for state
discrimination and active reset only with fidelity >= 0.80
(`analysis.07b.min_readout_fidelity`) **and** two separated Gaussian blobs (the
two-blob morphology test). Otherwise the 07b analysis warns that the readout is
not usable for state discrimination or active reset and names the reasons
(`state_discrimination_reasons`). Every node with `use_state_discrimination`
then runs that qubit on the raw I/Q signal: `jy_get_next_action` proposes
`use_state_discrimination=false` whenever its batch holds such a qubit, and the
server refuses `true` (without halting the lease).

A long line/tail, crescent, or strongly elongated cloud is still a
power-saturation signature; it now costs discrimination and active reset, not
the angle commit.

The 99th-percentile radius must remain within `4x` the median radius for each
prepared-state cloud. This intentionally rejects sparse or curved tails that can
look visually dominant while a covariance-axis test still appears acceptable.

For morphology failure, use `jy_request_07b_tail_power_reduction` to reduce each
affected target's readout amplitude by the policy-fixed factor `sqrt(1/2)` (about
-3 dB power) and repeat 07b. This deterministic recovery requires a saved failed
07b snapshot, may be applied through an active bounded-autonomy lease, and stops
at the configured minimum amplitude. Repeat one fixed step at a time until the
plot contains two compact clouds; do not compensate a long tail with fidelity or
threshold fitting alone.

If an isolated target still fails fidelity or compact-cloud morphology at the
configured minimum amplitude, record `manual_review` on that evidence run and
call `jy_mark_scientifically_unmeasurable` for only the failed target. This marks
the target incomplete at 07b without spending the rest of its attempt quota and
allows already resolved targets to continue downstream. Never use this boundary
for a passing target, a batch target without target-specific evidence, or while
another policy-authorized recovery remains.

Operator choice 2026-10-14 (q3 06): when a node's own fit crashes after the
measurement ran (`termination_cause` worker_exception, `execution_phase`
hardware_execution), the failed run may serve as the evidence run for
`jy_mark_scientifically_unmeasurable`, after the operator chooses to skip the
node for that qubit. Then record `advance` on that run.

State updates for integration angle, thresholds, confusion matrix, and
readout fidelity remain evidence-gated commits. Before the first 07b run for a
target subgroup, call `jy_request_07b_prerequisites` and apply its separately
authorized deterministic setup if any `extras/readout_fidelity` mapping key is
missing. The setup value is a non-passing zero sentinel used only so Qualibrate's
state recorder can observe an old value; 07b must replace it with measured evidence.
Normalize library fidelity values expressed as percentages (for example `88.6`)
to fractions (`0.886`) before acceptance or state commit. Retry unresolved
qubits that can share the same readout-amplitude recovery together; do not
walk failed 07b targets one isolated qubit at a time.

### Reset selection — thermal by default, active only after 07b evidence

Every registered experiment defaults to thermal reset. An operator or agent may
deliberately run 07b with active reset to qualify a per-qubit switch. Active reset
is unlocked for a qubit only when that active-reset 07b run retains normalized
readout fidelity of at least `0.80` (`reset_policy.active_min_readout_fidelity`;
higher is preferred) with two Gaussian blobs
(`analysis.07b.active_reset_requires_morphology: true`, operator 2026-10-14). A
thermal 07b result or an active result below `0.80` does not unlock active
reset.

After qualification, later experiments may explicitly request active reset; the
switch is optional and is never inferred from an old default. Before any 05st or
06st run uses active reset, repeat and accept all three baseline nodes—05, 06, and
06b—with active reset for the same qubit(s). These conditional verification runs
may be performed without discarding the workflow's current downstream node. A
statistics request must use the same reset type as its accepted 05/06/06b
baselines.

### 06 / 06b — Ramsey T2* and echo T2

Operator instruction, 2026-10-02: the Ramsey detuning is inversely
proportional to the window, `frequency_detuning_in_mhz x max_wait_time_in_ns/1000
~= 4` (1 us -> 4 MHz, 10 us -> 0.4 MHz, 40 us -> 0.1 MHz), so every 06 plot shows
about four fringes and can be read by eye. Choose the window from T2* (at least
3.5 lifetimes) and omit `frequency_detuning_in_mhz`: the policy derives it. An
explicit detuning more than 25% off the rule
(`nodes."06".detuning_wait_product_*`) is refused before the hardware. Use a
linear sweep with enough points to resolve the fringe (the defaults are 40 us,
0.1 MHz, 400 linear points). `06_Ramsey.py` measures one qubit per batch, so the
run time grows with qubits x points x averages; keep the window no longer than
needed.

Operator instruction 2026-10-10: a qubit passes 06 only when both fitted
fringe frequencies (detuning signs −1 and +1) lie within 20% of the set
detuning (`analysis."06".max_fringe_detuning_error_fraction`; 1 MHz → ±200
kHz). A first run that already meets it passes; there is no fixed second run.
A good fit that misses it stays unresolved; record its analysis
`frequency_correction_patch` (IF and `idle_freq` only, never `T2ramsey`) in the
decision so the corrected frequency is committed, then run 06 again for it.

Run Ramsey followed by T2 echo. Require a finite positive lifetime, relative
uncertainty below 0.25, at least two samples per lifetime, and at least 3.5
lifetimes of coverage before resolving a qubit. T2 echo additionally requires
R-squared of at least 0.8 (`analysis.06b.min_r_squared`) and a non-boundary fit. This longer coverage is
mandatory because both values feed downstream 100-run statistics. The first 06
and 06b runs must multiplex every active target with the same wait window;
retry unresolved qubits together when they can share that window. Use the
emitted 4.5x recommended statistics wait time when forming compatible target
subgroups. The Ramsey frequency correction and T2 state values remain separately
committed.

A node fit that fails records NaN (the 06b node's `fit_decay_exp` assumes a
falling trace and fails on a rising, inverted-contrast I quadrature). JY stores
it as null; its own sign-agnostic decay fit then supplies the value, and if
that fit also fails only that qubit is refused, not the whole run.

Quadrature (operator decision 2026-10-14): 05 and 06b keep the I (or state)
fit whenever it meets the acceptance rules above. Q or |IQ| is only a fallback
when I fails them, and the analysis then warns that a non-I quadrature was
used -- after 07b's always-committed angle that should not happen.

### 10a — single-qubit randomized benchmarking

Require a successful fit and finite EPC/EPG values in [0, 1] for every resolved
qubit. The first 10a run must multiplex every active target with the same depth
settings. The depth must be divisible by `delta_clifford`, and later retries may
use a shared-parameter unresolved subgroup.

### 05st / 06st_t2star / 06st_t2e — 100-run statistics

Operator instruction 2026-10-12: a ticked statistics node runs only for the
qubits whose own single measurement passed -- 05st after 05, 06st_t2star after
06, 06st_t2e after 06b (`shared_xy_drive.statistics_sources`). A qubit that
missed one of them skips that statistics node and goes on to the next; a node
no qubit qualifies for is skipped. 10a always runs.

Readout (`shared_xy_drive.statistics_readout_node`, operator choices
2026-10-12/13): each statistics node fits 100 short runs; with a failed
discriminator (q1 07b 55%) none of them fits and the 05st script crashed on an
empty histogram. A qubit whose 07b in this workflow did not reach 0.80 with two
Gaussian blobs therefore runs statistics (like every node with
`use_state_discrimination`) only with `use_state_discrimination=false` -- the
raw I/Q signal its 05/06b passed on -- and enough averages (1000 per point on
CRSQ_26009_die(5,-1)); the server refuses a discriminated request for it
without halting the lease. A later workflow can start such a qubit directly at
a statistics node whose single measurement passed in the chip's previous
workflow (lease-page per-qubit start); with no 07b in that workflow it runs on
raw I/Q too.

Completion and summary (operator choices 2026-10-13): a statistics run counts
only when at least `analysis.statistics.min_completed_iterations` (90) of its
100 repetitions completed -- a lost instrument link otherwise saves a summary
of a few repetitions. When the node's summary has sigma > mu (failed single
fits dominate the mean, e.g. q1 06st_t2e #494: 26481 +- 119514 us), the run
keeps its status but is annotated: the analyzer refits every repetition (I, Q
or state, the larger decay in either direction) and reports the median and
IQR in a warning and in the chip history. Do not commit the node's mean then.

Filtered statistics (operator decision 2026-10-13): JY refits every repetition
(I, Q or state; decay in either direction), drops fits below
`analysis.statistics.refit_min_r_squared` (0.5) or `refit_min_snr` (3.0) or
with a lifetime beyond the sweep window, and redraws the node's histogram from
the kept ones (`report_assets/<snapshot>-<node>-<qubit>-filtered.png`). The
statistics record (extras.T1/T1_dev, T2/T2_dev, T2ramsey/T2ramsey_dev) takes
that filtered mean and spread; the working T1/T2echo/T2ramsey stay with the
single 05/06/06b fits (they also set the thermal wait).

07b below its fidelity target or without two blobs (operator decisions
2026-10-13/14): 07b still passes, so its rotation angle and threshold commit
and 05/06b/statistics see the qubit signal on I -- the node scripts fit I only,
and an uncommitted angle leaves the contrast split between I and Q (q2
2026-10-13: both quadratures rose). State discrimination and active reset need
a 07b fidelity >= 0.80 with two Gaussian blobs.

06b passes at R-squared >= 0.8 (`analysis.06b.min_r_squared`, lowered from 0.9
the same day: q1 on CRSQ_26009_die(5,-1) showed clean echo decays at 0.88 with
the poor readout). 06b has no robust-SNR floor.

Each statistics node requires exactly `histo_num=100`. It may start only after the
05, 06, and 06b results are all resolved with the same reset type. Each baseline
fit must cover at least 3.5 lifetimes. For the statistic being measured, the
requested `max_wait_time_in_ns` must lie between 3.5 and 5.5 times its accepted
T1/T2 value. This ensures that the baseline has reached equilibrium and the
statistics range is approximately four lifetimes; prefer 4.5 times the baseline.
Start from the largest compatible multiplex group that can share one
`max_wait_time_in_ns`; do not begin with isolated one-qubit statistics runs.

Once those prerequisites pass, run each 05st/06st node once. JY does not refit all
100 raw iterations, does not impose a 100/100 secondary fitting gate, and does not
reject the entire statistics run because one independently re-analysed iteration
would fail. The protected node's summaries and plots remain informational; normal
worker completion, saved plot, exactly `histo_num=100`, and the already-validated
baseline prerequisites are sufficient to complete that statistics node. Any state
candidate produced by the registered statistics node remains subject to the usual
state-patch policy and commit authorization, but no extra statistics-specific
fitting check is added.

## A `needs_review` run still commits the qubits that passed

Operator instruction, 2026-09-20. `analysis_status` is a single run-level
verdict, but every acceptance check is per qubit. A run that is `needs_review`
because one target failed must still commit the targets that passed: the node
refuses a further run once every target is resolved, so suppressing the whole
patch lost calibrated values for good. This happened three times in workflow
`0892399eba47422b96c9203e8ead1523` — 02c committed none of nine passing
targets, 03a none of seven, and 04's `q1`/`q2` were never recovered.

The analyzer therefore reports, alongside `failure_reasons`:

- `failure_reasons_by_target` — each failure attributed to the qubit it names.
- `run_level_failure_reasons` — failures that belong to the run as a whole
  (no saved plot, unreadable dataset, invalid recorded state updates). Any of
  these still suppresses every target.
- `passing_targets` — the qubits with no failure of their own in a run that has
  no run-level failure. `candidate_state_patch` is filtered to these qubits.
- `suppressed_patch_targets` — the qubits whose changes were dropped, and why.

`jy_autonomy_commit_state` accepts a `needs_review` run only for
`passing_targets`, and refuses any other target without halting the lease. Read
`passing_targets` before proposing the commit, and say in the Reason which
targets it covers and which are still unresolved. Only passing evidence is ever
committed; this relaxes nothing else.

## Status meanings

- `run.status`: worker lifecycle (`starting`, `running`, `stopping`, `completed`,
  `failed`, `stopped`, `cancelled_by_shutdown`, `force_stopped`).
- `analysis_status`: deterministic evidence result (`pass`, `needs_review`,
  `failed`). It is run-level; `passing_targets` carries the per-qubit verdict.
- `workflow.status`: bring-up progress (`active`, `completed`, `stopped`,
  `recovery_required`).
- `proposal.status`: approval lifecycle (`pending`, `approved`, `expired`,
  `consumed`).

Do not substitute one status for another.

## Instrument-connectivity failures never engage the hardware lock

Operator instruction, 2026-09-19. A recognized instrument-connectivity failure
(`failure_category: instrument_unreachable`) pauses the run and scheduling, but
it must never retain the hardware lock and must never put JY into recovery-only
quarantine — not even when the worker could not confirm hardware cleanup
because the instrument was already unreachable. The worker still makes its
best-effort `_stop_active_node_with_evidence()` attempt and still refuses to
release the lock if `state.json` changed during the failed run; only the
unverifiable-cleanup condition is waived.

The operator's reasoning: when the cause is simply that the hardware is
unreachable, the lock adds an extra manual unlock step at the local console
without protecting anything — no job can be running on an instrument that
cannot be reached. Recovery must stay a plain `恢復量測` /
`jy_resume_measurement_mode` on the same paused lease.

This does not relax the other lock rules. A retained lock from any other cause
is still preserved for local recovery, is never deleted automatically, and
still requires the operator console.

## A run crash is classified: known ones continue, unknown ones pause

Operator instruction, 2026-09-20, replacing the 2026-09-19 rule that every
worker exception stopped for the operator. In automatic mode the lease must
survive everything that can be corrected, so a crash is now classified:

- **Registered node-parameter rejections continue.** A crash whose signature is
  listed in `autonomy.recoverable_worker_exceptions` is a parameter the
  protected node itself rejects: deterministic, reproducible, and fixed by a
  different value. The lease stays active, the server records
  `autonomy_recoverable_worker_exception` with the registered remedy, and you
  choose a corrected parameter. Never resend the value that crashed.
- **Everything else pauses.** An unrecognized worker exception pauses the lease
  and scheduling, exactly like an instrument outage. Report to the operator
  what crashed and why, and resume the same lease with `恢復量測` once they
  decide. No new approval is needed, and the lease is not lost.

Both currently registered signatures are also prevented before they run:
fractional `min_power_dbm`/`max_power_dbm` in 02c is refused because the node
types them as `int`, and `arbitrary_qubit_frequency_in_ghz` in 03a is refused
while `freq_vs_flux_01_quad_term` is zero. Both were crashes in workflow
`0892399eba47422b96c9203e8ead1523` and its predecessor.

Unchanged: record the failure as a labeled lesson in
`rules/experiences/<node>.md`, state plainly that no snapshot or plot exists,
and never retry the exact parameter that crashed. Add a new entry to
`recoverable_worker_exceptions` only together with its lesson.

## A refusal before the hardware never ends the lease

Operator instruction, 2026-09-20. A request that is rejected before the worker
reaches the instrument -- an out-of-policy parameter, an out-of-scope target
set, a state patch that fails validation, a premature `jy_analyze_run` -- does
not end the authorization. Nothing ran, so refusing the action is the whole
protection; ending the lease on top of that only costs a fresh human approval
and stops a bring-up that could have continued. Every refusal is recorded as
`autonomy_action_refused` and stays visible on the Dashboard.

Correct the request and continue. The conditions that still halt the lease are
only the ones where JY can no longer account for the hardware or the state
file: `snapshot_missing`, `state_hash_conflict`, and `hardware_lock_anomaly`.

One exception remains in the server: `jy_autonomy_start_run` still halts the
lease when its refusal is a plain service error rather than a policy, scope, or
quota refusal ("Authorized run was refused or failed to launch: ..."). Known
cases are a node that is not yet `current_node` ("Current workflow node is X;
record a decision first") and a later-node target set that does not exactly
match the resolved targets carried into the node ("Run qubits must exactly
match the resolved active targets"). Check `current_node` and
`jy_get_next_action` before requesting the first run of a node, and never send
it in parallel with the decision that advances to it.

## Ask the server what to do next

`jy_get_next_action(workflow_id)` returns the deterministic plan: the current
node, the active/resolved/incomplete/unresolved target sets, attempts used and
left per target, any required deterministic setup tool, and one of `wait`,
`analyze`, `record_decision`, `setup`, `run`, `advance`, or `blocked`. For a
`run` it gives the qubits and only the parameters that differ from the node
defaults, plus the rule it applied.

It is an aggregator over the same state the guards enforce, not a second
opinion. When `notes` says no registered deterministic rule covers the
situation, that is the honest answer: read the node's section above and
`rules/experiences/<node>.md` and choose the parameters yourself. Use
`jy_get_decision_experience` for what was tried before; each entry carries an
`evidence` digest of the gated numbers, and `include_analysis=True` only when
the digest is genuinely not enough.

## A shared XY drive output runs one qubit at a time

Operator instruction, 2026-09-22. Read `wiring.json` before planning anything.
If any two qubits name the same `wiring.qubits.<q>.xy.opx_output`, those qubits
share one physical drive line and therefore one LO. Such a chip cannot be swept
node by node across all of its targets, because placing one qubit at zero IF
moves every other qubit on the port.

On such a chip the workflow changes shape:

- 02, 02x, 02c and 02a are unchanged. They only use the readout line, so they still
  multiplex every active target and the multiplex section above applies to them
  in full.
- From 03a the workflow takes ONE qubit through the whole remaining sequence --
  03a, 04, 05, 07d, 07b, 06, 06b, 10a and the statistics nodes -- and only then
  returns to 03a for the next qubit. A pass ends either at the end of the
  sequence or as soon as its qubit stops being measurable.
- The LO therefore only ever moves between passes, never inside one. That is
  what makes the shared line workable: when a new pass starts at 03a, its qubit
  goes to zero IF and the shared LO is free to move to that qubit's RF.
- Every other qubit on that port has its `xy/intermediate_frequency` rewritten
  in the same patch so its RF is preserved. The server does this automatically
  in `jy_request_initial_03a_zero_if`, `jy_request_drive_lo_recenter`,
  `jy_request_03a_window_shift` and `jy_request_03a_candidate_center`; it
  refuses the move if any of those compensated IFs would exceed the IF limit.

The rule lives in `rules/policies.yaml` under `shared_xy_drive`
(`serialize_from_node: "03a"`, `per_qubit_pass: true`,
`max_multiplex_targets: 1`). It is derived from the wiring, not from the chip
name, so it switches itself on and off when the active QuAM state changes.

Operator request 2026-10-04: the lease approval page shows what the wiring
implies and asks the operator to choose `per_qubit` (this per-qubit pass) or
`multiplex` (the ordinary node-by-node sweep) before automatic measurement
starts. `per_qubit` serializes even a chip with one XY output per qubit. The
workflow keeps its mode; a later lease only confirms it.

Operator instruction 2026-10-08: on a chip that shares an XY output,
`multiplex` is no longer possible. The page disables it, the server refuses a
lease that asks for it, and even a workflow stored as `multiplex` before this
rule is capped at one qubit per run from 03a on.

Driving the pass:

- `jy_get_next_action` carries a `serial_pass` block: which qubit the pass is
  on, its position in the order, which qubits are finished and which are still
  waiting. The node's active target set is that one qubit; the others are not
  eligible here until their own pass starts.
- A run proposal naming more than one qubit at a tail node is refused before the
  worker reaches hardware. That is a correctable planning error, not a halt.
- `advance` keeps its usual meaning while the qubit still has usable evidence:
  the next node in the sequence, same qubit.
- When the qubit's pass is over -- the last node is done, or the qubit is
  resolved nowhere further -- `advance` names 03a again, and the next qubit
  starts there. On the last qubit it names nothing and the workflow completes.
  The server tells you which of these it expects; a wrong `next_node` is
  refused with the reason.
- A qubit that cannot be measured ends its pass rather than the workflow. Mark
  it with `jy_mark_scientifically_unmeasurable` and then record the `advance`
  that returns to 03a. That mark may be taken either after a `manual_review`
  decision or directly against the unreviewed evidence run, because a run holds
  exactly one decision and a single-qubit pass has no other run to carry the
  advance.

Operator instruction 2026-10-12 refines when a pass ends:

- Only 03a without a line and 04 without a pi pulse end a pass early
  (`shared_xy_drive.pass_prerequisite_nodes`). A qubit that fails or is marked
  incomplete at any other node -- 05, 07d, 07b, 06, 06b, 10a, statistics --
  still continues to the next node; 04 working means the qubit is alive and
  the later nodes only lose SNR. 05 follows 07b in the sequence (see 05), and
  04 may first return to 03a (see 04).
- Each qubit records its drive LO in `extras.xy_lo_freq`. Before every node of
  its pass after 03a, `jy_get_next_action` requires `jy_request_xy_lo_plan`: one
  100 MHz-grid LO for every measured qubit when all of them fit within
  +-300 MHz (IFs kept >= 20 MHz from the LO leakage where possible), otherwise
  the qubit's own LO. A run is refused until the plan is applied.
- Moving the shared LO no longer fails when another qubit would leave the
  +-400 MHz IF range: that qubit is parked at the limit (it is not played) and
  its real frequency stays in `extras.idle_freq`; its own LO plan or zero-IF
  setup restores it. This removed the q4 deadlock of 2026-10-12.
- A fine 03a line commits only when an earlier 03a run of the same qubit showed
  it within max(5 MHz, FWHM) at robust SNR >= 4.5; otherwise repeat the fine
  scan (`analysis.03a.reproduce_*`).
- A coarse line below the 03a SNR floor is still a credible candidate when two
  runs of the qubit put it at the same RF (within 5 MHz) from IFs at least
  50 MHz apart -- under different LOs, so not a fixed-IF artefact -- each with
  robust SNR >= 5 (`analysis.03a.cross_lo_*`, operator choice 2026-10-12, q3 at
  2.226 GHz). It then takes `jy_request_03a_candidate_center`, the refinement
  and fine scans, and 04 as usual; lines 04 rejected stay masked. Its fine
  scan, within 5 MHz of the confirmed RF, passes from robust SNR 5
  (`cross_lo_fine_min_robust_snr`); reproducibility still applies.
- The lease page lets each qubit restart its pass at any node it reached in the
  chip's last workflow, when the measurement starts at 03a ("Chip history and
  the start node"). A qubit's pass then begins at its chosen node, and the pass
  before it hands over to that node instead of 03a.

## Flux calibration (02b, 03b, 06a)

Operator request 2026-10-08. The lease approval page asks whether to calibrate
flux. Without a yes the flux nodes are not in the workflow at all (also in
conversational mode). Any Z output shared by two qubits in `wiring.json`
(`wiring.qubits.<q>.z.opx_output`) disables the yes, and the server refuses
02b/03b/06a on such a chip.

- Yes runs **02b** after 02a and **03b** after 03a. **06a** (after 06) is an
  extra tick on the same page and needs the yes.
- All three use `flux_point_joint_or_independent: "joint"` and write
  `z.joint_offset`; the rest of the sequence runs at joint flux, so it uses the
  new offset.
- Operator sweep limits: 02b ±0.9 V with at most 100 flux points; 03b ±0.4 V
  with at most 100 points; 06a `flux_span` ≤ 0.2 V (±0.1 V) with about 100
  points. 03b and 06a scale the z `const` pulse, so the sweep must also stay
  below 1.99 × its amplitude.
- Only the statistics nodes may run long. 02b, 03b and 06a carry a runtime
  estimate and are refused when it exceeds `instrument_limits.max_run_minutes`
  (10 min): lower the averages, points or targets per run.
- `jy_request_node_extras_prerequisites` adds `FluxPeriodV` and
  `sweetspot_freq` before these nodes, as for the other extras keys.

### 02b — resonator spectroscopy versus flux

Each qubit is classified from its resonator trace (`fit_quality.results.<q>.classification`):

- `tunable`: the trace follows a cosine (inlier R² ≥ 0.8), and the node's idle
  offset lies on an extremum of it, at least 5% from both sweep edges. It
  commits `z.joint_offset`, `z.min_offset`, `phi0_voltage`, `phi0_current`,
  the resonator IF at that offset and `extras.dressed_resonator_freq`. The
  aSWAP amplitude is never committed.
- `flat`: the resonator does not move with flux. This counts as resolved.
  Nothing commits, and the qubit skips 03b and 06a.
- `avoided_crossing`: two or more consecutive flux points jump far off the
  cosine and come back from the other side. This also counts as resolved and
  nothing commits; the qubit skips 03b and 06a. The analysis adds an
  `operator_notices` entry and an annotated plot. The results page shows it in
  an orange card telling the operator to tune the neighbouring couplers' flux
  offsets. Do not retry it; it is not something the agent can fix.
  Operator rule 2026-10-09: the excursion must come back inside the window.
  Off-cosine points that touch the first or last flux point are a settling
  transient at the sweep edge, reported as `edge_outliers`, and do not stop a
  clean cosine from classifying `tunable` (q8-q10 in run #334).
- `failed`: no visible dip (median dip SNR < 4) or an unusable fit. Retry it as
  for any node: wider or shifted flux window, more averages. Alternatively mark
  it scientifically unmeasurable.

A flux node never removes a qubit from the rest of the sequence. A flat,
crossing or failed qubit still runs 03a, 04, … with its old flux settings.

### 03b — qubit spectroscopy versus flux

Only qubits 02b found `tunable` run 03b. A pass needs:

- the qubit line visible at ≥ 50% of the flux points;
- a parabola through it with R² ≥ 0.7;
- a negative quadratic term;
- the sweet spot at least 5% from the flux edges;
- the sweet-spot frequency inside the window.

It commits `z.joint_offset`, the drive IF, `freq_vs_flux_01_quad_term`,
`extras.idle_freq` and `extras.sweetspot_freq`. If the parabola runs off the
window, widen `frequency_span_in_mhz` or the flux window within ±0.4 V.
Keep `frequency_shift_in_mhz` at 0 (finding 2026-10-09, run #382): the node
fits on the unshifted detuning axis, so every committed frequency would be
off by the shift. The node's own vertex must also agree with the tracked
line's vertex within 10% of the flux span, or the qubit fails.

### 06a — Ramsey versus flux (optional)

Only qubits that passed 03b run 06a. It reads the state before and after every
shot, so it needs no reset and stays short. A pass needs:

- Ramsey fringes at ≥ 50% of the flux points;
- a fringe-frequency parabola with R² ≥ 0.5;
- a fitted flux offset within 0.4 × `flux_span`.

It commits `z.joint_offset`, the drive IF, `extras.idle_freq` and
`extras.sweetspot_freq`. The quadratic term stays with 03b.

Inside a per-qubit pass, a flux node never ends the pass. `advance` goes to the
next node even when the qubit did not resolve there. A flux node with no
eligible qubit is skipped by the expected `next_node`.

## The 03a drive LO stays inside its band

Operator instruction, 2026-09-22. Every MW upconverter frequency has to stay at
least `instrument_limits.mw_band_edge_margin_hz` (100 MHz) inside the frequency
range of the band its port is configured for, so that an LO placed near an edge
never turns into a hardware error. The band ranges are in
`instrument_limits.mw_band_frequency_hz`; with the default margin band 1 gives
an LO window of 0.15-5.4 GHz, band 2 gives 4.6-7.4 GHz and band 3 gives
6.6-10.4 GHz.

A state commit that puts an upconverter frequency outside its band window is
refused. An RF-preserving recenter that would need such an LO is refused too,
because moving the LO anyway would falsify the qubit's frequency. The one
exception is an explicit 03a coarse-window shift, which is clamped to the edge
of the window instead: the search keeps stepping along the band rather than
stalling, and the applied centre is reported in the setup details as
`clamped_to_band`.

## 04 never multiplexes more than five targets

Operator instruction, 2026-09-21. Power Rabi caps its multiplex group at five
qubits, whatever the active set is, via `nodes."04".max_multiplex_targets: 5`
in `rules/policies.yaml`. 07d is the only other node with its own cap (four;
see "07d and 07b run once on defaults"), and a shared XY drive output caps
every node from 03a at one target; every other node still multiplexes all
active targets and the rule below applies to them unchanged.

For a capped node the cap replaces the shared-first-batch requirement: the
first 04 run is the first group of five, not every active target, and the node
is completed by repeating it for the remaining groups. A request above the cap
is refused before the worker reaches hardware, which is a correctable planning
error and does not halt the lease.

Evidence, `as-qpu-10qV2_agent`: a ten-qubit 04 program has never once been
accepted by this QOP. Workflow `2dba0368b00c4b06aeed1a1ceabbb3d9` spent seven
runs walking 10 -> 5 -> 3 -> 2 -> 1 twice, either side of a genuine QOP outage
and restart, without a single successful submission. Starting at five skips the
first rung of that ladder every time the node begins.

## Halve the multiplex group when a run times out on submission

Operator instruction, 2026-09-20, replacing the earlier "three or four qubits"
wording. Except at a capped node, every node starts the same way: the first run
of a new node always multiplexes ALL active targets with one shared parameter
set.

If that run dies with `QMTimeoutError: A timeout of 100 seconds was reached`
raised from `qm.execute()` / `add_to_queue`, assume the program carries too
many qubits and halve the group. Measure the targets in two halves; if a half
still times out, halve again, and keep halving.

If a run carrying a SINGLE qubit still times out on submission, the cause is no
longer program size. Treat it as a genuine instrument-side disconnection, stop
scheduling, and hand it to the operator.

So the ladder for ten active targets is 10 -> 5 -> 3/2 -> 1, and only a
one-qubit timeout means the instrument itself is at fault. Keep one shared
parameter set across the halves so their evidence stays comparable, and never
narrow the sweep instead of the group: coarsening the swept axis does not fix a
submission timeout.

`service.py` waives the shared-first-batch rule once a full-width attempt has
failed with `program_submission_timeout` or, for runs recorded before that
category existed, `instrument_unreachable`. That waiver is what lets the halves
be scheduled.

### Which timeout is this? Three causes, one of them halves

A `QMTimeoutError` alone does not say whether the program was too large, the
QOP is wedged, or the network is down, so the worker decides with evidence
rather than with the exception type. Do not re-derive this by hand: read
`failure_category` and `instrument_probe` in the run's analysis.

1. **Program too large** — `failure_category: program_submission_timeout`.
   Requires all of: the deadline expired in a `qm` package frame that submits a
   program (`execute` / `add_to_queue` / `_add_program` / `compile`), the gRPC
   status was `DEADLINE_EXCEEDED` rather than a transport failure, and a live
   probe right after the failure got the QOP to answer a health check. Only
   this case halves, and only while more than one target was attempted. The
   lease stays active and no operator is needed.
2. **Instrument wedged or down** — `failure_category: instrument_unreachable`
   with `instrument_probe.cause: "instrument"`. The socket was refused, or it
   opened but the QOP never completed a health check. Scheduling pauses and the
   operator restarts or inspects the QOP.
3. **Network fault** — `failure_category: instrument_unreachable` with
   `instrument_probe.cause: "network"`. Name resolution failed, the host was
   unroutable, or nothing answered at the socket level. Scheduling pauses and
   the operator checks the lab network.

A single-target submission timeout whose probe still succeeds is out of halving
room: the worker pauses it with `measurement_paused: true` and the operator
message names the QOP job queue, not the network. Never halve in cases 2 and 3
— there is nothing to shrink.

Evidence behind the rule, node 04 on `as-qpu-10qV2_agent`: ten multiplexed
targets failed at 171, 86 and 18 amplitude points (runs `cb707ffb…`,
`9a09832b…`, `db832677…`), all with the identical 100 s message, so the
eighteen-point program -- a tenth the size of the first -- bought nothing. The
first four-qubit subgroup then submitted and completed in 58 seconds.

The 100 s itself belongs to the `QuantumMachinesManager` created inside
`machine.connect()` ([quam_root.py:215]); `qm_session(timeout=...)` and the
node's `timeout` parameter do not affect it.

## Record `advance`, not `manual_review`, on the run that finishes a node

Lesson, 2026-09-20, workflow `0892399eba47422b96c9203e8ead1523` at node 04.

A run carries exactly one decision, and `current_node` only moves when a
decision names the next node. If the last run of a node is given
`manual_review` with `next_node` still pointing at the current node, and every
other run of that node already has a decision, the workflow cannot leave the
node: `jy_autonomy_start_run` refuses a fresh run with "All active <node>
targets are already resolved or incomplete", and `jy_record_decision` refuses a
second decision on the same run.

So when a target hits its scientific boundary and that closes the node, the
order is:

1. `jy_mark_scientifically_unmeasurable` for that target, then
2. `jy_record_decision` with `advance` and the real `next_node`, whose Reason
   states the target is incomplete at the boundary.

Never `manual_review` on the run that completes a node.

If the dead-end has already happened, an `advance` decision can be recorded
against an earlier run of the same node that never received one -- a failed run
qualifies, since `jy_record_decision` accepts `analysis_status: failed`. State
plainly in the Reason that the decision is node-level and that this particular
run produced no snapshot.

Related trap: a `jy_autonomy_start_run` for a node that is not yet
`current_node` is refused with a plain service error, which still ends the
lease ("Authorized run was refused or failed to launch: ..."; see the
exception in "A refusal before the hardware never ends the lease"). Check
`current_node` before requesting the first run of a new node; a request sent
too early costs the lease and a fresh operator approval.

## Success is counted once, and only when committed

Operator instruction 2026-10-10. At 03a, 03b, 04, 05, 06, 06a, 06b and 10a a
qubit that passed analysis but whose proposed values the decision left out of
the state patch is not resolved (it can be measured again or marked
unmeasurable). A resolved qubit is not measured again at that node (03a
included). The results page lists a qubit as successful only from JY's
`passing_targets` — never from the node's own outcomes, never every target of
a single-pass node, never a 03a coarse/refinement candidate, never a withheld
qubit — and once per node.

## Chip history and the start node

Operator request 2026-10-10. Every workflow records its chip (the QuAM state
folder `config.toml` names). The lease approval page shows the chip's latest
workflow: each node it reached, whether it finished, and the qubits that
passed. The operator starts from the beginning or at any node up to the one it
stopped at; the agent never chooses this. Runs and decisions before the start
node are copied in as inherited evidence (`runs.inherited_from_run_id`) and
are not shown in this session's results. Starting after 02b keeps that
workflow's flux choice; starting after the serialization node keeps its
measurement mode, and a per-qubit workflow can only restart at or before it.
If `state.json` changed since that workflow ended, the page warns.

The approval is two-step: the operator chooses, presses Preview to see which
experiments will run, then ticks the agree box and enters the approval code.
Changing a choice after the preview needs a new preview. A lease is approved
only on the Dashboard approval page.

## 02x probe headroom and plot cells

Operator debug request 2026-10-11 (CRSQ_26009_die(5,-1), runs #385-#387).
`02x_Bare_Resonator_Spectroscopy.py` plays every line tone together at a fixed
-10 dBm. `set_output_power` only raises the port full scale, so on a -11 dBm
line each tone sat at amplitude 0.79; six summed to 4.7 and the output clipped,
and all six 02x traces came out the same. Shared `grid_location` values also
made every node plot draw qubits on top of each other (only q6 and q4 showed).
When `jy_get_next_action` lists `jy_request_02x_probe_setup`, call it with the
lease id and apply it: it raises the line full scale (3 dB grid, at most
+10 dBm) until the summed probe is at most 1, scales each qubit's readout
amplitude so its delivered power is unchanged, and gives each qubit its own
grid cell. 02c at high power can clip the same way: six tones above about
-12 dBm each sum above 1, so judge punch-out below that.

## When 02 is required

Operator choice 2026-10-11. 02 is required (ticked, locked, `res_num` and
`res_design_freq` mandatory) only when the measurement starts from the
beginning and a target's readout IF is 0 or duplicated; the page then prefills
the chip's last 02 parameters. A later start point skips 02: the page hides the
02 box and the server ignores its tick and fields. A later start is offered only
when every unset-IF target was found absent at 02 in the chip's latest
workflow (that mark is carried over); otherwise the page explains which qubits
force a start from the beginning.

## One QUA thread per qubit

Operator decision 2026-10-11. The CRSQ_26009 states put every resonator on
QUA thread "a" (most XY elements on "b"). Elements on one thread run on one
core: multiplexed 02x (one shared loop) then reported the last element's (q6)
response on every channel, while 02c (one loop per qubit) ran serially and
looked fine. Whenever several qubits' resonators share a thread,
`jy_get_next_action` requires `jy_request_02x_probe_setup` at any node; it
gives each qubit its own thread (resonator and XY share it, as on
as-qpu-10qV2) together with the unique grid cells and the 02x probe headroom.

## 02c ignores low-power rows lost in noise

Operator rule 2026-10-11. Each 02c power row gets a dip contrast
(90th percentile minus minimum of the 3-point-smoothed |IQ|, over the
point-to-point noise). Rows below the lowest power from which two consecutive
rows reach `analysis."02c".min_power_row_snr` (6) are ignored, and the dressed
plateau is read from the usable range (`usable_min_power_dbm`). The bare-only
absence rule still judges the whole sweep and never applies when the usable
rows show a transition. Do not chase the dressed plateau below that power.
