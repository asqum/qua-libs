# Calibration experience log

This directory preserves reproducible experience for every bring-up node, in
the `config/agent.yaml` `workflow.sequence` order:
`02`, `02x`, `02c`, `02a`, `02b`, `03a`, `03b`, `04`, `07d`, `07b`, `05`,
`06`, `06a`, `06b`, `10a`, `05st`, `06st_t2star`, and `06st_t2e`.

Before proposing a run, read the relevant `<node>.md` file when it exists.
Accepted settings are evidence-backed starting points, not universal defaults;
the current qubit must independently pass every policy and node acceptance
check. Prefer experience from the same device/project and similar wiring or
qubit type. Never copy a learned frequency, amplitude, power, or lifetime into
active state without the normal run analysis and separately approved commit.

Scheduling: start every node with one multiplexed run of all active targets and
the same parameters. After analysis, omit already-resolved qubits and retry
unresolved targets together with modified settings. Split into subgroups only
when they cannot share the sweep. Do not begin a node with sequential isolated
one-qubit runs. Exceptions: the policy caps runs of `04` at 5 targets and `07d`
at 4 (`max_multiplex_targets` in `rules/policies.yaml`), so the first run is
the first capped group; after `07b` the active- and thermal-reset groups each
get their own first batch; and in per-qubit mode (always with a shared XY
output) one qubit at a time goes through every node from `03a` on.

Append an `Accepted setting` only after a saved snapshot passes the complete
node-specific checks. Include:

- Taipei timestamp, device/project, workflow ID, node, qubit subgroup, run ID,
  and snapshot ID/path.
- Complete run parameters needed to reproduce the result.
- Accepted fitted or manually selected values and the decisive evidence, such
  as SNR, edge fraction, FWHM, plateau/depletion evidence, Rabi fit, or T1 fit.
- Decision, reason, result-plot path, and any separately approved state commit.
- Applicability limits and what must still be revalidated on another qubit.

A useful failed attempt or recovery technique may be appended as a `Lesson`,
with the same run and snapshot references plus the observed failure and recovery
conditions. A lesson must never be presented as an accepted setting. Do not log
speculative conclusions or a run that lacks enough evidence to support them.

In addition to curated Markdown accepted settings, every run decision is
automatically retained in SQLite with its full parameters, elapsed time, snapshot,
analysis, Decision, Reason, and Next action. Agents should query this complete
decision history with `jy_get_decision_experience`; the Markdown files remain the
smaller human-curated subset of accepted settings and lessons.
