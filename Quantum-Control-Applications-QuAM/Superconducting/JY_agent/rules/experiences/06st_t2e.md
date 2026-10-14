
## Lesson — 2026-10-13: max_wait_time_in_ns is a half-wait here

- 06st_T2e_statics.py waits t on both sides of the pi pulse, so the total echo
  time is 2 x max_wait_time_in_ns, while the server's prerequisite check
  requires max_wait / T2E in 3.5-5.5 (treating it as total). With the check as
  is, use max_wait = 4 x T2E (total ~8 T2E). q1 2026-10-13: 186400 ns for T2E
  46.6 us.

## Failure — 2026-10-13 10:58 Asia/Taipei: q1 run 6c686000 (connection loss + KeyError 'T2')

- 1000 avg, max_wait 186400 ns, raw I/Q. After ~43 repetitions the QM server
  stopped answering (gRPC UNAVAILABLE, socket 10060, 8 occurrences); the script
  quit on "Too many consecutive failures" and then crashed writing
  q1.extras["T2"] (key absent -> KeyError). Lease paused; hardware lock retained
  (cleanup not verifiable while the connection was down).
- Fix: node_extras_prerequisites now lists 05st T1/T1_dev, 06st_t2star
  T2ramsey/T2ramsey_dev, 06st_t2e T2/T2_dev, so placeholders are created first.

## Failure — 2026-10-13 12:44 Asia/Taipei: q1 run 07fe69d5 (#493), 3 of 100 repetitions

- Workflow `c7f5365c…` after recovery; T2 placeholders applied first. The QOP
  10.21.30.201:9517 kept timing out / refusing (UNAVAILABLE, OpenQmException);
  only 3 repetitions ran, yet the node saved mu 33.3 us, sigma 0 and the old
  analyzer passed it. Recorded `repeat`, no commit, lease paused.
- Fix: `analysis.statistics.min_completed_iterations: 90` -- a statistics run
  needs the dataset's `iteration` size >= 90 to pass.
- The instrument link must be fixed before any long statistics run.

## Failure — 2026-10-13 13:18 / 13:30 Asia/Taipei: QM ports still locked by another QM

- Runs `4381d127…` and `3687870d…` (workflows c7f5365c, 20f0c90c): zero
  repetitions. The qm log shows "PHYSICAL CONFIG ERROR ... Resources already
  locked: device con1, fem number 2, analog output port 1/2/4 ..." -- another QM
  (likely left open by the 10:58 connection drop) holds the FEM 2 outputs. The
  script only prints "'OpenQmException' object has no attribute 'errors'".
- Read the qm ERROR line in the run log, not the script's message. The
  operator must close the stale QM (QM UI or close_all_qms) before rerunning;
  JY must not close another QM itself.
- 2026-10-13 13:55: qmm.list_open_qms() showed one QM (QM-85d1c25a…) with no
  running job -- the orphan of the 10:58 dropped run. Closed it at the operator's
  request after re-checking it was the only open QM; the list was then empty.
  Releasing the JY hardware lock does not close a QM on the server.

## Result — 2026-10-13 14:58 Asia/Taipei: q1 run 02f9c134 (#494), 100/100

- Workflow `e7ca2f4e…` after closing the orphan QM; raw I, 1000 avg, max_wait
  186400 ns (total echo to 373 us), thermal. ~59 min.
- Node summary mu 26481 us, sigma 119514 us -- 9 runaway single fits on flat
  repetitions (529 us to 1.3 s) dominate the mean; its T2echo = 26.5 ms
  candidate is unphysical and was not committed.
- Independent refit of ds.h5 (idle_time is already total echo time in us):
  99/100 fits, median 40.6 us, IQR 30.4-65.8 us; 88 within 0.2-5x median give
  48.3 +- 27.2 us -- consistent with 06b 46.6 us. Report the median/IQR for a
  raw-I statistics run; the node's mean/sigma is not robust to failed fits.

## Failure — 2026-10-13 23:47 Asia/Taipei: q2 run 56831f01, 100/100 then crash

- After committing q2's 07b rotation (e at higher I than g), 05st worked (T1
  32.5 +- 5.1 us) but 06st_t2e crashed on min([]): the echo (x90-x180--x90)
  returns the qubit to g at zero wait, so the rotated I RISES toward the mixed
  state, and the script's decay-only I fit fails on every repetition. T1 and
  the echo need opposite I directions; one readout angle cannot serve both
  without node changes (q1's echo worked because its unrotated angle put g at
  higher I).
- Operator decision 2026-10-14: skip q2's T2E statistics; single measurements
  only from now on (the operator unticks the statistics nodes on the lease
  page). Option kept for later: angle + pi (threshold negated) for 06st_t2e
  only, then restore.
