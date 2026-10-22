from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Iterable


@dataclass(frozen=True)
class FailureClassification:
    category: str | None
    safe_to_release_lock: bool = False


_INSTRUMENT_CONNECTION_TYPES = {
    "qmconnectionerror",
    "qmtimeouterror",
    "qmhealthcheckerror",
    "qmfailedtogetquantummachineerror",
    "qmfailedtoclosequantummachineerror",
    "qmfailedtocloseallquantummachineserror",
    "qmserverdetectionerror",
    "octaveconnectionerror",
    "octaveloopbackerror",
    "connecterror",
    "connecttimeout",
    "newconnectionerror",
    "nameresolutionerror",
}

_NO_CONNECTION_ESTABLISHED_TYPES = {
    "connectionrefusederror",
    "gaierror",
    "qmserverdetectionerror",
    "connecterror",
    "connecttimeout",
    "newconnectionerror",
    "nameresolutionerror",
}

_CONNECTION_MESSAGE_MARKERS = (
    "failed to connect",
    "unable to connect",
    "could not connect",
    "connection refused",
    "connection reset",
    "connection aborted",
    "network is unreachable",
    "host is unreachable",
    "name or service not known",
    "temporary failure in name resolution",
    "server detection",
    "failed to get quantum machine",
    "qop server",
    "opx connection",
    "octave connection",
    "grpc_status:14",
    "statuscode.unavailable",
    "deadline exceeded",
)

_QOP_COMPILE_FAILURE_TYPES = {
    "qopresponseerror",
    "failedtoexecutejobexception",
}

# Frames inside the ``qm`` package that mean the client was handing a program
# to the QOP rather than opening a connection to it.  Reaching any of them
# proves the manager was created, the config was uploaded, and a quantum
# machine was opened, because each of those is an earlier round trip.
_PROGRAM_SUBMISSION_FUNCTIONS = {
    "execute",
    "add_to_queue",
    "add_compiled",
    "_add_program",
    "compile",
    "_compile",
}

_QOP_COMPILE_MESSAGE_MARKERS = (
    "compilation failed",
    "internal error. please report it to qm",
)


def classify_failure(exc: BaseException) -> FailureClassification:
    """Classify recognizable instrument and compile-time QOP failures.

    Generic timeouts and arbitrary ``OSError`` values are intentionally not
    enough on their own: a fitting or file timeout must remain a normal worker
    failure.  ``safe_to_release_lock`` is true when either:

    - a transport connection was never established, or
    - QOP rejected the program at compile/queue time with an Internal error
      (job never owned the hardware, so retaining the lock is unnecessary).

    ``program_submission_timeout`` is reported separately from
    ``instrument_unreachable``.  A deadline that expires while the program is
    being handed to the QOP is not evidence that the instrument is
    unreachable: every earlier call in the same run was answered.  It usually
    means the program is too large for one submission, so the caller may
    retry with a smaller multiplex group.  The classifier cannot tell that
    apart from a QOP that is alive but wedged, so the caller must confirm with
    a live reachability probe before acting on this category.
    """

    chain = tuple(_exception_chain(exc))
    type_names = {type(item).__name__.casefold() for item in chain}
    messages = "\n".join(str(item).casefold() for item in chain)

    # Operator instruction 2026-10-19 (q2 06 run 842fcbcf, Ramsey fitting.py
    # KeyError): a crash raised inside a node's analysis/fitting code happens
    # after the instrument has returned the data, so it never needs the
    # hardware lock; the run is simply repeated.
    if _raised_in_analysis(chain):
        return FailureClassification("analysis_failure", safe_to_release_lock=True)

    if _looks_like_qop_compile_failure(type_names, messages):
        return FailureClassification(
            "qop_compile_failure",
            safe_to_release_lock=True,
        )

    # Checked before the connectivity markers below, which deliberately still
    # match "deadline exceeded" for a deadline that expires anywhere else.
    if _looks_like_program_submission_timeout(exc, type_names, messages):
        return FailureClassification(
            "program_submission_timeout",
            safe_to_release_lock=False,
        )

    specific_type = bool(type_names & _INSTRUMENT_CONNECTION_TYPES)
    builtin_connection = bool(
        type_names
        & {
            "connectionerror",
            "connectionrefusederror",
            "connectionreseterror",
            "connectionabortederror",
            "gaierror",
        }
    )
    marked_message = any(marker in messages for marker in _CONNECTION_MESSAGE_MARKERS)
    if not (specific_type or builtin_connection or marked_message):
        return FailureClassification(None)

    safe_to_release = bool(type_names & _NO_CONNECTION_ESTABLISHED_TYPES)
    return FailureClassification(
        "instrument_unreachable",
        safe_to_release_lock=safe_to_release,
    )


def _looks_like_program_submission_timeout(
    exc: BaseException, type_names: set[str], messages: str
) -> bool:
    """Report a deadline that expired while submitting a program.

    Requires all three of: a QM timeout type, a gRPC ``DEADLINE_EXCEEDED``
    status rather than a transport failure, and a ``qm`` package frame that
    submits a program.  A transport-level marker anywhere in the chain vetoes
    it, so a connection that dropped mid-submission stays an outage.
    """

    if "qmtimeouterror" not in type_names:
        return False
    if _has_transport_failure(type_names, messages):
        return False
    status = _grpc_status_name(exc)
    if status is not None and status != "DEADLINE_EXCEEDED":
        return False
    if status is None and "deadline exceeded" not in messages:
        return False
    return _failed_during_program_submission(exc)


def _has_transport_failure(type_names: set[str], messages: str) -> bool:
    """Report evidence that the transport itself failed, not just a deadline."""

    if type_names & _NO_CONNECTION_ESTABLISHED_TYPES:
        return True
    markers = (
        "statuscode.unavailable",
        "grpc_status:14",
        "connection refused",
        "connection reset",
        "connection aborted",
        "network is unreachable",
        "host is unreachable",
        "name or service not known",
        "temporary failure in name resolution",
    )
    return any(marker in messages for marker in markers)


def _grpc_status_name(exc: BaseException) -> str | None:
    """Return the gRPC status name carried by any error in the chain."""

    for item in _exception_chain(exc):
        code = getattr(item, "code", None)
        if not callable(code):
            continue
        try:
            value = code()
        except Exception:  # pragma: no cover - defensive
            continue
        name = getattr(value, "name", None)
        if isinstance(name, str) and name:
            return name
    return None


def _failed_during_program_submission(exc: BaseException) -> bool:
    for name, filename in _traceback_frames(exc):
        if name in _PROGRAM_SUBMISSION_FUNCTIONS and _is_qm_frame(filename):
            return True
    return False


def _raised_in_analysis(chain: tuple[BaseException, ...]) -> bool:
    """True when the traceback runs through fitting code and never into ``qm``."""

    frames = [filename for item in chain for _name, filename in _traceback_frames(item)]
    if any(_is_qm_frame(filename) for filename in frames):
        return False
    return any(_is_analysis_frame(filename) for filename in frames)


def _is_analysis_frame(filename: str) -> bool:
    try:
        parts = [part.casefold() for part in PurePath(filename).parts]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    name = parts[-1] if parts else ""
    if "quam_libs" in parts and (
        "analysis" in parts or name in {"fit.py", "fitting.py"} or "fit" in parts
    ):
        return True
    if "lmfit" in parts or "qiskit_experiments" in parts:
        return True
    return "scipy" in parts and "optimize" in parts


def _is_qm_frame(filename: str) -> bool:
    try:
        parts = PurePath(filename).parts
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    return "qm" in parts


def _traceback_frames(exc: BaseException) -> Iterable[tuple[str, str]]:
    for item in _exception_chain(exc):
        tb = item.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            yield code.co_name, code.co_filename
            tb = tb.tb_next


def _looks_like_qop_compile_failure(
    type_names: set[str], messages: str
) -> bool:
    has_compile_type = bool(type_names & _QOP_COMPILE_FAILURE_TYPES)
    has_compile_message = any(
        marker in messages for marker in _QOP_COMPILE_MESSAGE_MARKERS
    )
    # Require both a QM execute/compile exception type and the known markers so
    # unrelated "internal error" strings elsewhere do not release the lock.
    return has_compile_type and has_compile_message


def _exception_chain(exc: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__
