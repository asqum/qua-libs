"""Keep one quantum machine open so that the flux DC offsets never drop to 0 V between experiments.

Why this is needed
    On the OPX1000 a port goes back to its default (0 V) when the quantum machine that owns it is closed.
    `qm_session` closes the QM at the end of every node, so every node starts with 0 V on the flux lines.

How it works (needs QOP >= 3.5, i.e. `QopCaps.config_v2`, and qm-qua >= 1.2)
    1. `activate()` opens ONE quantum machine with a controller-only config (the ports of your FEMs, with the flux
       offsets as the idle values) and leaves it open. Its id is written to a small flag file.
    2. The same offsets are written into the QuAM state (`port.offset`), so every config generated afterwards carries them.
    3. `QuAM.connect()` calls `install_hold_hook()`. While the flag file exists, `qmm.open_qm(config, ...)` (which is what
       `qm_session` calls) does not open anything: it returns the held QM wrapped in `HeldQM`.
       `HeldQM.execute(program)` sends `config=<the node's config>` with the job (QOP 3.5 feature), and
       `HeldQM.close()` does nothing.
       The held QM already owns the port settings (offsets, delays, filters, upconverters...), so by default only the
       LOGICAL part (elements, pulses, waveforms...) is sent. If a node changed port settings in memory (they differ
       from the fingerprint stored in the flag file) the physical part is sent too, and the differences are logged.
    4. `deactivate()` restores the state, closes the held QM and deletes the flag file: everything is back to normal.

Nothing in site-packages is modified. Without the flag file `install_hold_hook()` does nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# One flag file per OS user. It is only honoured by a QMM that points to the same host / cluster.
FLAG_PATH = Path.home() / ".qualibrate" / "qm_hold.json"

FemKey = Tuple[str, int]

# Sections of a QUA config. The held QM owns the physical part; each job brings the logical part.
LOGICAL_KEYS = ("elements", "oscillators", "integration_weights", "waveforms", "digital_waveforms", "pulses")
PHYSICAL_KEYS = ("controllers", "octaves", "mixers")


def _normalize(obj: Any) -> Any:
    """JSON-comparable copy of a config part (str keys, lists instead of tuples, python scalars)."""
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):  # numpy scalar
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    return obj


def _diff_paths(node: Any, held: Any, path: str = ""):
    """Yield 'path: held=... node=...' for every leaf that differs (both sides already normalized)."""
    if isinstance(node, dict) and isinstance(held, dict):
        for key in sorted(set(node) | set(held)):
            yield from _diff_paths(node.get(key, "<missing>"), held.get(key, "<missing>"), f"{path}/{key}")
    elif node != held:
        yield f"{path}: held={held!r} node={node!r}"


# --------------------------------------------------------------------------------------------------------------------
# Flag file
# --------------------------------------------------------------------------------------------------------------------
def read_flag() -> Optional[dict]:
    """Return the content of the flag file, or None when no QM is being held."""
    if not FLAG_PATH.exists():
        return None
    try:
        return json.loads(FLAG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning(f"Could not read {FLAG_PATH}, ignoring it.")
        return None


def write_flag(data: dict) -> None:
    FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FLAG_PATH.write_text(json.dumps(_normalize(data), indent=2), encoding="utf-8")


def clear_flag() -> None:
    try:
        FLAG_PATH.unlink()
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------------------------------------------------
# Config helpers (pure functions, no hardware)
# --------------------------------------------------------------------------------------------------------------------
def _collect_fems(obj: Any, controllers: Set[str], out: Set[FemKey]) -> None:
    """Collect every (controller, fem) that appears as a port tuple ("con1", fem, port) inside `obj`."""
    if isinstance(obj, (tuple, list)):
        if (
            len(obj) == 3
            and isinstance(obj[0], str)
            and obj[0] in controllers
            and isinstance(obj[1], int)
            and isinstance(obj[2], int)
        ):
            out.add((obj[0], int(obj[1])))
            return
        for item in obj:
            _collect_fems(item, controllers, out)
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_fems(value, controllers, out)


def fems_of_elements(config: dict, element_names: Iterable[str]) -> Set[FemKey]:
    """The (controller, fem) pairs used by the given elements of a full config."""
    controllers = set(config.get("controllers", {}))
    out: Set[FemKey] = set()
    for name in element_names:
        if name in config.get("elements", {}):
            _collect_fems(config["elements"][name], controllers, out)
    return out


def prune_config(config: dict, fems: Iterable[FemKey]) -> dict:
    """Keep only the given FEMs: their controller entries and the elements that use nothing else.

    Returns a shallow copy, the input is not modified.
    """
    keep = {(con, int(fem)) for con, fem in fems}
    controllers = set(config.get("controllers", {}))

    pruned_controllers: Dict[str, Any] = {}
    for con, con_cfg in config.get("controllers", {}).items():
        if "fems" not in con_cfg:  # OPX+ style controller: not touched
            pruned_controllers[con] = con_cfg
            continue
        kept = {fem: fem_cfg for fem, fem_cfg in con_cfg["fems"].items() if (con, int(fem)) in keep}
        if kept:
            new_cfg = dict(con_cfg)
            new_cfg["fems"] = kept
            pruned_controllers[con] = new_cfg

    out = dict(config)
    out["controllers"] = pruned_controllers
    if "elements" in config:
        kept_elements = {}
        for name, element in config["elements"].items():
            used: Set[FemKey] = set()
            _collect_fems(element, controllers, used)
            if used <= keep:
                kept_elements[name] = element
        out["elements"] = kept_elements
    return out


def controller_only_config(config: dict, fems: Iterable[FemKey]) -> dict:
    """The config used to open the held QM: only the physical part, only the given FEMs."""
    pruned = prune_config(config, fems)
    out = {"controllers": pruned["controllers"]}
    for key in ("octaves", "mixers"):
        if pruned.get(key):
            out[key] = pruned[key]
    return out


def logical_config(config: dict, fems: Iterable[FemKey]) -> dict:
    """The part of a node's config that must travel with every job: elements, pulses, waveforms... of the given FEMs."""
    pruned = prune_config(config, fems)
    return {k: v for k, v in pruned.items() if k in LOGICAL_KEYS}


# --------------------------------------------------------------------------------------------------------------------
# The wrapper handed to qm_session / the nodes
# --------------------------------------------------------------------------------------------------------------------
class HeldQM:
    """The held quantum machine, as the nodes see it.

    - execute / compile / add_to_queue / simulate: the node's own config is sent with the job.
    - close: does nothing (the QM stays open).
    - everything else is forwarded to the real quantum machine.
    """

    def __init__(self, qm: Any, config: dict):
        object.__setattr__(self, "_qm", qm)
        object.__setattr__(self, "_config", config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._qm, name)

    def execute(self, program, *args, **kwargs):
        kwargs.setdefault("config", self._config)
        return self._qm.execute(program, *args, **kwargs)

    def compile(self, program, *args, **kwargs):
        kwargs.setdefault("config", self._config)
        return self._qm.compile(program, *args, **kwargs)

    def add_to_queue(self, program, *args, **kwargs):
        # A compiled program id (str) already carries the config given to compile(); qm-qua rejects a config here.
        if not isinstance(program, str):
            kwargs.setdefault("config", self._config)
        return self._qm.add_to_queue(program, *args, **kwargs)

    def simulate(self, program, *args, **kwargs):
        kwargs.setdefault("config", self._config)
        return self._qm.simulate(program, *args, **kwargs)

    def close(self) -> bool:
        logger.info(f"Held QM {self._qm.id} is kept open (deactivate it with quam_libs.lib.qm_hold.deactivate()).")
        return True

    def __repr__(self) -> str:
        return f"HeldQM({self._qm.id})"


def install_hold_hook(qmm: Any, host: Optional[str] = None, cluster_name: Optional[str] = None) -> None:
    """Make `qmm.open_qm(config, ...)` return the held QM while the flag file exists. Safe to call many times."""
    if getattr(qmm, "_qm_hold_original_open_qm", None) is not None:
        return
    original_open_qm = qmm.open_qm

    def open_qm(config, *args, **kwargs):
        flag = read_flag()
        if flag is None:
            return original_open_qm(config, *args, **kwargs)
        if (host is not None and flag.get("host") != host) or (
            cluster_name is not None and flag.get("cluster_name") != cluster_name
        ):
            return original_open_qm(config, *args, **kwargs)
        if flag["qm_id"] not in qmm.list_open_qms():
            logger.warning(
                f"qm_hold: the held QM {flag['qm_id']} is no longer open (closed by someone else?). "
                f"Opening a normal QM. Run deactivate() to clear {FLAG_PATH}."
            )
            return original_open_qm(config, *args, **kwargs)
        fems = [(con, int(fem)) for con, fem in flag["fems"]]
        held_physical = flag.get("physical")
        if held_physical is None:
            logger.info("qm_hold: no port fingerprint in the flag file (activated with an older version); "
                        "sending only the logical config. Re-activate to detect changed port settings.")
            job_config = logical_config(config, fems)
        else:
            node_physical = _normalize(controller_only_config(config, fems))
            differences = list(_diff_paths(node_physical, held_physical))
            if not differences:
                job_config = logical_config(config, fems)
            else:
                logger.warning(
                    f"qm_hold: this node changed {len(differences)} port setting(s) compared with the held QM; "
                    "the physical config is sent with the job:\n  " + "\n  ".join(differences[:8])
                )
                job_config = prune_config(config, fems)
        logger.info(f"qm_hold: using the held QM {flag['qm_id']} instead of opening a new one.")
        return HeldQM(qmm.get_qm(flag["qm_id"]), job_config)

    qmm._qm_hold_original_open_qm = original_open_qm
    qmm.open_qm = open_qm


# --------------------------------------------------------------------------------------------------------------------
# Planning / activating / deactivating (used by the notebook)
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class HoldTarget:
    kind: str  # "qubit" (z line) or "pair" (coupler)
    name: str  # qubit name or qubit-pair name
    channel: str  # element name in the config, e.g. "q4.z"
    port_key: str  # "con1/2/1"
    value: float  # V
    original: Optional[float]  # port.offset in the state before activation (None = not present)
    port: Any = field(default=None, repr=False)


@dataclass
class HoldPlan:
    targets: List[HoldTarget]
    fems: List[FemKey]


def _port_key(port: Any) -> str:
    return f"{port.controller_id}/{port.fem_id}/{port.port_id}"


def plan_hold(
    machine: Any,
    qubit_names: Optional[Sequence[str]] = None,
    pair_names: Optional[Sequence[str]] = None,
    overrides: Optional[Dict[str, float]] = None,
    extra_fems: Iterable[FemKey] = (),
) -> HoldPlan:
    """Decide which flux lines are held, at which voltage, and which FEMs the held QM owns.

    Voltages: qubit z line -> `z.joint_offset`; coupler -> `decouple_offset`. `overrides` is {qubit or pair name: volts}.
    qubit_names=None means every qubit of the machine (use `machine.active_qubit_names` for the active ones only).
    pair_names=None means every pair whose two qubits are both in `qubit_names` (and that has a coupler).
    FEMs: every FEM used by the xy / z / resonator of the qubits and by the couplers, plus `extra_fems`.
    """
    overrides = overrides or {}
    qubit_names = list(machine.qubits) if qubit_names is None else list(qubit_names)
    if pair_names is None:
        pair_names = [
            pn
            for pn, qp in machine.qubit_pairs.items()
            if qp.coupler is not None
            and qp.qubit_control.name in qubit_names
            and qp.qubit_target.name in qubit_names
        ]

    targets: List[HoldTarget] = []
    element_names: List[str] = []
    for name in qubit_names:
        q = machine.qubits[name]
        for ch in (q.xy, q.z, q.resonator):
            if ch is not None:
                element_names.append(ch.name)
        if q.z is not None:
            port = q.z.opx_output
            targets.append(
                HoldTarget(
                    "qubit", name, q.z.name, _port_key(port),
                    float(overrides.get(name, q.z.joint_offset)), port.offset, port,
                )
            )
    for pn in pair_names:
        qp = machine.qubit_pairs[pn]
        if qp.coupler is None:
            continue
        element_names.append(qp.coupler.name)
        port = qp.coupler.opx_output
        targets.append(
            HoldTarget(
                "pair", pn, qp.coupler.name, _port_key(port),
                float(overrides.get(pn, qp.coupler.decouple_offset)), port.offset, port,
            )
        )

    config = machine.generate_config()
    fems = fems_of_elements(config, element_names) | {(c, int(f)) for c, f in extra_fems}
    return HoldPlan(targets=targets, fems=sorted(fems))


def print_plan(plan: HoldPlan) -> None:
    print("Flux lines that will be held:")
    for t in plan.targets:
        print(f"  {t.channel:<18} ({t.kind} {t.name})  port {t.port_key:<8}  {t.value:+.5f} V   (state now: {t.original})")
    print("FEMs owned by the held QM (nobody else can use their ports meanwhile):")
    for con, fem in plan.fems:
        print(f"  {con} / FEM {fem}")


def _apply_offsets_in_state(plan: HoldPlan, restore: bool) -> None:
    for t in plan.targets:
        t.port.offset = t.original if restore else t.value


def activate(machine: Any, qmm: Any, plan: HoldPlan) -> dict:
    """Open the held QM, write the offsets into the state, and create the flag file."""
    from qm import QopCaps

    if read_flag() is not None:
        raise RuntimeError(f"A QM is already being held ({FLAG_PATH}). Run deactivate() first.")
    if not qmm.capabilities.supports(QopCaps.config_v2):
        raise RuntimeError(
            "This QOP does not support passing a config with each job (needs QOP >= 3.5), "
            "so a QM cannot be kept open across nodes this way."
        )
    if not plan.targets:
        raise ValueError("Nothing to hold: the plan has no flux line.")

    open_qm = getattr(qmm, "_qm_hold_original_open_qm", None) or qmm.open_qm

    _apply_offsets_in_state(plan, restore=False)
    try:
        hold_config = controller_only_config(machine.generate_config(), plan.fems)
        qm = open_qm(hold_config, close_other_machines=False)
    except Exception:
        _apply_offsets_in_state(plan, restore=True)
        raise

    try:
        machine.save()
        flag = {
            "qm_id": qm.id,
            "host": machine.network.get("host"),
            "cluster_name": machine.network.get("cluster_name"),
            "fems": [[c, f] for c, f in plan.fems],
            "physical": _normalize(hold_config),  # fingerprint of the port settings the held QM was opened with
            "targets": [
                {"kind": t.kind, "name": t.name, "channel": t.channel, "port_key": t.port_key,
                 "value": t.value, "original": t.original}
                for t in plan.targets
            ],
        }
        write_flag(flag)
    except Exception:
        _apply_offsets_in_state(plan, restore=True)
        qm.close()
        raise
    print(f"Holding QM {qm.id}. Offsets written to the state. Flag file: {FLAG_PATH}")
    return flag


def status(qmm: Any) -> str:
    flag = read_flag()
    if flag is None:
        return "No QM is being held. Nodes open and close their own QM as usual."
    is_open = flag["qm_id"] in qmm.list_open_qms()
    lines = [f"Holding QM {flag['qm_id']}  (open on the QOP: {is_open})"]
    lines += [f"  {t['channel']:<18} {t['value']:+.5f} V" for t in flag["targets"]]
    return "\n".join(lines)


def deactivate(qmm: Any, close_qm: bool = True) -> None:
    """Restore the state, close the held QM, delete the flag file. After this nodes work as before."""
    from quam_libs.components import QuAM

    flag = read_flag()
    if flag is None:
        print("Nothing is being held.")
        return

    machine = QuAM.load()
    for t in flag["targets"]:
        if t["kind"] == "qubit":
            port = machine.qubits[t["name"]].z.opx_output
        else:
            port = machine.qubit_pairs[t["name"]].coupler.opx_output
        port.offset = t["original"]
    machine.save()
    print("State restored (the held offsets are removed from the ports).")

    try:
        if close_qm and flag["qm_id"] in qmm.list_open_qms():
            qmm.get_qm(flag["qm_id"]).close()
            print(f"Closed QM {flag['qm_id']} (its ports go back to the default value).")
        elif close_qm:
            print(f"QM {flag['qm_id']} was already closed.")
    finally:
        clear_flag()
        print(f"Flag file removed: {FLAG_PATH}")
