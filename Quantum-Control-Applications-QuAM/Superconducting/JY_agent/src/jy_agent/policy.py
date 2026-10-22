from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .config import Settings, load_policies
from .state import (
    apply_json_patch,
    channel_kind,
    load_state,
    operation_amplitude,
    operation_backing_name,
    pointer_get,
    shared_xy_drive_groups,
    shared_z_line_groups,
    z_offset,
)
from .util import is_finite_number


class PolicyError(ValueError):
    pass


class _FactorCaps:
    def __init__(self, shared: set[str], shared_cap: float, private_cap: float):
        self._shared = shared
        self._shared_cap = shared_cap
        self._private_cap = private_cap

    def shared(self, name: str) -> bool:
        return name in self._shared

    def __getitem__(self, name: str) -> float:
        return self._shared_cap if name in self._shared else self._private_cap


class PolicyEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.raw = load_policies(settings)
        self.nodes: dict[str, dict[str, Any]] = self.raw["nodes"]
        self.limits: dict[str, Any] = self.raw["instrument_limits"]
        self.shared_xy_drive: dict[str, Any] = self.raw.get("shared_xy_drive", {})

    def node_definition(self, node_id: str) -> dict[str, Any]:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise PolicyError(
                f"Node {node_id!r} is not allowlisted. Allowed: {sorted(self.nodes)}"
            ) from exc

    def node_script(self, node_id: str) -> Path:
        definition = self.node_definition(node_id)
        script_root = str(definition.get("script_root", "calibration_graph"))
        roots = {
            "calibration_graph": self.settings.calibration_graph.resolve(),
            "side_project": (self.settings.superconducting_root / "side_project").resolve(),
        }
        if script_root not in roots:
            raise PolicyError(f"Unsupported script_root for {node_id}: {script_root}")
        root = roots[script_root]
        script = (root / definition["script"]).resolve()
        try:
            script.relative_to(root)
        except ValueError as exc:
            raise PolicyError(f"Allowlisted script for {node_id} escapes its root") from exc
        if not script.is_file():
            raise PolicyError(f"Invalid or missing allowlisted script for {node_id}")
        return script

    def validate_run(
        self,
        node_id: str,
        parameters: dict[str, Any],
        *,
        drive_mode: str | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(parameters, dict):
            raise PolicyError("parameters must be an object")
        definition = self.node_definition(node_id)
        rules: dict[str, str] = definition["allowed_parameters"]
        unknown = sorted(set(parameters) - set(rules))
        if unknown:
            raise PolicyError(f"Unsupported parameters for {node_id}: {unknown}")

        merged = {**definition.get("defaults", {}), **parameters}
        product = definition.get("detuning_wait_product_mhz_us")
        if (
            product is not None
            and "frequency_detuning_in_mhz" not in parameters
            and is_finite_number(merged.get("max_wait_time_in_ns"))
            and float(merged["max_wait_time_in_ns"]) > 0
        ):
            # Operator instruction 2026-10-02: the Ramsey detuning follows the
            # window, so a run that does not name one gets the matching value.
            # 2026-10-15: the window is max - min wait.
            window_ns = float(merged["max_wait_time_in_ns"]) - float(
                merged.get("min_wait_time_in_ns") or 0
            )
            if window_ns > 0:
                merged["frequency_detuning_in_mhz"] = round(
                    float(product) / (window_ns / 1000.0), 6
                )
        if self.settings.require_explicit_qubits and "qubits" not in parameters:
            raise PolicyError("An explicit qubits list is required for every run")
        for key, value in merged.items():
            if key == "qubits" or key in rules:
                self._validate_type(key, value, rules.get(key, "qubit_list"))

        state = load_state(self.settings.active_state)
        qubits = merged.get("qubits")
        if not isinstance(qubits, list) or not qubits:
            raise PolicyError("qubits must contain at least one explicit target")
        missing = [name for name in qubits if name not in state.get("qubits", {})]
        if missing:
            raise PolicyError(f"Unknown qubits in active state: {missing}")
        if len(set(qubits)) != len(qubits):
            raise PolicyError("qubits must not contain duplicates")

        if self.is_flux_calibration_node(node_id):
            shared_z = self.shared_z_line_groups()
            if shared_z:
                described = "; ".join(
                    f"{output} drives {', '.join(names)}"
                    for output, names in sorted(shared_z.items())
                )
                raise PolicyError(
                    f"{node_id} is flux calibration, which is disabled because "
                    f"this chip shares Z lines ({described})."
                )
        self._validate_common_sweep(merged)
        self._validate_node_caps(node_id, definition, merged)
        self._validate_detuning_wait_product(definition, merged)
        self._validate_node_specific(node_id, merged, state)
        self._validate_runtime_estimate(node_id, merged, state)
        warnings: list[str] = []
        if definition["flux_behavior"] == "hardcoded_min":
            warnings.append(
                f"{node_id} applies the node's hardcoded minimum-flux behavior; "
                "the runner cannot override it."
            )
        else:
            warnings.append(f"{node_id} is constrained to joint flux mode.")
        self._validate_shared_xy_drive_targets(node_id, qubits, drive_mode)
        if merged.get("multiplexed") is True and len(qubits) > 1:
            # 02 is the broadband search that assigns the readout IFs, so it
            # may run while they are still unset (operator rule 2026-10-07).
            warnings.extend(
                self.validate_multiplex_targets(
                    qubits, state, allow_unset_readout=node_id == "02"
                )
            )
        return merged, warnings

    def _validate_shared_xy_drive_targets(
        self, node_id: str, qubits: list[str], drive_mode: str | None = None
    ) -> None:
        """Hold a shared-drive node to one qubit per run."""

        cap = self.shared_xy_drive_cap(node_id, drive_mode)
        if cap is None or len(qubits) <= cap:
            return
        groups = self.shared_xy_drive_groups()
        shared = "; ".join(
            f"{output} drives {', '.join(names)}"
            for output, names in sorted(groups.items())
        ) or "the operator chose per-qubit measurement"
        raise PolicyError(
            f"{node_id} may measure at most {cap} qubit(s) per run because this "
            f"chip shares XY drive outputs ({shared}); requested {len(qubits)}. "
            "Run the unresolved qubits one at a time and finish the node before "
            "advancing."
        )

    def mw_band_lo_window(self, band: Any) -> tuple[float, float] | None:
        """The LO range this MW band allows, keeping clear of its edges.

        Operator instruction 2026-09-22: an LO within
        `mw_band_edge_margin_hz` of a band edge risks a hardware error, so the
        usable window is the band range shrunk by that margin at both ends.
        `None` means the band is unknown and only the global limits apply.
        """

        bands = self.limits.get("mw_band_frequency_hz")
        if not isinstance(bands, dict) or isinstance(band, bool):
            return None
        if not isinstance(band, int):
            return None
        span = bands.get(band, bands.get(str(band)))
        if not isinstance(span, dict):
            return None
        low, high = span.get("min"), span.get("max")
        if not is_finite_number(low) or not is_finite_number(high):
            return None
        margin = self.limits.get("mw_band_edge_margin_hz", 0)
        if not is_finite_number(margin):
            margin = 0
        low, high = float(low) + float(margin), float(high) - float(margin)
        return (low, high) if low <= high else None

    def spectroscopy_amplitude_factor_caps(self) -> "_FactorCaps":
        """03a operation_amplitude_factor cap per qubit, from the XY wiring.

        Operator instruction 2026-10-14: a qubit on a shared XY line gets
        `max_operation_amplitude_factor_shared_xy`, every other one
        `max_operation_amplitude_factor_private_xy`.
        """

        rules = self.raw["analysis"]["03a"]
        shared = {
            name for names in self.shared_xy_drive_groups().values() for name in names
        }
        return _FactorCaps(
            shared,
            float(rules["max_operation_amplitude_factor_shared_xy"]),
            float(rules["max_operation_amplitude_factor_private_xy"]),
        )

    def shared_xy_drive_groups(self) -> dict[str, list[str]]:
        """XY drive outputs that more than one qubit of this chip shares."""

        try:
            wiring = load_state(self.settings.wiring_path)
        except Exception:  # pragma: no cover - a missing wiring file is reported elsewhere
            return {}
        return shared_xy_drive_groups(wiring)

    def serializes_on_shared_xy_drive(self, node_id: str) -> bool:
        """True when this node uses the drive line the chip shares.

        Nodes before `shared_xy_drive.serialize_from_node` only touch the
        readout line, so a shared drive output does not constrain them.
        """

        first = self.shared_xy_drive.get("serialize_from_node")
        if not isinstance(first, str):
            return False
        sequence = list(self.settings.workflow_sequence)
        if first not in sequence or node_id not in sequence:
            return False
        return sequence.index(node_id) >= sequence.index(first)

    def shared_xy_drive_cap(
        self, node_id: str, drive_mode: str | None = None
    ) -> int | None:
        """Targets per run this node allows on a shared XY drive output.

        `None` means the shared-drive rule does not apply: either the chip
        wires every qubit to its own XY output, or the node runs before the
        drive line is used.

        Operator request 2026-10-04: `drive_mode`, chosen on the lease
        approval page, overrides the wiring -- `per_qubit` applies the cap on
        any chip and `multiplex` lifts it. `None` keeps the wiring default.

        Operator instruction 2026-10-08: `multiplex` cannot lift the cap on a
        chip that shares an XY output; such a chip is always one at a time.
        """

        cap = self.shared_xy_drive.get("max_multiplex_targets")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            return None
        if not self.serializes_on_shared_xy_drive(node_id):
            return None
        if drive_mode == "per_qubit" or self.shared_xy_drive_groups():
            return cap
        return None

    def shared_z_line_groups(self) -> dict[str, list[str]]:
        """Z (flux) outputs that more than one qubit of this chip shares."""

        try:
            wiring = load_state(self.settings.wiring_path)
        except Exception:  # pragma: no cover - a missing wiring file is reported elsewhere
            return {}
        return shared_z_line_groups(wiring)

    def flux_calibration_nodes(self) -> tuple[list[str], list[str]]:
        """The flux nodes a lease-page yes enables, and the optional extras."""

        config = self.raw.get("workflow_options", {}).get("flux_calibration") or {}
        return (
            [str(node) for node in config.get("nodes", [])],
            [str(node) for node in config.get("optional", [])],
        )

    def is_flux_calibration_node(self, node_id: str) -> bool:
        nodes, optional = self.flux_calibration_nodes()
        return node_id in nodes or node_id in optional

    def readout_frequencies_unset(
        self,
        qubits: list[str],
        state: dict[str, Any] | None = None,
    ) -> list[str]:
        """Targets whose readout IF is zero or shared with a line-mate.

        Operator rule 2026-10-07: such a chip has not had its resonators
        assigned yet, so the workflow must start with the broadband 02 search.
        """
        active_state = state or load_state(self.settings.active_state)
        wiring = load_state(self.settings.wiring_path)
        wiring_qubits = wiring.get("wiring", wiring).get("qubits", {})
        by_line: dict[tuple[str, float], list[str]] = {}
        unset: set[str] = set()
        for name in qubits:
            qubit = active_state.get("qubits", {}).get(name, {})
            frequency = (qubit.get("resonator") or {}).get("intermediate_frequency")
            if not is_finite_number(frequency) or abs(float(frequency)) < 1.0:
                unset.add(name)
                continue
            output = str(wiring_qubits.get(name, {}).get("rr", {}).get("opx_output"))
            by_line.setdefault((output, float(frequency)), []).append(name)
        for names in by_line.values():
            if len(names) > 1:
                unset.update(names)
        return sorted(unset)

    def readout_amplitude_budget_patch(
        self,
        qubits: list[str],
        state: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Scale each over-budget readout line down to its target sum.

        Operator choice 2026-10-07: a new chip's template amplitudes can sum
        above `max_aggregate_readout_amplitude`. Instead of refusing entry, JY
        scales every amplitude on such a line by one factor so the line sums to
        `readout_amplitude_budget_target`; 02c later sets each amplitude from
        its dressed limit. Lines already within the limit are left alone.
        """
        active_state = state or load_state(self.settings.active_state)
        wiring = load_state(self.settings.wiring_path)
        wiring_qubits = wiring.get("wiring", wiring).get("qubits", {})
        limits = self.limits["multiplex"]
        maximum = float(limits["max_aggregate_readout_amplitude"])
        target = float(limits.get("readout_amplitude_budget_target", 0.9 * maximum))
        lines: dict[str, list[tuple[str, str, float]]] = {}
        for name in qubits:
            qubit = active_state.get("qubits", {}).get(name, {})
            operations = (qubit.get("resonator") or {}).get("operations", {})
            key = "readout"
            readout = operations.get(key)
            if isinstance(readout, str) and readout.startswith("#./"):
                key = readout[3:]
                readout = operations.get(key)
            amplitude = readout.get("amplitude") if isinstance(readout, dict) else None
            if not is_finite_number(amplitude):
                continue
            output = str(wiring_qubits.get(name, {}).get("rr", {}).get("opx_output"))
            path = f"/qubits/{name}/resonator/operations/{key}/amplitude"
            lines.setdefault(output, []).append((name, path, float(amplitude)))
        patch: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        for output, members in sorted(lines.items()):
            total = sum(abs(amplitude) for _, _, amplitude in members)
            if total <= maximum:
                continue
            factor = target / total
            details[output] = {
                "old_sum": total,
                "new_sum": target,
                "factor": factor,
                "qubits": {name: amplitude * factor for name, _, amplitude in members},
            }
            for _name, path, amplitude in members:
                patch.append({"op": "replace", "path": path, "value": amplitude * factor})
        return patch, details

    def validate_multiplex_targets(
        self,
        qubits: list[str],
        state: dict[str, Any] | None = None,
        *,
        allow_unset_readout: bool = False,
        allow_over_budget: bool = False,
    ) -> list[str]:
        """Validate shared-readout topology before a multi-qubit run.

        ``allow_unset_readout`` skips the resonator-spacing check, for the
        broadband 02 search that assigns those frequencies.
        """
        limits = self.limits["multiplex"]
        if len(qubits) > int(limits["max_qubits_per_run"]):
            raise PolicyError(
                f"Multiplex run has {len(qubits)} qubits; policy maximum is "
                f"{limits['max_qubits_per_run']}"
            )
        active_state = state or load_state(self.settings.active_state)
        wiring = load_state(self.settings.wiring_path)
        wiring_qubits = wiring.get("wiring", wiring).get("qubits", {})
        groups: dict[str, dict[str, Any]] = {}
        for name in qubits:
            qubit = active_state.get("qubits", {}).get(name)
            topology = wiring_qubits.get(name, {}).get("rr", {})
            if not isinstance(qubit, dict):
                raise PolicyError(f"Multiplex target {name} is missing from state")
            output = topology.get("opx_output")
            input_ = topology.get("opx_input")
            if not isinstance(output, str) or not isinstance(input_, str):
                raise PolicyError(
                    f"{name} has no complete resonator OPX input/output wiring"
                )
            resonator = qubit.get("resonator", {})
            frequency = resonator.get("intermediate_frequency")
            if not is_finite_number(frequency):
                raise PolicyError(f"{name} resonator IF is not numeric")
            operations = resonator.get("operations", {})
            readout = operations.get("readout")
            if isinstance(readout, str) and readout.startswith("#./"):
                readout = operations.get(readout[3:])
            amplitude = readout.get("amplitude") if isinstance(readout, dict) else None
            if not is_finite_number(amplitude) or float(amplitude) <= 0:
                raise PolicyError(
                    f"{name} readout amplitude must be finite and positive"
                )
            group = groups.setdefault(
                output,
                {"input": input_, "qubits": []},
            )
            if group["input"] != input_:
                raise PolicyError(
                    f"Readout output {output} maps to multiple acquisition inputs"
                )
            group["qubits"].append(
                {
                    "name": name,
                    "frequency": float(frequency),
                    "amplitude": abs(float(amplitude)),
                }
            )

        descriptions: list[str] = []
        for output, group in sorted(groups.items()):
            members = group["qubits"]
            if len(members) > int(limits["max_qubits_per_readout_line"]):
                raise PolicyError(
                    f"Readout line {output} has {len(members)} targets; policy "
                    f"maximum is {limits['max_qubits_per_readout_line']}"
                )
            ordered = sorted(members, key=lambda item: item["frequency"])
            pairs = [] if allow_unset_readout else list(zip(ordered, ordered[1:]))
            for left, right in pairs:
                spacing = right["frequency"] - left["frequency"]
                if spacing < float(limits["min_resonator_spacing_hz"]):
                    raise PolicyError(
                        f"{left['name']} and {right['name']} are only "
                        f"{spacing / 1e6:g} MHz apart on {output}"
                    )
            aggregate = sum(item["amplitude"] for item in members)
            if not allow_over_budget and aggregate > float(
                limits["max_aggregate_readout_amplitude"]
            ):
                raise PolicyError(
                    f"Aggregate readout amplitude {aggregate:g} on {output} "
                    f"exceeds {limits['max_aggregate_readout_amplitude']:g}"
                )
            names = ",".join(item["name"] for item in members)
            descriptions.append(f"{names} on {output}")
        return [
            f"Multiplex topology validated for {len(qubits)} qubits across "
            f"{len(groups)} readout lines: {'; '.join(descriptions)}."
        ]

    def _validate_type(self, key: str, value: Any, rule: str) -> None:
        valid = True
        if rule == "qubit_list":
            valid = (
                isinstance(value, list)
                and all(isinstance(item, str) and re.fullmatch(r"q[0-9]+", item) for item in value)
            )
        elif rule == "positive_int":
            valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
        elif rule == "optional_positive_int":
            valid = value is None or (
                isinstance(value, int) and not isinstance(value, bool) and value > 0
            )
        elif rule == "optional_int":
            valid = value is None or (
                isinstance(value, int) and not isinstance(value, bool)
            )
        elif rule == "int":
            # For a protected node that types the field as `int`. A fractional
            # value crashes its Parameters model before any hardware runs; see
            # the 2026-09-18 02c power-limit lesson.
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif rule == "bool":
            valid = isinstance(value, bool)
        elif rule == "number":
            valid = is_finite_number(value)
        elif rule == "optional_number":
            valid = value is None or is_finite_number(value)
        elif rule == "optional_positive_number_list":
            valid = value is None or (
                isinstance(value, list)
                and bool(value)
                and all(is_finite_number(item) and float(item) > 0 for item in value)
            )
        elif rule == "positive_number":
            valid = is_finite_number(value) and float(value) > 0
        elif rule == "nonnegative_number":
            valid = is_finite_number(value) and float(value) >= 0
        elif rule == "joint_literal":
            valid = value == "joint"
        elif rule == "saturation_literal":
            valid = value == "saturation"
        elif rule == "xy_operation_literal":
            valid = value in {"x180", "x90", "-x90", "y90", "-y90"}
        elif rule == "reset_literal":
            valid = value in {"thermal", "active"}
        elif rule == "rb_reset_literal":
            valid = value in {"thermal", "active", "active_gef"}
        elif rule == "readout_mode_literal":
            valid = value in {"ge", "gef"}
        elif rule == "linear_log_literal":
            valid = value in {"linear", "log"}
        elif rule == "log_lin_literal":
            valid = value in {"log", "lin"}
        elif rule == "plot_dimension_literal":
            valid = value in {"2D", "3D"}
        else:
            raise PolicyError(f"Unknown policy type rule {rule!r}")
        if not valid:
            raise PolicyError(f"Parameter {key!r} violates rule {rule!r}: {value!r}")

    @staticmethod
    def _validate_node_caps(
        node_id: str, definition: dict[str, Any], parameters: dict[str, Any]
    ) -> None:
        """Per-node averaging and time-point caps (operator 2026-10-20).

        T1/T2*/T2e runs were too slow: `max_num_averages` caps num_averages,
        and `max_time_points` caps the sweep length -- `num_time_points` where
        the node has it, otherwise (max - min) / step + 1.
        """

        cap = definition.get("max_num_averages")
        averages = parameters.get("num_averages")
        if cap is not None and is_finite_number(averages) and averages > int(cap):
            raise PolicyError(
                f"{node_id} num_averages {averages} exceeds the cap {cap}; a run too "
                "noisy at the cap counts as failed"
            )
        points_cap = definition.get("max_time_points")
        if points_cap is None:
            return
        points = parameters.get("num_time_points")
        if points is None:
            low = parameters.get("min_wait_time_in_ns")
            high = parameters.get("max_wait_time_in_ns")
            step = parameters.get("wait_time_step_in_ns")
            if all(is_finite_number(value) for value in (low, high, step)) and step > 0:
                points = math.floor((float(high) - float(low)) / float(step)) + 1
        if is_finite_number(points) and points > int(points_cap):
            raise PolicyError(
                f"{node_id} sweep has {int(points)} time points; the cap is "
                f"{points_cap} (raise wait_time_step_in_ns or lower num_time_points)"
            )

    def _validate_detuning_wait_product(
        self, definition: dict[str, Any], parameters: dict[str, Any]
    ) -> None:
        """Keep the Ramsey detuning inversely proportional to its window.

        Operator instruction 2026-10-02: detuning [MHz] x window [us] stays at
        a fixed product (4 MHz*us: 1 us -> 4 MHz, 10 us -> 0.4 MHz), so every
        Ramsey plot shows the same number of fringes and is easy to read by
        eye. 2026-10-15: the window is max - min wait, the product is exact,
        min wait is fixed and max wait sits on min + a multiple of the step.
        """

        product = definition.get("detuning_wait_product_mhz_us")
        if product is None:
            return
        tolerance = float(definition.get("detuning_wait_product_tolerance", 0.25))
        detuning = parameters.get("frequency_detuning_in_mhz")
        wait_ns = parameters.get("max_wait_time_in_ns")
        minimum_ns = parameters.get("min_wait_time_in_ns")
        fixed_min = definition.get("fixed_min_wait_time_in_ns")
        step = definition.get("max_wait_time_step_in_ns")
        if fixed_min is not None and minimum_ns != fixed_min:
            raise PolicyError(f"min_wait_time_in_ns must be {fixed_min} ns")
        if (
            step is not None
            and is_finite_number(wait_ns)
            and is_finite_number(minimum_ns)
            and (
                float(wait_ns) <= float(minimum_ns)
                or (float(wait_ns) - float(minimum_ns)) % float(step)
            )
        ):
            raise PolicyError(
                f"max_wait_time_in_ns must be {minimum_ns} + a positive multiple "
                f"of {step} ns (e.g. {float(minimum_ns) + 40 * float(step):g})"
            )
        cap = definition.get("max_wait_time_cap_ns")
        if cap is not None and is_finite_number(wait_ns) and float(wait_ns) > float(cap):
            raise PolicyError(f"max_wait_time_in_ns {float(wait_ns):g} exceeds {cap} ns")
        if not is_finite_number(detuning) or not is_finite_number(wait_ns):
            return
        window_us = (float(wait_ns) - float(minimum_ns or 0)) / 1000.0
        actual = float(detuning) * window_us
        # 1e-4 covers the derived value rounded to 6 decimals.
        if abs(actual - float(product)) > max(tolerance, 1e-4) * float(product):
            expected = float(product) / window_us
            raise PolicyError(
                f"frequency_detuning_in_mhz {float(detuning):g} with a "
                f"{window_us:g} us window gives {actual:g} MHz*us; the Ramsey "
                f"rule needs about {float(product):g} MHz*us, i.e. "
                f"{expected:.4g} MHz for this window (or omit the detuning and "
                "it is derived)"
            )

    def estimate_run_seconds(
        self, node_id: str, parameters: dict[str, Any], state: dict[str, Any]
    ) -> float | None:
        """Lower-bound hardware time of one run, or None without an estimator.

        Counts shots times the pulses, readout and waits each shot plays; the
        program, data transfer and analysis come on top. Qubits are measured
        one after another unless the node runs them multiplexed.
        """

        if node_id not in {"02b", "03b", "06a"}:
            return None
        qubits = [state["qubits"][name] for name in parameters.get("qubits", [])]
        if not qubits:
            return None

        def readout_ns(qubit: dict[str, Any]) -> float:
            operations = (qubit.get("resonator") or {}).get("operations") or {}
            readout = operations.get("readout")
            if isinstance(readout, str) and readout.startswith("#./"):
                readout = operations.get(readout[3:])
            length = readout.get("length") if isinstance(readout, dict) else None
            return float(length) if is_finite_number(length) else 2000.0

        def depletion_ns(qubit: dict[str, Any]) -> float:
            value = (qubit.get("resonator") or {}).get("depletion_time")
            return float(value) if is_finite_number(value) else 2000.0

        averages = int(parameters["num_averages"])
        per_qubit: list[float] = []
        for qubit in qubits:
            shot = readout_ns(qubit) + depletion_ns(qubit)
            if node_id == "02b":
                span = float(parameters["frequency_span_in_mhz"])
                step = float(parameters["frequency_step_in_mhz"])
                frequencies = max(1, math.ceil(span / step))
                settle = (qubit.get("z") or {}).get("settle_time")
                settle_ns = float(settle) if is_finite_number(settle) else 0.0
                per_flux = settle_ns + frequencies * shot
                total_ns = averages * int(parameters["num_flux_points"]) * per_flux
            elif node_id == "03b":
                span = float(parameters["frequency_span_in_mhz"])
                step = float(parameters["frequency_step_in_mhz"])
                frequencies = max(1, math.ceil(span / step))
                length = parameters.get("operation_len_in_ns")
                if not is_finite_number(length):
                    saturation = ((qubit.get("xy") or {}).get("operations") or {}).get(
                        "saturation"
                    )
                    length = (
                        saturation.get("length") if isinstance(saturation, dict) else None
                    )
                drive_ns = float(length) if is_finite_number(length) else 0.0
                total_ns = (
                    averages
                    * frequencies
                    * int(parameters["num_flux_points"])
                    * (drive_ns + shot)
                )
            else:
                span = float(parameters["flux_span"])
                fluxes = math.floor(span / float(parameters["flux_step"]) + 1e-9) + 1
                waits = range(
                    int(parameters["min_wait_time_in_ns"]) // 4,
                    int(parameters["max_wait_time_in_ns"]) // 4,
                    max(1, int(parameters["wait_time_step_in_ns"]) // 4),
                )
                idle_ns = sum(4 * value for value in waits)
                # Two state readouts per shot, plus the two x90 and gaps.
                total_ns = averages * fluxes * (
                    len(waits) * (2 * shot + 400.0) + idle_ns
                )
            per_qubit.append(total_ns * 1e-9)
        if parameters.get("multiplexed") is True:
            return max(per_qubit)
        return sum(per_qubit)

    def _validate_runtime_estimate(
        self, node_id: str, parameters: dict[str, Any], state: dict[str, Any]
    ) -> None:
        """Operator instruction 2026-10-08: only statistics nodes run long."""

        limit_minutes = self.limits.get("max_run_minutes")
        if not is_finite_number(limit_minutes):
            return
        estimate = self.estimate_run_seconds(node_id, parameters, state)
        if estimate is not None and estimate > float(limit_minutes) * 60:
            raise PolicyError(
                f"{node_id} would take at least {estimate / 60:.1f} min on the "
                f"hardware; non-statistics runs must finish within "
                f"{float(limit_minutes):g} min. Reduce averages, points or "
                "targets per run."
            )

    def _validate_common_sweep(self, parameters: dict[str, Any]) -> None:
        averages = parameters.get("num_averages")
        if averages is not None and averages > int(self.limits["max_num_averages"]):
            raise PolicyError("num_averages exceeds the configured policy limit")
        span = parameters.get("frequency_span_in_mhz")
        step = parameters.get("frequency_step_in_mhz")
        if span is not None and step is not None:
            points = math.floor(float(span) / float(step)) + 1
            if points > int(self.limits["max_sweep_points"]):
                raise PolicyError(
                    f"Frequency sweep has {points} points; policy maximum is "
                    f"{self.limits['max_sweep_points']}"
                )
        if parameters.get("simulate") and parameters.get("load_data_id") is not None:
            raise PolicyError("simulate and load_data_id cannot be used together")

    def _validate_node_specific(
        self, node_id: str, parameters: dict[str, Any], state: dict[str, Any]
    ) -> None:
        qubits = parameters["qubits"]
        if node_id == "02":
            # The broadband sweep is centred on each line's LO and must stay
            # inside the resonator IF limit.
            half_span_hz = float(parameters["max_if_in_mhz"]) * 1e6
            limit = float(self.limits["resonator_if_abs_hz"])
            if half_span_hz > limit:
                raise PolicyError(
                    f"max_if_in_mhz {half_span_hz / 1e6:g} exceeds the "
                    f"±{limit / 1e6:g} MHz resonator IF limit"
                )
            points = (
                math.floor(2 * half_span_hz / (float(parameters["frequency_step_in_mhz"]) * 1e6))
                + 1
            )
            if points > int(self.limits["max_sweep_points"]):
                raise PolicyError(
                    f"Broadband sweep has {points} points; policy maximum is "
                    f"{self.limits['max_sweep_points']}"
                )
            design = parameters.get("res_design_freq")
            res_num = parameters.get("res_num")
            expected = res_num if res_num is not None else len(state.get("qubits", {}))
            if design is not None and len(design) != expected:
                raise PolicyError(
                    f"res_design_freq has {len(design)} entries but res_num is {expected}"
                )
        if node_id in {"02x", "02a", "02c", "02b"}:
            span_mhz = float(parameters["frequency_span_in_mhz"])
            resonator_rules = self.raw.get("analysis", {}).get("resonator", {})
            max_span_mhz = float(resonator_rules.get("max_frequency_span_mhz", 60.0))
            exclusion_hz = float(
                resonator_rules.get("upconverter_exclusion_hz", 1_000_000.0)
            )
            if span_mhz > max_span_mhz:
                raise PolicyError(
                    f"frequency_span_in_mhz {span_mhz:g} exceeds the resonator "
                    f"policy maximum of {max_span_mhz:g} MHz"
                )
            span_hz = span_mhz * 1e6
            limit = float(self.limits["resonator_if_abs_hz"])
            for name in qubits:
                current_if = state["qubits"][name]["resonator"]["intermediate_frequency"]
                if not is_finite_number(current_if):
                    raise PolicyError(f"{name} resonator IF is not numeric")
                if abs(float(current_if)) + span_hz / 2 > limit:
                    raise PolicyError(
                        f"{name} resonator sweep exceeds ±{limit / 1e6:g} MHz IF"
                    )
                if abs(float(current_if)) <= span_hz / 2 + exclusion_hz:
                    raise PolicyError(
                        f"{name} resonator sweep includes the readout "
                        "upconverter frequency; reduce frequency_span_in_mhz or "
                        "move resonator IF farther from the LO"
                    )

        if node_id == "02c":
            if parameters["min_power_dbm"] >= parameters["max_power_dbm"]:
                raise PolicyError("min_power_dbm must be lower than max_power_dbm")
            full_scale = self.limits["opx1000_full_scale_power_dbm"]
            if parameters["max_power_dbm"] > full_scale["max"]:
                raise PolicyError("max_power_dbm exceeds the OPX1000 policy limit")
            if parameters["min_power_dbm"] < -80:
                raise PolicyError("min_power_dbm is below the policy floor")
            # Operator rule 2026-10-05: the power window may be shifted but
            # keeps its configured width.
            window_db = (
                self.raw.get("analysis", {}).get("02c", {}).get("power_window_db")
            )
            if window_db is not None and not math.isclose(
                float(parameters["max_power_dbm"]) - float(parameters["min_power_dbm"]),
                float(window_db),
                abs_tol=1e-9,
            ):
                raise PolicyError(
                    f"02c power window must span exactly {float(window_db):g} dB "
                    "(shift min_power_dbm and max_power_dbm together)"
                )
            if parameters["max_amp"] > 1:
                raise PolicyError("max_amp cannot exceed normalized amplitude 1")
            if parameters["num_power_points"] > self.limits["max_sweep_points"]:
                raise PolicyError("num_power_points exceeds the sweep-point limit")
            resonator_02c = self.raw.get("analysis", {}).get("02c", {})
            max_power_points = int(resonator_02c.get("max_num_power_points", 30))
            max_averages = int(resonator_02c.get("max_num_averages", 100))
            if int(parameters["num_power_points"]) > max_power_points:
                raise PolicyError(
                    f"num_power_points {parameters['num_power_points']} exceeds "
                    f"the 02c policy maximum of {max_power_points}"
                )
            if int(parameters["num_averages"]) > max_averages:
                raise PolicyError(
                    f"num_averages {parameters['num_averages']} exceeds "
                    f"the 02c policy maximum of {max_averages}"
                )

        if node_id == "03a":
            span_hz = float(parameters["frequency_span_in_mhz"]) * 1e6
            limit = float(self.limits["qubit_if_abs_hz"])
            arbitrary_frequency = parameters.get("arbitrary_qubit_frequency_in_ghz")
            factor_cap = self.spectroscopy_amplitude_factor_caps()
            factor = float(parameters["operation_amplitude_factor"])
            for name in qubits:
                if factor > factor_cap[name] + 1e-12:
                    raise PolicyError(
                        f"{name} 03a operation_amplitude_factor {factor:g} exceeds "
                        f"{factor_cap[name]:g}, the cap for its "
                        f"{'shared' if factor_cap.shared(name) else 'private'} XY line"
                    )
                qubit = state["qubits"][name]
                if arbitrary_frequency is not None:
                    # 03a_Qubit_Spectroscopy.py computes
                    # sqrt(detuning / freq_vs_flux_01_quad_term) for an
                    # arbitrary center. That term is still zero until flux
                    # spectroscopy has run, and the node raises
                    # ZeroDivisionError mid-run; see the 2026-09-19 lesson.
                    quad_term = qubit.get("freq_vs_flux_01_quad_term")
                    if not is_finite_number(quad_term) or float(quad_term) == 0:
                        raise PolicyError(
                            f"{name} freq_vs_flux_01_quad_term is zero, so "
                            "arbitrary_qubit_frequency_in_ghz would divide by "
                            "zero inside the node; use "
                            "jy_request_03a_candidate_center or "
                            "jy_request_03a_window_shift to move the scan"
                        )
                current_if = qubit["xy"]["intermediate_frequency"]
                if not is_finite_number(current_if):
                    raise PolicyError(f"{name} qubit IF is not numeric")
                if abs(float(current_if)) + span_hz / 2 > limit:
                    raise PolicyError(
                        f"{name} qubit sweep exceeds ±{limit / 1e6:g} MHz IF"
                    )
                if operation_amplitude(qubit, "x180") == 0:
                    raise PolicyError(
                        f"{name} x180 amplitude is zero; approve the bootstrap proposal first"
                    )
                saturation = operation_amplitude(qubit, "saturation")
                kind = channel_kind(qubit)
                maximum = float(self.limits[kind]["max_wf_amplitude"])
                driven = abs(saturation * float(parameters["operation_amplitude_factor"]))
                if driven > maximum:
                    raise PolicyError(
                        f"{name} spectroscopy waveform amplitude {driven:g} exceeds {maximum:g}"
                    )

        sweep = self.node_definition(node_id).get("flux_sweep")
        if isinstance(sweep, dict):
            # Operator limits 2026-10-08: 02b +-0.9 V, 03b +-0.4 V, each at
            # most 100 flux points.
            low = float(parameters[sweep["min_key"]])
            high = float(parameters[sweep["max_key"]])
            points = int(parameters[sweep["points_key"]])
            limit = float(sweep["max_abs_v"])
            if low >= high:
                raise PolicyError(
                    f"{sweep['min_key']} must be lower than {sweep['max_key']}"
                )
            if max(abs(low), abs(high)) > limit + 1e-12:
                raise PolicyError(
                    f"{node_id} flux sweep {low:g}..{high:g} V exceeds the "
                    f"+-{limit:g} V policy limit"
                )
            if points < 5 or points > int(sweep["max_points"]):
                raise PolicyError(
                    f"{node_id} needs between 5 and {sweep['max_points']} flux "
                    f"points; requested {points}"
                )
        if node_id in {"03b", "06a"}:
            # These play the z `const` pulse scaled by sweep / amplitude; QUA
            # accepts amplitude_scale only inside [-2, 2).
            reach = (
                max(
                    abs(float(parameters["min_flux_offset_in_v"])),
                    abs(float(parameters["max_flux_offset_in_v"])),
                )
                if node_id == "03b"
                else float(parameters["flux_span"]) / 2
            )
            max_scale = float(
                (self.limits.get("flux") or {}).get("max_amplitude_scale", 1.99)
            )
            for name in qubits:
                const = (
                    (state["qubits"][name].get("z") or {}).get("operations") or {}
                ).get("const")
                amplitude = const.get("amplitude") if isinstance(const, dict) else None
                if not is_finite_number(amplitude) or float(amplitude) == 0:
                    raise PolicyError(
                        f"{name} has no z const pulse amplitude for the flux sweep"
                    )
                if reach / abs(float(amplitude)) > max_scale:
                    raise PolicyError(
                        f"{name} flux sweep of {reach:g} V needs amplitude_scale "
                        f"{reach / abs(float(amplitude)):g} of the {float(amplitude):g} V "
                        f"z const pulse; QUA allows at most {max_scale:g}"
                    )
        if node_id in {"02b", "03b", "06a"}:
            for name in qubits:
                if z_offset(state["qubits"][name], "joint_offset") is None:
                    raise PolicyError(
                        f"{name} has no numeric z joint_offset for flux calibration"
                    )
        if node_id == "03b":
            span_hz = float(parameters["frequency_span_in_mhz"]) * 1e6
            shift_hz = abs(float(parameters["frequency_shift_in_mhz"])) * 1e6
            limit = float(self.limits["qubit_if_abs_hz"])
            for name in qubits:
                qubit = state["qubits"][name]
                current_if = qubit["xy"]["intermediate_frequency"]
                if not is_finite_number(current_if):
                    raise PolicyError(f"{name} qubit IF is not numeric")
                if abs(float(current_if)) + shift_hz + span_hz / 2 > limit:
                    raise PolicyError(
                        f"{name} qubit sweep exceeds ±{limit / 1e6:g} MHz IF"
                    )
                saturation = operation_amplitude(qubit, "saturation")
                maximum = float(self.limits[channel_kind(qubit)]["max_wf_amplitude"])
                driven = abs(saturation * float(parameters["operation_amplitude_factor"]))
                if driven > maximum:
                    raise PolicyError(
                        f"{name} spectroscopy waveform amplitude {driven:g} exceeds {maximum:g}"
                    )
        if node_id == "06a":
            span = float(parameters["flux_span"])
            step = float(parameters["flux_step"])
            if span > 0.2 + 1e-12:
                raise PolicyError(
                    f"06a flux_span {span:g} V exceeds the +-0.1 V policy limit"
                )
            if step > span:
                raise PolicyError("06a flux_step must not exceed flux_span")
            if math.floor(span / step + 1e-9) + 1 > 101:
                raise PolicyError(
                    f"06a flux sweep has {math.floor(span / step + 1e-9) + 1} "
                    "points; the policy allows about 100"
                )
            minimum = parameters["min_wait_time_in_ns"]
            maximum = parameters["max_wait_time_in_ns"]
            wait_step = parameters["wait_time_step_in_ns"]
            if minimum < 16 or minimum >= maximum:
                raise PolicyError("06a wait-time range is invalid")
            if any(value % 4 for value in (minimum, maximum, wait_step)):
                raise PolicyError("06a wait times and step must be multiples of 4 ns")
            if (maximum - minimum) // wait_step < 8:
                raise PolicyError("06a wait sweep must contain at least 8 points")
            for name in qubits:
                if operation_amplitude(state["qubits"][name], "x180") == 0:
                    raise PolicyError(
                        f"{name} x180 amplitude is zero; calibrate it before 06a"
                    )

        if node_id == "04":
            if parameters["min_amp_factor"] >= parameters["max_amp_factor"]:
                raise PolicyError("min_amp_factor must be lower than max_amp_factor")
            points = (
                math.floor(
                    (parameters["max_amp_factor"] - parameters["min_amp_factor"])
                    / parameters["amp_factor_step"]
                )
                + 1
            )
            if points > int(self.limits["max_sweep_points"]):
                raise PolicyError("Power-Rabi amplitude sweep exceeds the point limit")
            operation = parameters["operation_x180_or_any_90"]
            for name in qubits:
                qubit = state["qubits"][name]
                amplitude = operation_amplitude(qubit, operation)
                if amplitude == 0:
                    raise PolicyError(
                        f"{name} {operation} amplitude is zero; bootstrap or calibrate it first"
                    )
                kind = channel_kind(qubit)
                maximum = float(self.limits[kind]["max_wf_amplitude"])
                swept = abs(amplitude) * float(parameters["max_amp_factor"])
                if swept > maximum:
                    raise PolicyError(
                        f"{name} sweep reaches {swept:g}, above {maximum:g}; "
                        "reduce max_amp_factor"
                    )

        if node_id in {"05", "06b"}:
            minimum = parameters["min_wait_time_in_ns"]
            maximum = parameters["max_wait_time_in_ns"]
            step = parameters["wait_time_step_in_ns"]
            if minimum < 16:
                raise PolicyError("min_wait_time_in_ns must be at least 16 ns")
            if any(value % 4 for value in (minimum, maximum, step)):
                raise PolicyError("Decay wait times and step must be multiples of 4 ns")
            if minimum >= maximum:
                raise PolicyError(
                    "min_wait_time_in_ns must be lower than max_wait_time_in_ns"
                )
            points = math.ceil((maximum - minimum) / step)
            if points < 8:
                raise PolicyError("Decay sweep must contain at least 8 points")
            if points > int(self.limits["max_sweep_points"]):
                raise PolicyError("Decay wait-time sweep exceeds the point limit")
            for name in qubits:
                qubit = state["qubits"][name]
                amplitude = operation_amplitude(qubit, "x180")
                if amplitude == 0:
                    raise PolicyError(
                        f"{name} x180 amplitude is zero; calibrate it before "
                        f"{'T1' if node_id == '05' else 'decay measurement'}"
                    )
                kind = channel_kind(qubit)
                amplitude_limit = float(
                    self.limits[kind]["max_x180_wf_amplitude"]
                )
                if abs(amplitude) > amplitude_limit:
                    raise PolicyError(
                        f"{name} x180 amplitude {amplitude:g} exceeds "
                        f"{amplitude_limit:g}"
                    )

        if node_id == "06":
            minimum = parameters["min_wait_time_in_ns"]
            maximum = parameters["max_wait_time_in_ns"]
            if minimum < 16 or minimum >= maximum:
                raise PolicyError("Ramsey wait-time range is invalid")
            if any(value % 4 for value in (minimum, maximum)):
                raise PolicyError("Ramsey wait times must be multiples of 4 ns")
            if parameters["num_time_points"] < 8:
                raise PolicyError("Ramsey sweep must contain at least 8 points")
            if parameters["num_time_points"] > int(self.limits["max_sweep_points"]):
                raise PolicyError("Ramsey wait-time sweep exceeds the point limit")

        if node_id == "07b":
            if parameters["num_runs"] > int(self.limits["max_num_averages"]):
                raise PolicyError("IQ-blob num_runs exceeds the configured policy limit")

        if node_id == "10a":
            depth = parameters["max_circuit_depth"]
            delta = parameters["delta_clifford"]
            if delta < 2 or depth < delta or depth % delta:
                raise PolicyError(
                    "Randomized-benchmarking depth must be divisible by delta_clifford >= 2"
                )
            if depth > 5000 or parameters["num_random_sequences"] > 1000:
                raise PolicyError("Randomized-benchmarking request exceeds the safety limit")

        if node_id in {"05st", "06st_t2star", "06st_t2e"}:
            minimum = parameters["min_wait_time_in_ns"]
            maximum = parameters["max_wait_time_in_ns"]
            if minimum < 16 or minimum >= maximum:
                raise PolicyError("Statistics wait-time range is invalid")
            if any(value % 4 for value in (minimum, maximum)):
                raise PolicyError("Statistics wait times must be multiples of 4 ns")
            if parameters["histo_num"] != 100:
                raise PolicyError("Statistics nodes require exactly histo_num=100")

    def validate_state_patch(
        self, patch: list[dict[str, Any]], base_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not patch:
            raise PolicyError("State patch cannot be empty")
        config = self.raw["state_commit"]
        patterns = [re.compile(item) for item in config["allowed_json_pointer_patterns"]]
        allowed_ops = set(config["allowed_operations"])
        for item in patch:
            if item.get("op") not in allowed_ops:
                raise PolicyError(f"State patch operation is not allowed: {item}")
            path = item.get("path")
            if not isinstance(path, str) or not any(p.fullmatch(path) for p in patterns):
                raise PolicyError(f"State path is not allowlisted: {path!r}")
            value = item.get("value")
            if isinstance(value, float) and not math.isfinite(value):
                raise PolicyError(f"Non-finite state value at {path}")

        original = base_state or load_state(self.settings.active_state)
        try:
            updated = apply_json_patch(original, patch)
        except Exception as exc:
            raise PolicyError(str(exc)) from exc

        for item in patch:
            path = item["path"]
            value = pointer_get(updated, path)
            if re.fullmatch(
                r"/ports/mw_outputs/[^/]+/[0-9]+/[0-9]+/upconverter_frequency",
                path,
            ):
                limits = self.limits["mw_upconverter_frequency_hz"]
                if (
                    not is_finite_number(value)
                    or not float(limits["min"]) <= float(value) <= float(limits["max"])
                ):
                    raise PolicyError(
                        f"Unsupported MW upconverter frequency at {path}"
                    )
                port = pointer_get(updated, path.rsplit("/", 1)[0])
                window = self.mw_band_lo_window(
                    port.get("band") if isinstance(port, dict) else None
                )
                if window is not None and not window[0] <= float(value) <= window[1]:
                    raise PolicyError(
                        f"MW upconverter frequency {float(value) / 1e9:g} GHz at "
                        f"{path} is outside the usable part of band "
                        f"{port['band']}: keep it between "
                        f"{window[0] / 1e9:g} GHz and {window[1] / 1e9:g} GHz."
                    )
                continue
            if re.fullmatch(
                r"/ports/mw_inputs/[^/]+/[0-9]+/[0-9]+/downconverter_frequency",
                path,
            ):
                # Readout-LO recentre (2026-10-07): the input LO must equal
                # an output LO written in the same patch on the same FEM.
                controller, fem = path.split("/")[3:5]
                paired = re.compile(
                    rf"/ports/mw_outputs/{re.escape(controller)}/{fem}/[0-9]+/upconverter_frequency"
                )
                if not is_finite_number(value) or not any(
                    paired.fullmatch(str(other.get("path"))) and other.get("value") == value
                    for other in patch
                ):
                    raise PolicyError(
                        f"{path} must equal the readout output LO written in "
                        "the same patch"
                    )
                continue
            if re.fullmatch(
                r"/ports/mw_outputs/[^/]+/[0-9]+/[0-9]+/full_scale_power_dbm",
                path,
            ):
                limits = self.limits["opx1000_full_scale_power_dbm"]
                if not isinstance(value, int) or not limits["min"] <= value <= limits["max"]:
                    raise PolicyError(f"Unsupported full-scale power at {path}")
                continue
            parts = path.split("/")
            if len(parts) < 4:
                continue
            name = parts[2]
            qubit = updated["qubits"][name]
            if path.endswith("/intermediate_frequency"):
                limit = (
                    self.limits["qubit_if_abs_hz"]
                    if "/xy/" in path
                    else self.limits["resonator_if_abs_hz"]
                )
                if not is_finite_number(value) or abs(float(value)) > float(limit):
                    raise PolicyError(f"IF update exceeds the limit at {path}")
            elif path.endswith("/amplitude"):
                if not is_finite_number(value):
                    raise PolicyError(f"Amplitude must be numeric at {path}")
                kind = channel_kind(qubit)
                maximum = float(self.limits[kind]["max_wf_amplitude"])
                backing = parts[-2]
                try:
                    x180_backing = operation_backing_name(qubit, "x180")
                except Exception:
                    x180_backing = ""
                if backing == x180_backing:
                    maximum = float(self.limits[kind]["max_x180_wf_amplitude"])
                if abs(float(value)) > maximum:
                    raise PolicyError(
                        f"Amplitude {value} exceeds {maximum} at {path}"
                    )
            elif path.endswith("/full_scale_power_dbm"):
                limits = self.limits["opx1000_full_scale_power_dbm"]
                if not isinstance(value, int) or not limits["min"] <= value <= limits["max"]:
                    raise PolicyError(f"Unsupported full-scale power at {path}")
            elif path.endswith(("/threshold", "/rus_exit_threshold")):
                if not is_finite_number(value):
                    raise PolicyError(f"Readout threshold must be numeric at {path}")
            elif path.endswith("/confusion_matrix"):
                if (
                    not isinstance(value, list)
                    or len(value) != 2
                    or any(not isinstance(row, list) or len(row) != 2 for row in value)
                    or any(
                        not is_finite_number(item) or not 0 <= float(item) <= 1
                        for row in value
                        for item in row
                    )
                ):
                    raise PolicyError(f"Confusion matrix must be a finite 2x2 probability matrix at {path}")
            elif path.endswith("/extras/bare_resonator_freq"):
                if not is_finite_number(value) or float(value) <= 0:
                    raise PolicyError(
                        f"Bare resonator frequency must be finite and positive at {path}"
                    )
            elif path.endswith(("/T1", "/T2ramsey", "/T2echo")) or re.search(
                r"/extras/(T1|T1_dev|T2ramsey|T2ramsey_dev|T2|T2_dev)$", path
            ):
                if not is_finite_number(value) or float(value) <= 0:
                    raise PolicyError(f"Coherence time must be finite and positive at {path}")
            elif path.endswith("/extras/readout_fidelity"):
                if not is_finite_number(value) or not 0 <= float(value) <= 1:
                    raise PolicyError(f"Readout fidelity must be between 0 and 1 at {path}")
            elif re.search(r"/z/(joint_offset|independent_offset|min_offset)$", path):
                limit = float((self.limits.get("flux") or {}).get("max_offset_abs_v", 0.9))
                if not is_finite_number(value) or abs(float(value)) > limit:
                    raise PolicyError(
                        f"Flux offset must be a number within +-{limit:g} V at {path}"
                    )
            elif path.endswith(("/phi0_voltage", "/phi0_current")):
                if not is_finite_number(value) or float(value) <= 0:
                    raise PolicyError(f"Flux period must be finite and positive at {path}")
            elif path.endswith("/freq_vs_flux_01_quad_term"):
                if not is_finite_number(value):
                    raise PolicyError(f"Quadratic flux term must be numeric at {path}")
            elif re.search(r"/extras/(sweetspot_freq|FluxPeriodV)$", path):
                # 0.0 is the placeholder the extras prerequisite writes.
                if not is_finite_number(value) or float(value) < 0:
                    raise PolicyError(f"{path} must be finite and non-negative")
            # Fresh-qubit defaults (operator 2026-10-14).
            elif re.fullmatch(r"/qubits/q[0-9]+/extras", path):
                if value != {}:
                    raise PolicyError(f"{path} may only be emptied")
            elif path.endswith("/anharmonicity"):
                if not is_finite_number(value) or float(value) <= 0:
                    raise PolicyError(f"Anharmonicity must be finite and positive at {path}")
            elif path.endswith(("/alpha", "/detuning")):
                if not is_finite_number(value):
                    raise PolicyError(f"DRAG term must be numeric at {path}")
        return updated
