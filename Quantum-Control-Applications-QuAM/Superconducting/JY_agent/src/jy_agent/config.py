from __future__ import annotations

import ipaddress
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class Settings:
    agent_root: Path
    superconducting_root: Path
    calibration_graph: Path
    data_root: Path
    quam_state_root: Path
    active_state: Path
    wiring_path: Path
    qualibrate_config_path: Path
    qualibrate_project: str
    runtime: Path
    policy_path: Path
    playbook_path: Path
    host: str
    port: int
    mcp_path: str
    qualibrate_python: Path
    workflow_sequence: tuple[str, ...]
    require_explicit_qubits: bool
    measurement_mode_entry_phrase: str
    measurement_mode_exit_phrase: str
    measurement_mode_shutdown_phrase: str = "結束量測"
    measurement_mode_shutdown_phrases: tuple[str, ...] = (
        "結束量測",
        "結束量測模式",
        "結束 JY 量測模式",
        "結束JY量測模式",
        "結束 JY 自動量測模式",
        "結束JY自動量測模式",
        "結束 JY 自動量測",
        "結束JY自動量測",
        "退出 JY 量測模式",
        "退出JY量測模式",
        "退出 JY 自動量測模式",
        "退出JY自動量測模式",
        "退出 JY 自動量測",
        "退出JY自動量測",
        "停止量測",
        "停止量測模式",
        "停止 JY 量測模式",
        "停止JY量測模式",
        "停止 JY 自動量測模式",
        "停止JY自動量測模式",
        "停止 JY 自動量測",
        "停止JY自動量測",
    )
    autonomy_mode_entry_phrase: str = "進入 JY 自動量測模式"
    measurement_mode_entry_phrases: tuple[str, ...] = (
        "進入 JY 量測模式",
        "進入JY量測模式",
        "Enter JY measurement mode",
    )
    autonomy_mode_entry_phrases: tuple[str, ...] = (
        "進入 JY 自動量測模式",
        "進入JY自動量測模式",
        "Enter JY automatic measurement mode",
    )
    recovery_phrases: tuple[str, ...] = (
        "恢復",
        "恢复",
        "Recover",
    )
    measurement_resume_phrases: tuple[str, ...] = (
        "恢復量測",
        "恢复量测",
        "Resume measurement",
    )
    approval_transport: str = "local"
    approval_host: str = "127.0.0.1"
    approval_port: int = 8766
    approval_public_base_url: str | None = None
    approval_trusted_origins: tuple[str, ...] = ()
    approval_trusted_proxy_cidrs: tuple[str, ...] = (
        "127.0.0.0/8",
        "::1/128",
    )
    approval_access_token: str | None = None
    # Operator instruction 2026-10-14: a qubit this chip never measured gets
    # the fresh-qubit defaults before its first run. Unit tests that are not
    # about that rule switch it off.
    fresh_qubit_defaults: bool = True

    @classmethod
    def load(cls, agent_root: Path | None = None) -> "Settings":
        root = (agent_root or Path(__file__).resolve().parents[2]).resolve()
        raw = _load_yaml(root / "config" / "agent.yaml")
        qualibrate_config_path = (Path.home() / ".qualibrate" / "config.toml").resolve()
        qualibrate_raw = _load_toml(qualibrate_config_path)

        quam = _mapping(qualibrate_raw, "quam")
        qualibrate = _mapping(qualibrate_raw, "qualibrate")
        storage = _mapping(qualibrate, "storage")
        calibration_library = _mapping(qualibrate, "calibration_library")
        if storage.get("type", "local_storage") != "local_storage":
            raise ValueError("JY requires qualibrate.storage.type='local_storage'")

        quam_state_root = _external_path(
            _required_string(quam, "state_path"),
            qualibrate_config_path.parent,
        )
        data_root = _external_path(
            _required_string(storage, "location"),
            qualibrate_config_path.parent,
        )
        calibration_graph = _external_path(
            _required_string(calibration_library, "folder"),
            qualibrate_config_path.parent,
        )

        path_cfg = _mapping(raw, "paths")

        def agent_path(value: str) -> Path:
            path = Path(value).expanduser()
            return (path if path.is_absolute() else root / path).resolve()

        python_value = (
            os.environ.get("JY_QUALIBRATE_PYTHON")
            or raw.get("qualibrate_python")
            or sys.executable
        )
        server = _mapping(raw, "server")
        approval = _mapping(raw, "approval")
        workflow = _mapping(raw, "workflow")
        interaction = _mapping(raw, "interaction")
        approval_transport = str(
            os.environ.get("JY_APPROVAL_TRANSPORT")
            or approval.get("transport", "local")
        ).strip().casefold()
        if approval_transport not in {"local", "public"}:
            raise ValueError("approval.transport must be local or public")
        public_base_url = _optional_origin(
            os.environ.get("JY_APPROVAL_PUBLIC_BASE_URL")
            or approval.get("public_base_url"),
        )
        configured_origins = _string_tuple(
            approval.get("trusted_origins", []), "approval.trusted_origins"
        )
        trusted_origins = tuple(
            _required_origin(value)
            for value in configured_origins
        )
        approval_host = _approval_host(
            str(
                os.environ.get("JY_APPROVAL_HOST")
                or approval.get("host", "127.0.0.1")
            ),
        )
        trusted_proxy_override = os.environ.get(
            "JY_APPROVAL_TRUSTED_PROXY_CIDRS"
        )
        trusted_proxy_cidrs = _cidr_tuple(
            [
                item.strip()
                for item in trusted_proxy_override.split(",")
                if item.strip()
            ]
            if trusted_proxy_override is not None
            else approval.get(
                "trusted_proxy_cidrs", ["127.0.0.0/8", "::1/128"]
            ),
            "approval.trusted_proxy_cidrs",
        )
        access_token = (
            os.environ.get("JY_APPROVAL_ACCESS_TOKEN")
            or approval.get("access_token")
            or None
        )
        if access_token is not None:
            access_token = str(access_token).strip() or None
        approval_port = int(
            os.environ.get("JY_APPROVAL_PORT") or approval.get("port", 8766)
        )
        if not 1 <= approval_port <= 65535:
            raise ValueError("approval.port must be between 1 and 65535")
        measurement_mode_entry_phrase = str(
            interaction["measurement_mode_entry_phrase"]
        )
        measurement_mode_exit_phrase = str(
            interaction["measurement_mode_exit_phrase"]
        )
        measurement_mode_shutdown_phrase = str(
            interaction.get("measurement_mode_shutdown_phrase", "結束量測")
        )
        measurement_entry_phrases = _string_tuple(
            interaction.get(
                "measurement_mode_entry_phrases",
                [measurement_mode_entry_phrase],
            ),
            "interaction.measurement_mode_entry_phrases",
        )
        autonomy_mode_entry_phrase = str(
            interaction.get(
                "autonomy_mode_entry_phrase", "進入 JY 自動量測模式"
            )
        )
        autonomy_entry_phrases = _string_tuple(
            interaction.get(
                "autonomy_mode_entry_phrases",
                [autonomy_mode_entry_phrase],
            ),
            "interaction.autonomy_mode_entry_phrases",
        )
        recovery_phrases = _string_tuple(
            interaction.get("recovery_phrases", ["恢復", "Recover"]),
            "interaction.recovery_phrases",
        )
        measurement_resume_phrases = _string_tuple(
            interaction.get(
                "measurement_resume_phrases",
                ["恢復量測", "恢复量测", "Resume measurement"],
            ),
            "interaction.measurement_resume_phrases",
        )
        shutdown_phrases = _string_tuple(
            interaction.get(
                "measurement_mode_shutdown_phrases",
                [measurement_mode_shutdown_phrase, measurement_mode_exit_phrase],
            ),
            "interaction.measurement_mode_shutdown_phrases",
        )
        if measurement_mode_exit_phrase not in shutdown_phrases:
            raise ValueError(
                "measurement_mode_exit_phrase must be included in "
                "measurement_mode_shutdown_phrases"
            )
        if measurement_mode_shutdown_phrase not in shutdown_phrases:
            raise ValueError(
                "measurement_mode_shutdown_phrase must be included in "
                "measurement_mode_shutdown_phrases"
            )
        if measurement_mode_entry_phrase not in measurement_entry_phrases:
            raise ValueError(
                "measurement_mode_entry_phrase must be included in "
                "measurement_mode_entry_phrases"
            )
        if autonomy_mode_entry_phrase not in autonomy_entry_phrases:
            raise ValueError(
                "autonomy_mode_entry_phrase must be included in "
                "autonomy_mode_entry_phrases"
            )
        settings = cls(
            agent_root=root,
            superconducting_root=calibration_graph.parent,
            calibration_graph=calibration_graph,
            data_root=data_root,
            quam_state_root=quam_state_root,
            active_state=quam_state_root / "state.json",
            wiring_path=quam_state_root / "wiring.json",
            qualibrate_config_path=qualibrate_config_path,
            qualibrate_project=_required_string(qualibrate, "project"),
            runtime=agent_path(str(path_cfg.get("runtime", "runtime"))),
            policy_path=root / "rules" / "policies.yaml",
            playbook_path=root / "rules" / "PLAYBOOK.md",
            host=str(server["host"]),
            port=int(server["port"]),
            mcp_path=str(server["path"]),
            qualibrate_python=Path(python_value).expanduser().resolve(),
            workflow_sequence=tuple(workflow["sequence"]),
            require_explicit_qubits=bool(workflow["require_explicit_qubits"]),
            measurement_mode_entry_phrase=measurement_mode_entry_phrase,
            measurement_mode_exit_phrase=measurement_mode_exit_phrase,
            measurement_mode_shutdown_phrase=measurement_mode_shutdown_phrase,
            measurement_mode_shutdown_phrases=shutdown_phrases,
            autonomy_mode_entry_phrase=autonomy_mode_entry_phrase,
            measurement_mode_entry_phrases=measurement_entry_phrases,
            autonomy_mode_entry_phrases=autonomy_entry_phrases,
            recovery_phrases=recovery_phrases,
            measurement_resume_phrases=measurement_resume_phrases,
            approval_transport=approval_transport,
            approval_host=approval_host,
            approval_port=approval_port,
            approval_public_base_url=public_base_url,
            approval_trusted_origins=trusted_origins,
            approval_trusted_proxy_cidrs=trusted_proxy_cidrs,
            approval_access_token=access_token,
        )
        settings.validate_approval_transport()
        settings.validate_external_paths()
        return settings

    def validate_approval_transport(self) -> None:
        _approval_host(self.approval_host)
        if self.approval_transport == "public":
            if not self.approval_public_base_url:
                raise ValueError(
                    "Public approval requires approval.public_base_url"
                )
            if not self.approval_access_token or len(self.approval_access_token) < 32:
                raise ValueError(
                    "Public approval requires a random access token of at least 32 characters"
                )

    def validate_external_paths(self) -> None:
        files = {
            "Qualibrate config": self.qualibrate_config_path,
            "QuAM state.json": self.active_state,
            "QuAM wiring.json": self.wiring_path,
            "Qualibrate Python": self.qualibrate_python,
        }
        directories = {
            "QuAM state directory": self.quam_state_root,
            "Qualibrate storage": self.data_root,
            "Qualibrate calibration library": self.calibration_graph,
        }
        missing = [
            f"{label}: {path}"
            for label, path in files.items()
            if not path.is_file()
        ]
        missing.extend(
            f"{label}: {path}"
            for label, path in directories.items()
            if not path.is_dir()
        )
        if missing:
            raise FileNotFoundError(
                "Qualibrate environment validation failed:\n- "
                + "\n- ".join(missing)
            )

    def ensure_runtime(self) -> None:
        self.runtime.mkdir(parents=True, exist_ok=True)
        (self.runtime / "requests").mkdir(exist_ok=True)
        (self.runtime / "logs").mkdir(exist_ok=True)
        (self.runtime / "backups").mkdir(exist_ok=True)
        (self.runtime / "report_assets").mkdir(exist_ok=True)

    @property
    def database_path(self) -> Path:
        return self.runtime / "jy_agent.sqlite3"

    @property
    def lock_path(self) -> Path:
        return self.runtime / "hardware.lock"


def _external_path(value: str, config_parent: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_parent / path).resolve()


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a [{key}] mapping in configuration")
    return result


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"Configuration value {key!r} must be a non-empty string")
    return result


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Configuration value {label!r} must be a list of strings")
    return tuple(item.strip() for item in value if item.strip())


def _optional_origin(value: Any) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError("approval.public_base_url must be a URL string")
    return _required_origin(value)


def _required_origin(value: str) -> str:
    candidate = value.strip().rstrip("/")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Approval origin must be an absolute HTTP(S) URL: {value}")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(f"Approval origin must not contain a path/query/fragment: {value}")
    if parsed.scheme != "https":
        hostname = parsed.hostname or ""
        is_loopback = hostname in {"127.0.0.1", "localhost", "::1"}
        if not is_loopback:
            raise ValueError(
                "Remote approval origins must use HTTPS"
            )
    return candidate


def _approval_host(value: str) -> str:
    host = value.strip()
    try:
        address = ipaddress.ip_address(host)
        is_loopback = address.is_loopback
    except ValueError:
        is_loopback = host.casefold() == "localhost"
    if not is_loopback:
        raise ValueError("approval.host must remain loopback")
    return host


def _cidr_tuple(
    value: Any, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    entries = _string_tuple(value, label)
    for entry in entries:
        try:
            ipaddress.ip_network(entry)
        except ValueError as exc:
            raise ValueError(f"Invalid CIDR in {label}: {entry}") from exc
    if not entries and not allow_empty:
        raise ValueError(f"Configuration value {label!r} must not be empty")
    return entries


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Qualibrate config not found: {path}")
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a TOML mapping in {path}")
    return value


def load_policies(settings: Settings) -> dict[str, Any]:
    return _load_yaml(settings.policy_path)
