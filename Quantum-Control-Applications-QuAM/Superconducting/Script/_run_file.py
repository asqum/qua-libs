"""
執行單一或多個 Qualibration 實驗 .py 檔。

單一實驗:
    python Script/_run_file.py <實驗.py> [key=value ...]

批次（整段序列只啟動一次 Python，不掃描整個 calibration_graph）:
    python Script/_run_file.py --batch --batch-file queue.json

參數覆寫時僅 inspect 該檔案以取得 Parameters 定義，再 node.run(interactive=False)。
無參數覆寫時直接 subprocess 執行 .py，完全不經過 qualibrate 掃描。
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from time import sleep, time
from typing import Any, Generator

SCRIPT_DIR = Path(__file__).resolve().parent
SUPERCONDUCTING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SUPERCONDUCTING_DIR))

from qualibrate import QualibrationNode  # noqa: E402
from qualibrate.models.run_mode import RunModes  # noqa: E402
from qualibrate.q_runnnable import run_modes_ctx  # noqa: E402

# --- 批次中斷時追蹤 QOP / 子行程（僅 --batch 啟用）---
_active_qm: Any = None
_active_subprocess: subprocess.Popen[bytes] | None = None
_interrupt_stop_batch: bool = False
_batch_hooks_installed: bool = False
_original_qm_session: Any = None

_MSG_BUSY = (
    "A quantum machine cannot be opened because an existing quantum machine, using the same ports, is currently "
    "running a program. Please close the currently open quantum machine."
)
_MSG_BUSY_OPX1000 = "Resources already locked"


def _parse_value(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _parse_overrides(argv: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in argv:
        if "=" not in item:
            raise ValueError(f"參數格式錯誤（需 key=value）: {item}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = _parse_value(value.strip())
    return overrides


def _resolve_script(script_arg: str | Path) -> Path:
    script_path = Path(script_arg)
    if not script_path.is_absolute():
        script_path = (SUPERCONDUCTING_DIR / script_path).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"找不到實驗檔案: {script_path}")
    return script_path


def _inspect_node(script_path: Path) -> QualibrationNode:
    """只 inspect 單一檔案以取得 Parameters 定義（不掃描整個 calibration_graph）。"""
    nodes: dict[str, QualibrationNode] = {}
    token = run_modes_ctx.set(RunModes(inspection=True))
    try:
        QualibrationNode.scan_node_file(script_path, nodes)
    finally:
        run_modes_ctx.reset(token)

    if not nodes:
        raise RuntimeError(
            f"無法從檔案載入 QualibrationNode: {script_path}\n"
            f"請確認檔案內有 QualibrationNode(...) 定義。"
        )
    return next(iter(nodes.values()))


def _merge_parameters(node: QualibrationNode, overrides: dict[str, Any]) -> dict[str, Any]:
    return {**node.parameters.model_dump(), **overrides}


def _halt_and_close_qm(qm: Any) -> None:
    try:
        from qm import QopCaps

        qmm = getattr(qm, "manager", None)
        if qmm is not None and qmm.capabilities.supports(QopCaps.qop3):
            for job in qm.get_jobs(status=["Running"]):
                if callable(getattr(job, "halt", None)):
                    job.halt()
        else:
            job = qm.get_running_job()
            if job is not None and callable(getattr(job, "halt", None)):
                job.halt()
    except Exception:
        pass
    try:
        qm.close()
    except Exception:
        pass


def _cleanup_hardware() -> None:
    global _active_qm, _active_subprocess

    if QualibrationNode.active_node is not None:
        try:
            QualibrationNode.active_node.stop()
        except Exception:
            pass

    if _active_qm is not None:
        _halt_and_close_qm(_active_qm)
        _active_qm = None

    if _active_subprocess is not None and _active_subprocess.poll() is None:
        _active_subprocess.terminate()
        try:
            _active_subprocess.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _active_subprocess.kill()
        _active_subprocess = None


def _signal_handler(signum: int, frame: Any) -> None:
    global _interrupt_stop_batch
    _interrupt_stop_batch = True
    print(f"\n>>> 收到中斷訊號 ({signum})，正在停止 QOP…", flush=True)
    _cleanup_hardware()
    raise KeyboardInterrupt


def _install_batch_shutdown_hooks() -> None:
    global _batch_hooks_installed, _original_qm_session
    if _batch_hooks_installed:
        return

    import qualang_tools.multi_user.multi_user_tools as mut

    _original_qm_session = mut.qm_session
    mut.qm_session = _qm_session_with_interrupt_flag
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_handler)
    _batch_hooks_installed = True


@contextmanager
def _qm_session_with_interrupt_flag(
    qmm: Any, config: dict, timeout: int = 100
) -> Generator[Any, None, None]:
    """與 qualang_tools.qm_session 相同，但量測中 Ctrl+C 會標記整批結束。"""
    global _active_qm, _interrupt_stop_batch

    from qualang_tools.multi_user.multi_user_tools import BusyFilter
    from qm import QopCaps
    from qm.logging_utils import set_logging_level

    if not timeout > 0:
        raise ValueError(f"{timeout=} must be positive")

    qm_log = logging.getLogger("qm.api.frontend_api")
    filt = BusyFilter()
    is_busy = True
    printed = False
    t0 = time()
    elapsed_time = 0
    set_logging_level("ERROR")
    qm: Any = None
    while is_busy and elapsed_time < timeout:
        try:
            qm_log.addFilter(filt)
            qm = qmm.open_qm(config, close_other_machines=False)
        except Exception as e:
            if (qmm.capabilities.supports(QopCaps.qop3) and _MSG_BUSY_OPX1000 in e.errors[0][2]) or (
                ~qmm.capabilities.supports(QopCaps.qop3) and _MSG_BUSY in e.errors[1][2]
            ):
                if not printed:
                    qm_log.error(f"QOP is busy. Waiting for it to free up for {timeout}s...")
                    printed = True
                sleep(0.2)
                elapsed_time = time() - t0
            else:
                raise Exception from e
        else:
            is_busy = False
            set_logging_level("INFO")
            qm_log.info("Opening QM")
        finally:
            qm_log.removeFilter(filt)

    if is_busy and elapsed_time >= timeout:
        qm_log.warning(f"While waiting for QOP to free, reached timeout: {timeout}s")
        raise TimeoutError(f"While waiting for QOP to free, reached timeout: {timeout}s")

    _active_qm = qm
    try:
        try:
            yield qm

            if qmm.capabilities.supports(QopCaps.qop3):
                while qm.get_jobs(status=["Running"]):
                    sleep(0.2)
            else:
                while qm.get_running_job() is not None:
                    sleep(0.2)
        except KeyboardInterrupt:
            _interrupt_stop_batch = True
    finally:
        qm_log.info("Closing QM")
        _halt_and_close_qm(qm)
        _active_qm = None


def _run_direct(script_path: Path) -> None:
    """直接執行 .py，與 IDE 手動 Run 相同，零 qualibrate 掃描。"""
    global _active_subprocess

    print(f">>> 直接執行: {script_path.relative_to(SUPERCONDUCTING_DIR)}")
    if _batch_hooks_installed:
        _active_subprocess = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=SUPERCONDUCTING_DIR,
        )
        try:
            returncode = _active_subprocess.wait()
        finally:
            _active_subprocess = None
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, [sys.executable, str(script_path)])
        return

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=SUPERCONDUCTING_DIR,
        check=True,
    )


def _run_with_overrides(script_path: Path, overrides: dict[str, Any]) -> None:
    """有參數覆寫時：inspect 單檔 → 合併參數 → run(interactive=False)。"""
    node = _inspect_node(script_path)
    merged = _merge_parameters(node, overrides)

    print(f">>> 執行: {script_path.relative_to(SUPERCONDUCTING_DIR)}")
    if overrides:
        print(f"    參數覆寫: {overrides}")

    node.copy(**merged).run(interactive=False)


def run_experiment(script_arg: str | Path, param_strings: list[str] | None = None) -> None:
    script_path = _resolve_script(script_arg)
    overrides = _parse_overrides(param_strings or [])

    if overrides:
        _run_with_overrides(script_path, overrides)
    else:
        _run_direct(script_path)


def run_batch(experiments: list[dict[str, Any]], skip_failed: bool = False) -> None:
    global _interrupt_stop_batch

    _install_batch_shutdown_hooks()
    total = len(experiments)
    try:
        for i, item in enumerate(experiments, 1):
            label = item.get("label", item["path"])
            params = item.get("params") or []
            print(f"\n{'=' * 60}")
            print(f"[{i}/{total}] {label}")
            print("=" * 60)
            try:
                run_experiment(item["path"], params)
            except KeyboardInterrupt:
                print("\n>>> 使用者中斷，結束批次序列。")
                raise SystemExit(130) from None
            except subprocess.CalledProcessError as exc:
                print(f"!!! 失敗 (exit {exc.returncode}): {label}")
                if not skip_failed:
                    raise
            except Exception as exc:
                print(f"!!! 失敗: {label} — {exc!r}")
                if not skip_failed:
                    raise

            if _interrupt_stop_batch:
                print("\n>>> 量測已中斷；當前節點若已完成分析則已存檔，結束批次序列。")
                raise SystemExit(130)
    finally:
        _cleanup_hardware()


def main() -> None:
    parser = argparse.ArgumentParser(description="執行 Qualibration 實驗 .py")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批次執行（JSON 佇列；建議搭配 --batch-file）",
    )
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=None,
        help="批次佇列 JSON 檔路徑（避免 PowerShell pipe 導致 Ctrl+C 無法優雅關閉 QM）",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="批次模式下單一實驗失敗時繼續",
    )
    parser.add_argument("script", nargs="?", help="實驗 .py 路徑")
    parser.add_argument("params", nargs="*", help="key=value 參數覆寫")
    args = parser.parse_args()

    if args.batch:
        if args.batch_file is not None:
            payload = json.loads(args.batch_file.read_text(encoding="utf-8-sig"))
        else:
            payload = json.load(sys.stdin)
        if isinstance(payload, dict):
            payload = [payload]
        run_batch(payload, skip_failed=args.skip_failed)
        return

    if not args.script:
        parser.print_help()
        raise SystemExit(1)

    run_experiment(args.script, args.params)


if __name__ == "__main__":
    main()
