"""
執行單一或多個 Qualibration 實驗 .py 檔。

單一實驗:
    python Script/_run_file.py <實驗.py> [key=value ...]

批次（整段序列只啟動一次 Python，不掃描整個 calibration_graph）:
    python Script/_run_file.py --batch < experiments.json

參數覆寫時僅 inspect 該檔案以取得 Parameters 定義，再 node.run(interactive=False)。
無參數覆寫時直接 subprocess 執行 .py，完全不經過 qualibrate 掃描。
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SUPERCONDUCTING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SUPERCONDUCTING_DIR))

from qualibrate import QualibrationNode  # noqa: E402
from qualibrate.models.run_mode import RunModes  # noqa: E402
from qualibrate.q_runnnable import run_modes_ctx  # noqa: E402


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


def _run_direct(script_path: Path) -> None:
    """直接執行 .py，與 IDE 手動 Run 相同，零 qualibrate 掃描。"""
    print(f">>> 直接執行: {script_path.relative_to(SUPERCONDUCTING_DIR)}")
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

    # interactive=False 才會真的寫入 machine / state.json（見前次說明）
    node.copy(**merged).run(interactive=False)


def run_experiment(script_arg: str | Path, param_strings: list[str] | None = None) -> None:
    script_path = _resolve_script(script_arg)
    overrides = _parse_overrides(param_strings or [])

    if overrides:
        _run_with_overrides(script_path, overrides)
    else:
        _run_direct(script_path)


def run_batch(experiments: list[dict[str, Any]], skip_failed: bool = False) -> None:
    total = len(experiments)
    for i, item in enumerate(experiments, 1):
        label = item.get("label", item["path"])
        params = item.get("params") or []
        print(f"\n{'=' * 60}")
        print(f"[{i}/{total}] {label}")
        print("=" * 60)
        try:
            run_experiment(item["path"], params)
        except subprocess.CalledProcessError as exc:
            print(f"!!! 失敗 (exit {exc.returncode}): {label}")
            if not skip_failed:
                raise
        except Exception as exc:
            print(f"!!! 失敗: {label} — {exc!r}")
            if not skip_failed:
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description="執行 Qualibration 實驗 .py")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="從 stdin 讀取 JSON 批次執行（整段序列只啟動一次 Python）",
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
