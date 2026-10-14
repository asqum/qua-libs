"""Chip history on the Experiment results page (operator request 2026-10-13)."""

from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from PIL import Image

from jy_agent.chip_history import (
    chip_history_asset,
    collect_chip_history,
    render_chip_history,
    write_static_chip_history,
)

from test_next_action_planner import analysis_document
from test_per_qubit_pass_rules import build, insert_run


def plot(folder: Path, name: str) -> str:
    path = folder / name
    Image.new("RGB", (8, 8), (40, 80, 120)).save(path)
    return str(path)


def passing(node_id: str, names: list[str], plots: list[str], snapshot: int) -> dict:
    document = analysis_document(node_id, {name: 999.0 for name in names})
    document["plots"] = plots
    document["snapshot_id"] = snapshot
    return document


class ChipHistoryTests(unittest.TestCase):
    def test_successes_by_node_and_qubit_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            wid = workflow["id"]
            data = service.settings.data_root
            shared = plot(data, "02x_all.png")
            insert_run(service, wid, "02x", "q1", passing("02x", ["q1", "q2"], [shared], 1))
            service.db.execute(
                "UPDATE runs SET parameters_json = '{\"qubits\": [\"q1\", \"q2\"]}' "
                "WHERE node_id = '02x'"
            )
            old = insert_run(service, wid, "03a", "q1", passing("03a", ["q1"], [plot(data, "old.png")], 2))
            new = insert_run(service, wid, "03a", "q1", passing("03a", ["q1"], [plot(data, "new.png")], 3))
            failed = analysis_document("04", {"q1": 2.0})
            insert_run(service, wid, "04", "q1", failed)
            inherited = insert_run(service, wid, "05", "q1", passing("05", ["q1"], [], 4))
            service.db.execute(
                "UPDATE runs SET inherited_from_run_id = 'x' WHERE id = ?", (inherited,)
            )

            history = collect_chip_history(service, str(workflow["device_key"]))
            self.assertEqual([node["node"] for node in history["nodes"]], ["02x", "03a"])
            x02 = history["nodes"][0]
            self.assertEqual([item["qubit"] for item in x02["qubits"]], ["q1", "q2"])
            self.assertEqual(x02["shared"][0]["qubits"], ["q1", "q2"])
            results = history["nodes"][1]["qubits"][0]["results"]
            self.assertEqual([item["run_id"] for item in results], [new, old])

            html = render_chip_history(history, lambda run, pos, thumb: f"/img/{run}/{pos}")
            self.assertIn("Snapshot", render_chip_history(history, lambda r, p, t: "", "en"))
            self.assertLess(html.index(">02x<"), html.index(">03a<"))

            # Assets: only this chip's runs, only plots under the data root.
            key = str(workflow["device_key"])
            self.assertEqual(chip_history_asset(service, key, new, 0).name, "new.png")
            with self.assertRaises(ValueError):
                chip_history_asset(service, "another chip", new, 0)
            with self.assertRaises(ValueError):
                chip_history_asset(service, key, uuid.uuid4().hex, 0)
            with self.assertRaises(ValueError):
                chip_history_asset(service, key, new, 5)

            page = write_static_chip_history(service, str(workflow["device_key"]))
            text = page.read_text(encoding="utf-8")
            self.assertIn("images/", text)
            self.assertTrue(any((page.parent / "images").iterdir()))

    def test_a_chip_without_success_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            service, workflow, _lease = build(folder, "03a")
            history = collect_chip_history(service, str(workflow["device_key"]))
            self.assertEqual(history["nodes"], [])
            self.assertIn("No successful result", render_chip_history(history, lambda r, p, t: "", "en"))


if __name__ == "__main__":
    unittest.main()
