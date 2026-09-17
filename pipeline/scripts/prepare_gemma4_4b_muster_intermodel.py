#!/usr/bin/env python3
"""Prepare intermodel-only manifests/config for the supplied Musterloesungen."""

from __future__ import annotations

import copy
import json
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
MUSTER_ROOT = PROJECT_ROOT / "Musterloesungen_ohne_Intermodel_RS"
OUTPUT_ROOT = (
    PIPELINE_ROOT / "Evaluation" / "inter" / "gemma4" / "4b" / "musterloesungen"
)

SCENARIOS = (
    "controlled_phase1", "controlled_phase2", "controlled_phase3",
    "controlled_phase4", "organized_phase1", "organized_phase2",
    "organized_phase3", "organized_phase4",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_manifest(scenario: str) -> Path:
    phase = scenario.removeprefix("controlled_").removeprefix("organized_")
    number = phase.removeprefix("phase")
    source_manifest = (
        MUSTER_ROOT
        / "intermodel_runs"
        / f"{scenario}_run"
        / "output_model_gen"
        / "artifacts.json"
    )
    data = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_slm_root = MUSTER_ROOT / f"S{number}"
    input_adl = MUSTER_ROOT / f"Controlled_S{number}_WITHOUT_INTERMODEL.adl"

    artifacts = []
    for item in data["artifacts"]:
        prepared = copy.deepcopy(item)
        prepared["source_scenario_file"] = str(
            (PIPELINE_ROOT / item["source_scenario_file"]).resolve()
        )
        prepared["slm_file"] = str(
            (source_slm_root / Path(item["slm_file"]).name).resolve()
        )
        prepared["adl_file"] = str(input_adl.resolve())
        prepared["status"] = "EXISTING"
        prepared["task_id"] = (
            f"muster:{scenario}:{item['model_id']}:{item['adl_model_name']}"
        )
        artifacts.append(prepared)

    manifest = {
        "format": "4em-pipeline-artifact-manifest-v1",
        "scenario_name": scenario,
        "scenario_directory": str((PIPELINE_ROOT / "scenarios" / scenario).resolve()),
        "run_directory": str((OUTPUT_ROOT / scenario).resolve()),
        "source": str(input_adl.resolve()),
        "note": "Prepared from Musterloesungen_ohne_Intermodel_RS for Gemma 4 4B.",
        "artifacts": artifacts,
    }
    target = OUTPUT_ROOT / scenario / "input" / "artifacts.json"
    write_json(target, manifest)
    return target


def main() -> int:
    base_path = PIPELINE_ROOT / "parameter.json"
    config = json.loads(base_path.read_text(encoding="utf-8"))
    config["execution"]["mode"] = "intermodel_only"
    config["execution"]["selected_scenario"] = "controlled_phase1"
    config["execution"]["overwrite"] = True
    config["slm"]["model"] = "gemma4:e4b"

    jobs = []
    for scenario in SCENARIOS:
        number = scenario.rsplit("phase", 1)[1]
        manifest = prepare_manifest(scenario)
        run_directory = OUTPUT_ROOT / scenario
        jobs.append({
            "enabled": True,
            "scenario_name": scenario,
            "scenario_directory": str(
                (PIPELINE_ROOT / "scenarios" / scenario).resolve()
            ),
            "artifact_manifest": str(manifest.resolve()),
            "input_adl": str(
                (MUSTER_ROOT / f"Controlled_S{number}_WITHOUT_INTERMODEL.adl").resolve()
            ),
            "output_run_directory": str(run_directory.resolve()),
        })
    config["intermodel_only"]["jobs"] = jobs

    target = PIPELINE_ROOT / "parameter.gemma4-4b-musterloesungen.json"
    write_json(target, config)
    print(f"Created {target}")
    print(f"Prepared {len(jobs)} intermodel-only jobs in {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
