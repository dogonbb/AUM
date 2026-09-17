#!/usr/bin/env python3
"""Run controlled phase 1-4 intermodel evaluations on simulated reference SLMs."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
MUSTER_RUNS = (
    PROJECT_ROOT / "Musterloesungen_ohne_Intermodel_RS" / "intermodel_runs"
)
EVALUATION_ROOT = PIPELINE_ROOT / "Evaluation" / "inter"

MODELS = {
    "gemma4-e4b": ("gemma4:e4b", Path("gemma4/e4b/run2")),
    "qwen35-4b": ("qwen3.5:4b", Path("qwen3.5/4b/run2")),
    "qwen35-9b": ("qwen3.5:9b", Path("qwen3.5/9b/run2")),
}
SCENARIOS = tuple(f"controlled_phase{number}" for number in range(1, 5))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def prepare_manifest(scenario: str, destination: Path) -> tuple[Path, Path]:
    source_run = MUSTER_RUNS / f"{scenario}_run"
    source_model_output = source_run / "output_model_gen"
    source_manifest = source_model_output / "artifacts.json"
    input_adl = source_model_output / "complete_adl_file_here.adl"
    data = json.loads(source_manifest.read_text(encoding="utf-8"))

    artifacts = []
    for item in data["artifacts"]:
        prepared = copy.deepcopy(item)
        prepared["source_scenario_file"] = str(
            (PIPELINE_ROOT / item["source_scenario_file"]).resolve()
        )
        # Use precisely the simulated SLM output bundled with this reference run.
        relative_slm = Path(item["slm_file"])
        marker = Path("output_model_gen") / "slm"
        parts = relative_slm.parts
        marker_parts = marker.parts
        start = next(
            index
            for index in range(len(parts) - len(marker_parts) + 1)
            if parts[index : index + len(marker_parts)] == marker_parts
        )
        slm_relative = Path(*parts[start + len(marker_parts) :])
        prepared["slm_file"] = str(
            (source_model_output / "slm" / slm_relative).resolve()
        )
        prepared["adl_file"] = str(input_adl.resolve())
        prepared["status"] = "EXISTING"
        prepared["task_id"] = (
            f"simulated-reference:{scenario}:{item['model_id']}:{item['adl_model_name']}"
        )
        artifacts.append(prepared)

    manifest = {
        "format": "4em-pipeline-artifact-manifest-v1",
        "scenario_name": scenario,
        "scenario_directory": str((PIPELINE_ROOT / "scenarios" / scenario).resolve()),
        "run_directory": str(destination.resolve()),
        "source": str(input_adl.resolve()),
        "note": (
            "Uses only simulated model-generation SLM outputs from "
            "Musterloesungen_ohne_Intermodel_RS/intermodel_runs; existing "
            "intermodel outputs from that directory are not reused."
        ),
        "artifacts": artifacts,
    }
    manifest_path = destination / "input" / "artifacts.json"
    write_json(manifest_path, manifest)
    return manifest_path, input_adl


def prepare_config(model_key: str, scenario: str) -> Path:
    ollama_model, relative_destination = MODELS[model_key]
    destination = EVALUATION_ROOT / relative_destination / scenario
    manifest, input_adl = prepare_manifest(scenario, destination)
    config = json.loads((PIPELINE_ROOT / "parameter.json").read_text(encoding="utf-8"))
    config["execution"].update(
        {"mode": "intermodel_only", "selected_scenario": scenario, "overwrite": True}
    )
    config["slm"]["model"] = ollama_model
    config["intermodel_only"]["jobs"] = [
        {
            "enabled": True,
            "scenario_name": scenario,
            "scenario_directory": str(
                (PIPELINE_ROOT / "scenarios" / scenario).resolve()
            ),
            "artifact_manifest": str(manifest.resolve()),
            "input_adl": str(input_adl.resolve()),
            "output_run_directory": str(destination.resolve()),
        }
    ]
    archived_config_path = destination / "input" / "parameters.json"
    write_json(archived_config_path, config)
    return archived_config_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), action="append")
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the prepared evaluations. Without this flag, only prepare folders.",
    )
    args = parser.parse_args()
    model_keys = args.model or list(MODELS)
    scenarios = args.scenario or list(SCENARIOS)

    failures: list[str] = []
    total = len(model_keys) * len(scenarios)
    current = 0
    for model_key in model_keys:
        for scenario in scenarios:
            current += 1
            label = f"{model_key}/{scenario}"
            archived_config = prepare_config(model_key, scenario)
            print(f"[{current}/{total}] PREPARED {label}", flush=True)
            if not args.run:
                continue
            # pipeline.py uses the parameter file's parent as project root.
            # Materialize an active root-level copy only for actual execution.
            config_path = PIPELINE_ROOT / "parameter.controlled-muster-intermodel-eval.json"
            config_path.write_text(
                archived_config.read_text(encoding="utf-8"), encoding="utf-8"
            )
            print(f"[{current}/{total}] START {label}", flush=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "pipeline.py",
                    "--parameters",
                    str(config_path),
                    "--mode",
                    "intermodel_only",
                    "--scenario",
                    scenario,
                ],
                cwd=PIPELINE_ROOT,
            )
            if result.returncode:
                failures.append(label)
                print(f"FAILED {label} (exit {result.returncode})", flush=True)
            else:
                print(f"COMPLETED {label}", flush=True)

    if failures:
        print("Failed evaluations: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
