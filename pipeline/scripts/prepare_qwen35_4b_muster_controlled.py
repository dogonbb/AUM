"""Prepare Qwen 3.5 4B controlled phase 3/4 intermodel-only runs."""

from __future__ import annotations

import copy
import json
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
MUSTER_ROOT = PROJECT_ROOT / "Musterloesungen_ohne_Intermodel_RS"
MUSTER_RUNS = MUSTER_ROOT / "intermodel_runs"
TARGET_CONFIG = (
    PIPELINE_ROOT / "parameter.qwen3.5-4b-musterloesungen-controlled.json"
)
PHASES = (3, 4)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_manifest(phase: int, destination: Path) -> tuple[Path, Path]:
    scenario = f"controlled_phase{phase}"
    source_model_output = (
        MUSTER_RUNS / f"{scenario}_run" / "output_model_gen"
    )
    source_manifest = source_model_output / "artifacts.json"
    input_adl = source_model_output / "complete_adl_file_here.adl"
    data = json.loads(source_manifest.read_text(encoding="utf-8-sig"))

    artifacts = []
    for item in data["artifacts"]:
        # BPM overview models are deliberately excluded from this evaluation.
        if "overview" in item["adl_model_name"].lower():
            continue

        prepared = copy.deepcopy(item)
        prepared["source_scenario_file"] = str(
            (PIPELINE_ROOT / item["source_scenario_file"]).resolve()
        )
        slm_relative = Path(item["slm_file"])
        marker_parts = ("output_model_gen", "slm")
        parts = slm_relative.parts
        start = next(
            index
            for index in range(len(parts) - len(marker_parts) + 1)
            if parts[index : index + len(marker_parts)] == marker_parts
        )
        prepared["slm_file"] = str(
            (source_model_output / "slm" / Path(*parts[start + 2 :])).resolve()
        )
        prepared["adl_file"] = str(input_adl.resolve())
        prepared["status"] = "EXISTING"
        prepared["task_id"] = (
            f"simulated-reference:{scenario}:"
            f"{item['model_id']}:{item['adl_model_name']}"
        )
        artifacts.append(prepared)

    manifest = {
        "format": "4em-pipeline-artifact-manifest-v1",
        "scenario_name": scenario,
        "scenario_directory": str(
            (PIPELINE_ROOT / "scenarios" / scenario).resolve()
        ),
        "run_directory": str(destination.resolve()),
        "source": str(input_adl.resolve()),
        "note": (
            "Uses the simulated Musterloesung model-generation SLM outputs. "
            "BPM overview artifacts are excluded. Existing intermodel outputs "
            "from the reference run are not reused."
        ),
        "artifacts": artifacts,
    }
    manifest_path = destination / "input" / "artifacts.json"
    write_json(manifest_path, manifest)
    return manifest_path, input_adl


def main() -> None:
    base_config = json.loads(
        (PIPELINE_ROOT / "parameter.json").read_text(encoding="utf-8-sig")
    )
    base_config["execution"].update(
        {
            "mode": "intermodel_only",
            "overwrite": True,
            "continue_after_failure": True,
            "selected_scenario": "controlled_phase3",
        }
    )
    base_config["slm"]["model"] = "qwen3.5:4b"

    jobs = []
    for phase in PHASES:
        scenario = f"controlled_phase{phase}"
        destination = (
            PIPELINE_ROOT / "output" / f"{scenario}_qwen3.5_4b_run2"
        )
        manifest_path, input_adl = prepare_manifest(phase, destination)
        job = {
            "enabled": True,
            "scenario_name": scenario,
            "scenario_directory": str(
                (PIPELINE_ROOT / "scenarios" / scenario).resolve()
            ),
            "artifact_manifest": str(manifest_path.resolve()),
            "input_adl": str(input_adl.resolve()),
            "output_run_directory": str(destination.resolve()),
        }
        jobs.append(job)

        archived_config = copy.deepcopy(base_config)
        archived_config["execution"]["selected_scenario"] = scenario
        archived_config["intermodel_only"]["jobs"] = [job]
        write_json(destination / "input" / "parameters.json", archived_config)
        print(f"Prepared: {destination}")

    base_config["intermodel_only"]["jobs"] = jobs
    write_json(TARGET_CONFIG, base_config)
    print(f"Prepared active config: {TARGET_CONFIG}")


if __name__ == "__main__":
    main()
