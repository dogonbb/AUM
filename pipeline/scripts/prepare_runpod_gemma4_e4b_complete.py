"""Build a self-contained Runpod bundle for all Gemma 4 e4b reference runs."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
MUSTER_RUNS = (
    PROJECT_ROOT / "Musterloesungen_ohne_Intermodel_RS" / "intermodel_runs"
)
BUNDLE = PIPELINE_ROOT / "runpod_gemma4_e4b_controlled_intermodel"
TEMPLATE_BUNDLE = PIPELINE_ROOT / "runpod_gemma4_e4b_complete"
SCENARIOS = tuple(f"controlled_phase{phase}" for phase in range(1, 5))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_project_files() -> None:
    BUNDLE.mkdir(parents=True)
    shutil.copy2(PIPELINE_ROOT / "pipeline.py", BUNDLE / "pipeline.py")
    shutil.copy2(
        TEMPLATE_BUNDLE / "run_all.sh",
        BUNDLE / "run_all.sh",
    )
    shutil.copy2(
        TEMPLATE_BUNDLE / "pack_results.py",
        BUNDLE / "pack_results.py",
    )
    shutil.copy2(
        TEMPLATE_BUNDLE / "README_RUNPOD.md",
        BUNDLE / "README_RUNPOD.md",
    )
    for directory in ("prompts", "scripts"):
        shutil.copytree(
            PIPELINE_ROOT / directory,
            BUNDLE / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for scenario in SCENARIOS:
        shutil.copytree(
            PIPELINE_ROOT / "scenarios" / scenario,
            BUNDLE / "scenarios" / scenario,
        )


def prepare_inputs(scenario: str) -> tuple[Path, Path, int]:
    source_output = MUSTER_RUNS / f"{scenario}_run" / "output_model_gen"
    source_manifest = source_output / "artifacts.json"
    source_adl = source_output / "complete_adl_file_here.adl"
    data = json.loads(source_manifest.read_text(encoding="utf-8-sig"))

    input_root = BUNDLE / "input" / scenario
    slm_root = input_root / "slm"
    input_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_adl, input_root / "complete_adl_file_here.adl")

    artifacts = []
    for item in data["artifacts"]:
        if "overview" in item["adl_model_name"].lower():
            continue

        source_slm = (
            source_output
            / "slm"
            / item["model_folder"]
            / Path(item["slm_file"]).name
        )
        target_slm = slm_root / item["model_folder"] / source_slm.name
        target_slm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_slm, target_slm)

        prepared = copy.deepcopy(item)
        prepared["source_scenario_file"] = (
            f"scenarios/{scenario}/{item['source_relative_path']}"
        )
        prepared["slm_file"] = target_slm.relative_to(BUNDLE).as_posix()
        prepared["adl_file"] = (
            f"input/{scenario}/complete_adl_file_here.adl"
        )
        prepared["status"] = "EXISTING"
        prepared["task_id"] = (
            f"runpod-reference:{scenario}:"
            f"{item['model_id']}:{item['adl_model_name']}"
        )
        artifacts.append(prepared)

    manifest = {
        "format": "4em-pipeline-artifact-manifest-v1",
        "scenario_name": scenario,
        "scenario_directory": f"scenarios/{scenario}",
        "run_directory": f"run/{scenario}",
        "source": f"input/{scenario}/complete_adl_file_here.adl",
        "note": (
            "Self-contained Runpod input using only Musterloesung SLM files; "
            "BPM overview artifacts are excluded."
        ),
        "artifacts": artifacts,
    }
    manifest_path = input_root / "artifacts.json"
    write_json(manifest_path, manifest)
    return manifest_path, input_root / "complete_adl_file_here.adl", len(artifacts)


def main() -> None:
    if BUNDLE.exists():
        raise SystemExit(f"Refusing to overwrite existing bundle: {BUNDLE}")

    copy_project_files()
    config = json.loads(
        (PIPELINE_ROOT / "parameter.json").read_text(encoding="utf-8-sig")
    )
    config["execution"].update(
        {
            "mode": "intermodel_only",
            "overwrite": True,
            "continue_after_failure": True,
            "selected_scenario": SCENARIOS[0],
        }
    )
    config["slm"].update(
        {
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "model": "gemma4:e4b",
            "timeout_seconds": 1800,
        }
    )

    jobs = []
    for scenario in SCENARIOS:
        manifest, input_adl, artifact_count = prepare_inputs(scenario)
        jobs.append(
            {
                "enabled": True,
                "scenario_name": scenario,
                "scenario_directory": f"scenarios/{scenario}",
                "artifact_manifest": manifest.relative_to(BUNDLE).as_posix(),
                "input_adl": input_adl.relative_to(BUNDLE).as_posix(),
                "output_run_directory": f"run/{scenario}",
            }
        )
        print(f"Prepared {scenario}: {artifact_count} artifacts")

    config["intermodel_only"]["jobs"] = jobs
    write_json(BUNDLE / "parameter.json", config)
    print(f"Bundle prepared: {BUNDLE}")


if __name__ == "__main__":
    main()
