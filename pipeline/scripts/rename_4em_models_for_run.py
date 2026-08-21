#!/usr/bin/env python3
"""Give every 4EM model in an ADL file a unique run-specific name.

The script renames the model declarations and every occurrence of the old
model names, including INTERREF values and decomposition references.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


MODEL_HEADER = re.compile(
    r"(?m)^BUSINESS PROCESS MODEL <(?P<name>[^>\r\n]+)>"
)
RUN_FOLDER = re.compile(
    r"^run_(?P<run_number>\d+)_(?P<suffix>.+)_(?:SUCCESS|FAILED)$",
    re.IGNORECASE,
)


def safe_token(value: str, label: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        raise ValueError(f"{label} does not contain a usable character.")
    return token


def read_adl(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252"), "cp1252"


def discover_adl_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.casefold() != ".adl":
            raise ValueError(f"Not an ADL file: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.adl"))
        if not files:
            raise ValueError(f"No ADL files found below: {path}")
        return files
    raise ValueError(f"Path does not exist: {path}")


def infer_run_context(path: Path) -> tuple[str, int, str]:
    for parent in path.parents:
        match = RUN_FOLDER.fullmatch(parent.name)
        if match:
            scenario_folder = parent.parent
            if not scenario_folder.name:
                break
            return (
                safe_token(scenario_folder.name, "scenario"),
                int(match.group("run_number")),
                safe_token(match.group("suffix"), "suffix"),
            )
    raise ValueError(
        f"Could not infer scenario/run/timestamp from {path}. Expected a path "
        "like <history>/<scenario>/run_<number>_<timestamp>_<status>/.../*.adl"
    )


def build_mapping(
    model_names: list[str],
    scenario: str,
    run_number: int,
    suffix: str,
) -> dict[str, str]:
    unique_marker = f"__{scenario}__run_{run_number:03d}__{suffix}"
    mapping: dict[str, str] = {}
    for old_name in model_names:
        if old_name.endswith(unique_marker):
            mapping[old_name] = old_name
        else:
            mapping[old_name] = f"{old_name}{unique_marker}"

    if len(set(mapping.values())) != len(mapping):
        raise ValueError("The generated model names are not unique.")
    return mapping


def replace_names(text: str, mapping: dict[str, str]) -> str:
    changed = {old: new for old, new in mapping.items() if old != new}
    if not changed:
        return text
    pattern = re.compile(
        "|".join(re.escape(name) for name in sorted(changed, key=len, reverse=True))
    )
    return pattern.sub(lambda match: changed[match.group(0)], text)


def mapping_path_for(adl_path: Path, multiple_files: bool) -> Path:
    if multiple_files:
        return adl_path.with_name(f"{adl_path.stem}_model_name_mapping.json")
    return adl_path.with_name("model_name_mapping.json")


def rename_file(
    path: Path,
    scenario: str,
    run_number: int,
    suffix: str,
    dry_run: bool,
    write_mapping: bool,
    multiple_files: bool,
) -> dict[str, str]:
    text, encoding = read_adl(path)
    model_names = list(dict.fromkeys(
        match.group("name").strip() for match in MODEL_HEADER.finditer(text)
    ))
    if not model_names:
        raise ValueError(f"No 4EM model declarations found in: {path}")

    mapping = build_mapping(model_names, scenario, run_number, suffix)
    result = replace_names(text, mapping)

    expected_names = list(mapping.values())
    actual_names = [
        match.group("name").strip() for match in MODEL_HEADER.finditer(result)
    ]
    if actual_names != expected_names:
        raise RuntimeError(f"Model header validation failed for: {path}")

    if not dry_run:
        temporary = path.with_suffix(path.suffix + ".rename_tmp")
        temporary.write_bytes(result.encode(encoding))
        temporary.replace(path)

        if write_mapping:
            report = {
                "scenario": scenario,
                "run_number": run_number,
                "suffix": suffix,
                "adl_file": path.name,
                "models": [
                    {"old_name": old, "new_name": new}
                    for old, new in mapping.items()
                ],
            }
            report_path = mapping_path_for(path, multiple_files)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rename all models and internal model references in 4EM ADL files "
            "using a unique scenario/run suffix."
        )
    )
    parser.add_argument("path", type=Path, help="ADL file or directory")
    parser.add_argument(
        "--scenario",
        default=None,
        help="Manual mode: scenario name. Omit in automatic history mode.",
    )
    parser.add_argument(
        "--run-number",
        default=None,
        type=int,
        help="Manual mode: run number. Omit in automatic history mode.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help=(
            "Unique suffix, preferably the run timestamp. "
            "Default: current timestamp with microseconds."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-mapping", action="store_true")
    args = parser.parse_args()

    manual_mode = args.scenario is not None or args.run_number is not None
    if manual_mode and (args.scenario is None or args.run_number is None):
        parser.error("--scenario and --run-number must be used together.")
    if not manual_mode and args.suffix is not None:
        parser.error("--suffix can only be used together with manual mode.")
    if args.run_number is not None and args.run_number < 1:
        parser.error("--run-number must be at least 1.")

    files = discover_adl_files(args.path)
    if manual_mode:
        manual_scenario = safe_token(args.scenario, "scenario")
        raw_suffix = args.suffix or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        manual_suffix = safe_token(raw_suffix, "suffix")

    for path in files:
        if manual_mode:
            scenario = manual_scenario
            run_number = args.run_number
            suffix = manual_suffix
        else:
            scenario, run_number, suffix = infer_run_context(path)

        mapping = rename_file(
            path=path,
            scenario=scenario,
            run_number=run_number,
            suffix=suffix,
            dry_run=args.dry_run,
            write_mapping=not args.no_mapping,
            multiple_files=len(files) > 1,
        )
        action = "Would rename" if args.dry_run else "Renamed"
        print(
            f"{action}: {path} "
            f"[scenario={scenario}, run={run_number}, suffix={suffix}]"
        )
        for old_name, new_name in mapping.items():
            print(f"  {old_name} -> {new_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
