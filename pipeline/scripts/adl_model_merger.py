#!/usr/bin/env python3
"""Mehrere 4EM-ADL-Exportdateien zu einer ADL-Datei zusammenfügen.

Die Zieldatei enthält:
  1. einen neu erzeugten Informationskopf,
  2. genau eine globale Zeile VERSION <...>,
  3. anschließend die vollständigen Modellblöcke aller ausgewählten Dateien.

Das Skript bietet standardmäßig einen Dateiauswahldialog. Alternativ können
Eingabedateien oder Ordner und die Ausgabe über die Kommandozeile angegeben
werden. Bei einem Ordner werden automatisch alle darin enthaltenen .adl-Dateien
verwendet.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

GLOBAL_VERSION_RE = re.compile(r"(?m)^VERSION\s*<([^>]*)>\s*$")
MODEL_START_RE = re.compile(
    r"(?m)^(?P<kind>[A-Z][A-Z0-9 &/()_-]*?\s+MODEL)\s+"
    r"<(?P<name>[^>]+)>\s*:\s*<(?P<context>[^>]*)>\s*$"
)


@dataclass(frozen=True)
class AdlModel:
    source: Path
    kind: str
    name: str
    context: str
    text: str


def read_text(path: Path) -> str:
    """ADL-Datei mit üblichen Kodierungen lesen."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not detect the encoding of {path}.")


def parse_adl_file(path: Path) -> tuple[str, list[AdlModel]]:
    """Globale Datenversion und alle Modellblöcke einer ADL-Datei lesen."""
    text = read_text(path).replace("\r\n", "\n").replace("\r", "\n")

    version_match = GLOBAL_VERSION_RE.search(text)
    if not version_match:
        raise ValueError(f"No global VERSION line found in {path.name}.")
    version = version_match.group(1).strip()

    starts = list(MODEL_START_RE.finditer(text))
    if not starts:
        raise ValueError(f"No model block found in {path.name}.")

    models: list[AdlModel] = []
    for index, match in enumerate(starts):
        start = match.start()
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start:end].strip() + "\n"
        models.append(
            AdlModel(
                source=path,
                kind=match.group("kind").strip(),
                name=match.group("name").strip(),
                context=match.group("context").strip(),
                text=block,
            )
        )

    return version, models


def make_header(models: Iterable[AdlModel], version: str) -> str:
    model_list = list(models)
    now = datetime.now().strftime("%d.%m.%Y  %H:%M")
    lines = [
        "/" * 63,
        "//",
        f"// Date: {now}",
        "//",
        "// Combined by adl_model_merger.py",
        f"// Data version {version}",
        "//",
        "/" * 63,
        "//",
        "// The file contains the following models:",
        "//",
    ]
    for model in model_list:
        lines.append(f"// {model.name} ({model.kind.title()})")
    lines.extend(["//", "/" * 62, "", f"VERSION <{version}>", "", ""])
    return "\n".join(lines)


def collect_adl_files(inputs: list[Path], output_path: Path | None = None) -> list[Path]:
    """ADL-Dateien aus einzelnen Dateien und Ordnern sammeln."""
    collected: list[Path] = []
    output_resolved = output_path.resolve() if output_path is not None else None

    for item in inputs:
        if not item.exists():
            raise ValueError(f"Input not found: {item}")

        if item.is_dir():
            candidates = sorted(
                (path for path in item.iterdir() if path.is_file() and path.suffix.casefold() == ".adl"),
                key=lambda path: path.name.casefold(),
            )
            if not candidates:
                raise ValueError(f"No ADL files found in directory: {item}")
            collected.extend(candidates)
        elif item.is_file():
            if item.suffix.casefold() != ".adl":
                raise ValueError(f"Not an ADL file: {item}")
            collected.append(item)
        else:
            raise ValueError(f"Invalid input: {item}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in collected:
        resolved = path.resolve()
        if output_resolved is not None and resolved == output_resolved:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    if not unique:
        raise ValueError("No ADL input files were found.")
    return unique


def merge_adl_files(input_paths: list[Path], output_path: Path) -> list[AdlModel]:
    if not input_paths:
        raise ValueError("No input files were selected.")

    versions: set[str] = set()
    all_models: list[AdlModel] = []

    for path in input_paths:
        version, models = parse_adl_file(path)
        versions.add(version)
        all_models.extend(models)

    if len(versions) != 1:
        details = ", ".join(sorted(versions))
        raise ValueError(
            "The files use different ADL data versions "
            f"({details}) and therefore cannot be merged automatically."
        )

    # Doppelte Modellnamen können in 4EM zu Konflikten führen.
    seen: dict[tuple[str, str], Path] = {}
    duplicates: list[str] = []
    for model in all_models:
        key = (model.kind.casefold(), model.name.casefold())
        if key in seen:
            duplicates.append(
                f"{model.kind} <{model.name}> in {seen[key].name} und {model.source.name}"
            )
        else:
            seen[key] = model.source
    if duplicates:
        raise ValueError("Duplicate model names found:\n- " + "\n- ".join(duplicates))

    version = next(iter(versions))
    result = make_header(all_models, version)
    result += "\n\n".join(model.text.rstrip() for model in all_models)
    result = result.rstrip() + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8", newline="\n")
    return all_models


def choose_files_gui() -> tuple[list[Path], Path | None]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("Tkinter is not installed; use CLI mode instead.") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()

    selected = filedialog.askopenfilenames(
        title="Select ADL model files",
        filetypes=[("4EM ADL files", "*.adl"), ("All files", "*.*")],
    )
    if not selected:
        root.destroy()
        return [], None

    output = filedialog.asksaveasfilename(
        title="Save merged ADL file",
        defaultextension=".adl",
        initialfile="Controlled_S3_merged.adl",
        filetypes=[("4EM ADL files", "*.adl"), ("All files", "*.*")],
    )
    root.destroy()
    return [Path(item) for item in selected], Path(output) if output else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple 4EM ADL files into one file."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input ADL files or directories containing ADL files",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output ADL file")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.inputs:
            if args.output is None:
                raise ValueError("Im CLI-Modus ist --output erforderlich.")
            output = args.output
            inputs = collect_adl_files(args.inputs, output)
        else:
            inputs, output = choose_files_gui()
            if not inputs or output is None:
                print("Cancelled: no files selected.")
                return 0

        models = merge_adl_files(inputs, output)
        print(f"Created: {output}")
        print(f"Merged models: {len(models)}")
        for model in models:
            print(f"  - {model.kind} <{model.name}> ({model.source.name})")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
