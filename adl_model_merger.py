#!/usr/bin/env python3
"""Mehrere 4EM-ADL-Exportdateien zu einer ADL-Datei zusammenfügen.

Die Zieldatei enthält:
  1. einen neu erzeugten Informationskopf,
  2. genau eine globale Zeile VERSION <...>,
  3. anschließend die vollständigen Modellblöcke aller ausgewählten Dateien.

Das Skript bietet standardmäßig einen Dateiauswahldialog. Alternativ können
Eingabedateien und Ausgabe über die Kommandozeile angegeben werden.
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
    raise ValueError(f"Kodierung von {path} konnte nicht erkannt werden.")


def parse_adl_file(path: Path) -> tuple[str, list[AdlModel]]:
    """Globale Datenversion und alle Modellblöcke einer ADL-Datei lesen."""
    text = read_text(path).replace("\r\n", "\n").replace("\r", "\n")

    version_match = GLOBAL_VERSION_RE.search(text)
    if not version_match:
        raise ValueError(f"Keine globale VERSION-Zeile in {path.name} gefunden.")
    version = version_match.group(1).strip()

    starts = list(MODEL_START_RE.finditer(text))
    if not starts:
        raise ValueError(f"Kein Modellblock in {path.name} gefunden.")

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


def merge_adl_files(input_paths: list[Path], output_path: Path) -> list[AdlModel]:
    if not input_paths:
        raise ValueError("Es wurden keine Eingabedateien ausgewählt.")

    versions: set[str] = set()
    all_models: list[AdlModel] = []

    for path in input_paths:
        version, models = parse_adl_file(path)
        versions.add(version)
        all_models.extend(models)

    if len(versions) != 1:
        details = ", ".join(sorted(versions))
        raise ValueError(
            "Die Dateien verwenden unterschiedliche ADL-Datenversionen "
            f"({details}) und werden deshalb nicht automatisch vermischt."
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
        raise ValueError("Doppelte Modellnamen gefunden:\n- " + "\n- ".join(duplicates))

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
        raise RuntimeError("Tkinter ist nicht installiert; bitte CLI-Modus verwenden.") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()

    selected = filedialog.askopenfilenames(
        title="ADL-Modelldateien auswählen",
        filetypes=[("4EM ADL-Dateien", "*.adl"), ("Alle Dateien", "*.*")],
    )
    if not selected:
        root.destroy()
        return [], None

    output = filedialog.asksaveasfilename(
        title="Zusammengefügte ADL-Datei speichern",
        defaultextension=".adl",
        initialfile="Controlled_S3_merged.adl",
        filetypes=[("4EM ADL-Dateien", "*.adl"), ("Alle Dateien", "*.*")],
    )
    root.destroy()
    return [Path(item) for item in selected], Path(output) if output else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mehrere 4EM-ADL-Dateien zu einer Datei zusammenfügen."
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="Eingabe-ADL-Dateien")
    parser.add_argument("-o", "--output", type=Path, help="Ausgabe-ADL-Datei")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.inputs:
            if args.output is None:
                raise ValueError("Im CLI-Modus ist --output erforderlich.")
            inputs, output = args.inputs, args.output
        else:
            inputs, output = choose_files_gui()
            if not inputs or output is None:
                print("Abgebrochen: keine Dateien ausgewählt.")
                return 0

        models = merge_adl_files(inputs, output)
        print(f"Erstellt: {output}")
        print(f"Zusammengefügte Modelle: {len(models)}")
        for model in models:
            print(f"  - {model.kind} <{model.name}> ({model.source.name})")
        return 0
    except Exception as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
