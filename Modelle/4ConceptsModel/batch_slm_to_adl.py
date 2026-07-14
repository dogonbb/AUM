#!/usr/bin/env python3
"""
Verarbeitet alle TXT-Dateien eines Eingabeordners mit slm_to_adl.py.

Voraussetzung:
    batch_slm_to_adl.py und slm_to_adl.py liegen im selben Ordner.

Aufruf:
    python batch_slm_to_adl.py <txt_ordner> <adl_output_ordner>

Beispiel:
    python batch_slm_to_adl.py ./input_txt ./output_adl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Konvertiert alle TXT-Dateien eines Ordners mithilfe der "
            "slm_to_adl.py im selben Verzeichnis in ADL-Dateien."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Ordner mit den zu verarbeitenden TXT-Dateien",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Ordner für die erzeugten ADL-Dateien",
    )
    parser.add_argument(
        "--model-name-prefix",
        default="",
        help=(
            "Optionaler Präfix für den Modellnamen. "
            "Standardmäßig wird der Name der TXT-Datei verwendet."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    converter_script = script_dir / "slm_to_adl.py"

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not converter_script.is_file():
        print(
            f"Fehler: Die Datei '{converter_script}' wurde nicht gefunden.\n"
            "Lege slm_to_adl.py in denselben Ordner wie dieses Batch-Skript.",
            file=sys.stderr,
        )
        return 1

    if not input_dir.is_dir():
        print(
            f"Fehler: Der Eingabeordner '{input_dir}' existiert nicht "
            "oder ist kein Ordner.",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )

    if not txt_files:
        print(f"Keine TXT-Dateien in '{input_dir}' gefunden.")
        return 0

    successful = 0
    failed = 0

    for txt_file in txt_files:
        output_file = output_dir / f"{txt_file.stem}__adl.adl"

        model_name = (
            f"{args.model_name_prefix}{txt_file.stem}"
            if args.model_name_prefix
            else txt_file.stem
        )

        command = [
            sys.executable,
            str(converter_script),
            str(txt_file),
            str(output_file),
            "--model-name",
            model_name,
        ]

        print(f"Verarbeite: {txt_file.name} -> {output_file.name}")

        try:
            subprocess.run(command, check=True)
            successful += 1
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(
                f"Fehler bei '{txt_file.name}' "
                f"(Exit-Code {exc.returncode}).",
                file=sys.stderr,
            )

    print()
    print(f"Erfolgreich verarbeitet: {successful}")
    print(f"Fehlgeschlagen: {failed}")
    print(f"Ausgabeordner: {output_dir}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
