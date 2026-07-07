#!/usr/bin/env python3
"""
run_scenarios.py

Dieses Script verarbeitet mehrere Szenario-TXT-Dateien mit einem Ollama-Modell
und konvertiert die erzeugten SLM-Ausgaben anschließend per slm_to_adl.py in ADL-Dateien.

Features:
- Projektordner kann angegeben werden
- Liest Prompt.txt
- Liest alle .txt-Dateien aus scenarios/
- Ruft Ollama mit aktiviertem Thinking auf
- Speichert standardmäßig NUR die finale Modellantwort in output_slm/
- Optional kann der Thinking-Block zusätzlich gespeichert werden
- Optional kann der Thinking-Block zusätzlich in der Konsole ausgegeben werden
- Ruft slm_to_adl.py auf
- Speichert ADL-Dateien in output_adl/
- Trackt die Laufzeit pro Scenario
- Speichert am Ende einen Runtime-Report als TXT

Erwartete Ordnerstruktur:

projektordner/
│
├── Prompt.txt
├── slm_to_adl.py
│
├── scenarios/
│   ├── scenario_1.txt
│   ├── scenario_2.txt
│   └── ...
│
├── output_slm/
│   └── wird automatisch erstellt
│
└── output_adl/
    └── wird automatisch erstellt

Beispielaufruf:

python run_scenarios.py "C:\\MeinProjekt" --model-name "qwen3:8b" --num-ctx 8192 --temperature 0.2

Mit GPT-OSS und hoher Thinking-Stufe:

python run_scenarios.py "C:\\MeinProjekt" --model-name "gpt-oss:20b" --thinking high --num-ctx 8192

Thinking zusätzlich in output_slm speichern:

python run_scenarios.py "C:\\MeinProjekt" --model-name "qwen3:8b" --include-thinking-in-output

Thinking zusätzlich in der Konsole ausgeben:

python run_scenarios.py "C:\\MeinProjekt" --model-name "qwen3:8b" --show-thinking-in-console

Interaktiv:

python run_scenarios.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass
class ScenarioRuntimeResult:
    scenario_file: str
    slm_output_file: str
    adl_output_file: str
    status: str
    slm_seconds: float
    adl_seconds: float
    total_seconds: float
    slm_output_chars: int
    error_message: str = ""


def safe_filename(name: str) -> str:
    """
    Erzeugt einen möglichst sicheren Dateinamen.
    """
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
    cleaned = "".join(char if char in allowed else "_" for char in name)
    return cleaned.strip("_") or "output"


def format_seconds(seconds: float) -> str:
    """
    Formatiert Sekunden benutzerfreundlich.
    """
    if seconds < 60:
        return f"{seconds:.2f} Sekunden"

    minutes = int(seconds // 60)
    rest_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes} Min {rest_seconds:.2f} Sek"

    hours = minutes // 60
    rest_minutes = minutes % 60
    return f"{hours} Std {rest_minutes} Min {rest_seconds:.2f} Sek"


def read_text_file(path: Path) -> str:
    """
    Liest eine Textdatei robust ein.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def strip_thinking_blocks(text: str) -> str:
    """
    Entfernt vorsichtshalber Thinking-Blöcke aus der finalen Antwort.

    Manche Modelle schreiben trotz API-Feld 'think' Thinking direkt in 'response',
    z. B. als:
    <think>...</think>
    <thinking>...</thinking>

    Diese Funktion entfernt solche Blöcke, damit in output_slm standardmäßig
    wirklich nur die finale Antwort landet.
    """

    patterns = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
    ]

    cleaned = text

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    return cleaned.strip()


def ensure_required_paths(base_dir: Path) -> tuple[Path, Path, Path]:
    """
    Prüft, ob die benötigten Dateien und Ordner vorhanden sind.
    """
    prompt_file = base_dir / "Prompt.txt"
    scenarios_dir = base_dir / "scenarios"
    converter_file = base_dir / "slm_to_adl.py"

    missing = []

    if not prompt_file.is_file():
        missing.append(prompt_file)

    if not scenarios_dir.is_dir():
        missing.append(scenarios_dir)

    if not converter_file.is_file():
        missing.append(converter_file)

    if missing:
        print("\nFehler: Folgende benötigte Dateien/Ordner fehlen:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(1)

    return prompt_file, scenarios_dir, converter_file


def normalize_ollama_url(url: str) -> str:
    """
    Entfernt einen abschließenden Slash von der Ollama-URL.
    """
    return url.rstrip("/")


def build_ollama_options(
    num_ctx: int | None,
    temperature: float | None,
    top_p: float | None,
    seed: int | None,
    num_predict: int | None,
) -> dict:
    """
    Baut den options-Block für die Ollama API.
    """
    options = {}

    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    if temperature is not None:
        options["temperature"] = temperature

    if top_p is not None:
        options["top_p"] = top_p

    if seed is not None:
        options["seed"] = seed

    if num_predict is not None:
        options["num_predict"] = num_predict

    return options


def print_thinking_to_console(thinking_text: str) -> None:
    """
    Gibt den Thinking-Text formatiert in der Konsole aus.
    """
    if not thinking_text:
        return

    print("\n  Thinking-Ausgabe des Modells")
    print("  " + "-" * 76)

    for line in thinking_text.splitlines():
        print(f"  {line}")

    print("  " + "-" * 76)
    print()


def run_ollama_with_thinking(
    model_name: str,
    prompt: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    thinking: str = "true",
    num_ctx: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
    timeout_seconds: int | None = None,
    include_thinking_in_output: bool = False,
    show_thinking_in_console: bool = False,
) -> str:
    """
    Ruft Ollama über die native API auf und aktiviert Thinking explizit.

    Wenn show_thinking_in_console=True ist, wird Streaming aktiviert,
    damit Thinking schon während der Generierung in der Konsole erscheinen kann.

    Wichtig:
    - Nicht jedes Modell liefert ein separates Feld 'thinking'.
    - Wenn kein 'thinking'-Feld kommt, kann auch nichts angezeigt werden.
    """

    ollama_url = normalize_ollama_url(ollama_url)
    endpoint = f"{ollama_url}/api/generate"

    options = build_ollama_options(
        num_ctx=num_ctx,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        num_predict=num_predict,
    )

    if thinking.lower() == "true":
        think_value: bool | str = True
    elif thinking.lower() in {"low", "medium", "high"}:
        think_value = thinking.lower()
    else:
        raise ValueError(
            "Ungültiger Thinking-Wert. Erlaubt sind: true, low, medium, high."
        )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": show_thinking_in_console,
        "think": think_value,
    }

    if options:
        payload["options"] = options

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if show_thinking_in_console:
                final_answer_parts: list[str] = []
                thinking_parts: list[str] = []
                printed_thinking_header = False
                received_thinking = False

                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()

                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise RuntimeError(
                            "Ollama hat im Streaming-Modus keine gültige JSON-Zeile zurückgegeben.\n"
                            f"Zeile:\n{line}"
                        ) from error

                    if "error" in chunk:
                        raise RuntimeError(
                            "Ollama hat einen Fehler zurückgegeben:\n"
                            f"{chunk['error']}"
                        )

                    thinking_chunk = chunk.get("thinking", "")
                    response_chunk = chunk.get("response", "")

                    if thinking_chunk:
                        received_thinking = True
                        thinking_parts.append(thinking_chunk)

                        if not printed_thinking_header:
                            print("\n  Thinking-Ausgabe des Modells")
                            print("  " + "-" * 76)
                            printed_thinking_header = True

                        print(thinking_chunk, end="", flush=True)

                    if response_chunk:
                        final_answer_parts.append(response_chunk)

                    if chunk.get("done", False):
                        break

                if printed_thinking_header:
                    print()
                    print("  " + "-" * 76)
                    print()

                if not received_thinking:
                    print(
                        "  Hinweis: Das Modell hat kein separates 'thinking'-Feld zurückgegeben. "
                        "Daher konnte kein Thinking in der Konsole ausgegeben werden.\n"
                    )

                final_answer = "".join(final_answer_parts).strip()
                thinking_text = "".join(thinking_parts).strip()

            else:
                raw_response = response.read().decode("utf-8")

                try:
                    result = json.loads(raw_response)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        "Ollama hat keine gültige JSON-Antwort zurückgegeben.\n"
                        f"Antwort:\n{raw_response}"
                    ) from error

                if "error" in result:
                    raise RuntimeError(
                        "Ollama hat einen Fehler zurückgegeben:\n"
                        f"{result['error']}"
                    )

                final_answer = result.get("response", "").strip()
                thinking_text = result.get("thinking", "").strip()

    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama API konnte nicht erreicht werden.\n"
            f"URL: {endpoint}\n"
            "Prüfe, ob Ollama läuft.\n"
            f"Details: {error}"
        ) from error

    if include_thinking_in_output and thinking_text:
        final_answer_cleaned = strip_thinking_blocks(final_answer)

        return (
            "<thinking>\n"
            f"{thinking_text}\n"
            "</thinking>\n\n"
            "<answer>\n"
            f"{final_answer_cleaned}\n"
            "</answer>"
        ).strip()

    return strip_thinking_blocks(final_answer)
def run_slm_to_adl(
    converter_file: Path,
    input_txt: Path,
    output_adl: Path,
    model_name: str,
) -> None:
    """
    Ruft slm_to_adl.py auf:

        python ".\\slm_to_adl.py" "input.txt" ".\\output.adl" --model-name "Modellname"
    """

    command = [
        sys.executable,
        str(converter_file),
        str(input_txt),
        str(output_adl),
        "--model-name",
        model_name,
    ]

    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "slm_to_adl.py-Aufruf fehlgeschlagen.\n"
            f"Befehl: {' '.join(command)}\n"
            f"Fehlerausgabe:\n{result.stderr}\n"
            f"Standardausgabe:\n{result.stdout}"
        )


def build_combined_prompt(base_prompt: str, scenario_text: str, scenario_name: str) -> str:
    """
    Baut den finalen Prompt aus Prompt.txt und dem jeweiligen Szenario.
    """

    return f"""\
{base_prompt.strip()}

---

Szenario-Datei: {scenario_name}

{scenario_text.strip()}
"""


def write_runtime_report(
    report_file: Path,
    results: list[ScenarioRuntimeResult],
    model_name: str,
    ollama_url: str,
    thinking: str,
    include_thinking_in_output: bool,
    show_thinking_in_console: bool,
    num_ctx: int | None,
    temperature: float | None,
    top_p: float | None,
    seed: int | None,
    num_predict: int | None,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    """
    Schreibt eine TXT-Datei mit allen Laufzeit-Informationen.
    """

    total_seconds = sum(result.total_seconds for result in results)
    total_slm_seconds = sum(result.slm_seconds for result in results)
    total_adl_seconds = sum(result.adl_seconds for result in results)

    success_count = sum(1 for result in results if result.status == "OK")
    failed_count = sum(1 for result in results if result.status == "ERROR")
    skipped_count = sum(1 for result in results if result.status == "SKIPPED")

    lines = []

    lines.append("SLM Runtime Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Startzeit:              {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Endzeit:                {finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Gesamtdauer:            {format_seconds((finished_at - started_at).total_seconds())}")
    lines.append("")
    lines.append("Konfiguration")
    lines.append("-" * 80)
    lines.append(f"Modell:                 {model_name}")
    lines.append(f"Ollama URL:             {ollama_url}")
    lines.append(f"Thinking aktiviert:     Ja")
    lines.append(f"Thinking-Wert:          {thinking}")
    lines.append(f"Thinking gespeichert:   {'Ja' if include_thinking_in_output else 'Nein'}")
    lines.append(f"Thinking in Konsole:    {'Ja' if show_thinking_in_console else 'Nein'}")
    lines.append(f"num_ctx:                {num_ctx if num_ctx is not None else 'Ollama-Standard'}")
    lines.append(f"temperature:            {temperature if temperature is not None else 'Ollama-Standard'}")
    lines.append(f"top_p:                  {top_p if top_p is not None else 'Ollama-Standard'}")
    lines.append(f"seed:                   {seed if seed is not None else 'kein fester Seed'}")
    lines.append(f"num_predict:            {num_predict if num_predict is not None else 'Ollama-Standard'}")
    lines.append("")
    lines.append("Zusammenfassung")
    lines.append("-" * 80)
    lines.append(f"Gesamtanzahl:           {len(results)}")
    lines.append(f"Erfolgreich:            {success_count}")
    lines.append(f"Übersprungen:           {skipped_count}")
    lines.append(f"Fehler:                 {failed_count}")
    lines.append(f"SLM-Gesamtzeit:         {format_seconds(total_slm_seconds)}")
    lines.append(f"ADL-Gesamtzeit:         {format_seconds(total_adl_seconds)}")
    lines.append(f"Scenario-Gesamtzeit:    {format_seconds(total_seconds)}")
    lines.append("")

    processed_results = [
        result for result in results
        if result.status in {"OK", "ERROR"} and result.total_seconds > 0
    ]

    if processed_results:
        avg_total = sum(result.total_seconds for result in processed_results) / len(processed_results)
        avg_slm = sum(result.slm_seconds for result in processed_results) / len(processed_results)
        avg_adl = sum(result.adl_seconds for result in processed_results) / len(processed_results)

        lines.append("Durchschnitt")
        lines.append("-" * 80)
        lines.append(f"Ø Gesamtzeit pro Datei: {format_seconds(avg_total)}")
        lines.append(f"Ø SLM-Zeit pro Datei:   {format_seconds(avg_slm)}")
        lines.append(f"Ø ADL-Zeit pro Datei:   {format_seconds(avg_adl)}")
        lines.append("")

    lines.append("Details pro Scenario")
    lines.append("=" * 80)
    lines.append("")

    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.scenario_file}")
        lines.append("-" * 80)
        lines.append(f"Status:                 {result.status}")
        lines.append(f"SLM-Output-Datei:       {result.slm_output_file}")
        lines.append(f"ADL-Output-Datei:       {result.adl_output_file}")
        lines.append(f"SLM-Output-Zeichen:     {result.slm_output_chars}")
        lines.append(f"SLM-Zeit:               {format_seconds(result.slm_seconds)}")
        lines.append(f"ADL-Zeit:               {format_seconds(result.adl_seconds)}")
        lines.append(f"Gesamtzeit:             {format_seconds(result.total_seconds)}")

        if result.error_message:
            lines.append("")
            lines.append("Fehler:")
            lines.append(result.error_message)

        lines.append("")

    report_file.write_text("\n".join(lines), encoding="utf-8")


def process_scenarios(
    base_dir: Path,
    model_name: str,
    ollama_url: str,
    thinking: str,
    include_thinking_in_output: bool,
    show_thinking_in_console: bool,
    num_ctx: int | None,
    temperature: float | None,
    top_p: float | None,
    seed: int | None,
    num_predict: int | None,
    timeout_seconds: int | None,
    overwrite: bool,
) -> None:
    """
    Hauptlogik:
    - Prompt.txt lesen
    - alle Szenario-TXT-Dateien verarbeiten
    - Ollama mit aktiviertem Thinking aufrufen
    - SLM-Output speichern
    - optional Thinking in Konsole ausgeben
    - ADL-Datei erzeugen
    - Runtime-Report schreiben
    """

    started_at = datetime.now()
    runtime_results: list[ScenarioRuntimeResult] = []

    prompt_file, scenarios_dir, converter_file = ensure_required_paths(base_dir)

    output_slm_dir = base_dir / "output_slm"
    output_adl_dir = base_dir / "output_adl"

    output_slm_dir.mkdir(exist_ok=True)
    output_adl_dir.mkdir(exist_ok=True)

    report_file = output_slm_dir / "slm_runtime_report.txt"

    base_prompt = read_text_file(prompt_file)

    scenario_files = sorted(scenarios_dir.glob("*.txt"))

    if not scenario_files:
        print(f"\nKeine .txt-Dateien im Ordner gefunden: {scenarios_dir}")
        sys.exit(1)

    print("\nStarte Verarbeitung")
    print(f"Projektordner:          {base_dir}")
    print(f"Modell:                 {model_name}")
    print(f"Ollama URL:             {ollama_url}")
    print(f"Thinking aktiviert:     Ja")
    print(f"Thinking-Wert:          {thinking}")
    print(f"Thinking gespeichert:   {'Ja' if include_thinking_in_output else 'Nein'}")
    print(f"Thinking in Konsole:    {'Ja' if show_thinking_in_console else 'Nein'}")
    print(f"Szenarien:              {len(scenario_files)}")
    print(f"SLM-Output:             {output_slm_dir}")
    print(f"ADL-Output:             {output_adl_dir}")
    print(f"Runtime-Report:         {report_file}")

    if num_ctx is not None:
        print(f"num_ctx:                {num_ctx}")

    if temperature is not None:
        print(f"temperature:            {temperature}")

    if top_p is not None:
        print(f"top_p:                  {top_p}")

    if seed is not None:
        print(f"seed:                   {seed}")

    if num_predict is not None:
        print(f"num_predict:            {num_predict}")

    if timeout_seconds is not None:
        print(f"timeout_seconds:        {timeout_seconds}")

    print()

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for index, scenario_file in enumerate(scenario_files, start=1):
        print(f"[{index}/{len(scenario_files)}] Verarbeite: {scenario_file.name}")

        scenario_total_start = time.perf_counter()

        scenario_stem = safe_filename(scenario_file.stem)

        slm_output_file = output_slm_dir / f"{scenario_stem}__slm.txt"
        adl_output_file = output_adl_dir / f"{scenario_stem}__adl.adl"

        if not overwrite and slm_output_file.exists() and adl_output_file.exists():
            scenario_total_seconds = time.perf_counter() - scenario_total_start

            print("  Übersprungen, da SLM- und ADL-Datei bereits existieren.\n")

            runtime_results.append(
                ScenarioRuntimeResult(
                    scenario_file=scenario_file.name,
                    slm_output_file=str(slm_output_file),
                    adl_output_file=str(adl_output_file),
                    status="SKIPPED",
                    slm_seconds=0.0,
                    adl_seconds=0.0,
                    total_seconds=scenario_total_seconds,
                    slm_output_chars=0,
                    error_message="Dateien existierten bereits. Nutze --overwrite zum Überschreiben.",
                )
            )

            skipped_count += 1
            continue

        slm_seconds = 0.0
        adl_seconds = 0.0
        slm_output_chars = 0

        try:
            scenario_text = read_text_file(scenario_file)
            combined_prompt = build_combined_prompt(
                base_prompt=base_prompt,
                scenario_text=scenario_text,
                scenario_name=scenario_file.name,
            )

            print("  Rufe Ollama mit aktiviertem Thinking auf...")

            slm_start = time.perf_counter()

            model_output = run_ollama_with_thinking(
                model_name=model_name,
                prompt=combined_prompt,
                ollama_url=ollama_url,
                thinking=thinking,
                num_ctx=num_ctx,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                num_predict=num_predict,
                timeout_seconds=timeout_seconds,
                include_thinking_in_output=include_thinking_in_output,
                show_thinking_in_console=show_thinking_in_console,
            )

            slm_seconds = time.perf_counter() - slm_start
            slm_output_chars = len(model_output)

            slm_output_file.write_text(model_output, encoding="utf-8")

            print(f"  SLM-Ausgabe gespeichert: {slm_output_file.name}")
            print(f"  SLM-Zeit: {format_seconds(slm_seconds)}")
            print(f"  SLM-Output-Zeichen: {slm_output_chars}")

            print("  Konvertiere SLM nach ADL...")

            adl_start = time.perf_counter()

            run_slm_to_adl(
                converter_file=converter_file,
                input_txt=slm_output_file,
                output_adl=adl_output_file,
                model_name=model_name,
            )

            adl_seconds = time.perf_counter() - adl_start

            scenario_total_seconds = time.perf_counter() - scenario_total_start

            print(f"  ADL-Datei gespeichert: {adl_output_file.name}")
            print(f"  ADL-Zeit: {format_seconds(adl_seconds)}")
            print(f"  Gesamtzeit: {format_seconds(scenario_total_seconds)}")
            print("  Fertig.\n")

            runtime_results.append(
                ScenarioRuntimeResult(
                    scenario_file=scenario_file.name,
                    slm_output_file=str(slm_output_file),
                    adl_output_file=str(adl_output_file),
                    status="OK",
                    slm_seconds=slm_seconds,
                    adl_seconds=adl_seconds,
                    total_seconds=scenario_total_seconds,
                    slm_output_chars=slm_output_chars,
                    error_message="",
                )
            )

            success_count += 1

        except Exception as error:
            scenario_total_seconds = time.perf_counter() - scenario_total_start
            failed_count += 1

            print("  Fehler bei dieser Datei:")
            print(f"  {error}")
            print(f"  Gesamtzeit bis Fehler: {format_seconds(scenario_total_seconds)}\n")

            runtime_results.append(
                ScenarioRuntimeResult(
                    scenario_file=scenario_file.name,
                    slm_output_file=str(slm_output_file),
                    adl_output_file=str(adl_output_file),
                    status="ERROR",
                    slm_seconds=slm_seconds,
                    adl_seconds=adl_seconds,
                    total_seconds=scenario_total_seconds,
                    slm_output_chars=slm_output_chars,
                    error_message=str(error),
                )
            )

    finished_at = datetime.now()

    write_runtime_report(
        report_file=report_file,
        results=runtime_results,
        model_name=model_name,
        ollama_url=ollama_url,
        thinking=thinking,
        include_thinking_in_output=include_thinking_in_output,
        show_thinking_in_console=show_thinking_in_console,
        num_ctx=num_ctx,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        num_predict=num_predict,
        started_at=started_at,
        finished_at=finished_at,
    )

    print("Verarbeitung abgeschlossen")
    print(f"Erfolgreich:     {success_count}")
    print(f"Übersprungen:    {skipped_count}")
    print(f"Fehler:          {failed_count}")
    print(f"Runtime-Report:  {report_file}")


def ask_interactively() -> argparse.Namespace:
    """
    Fragt Werte interaktiv ab, falls kein Ordner per Argument angegeben wurde.
    """

    print("Interaktiver Modus\n")

    base_dir = input("Projektordner angeben: ").strip().strip('"')
    model_name = input("Ollama-Modellname, z. B. qwen3:8b oder gpt-oss:20b: ").strip()

    ollama_url_raw = input(
        f"Ollama URL, leer lassen für {DEFAULT_OLLAMA_URL}: "
    ).strip()

    print("\nThinking wird immer aktiviert.")
    print("Für die meisten Modelle: true")
    print("Für GPT-OSS möglich: low, medium, high")
    thinking_raw = input("Thinking-Wert, leer lassen für true: ").strip().lower()

    include_thinking_raw = input(
        "Thinking-Text zusätzlich in die output_slm-Datei schreiben? [j/N]: "
    ).strip().lower()

    show_thinking_console_raw = input(
        "Thinking-Text zusätzlich in der Konsole ausgeben? [j/N]: "
    ).strip().lower()

    num_ctx_raw = input("num_ctx, leer lassen für Standardwert, z. B. 8192: ").strip()
    temperature_raw = input("temperature, leer lassen für Standardwert, z. B. 0.2: ").strip()
    top_p_raw = input("top_p, leer lassen für Standardwert, z. B. 0.9: ").strip()
    seed_raw = input("seed, leer lassen für keinen festen Seed, z. B. 42: ").strip()
    num_predict_raw = input("num_predict/max. Ausgabetokens, leer lassen für Standardwert: ").strip()
    timeout_raw = input("Timeout pro Ollama-Aufruf in Sekunden, leer lassen für kein Limit: ").strip()

    overwrite_raw = input("Vorhandene Ausgaben überschreiben? [j/N]: ").strip().lower()
    overwrite = overwrite_raw in {"j", "ja", "y", "yes"}

    if not base_dir:
        print("Fehler: Kein Projektordner angegeben.")
        sys.exit(1)

    if not model_name:
        print("Fehler: Kein Modellname angegeben.")
        sys.exit(1)

    return argparse.Namespace(
        base_dir=base_dir,
        model_name=model_name,
        ollama_url=ollama_url_raw or DEFAULT_OLLAMA_URL,
        thinking=thinking_raw or "true",
        include_thinking_in_output=include_thinking_raw in {"j", "ja", "y", "yes"},
        show_thinking_in_console=show_thinking_console_raw in {"j", "ja", "y", "yes"},
        num_ctx=int(num_ctx_raw) if num_ctx_raw else None,
        temperature=float(temperature_raw) if temperature_raw else None,
        top_p=float(top_p_raw) if top_p_raw else None,
        seed=int(seed_raw) if seed_raw else None,
        num_predict=int(num_predict_raw) if num_predict_raw else None,
        timeout_seconds=int(timeout_raw) if timeout_raw else None,
        overwrite=overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verarbeitet Szenario-TXT-Dateien mit Ollama Thinking "
            "und konvertiert sie anschließend zu ADL."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "base_dir",
        nargs="?",
        help="Projektordner, der Prompt.txt, scenarios/ und slm_to_adl.py enthält.",
    )

    parser.add_argument(
        "--model-name",
        "-m",
        help="Name des Ollama-Modells, z. B. qwen3:8b, deepseek-r1:8b, gpt-oss:20b.",
    )

    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="URL der Ollama API.",
    )

    parser.add_argument(
        "--thinking",
        default="true",
        choices=["true", "low", "medium", "high"],
        help=(
            "Thinking wird immer aktiviert. "
            "Für die meisten Modelle: true. "
            "Für GPT-OSS: low, medium oder high."
        ),
    )

    parser.add_argument(
        "--include-thinking-in-output",
        action="store_true",
        help=(
            "Thinking wird aktiviert und zusätzlich mit in die output_slm-Datei geschrieben. "
            "Ohne diese Option wird nur die finale Antwort gespeichert."
        ),
    )

    parser.add_argument(
        "--show-thinking-in-console",
        action="store_true",
        help=(
            "Thinking wird aktiviert und zusätzlich in der Konsole ausgegeben. "
            "Die output_slm-Datei bleibt davon unberührt."
        ),
    )

    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Kontextgröße für Ollama, z. B. 4096, 8192, 16384.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Kreativität des Modells. Niedriger ist deterministischer, z. B. 0.0 bis 0.3.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Sampling-Parameter. Typische Werte: 0.8 bis 1.0.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fester Seed für reproduzierbarere Ausgaben, z. B. 42.",
    )

    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="Maximale Anzahl Ausgabetokens. Leer lassen für Ollama-Standard.",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Maximale Laufzeit pro Ollama-Aufruf in Sekunden.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Vorhandene Ausgaben überschreiben.",
    )

    args = parser.parse_args()

    if not args.base_dir:
        return ask_interactively()

    if not args.model_name:
        print("Fehler: Bitte --model-name angeben.")
        print(
            'Beispiel: python run_scenarios.py "C:\\MeinProjekt" '
            '--model-name "qwen3:8b" --num-ctx 8192'
        )
        sys.exit(1)

    return args


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()

    if not base_dir.is_dir():
        print(f"Fehler: Der angegebene Projektordner existiert nicht: {base_dir}")
        sys.exit(1)

    process_scenarios(
        base_dir=base_dir,
        model_name=args.model_name,
        ollama_url=args.ollama_url,
        thinking=args.thinking,
        include_thinking_in_output=args.include_thinking_in_output,
        show_thinking_in_console=args.show_thinking_in_console,
        num_ctx=args.num_ctx,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        num_predict=args.num_predict,
        timeout_seconds=args.timeout_seconds,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()