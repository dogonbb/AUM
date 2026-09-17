"""Create the supplementary workbook for hardware-normalized evaluation times."""

from __future__ import annotations

import json
import re
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "Evaluation"
OUTPUT_DIR = ROOT.parents[1] / "overleaf"
TIMEOUT_SECONDS = 30 * 60


def model_and_run(path: Path) -> tuple[str, int] | None:
    rel = path.relative_to(EVALUATION).parts
    if rel[:2] == ("gemma", "4eb"):
        match = re.fullmatch(r"gemma_e4b_run(\d+)", rel[2])
        return ("Gemma 4 4B", int(match.group(1))) if match else None
    if rel[:2] == ("qwen3.5", "4b"):
        match = re.fullmatch(r"run(\d+)", rel[2])
        return ("Qwen3.5 4B", int(match.group(1))) if match else None
    if rel[:2] == ("qwen3.5", "9b"):
        match = re.fullmatch(r"run(\d+)", rel[2])
        return ("Qwen3.5 9B", int(match.group(1))) if match else None
    return None


def slm_attempts(value):
    """Yield each retry-level SLM call exactly once from the nested report."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "slm_attempts" and isinstance(child, list):
                yield from child
            else:
                yield from slm_attempts(child)
    elif isinstance(value, list):
        for child in value:
            yield from slm_attempts(child)


def is_reference(model: str, run: int, scenario: str, hostname: str) -> bool:
    if model == "Gemma 4 4B":
        return True
    if model == "Qwen3.5 4B":
        return run == 1
    # Qwen 9B run 1 controlled is the stated reference. Run 2 has the same
    # recorded host as that reference and is therefore already comparable.
    if model == "Qwen3.5 9B":
        return (run == 1 and scenario.startswith("controlled")) or hostname == "DariusPC"
    return False


def collect_rows():
    rows = []
    for path in EVALUATION.rglob("pipeline_*.json"):
        if not re.match(r"^(controlled|organized)_phase\d", path.parent.name):
            continue
        identity = model_and_run(path)
        if not identity:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        model, run = identity
        scenario_match = re.match(r"((?:controlled|organized)_phase\d)", path.parent.name)
        scenario = scenario_match.group(1)
        host = data.get("system", {}).get("hostname", "unknown")
        attempts = list(slm_attempts(data.get("scenarios", [])))
        successful_seconds = successful_tokens = missing_seconds = 0.0
        missing_count = timeout_count = 0
        timeout_seconds = 0.0
        configured_timeout = float(data.get("configuration", {}).get("slm", {}).get("timeout_seconds") or 0)
        for attempt in attempts:
            duration = float(attempt.get("duration_seconds") or 0)
            meta = (attempt.get("slm") or {}).get("metadata") or {}
            tokens = (meta.get("prompt_eval_count") or 0) + (meta.get("eval_count") or 0)
            if attempt.get("status") == "OK" and tokens:
                successful_seconds += duration
                successful_tokens += tokens
            elif attempt.get("status") == "TIMEOUT" and configured_timeout == TIMEOUT_SECONDS:
                # A timeout already represents the complete capped execution
                # time. It must not be multiplied by the hardware factor.
                timeout_seconds += duration
                timeout_count += 1
            else:
                missing_seconds += duration
                missing_count += 1
        summary = data.get("summary", {})
        rows.append({
            "model": model,
            "run": run,
            "scenario": scenario,
            "kind": scenario.split("_", 1)[0],
            "phase": int(scenario[-1]),
            "status": data.get("status"),
            "host": host,
            "reference": is_reference(model, run, scenario, host),
            "duration": float(data.get("duration_seconds") or 0),
            "slm_wall": float(summary.get("model_generation", {}).get("slm_wall_seconds") or 0),
            "tokens": int(summary.get("all_tokens") or 0),
            "successful_tokens": int(successful_tokens),
            "successful_seconds": successful_seconds,
            "missing_count": missing_count,
            "missing_seconds": missing_seconds,
            "timeout_count": timeout_count,
            "timeout_seconds": timeout_seconds,
            "source": str(path.relative_to(ROOT.parents[1])),
        })
    return sorted(rows, key=lambda r: (r["model"], r["run"], r["kind"], r["phase"]))


def weighted_rates(rows):
    ref = defaultdict(lambda: [0.0, 0.0])
    system = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        if row["successful_tokens"] and row["successful_seconds"]:
            key = (row["model"], row["host"])
            system[key][0] += row["successful_tokens"]
            system[key][1] += row["successful_seconds"]
            if row["reference"]:
                ref[row["model"]][0] += row["successful_tokens"]
                ref[row["model"]][1] += row["successful_seconds"]
    return ({k: t / s for k, (t, s) in ref.items()},
            {k: t / s for k, (t, s) in system.items()})


def style_sheet(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(wrap_text=True)
    for col in range(1, ws.max_column + 1):
        width = max(len(str(ws.cell(row, col).value or "")) for row in range(1, min(ws.max_row, 100) + 1))
        ws.column_dimensions[get_column_letter(col)].width = min(max(width + 2, 11), 42)


def create_phase_plot(rows):
    """Recreate the phase comparison used in the evaluation chapter."""
    models = ["Gemma 4 4B", "Qwen3.5 4B", "Qwen3.5 9B"]
    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    width = 0.24
    phases = [1, 2, 3, 4]
    for ax, kind, title in zip(axes, ("controlled", "organized"), ("Kontrolliert", "Organisiert")):
        for model_index, (model, color) in enumerate(zip(models, colors)):
            means, deviations = [], []
            for phase in phases:
                values = [
                    row["normalized_total"] / 3600
                    for row in rows
                    if row["model"] == model and row["kind"] == kind and row["phase"] == phase
                ]
                means.append(statistics.mean(values))
                deviations.append(statistics.pstdev(values))
            positions = [phase + (model_index - 1) * width for phase in phases]
            ax.bar(positions, means, width, yerr=deviations, capsize=3,
                   color=color, label=model, edgecolor="white", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Szenariophase")
        ax.set_xticks(phases)
        ax.set_yscale("log")
        ax.grid(axis="y", which="both", alpha=0.25)
    axes[0].set_ylabel("Normalisierte Laufzeit [h]")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    image_dir = OUTPUT_DIR / "Bilder"
    image_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_dir / "Laufzeiten_nach_Phase_und_SLM.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    rows = collect_rows()
    ref_rates, source_rates = weighted_rates(rows)
    for row in rows:
        ref_rate = ref_rates[row["model"]]
        source_rate = source_rates.get((row["model"], row["host"]), ref_rate)
        estimated_missing_tokens = row["missing_seconds"] * source_rate
        normalized_timeout_seconds = row["timeout_count"] * TIMEOUT_SECONDS
        # slm_wall contains only completed SLM calls. Failed/repetitive calls
        # and timeouts are part of the total duration as well, so subtract
        # them before deriving the actual non-SLM pipeline overhead. Otherwise
        # every aborted call would be counted once as SLM time and again as
        # overhead after normalization.
        overhead = max(
            0.0,
            row["duration"]
            - row["slm_wall"]
            - row["missing_seconds"]
            - row["timeout_seconds"],
        )
        if row["reference"]:
            normalized_slm = row["slm_wall"]
            normalized_total = row["duration"]
            method = "Messzeit übernommen (Referenzsystem)"
        else:
            normalized_slm = (
                (row["tokens"] + estimated_missing_tokens) / ref_rate
                + normalized_timeout_seconds
            )
            normalized_total = normalized_slm + overhead
            method = "Tokens → Referenzzeit"
            if row["missing_count"]:
                method += "; fehlende Tokens: Zeit → Tokens → Zeit"
            if row["timeout_count"]:
                method += "; 30-Min-Timeouts unskaliert addiert"
        row.update(ref_rate=ref_rate, source_rate=source_rate,
                   estimated_missing_tokens=estimated_missing_tokens,
                   normalized_timeout_seconds=normalized_timeout_seconds,
                   overhead=overhead, normalized_slm=normalized_slm,
                   normalized_total=normalized_total, method=method,
                   factor=normalized_total / row["duration"] if row["duration"] else 0)

    wb = Workbook()
    ws = wb.active
    ws.title = "Läufe"
    headers = ["Modell", "Run", "Szenario", "Status", "protokolliertes System", "Referenzsystem?",
               "gemessene Gesamtzeit [s]", "gemessene SLM-Zeit [s]", "protokollierte Tokens",
               "sonstige abgebrochene Aufrufe", "sonstige Zeit ohne Tokenwerte [s]", "geschätzte fehlende Tokens",
               "30-Min-Timeouts", "gemessene Timeout-Zeit [s]", "angesetzte Timeout-Zeit [s]",
               "Quelldurchsatz [Token/s]", "Referenzdurchsatz [Token/s]", "Nicht-SLM-Overhead [s]",
               "normalisierte SLM-Zeit [s]", "normalisierte Gesamtzeit [s]", "Faktor norm./gemessen",
               "Berechnung", "Quelldatei"]
    ws.append(headers)
    for r in rows:
        ws.append([r["model"], r["run"], r["scenario"], r["status"], r["host"], "ja" if r["reference"] else "nein",
                   r["duration"], r["slm_wall"], r["tokens"], r["missing_count"], r["missing_seconds"],
                   r["estimated_missing_tokens"], r["timeout_count"], r["timeout_seconds"],
                   r["normalized_timeout_seconds"], r["source_rate"], r["ref_rate"], r["overhead"],
                   r["normalized_slm"], r["normalized_total"], r["factor"], r["method"], r["source"]])
    for row in ws.iter_rows(min_row=2, min_col=7, max_col=21):
        for cell in row:
            cell.number_format = '#,##0.00'
    style_sheet(ws)

    summary = wb.create_sheet("Zusammenfassung")
    summary.append(["Modell", "Run", "Szenariotyp", "Anzahl Läufe", "Tokens", "gemessene Zeit [h]", "normalisierte Zeit [h]", "Differenz [h]"])
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["run"], r["kind"])].append(r)
    for key, values in sorted(groups.items()):
        measured = sum(v["duration"] for v in values) / 3600
        normalized = sum(v["normalized_total"] for v in values) / 3600
        summary.append([*key, len(values), sum(v["tokens"] for v in values), measured, normalized, normalized-measured])
    summary.append([])
    summary.append(["GESAMT", "", "", len(rows), sum(r["tokens"] for r in rows),
                    sum(r["duration"] for r in rows)/3600, sum(r["normalized_total"] for r in rows)/3600,
                    sum(r["normalized_total"]-r["duration"] for r in rows)/3600])
    for row in summary.iter_rows(min_row=2, min_col=5, max_col=8):
        for cell in row: cell.number_format = '#,##0.00'
    style_sheet(summary)
    method = wb.create_sheet("Methode")
    method_rows = [
        ["Laufzeitnormalisierung auf das Referenzsystem"],
        ["Grundannahme", "Laufzeit und erzeugte/verarbeitete Tokenmenge sind innerhalb desselben Modells und Systems linear proportional."],
        ["Referenzläufe", "Gemma 4 4B: alle Runs; Qwen3.5 4B: Run 1; Qwen3.5 9B: Run 1 controlled. Weitere Qwen-9B-Läufe auf demselben protokollierten Host DariusPC werden ebenfalls als bereits vergleichbar behandelt."],
        ["Referenzdurchsatz", "Summe der Tokens aller erfolgreichen SLM-Aufrufe der Referenzläufe geteilt durch deren aufsummierte Aufrufzeit; getrennt je Modell."],
        ["Reguläre Fremdsystemläufe", "Normalisierte SLM-Zeit = protokollierte Tokens / Referenzdurchsatz."],
        ["Vorzeitig abgebrochene Aufrufe", "Liefert Ollama bei einer erkannten Thinking-Wiederholung oder einem sonstigen vorzeitigen Abbruch keine abschließenden Tokenwerte, werden die fehlenden Tokens aus Abbruchzeit und beobachtetem Durchsatz desselben Modells und Quellsystems geschätzt. Danach wird diese Tokenmenge mit dem Referenzdurchsatz in Referenzzeit umgerechnet."],
        ["30-Minuten-Timeouts", "Ein regulärer Timeout nach 1.800 Sekunden repräsentiert bereits die vollständige gedeckelte Laufzeit. Daher wird er nicht mit dem Hardwarefaktor multipliziert, sondern je Timeout als feste 30 Minuten zur auf dem Referenzsystem berechneten Zeit addiert."],
        ["Gesamtzeit", "Normalisierte Gesamtzeit = normalisierte SLM-Zeit + gemessener Nicht-SLM-Overhead. Bei Referenzläufen wird die gemessene Gesamtzeit unverändert übernommen."],
        ["Einschränkung", "Die lineare Umrechnung ist eine Näherung. Promptverarbeitung, Antwortlänge, Caching, Modellneustarts und Systemlast können den tatsächlichen Durchsatz beeinflussen."],
        ["Quelle", "Die Werte stammen aus den pipeline_*.json-Berichten unter AUM/pipeline/Evaluation. Das Blatt 'Läufe' nennt für jede Zeile die konkrete Quelldatei."],
    ]
    for row in method_rows: method.append(row)
    method["A1"].font = Font(bold=True, size=14, color="FFFFFF")
    method["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    method.merge_cells("A1:B1")
    method.column_dimensions["A"].width = 24
    method.column_dimensions["B"].width = 120
    for row in method.iter_rows():
        for cell in row: cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in method["A"][1:]: cell.font = Font(bold=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filenames = {
        "Läufe": "Laufzeitnormalisierung.csv",
        "Zusammenfassung": "Laufzeitnormalisierung_Zusammenfassung.csv",
        "Methode": "Laufzeitnormalisierung_Methode.csv",
    }
    # UTF-8 with BOM and semicolon separation opens directly with the German
    # Excel locale while avoiding executable Office container content.
    for sheet_name, filename in filenames.items():
        with (OUTPUT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            for values in wb[sheet_name].iter_rows(values_only=True):
                writer.writerow("" if value is None else value for value in values)
    create_phase_plot(rows)
    print(f"Created three CSV files in {OUTPUT_DIR} with {len(rows)} runs")
    for model, rate in ref_rates.items():
        print(f"{model}: {rate:.3f} Token/s")


if __name__ == "__main__":
    main()
