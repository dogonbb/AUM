"""Create a standalone HTML dashboard from one pipeline runtime JSON report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def slm_values(slm: Mapping[str, Any] | None) -> tuple[float, int, int]:
    if not slm:
        return 0.0, 0, 0
    metadata = slm.get("metadata", {})
    return (
        number(slm.get("wall_seconds")),
        int(number(metadata.get("prompt_eval_count"))),
        int(number(metadata.get("eval_count"))),
    )


def rows_from_task(phase: str, task: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_id = str(task.get("task_id", "unknown"))
    status = str(task.get("status", "UNKNOWN"))
    for outer_index, attempt in enumerate(task.get("attempts", []), start=1):
        nested = attempt.get("slm_attempts", [])
        if nested:
            for call in nested:
                seconds, prompt_tokens, output_tokens = slm_values(call.get("slm"))
                rows.append(
                    {
                        "phase": phase,
                        "task": task_id,
                        "kind": "generation",
                        "attempt": f"{outer_index}.{call.get('attempt_number', '?')}",
                        "status": call.get("status", status),
                        "seconds": seconds or number(call.get("duration_seconds")),
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "restart": 1 if call.get("model_restart") is not None else 0,
                    }
                )
        else:
            seconds, prompt_tokens, output_tokens = slm_values(attempt.get("slm"))
            rows.append(
                {
                    "phase": phase,
                    "task": task_id,
                    "kind": "generation",
                    "attempt": outer_index,
                    "status": status,
                    "seconds": seconds or number(attempt.get("duration_seconds")),
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "restart": int(number(task.get("restart_counts", {}).get("timeout"))),
                }
            )
    for repair in task.get("format_repairs", []):
        nested = repair.get("slm_attempts", [])
        for call in nested:
            seconds, prompt_tokens, output_tokens = slm_values(call.get("slm"))
            rows.append(
                {
                    "phase": phase,
                    "task": task_id,
                    "kind": "format repair",
                    "attempt": f"R{repair.get('attempt_number', '?')}.{call.get('attempt_number', '?')}",
                    "status": call.get("status", repair.get("status", "UNKNOWN")),
                    "seconds": seconds or number(call.get("duration_seconds")),
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "restart": 1 if call.get("model_restart") is not None else 0,
                }
            )
    return rows


def collect_rows(report: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    batch_repairs = 0
    for scenario in report.get("scenarios", []):
        for task in scenario.get("model_generation", {}).get("tasks", []):
            rows.extend(rows_from_task("model", task))
        for batch in scenario.get("intermodel", {}).get("batches", []):
            for task in batch.get("task_reports", []):
                rows.extend(rows_from_task("intermodel", task))
            batch_repairs += len(batch.get("format_repairs", []))
            synthetic = {
                "task_id": f"intermodel-integrator-batch-{batch.get('batch_number', '?')}",
                "status": batch.get("status", "UNKNOWN"),
                "attempts": [],
                "format_repairs": batch.get("format_repairs", []),
            }
            rows.extend(rows_from_task("intermodel", synthetic))
    return rows, batch_repairs


def card(label: str, value: str) -> str:
    return f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'


def bar(value: float, maximum: float, css_class: str) -> str:
    width = 0 if maximum <= 0 else max(1.0, value / maximum * 100)
    return f'<div class="bar-bg"><div class="bar {css_class}" style="width:{width:.2f}%"></div></div>'


def generate_html(report: Mapping[str, Any]) -> str:
    rows, batch_repairs = collect_rows(report)
    total_prompt = sum(row["prompt_tokens"] for row in rows)
    total_output = sum(row["output_tokens"] for row in rows)
    total_restarts = sum(row["restart"] for row in rows)
    repair_calls = sum(1 for row in rows if row["kind"] == "format repair")
    max_seconds = max((row["seconds"] for row in rows), default=0.0)
    max_tokens = max((row["prompt_tokens"] + row["output_tokens"] for row in rows), default=0)
    body_rows: list[str] = []
    for row in rows:
        tokens = row["prompt_tokens"] + row["output_tokens"]
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(row['phase'])}</td>"
            f"<td class='task'>{html.escape(row['task'])}</td>"
            f"<td>{html.escape(row['kind'])}</td>"
            f"<td>{html.escape(str(row['attempt']))}</td>"
            f"<td><span class='status'>{html.escape(str(row['status']))}</span></td>"
            f"<td>{row['seconds']:.2f}{bar(row['seconds'], max_seconds, 'time')}</td>"
            f"<td>{row['prompt_tokens']:,}</td>"
            f"<td>{row['output_tokens']:,}</td>"
            f"<td>{tokens:,}{bar(tokens, max_tokens, 'tokens')}</td>"
            f"<td>{row['restart']}</td>"
            "</tr>"
        )
    scenario_names = ", ".join(
        str(item.get("scenario_name", "unknown")) for item in report.get("scenarios", [])
    ) or "–"
    cards = "".join(
        [
            card("Status", str(report.get("status", "UNKNOWN"))),
            card("Laufzeit", f"{number(report.get('duration_seconds')):.1f} s"),
            card("SLM-Aufrufe", str(len(rows))),
            card("Prompt-Tokens", f"{total_prompt:,}"),
            card("Output-Tokens", f"{total_output:,}"),
            card("Tokens gesamt", f"{total_prompt + total_output:,}"),
            card("Timeout-Neustarts", str(total_restarts)),
            card("Format-Reparaturaufrufe", str(repair_calls)),
        ]
    )
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>4EM Pipeline – Runtime-Visualisierung</title>
<style>
:root{{--bg:#0f172a;--panel:#172033;--line:#2d3a52;--text:#e5edf8;--muted:#9fb0c8;--blue:#38bdf8;--violet:#a78bfa}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}} h1{{margin:0 0 6px}} .meta{{color:var(--muted);margin-bottom:22px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin-bottom:24px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}} .card span{{display:block;color:var(--muted);font-size:12px}} .card strong{{display:block;font-size:22px;margin-top:5px}}
.table-wrap{{overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}} table{{border-collapse:collapse;width:100%;min-width:1150px}}
th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{position:sticky;top:0;background:#202b40}} .task{{max-width:430px;word-break:break-word}}
.bar-bg{{height:4px;background:#2a354a;margin-top:5px;border-radius:4px}} .bar{{height:100%;border-radius:4px}} .time{{background:var(--blue)}} .tokens{{background:var(--violet)}}
.status{{font-size:11px;padding:3px 6px;border:1px solid var(--line);border-radius:12px}} footer{{color:var(--muted);margin-top:15px}}
</style></head><body><main>
<h1>4EM Pipeline – Runtime-Visualisierung</h1>
<div class="meta">Run-ID: {html.escape(str(report.get('run_id', '–')))} · Modus: {html.escape(str(report.get('mode', '–')))} · Szenario: {html.escape(scenario_names)}</div>
<section class="cards">{cards}</section>
<div class="table-wrap"><table><thead><tr><th>Phase</th><th>SLM-Aufgabe</th><th>Art</th><th>Versuch</th><th>Status</th><th>Zeit (s)</th><th>Prompt</th><th>Output</th><th>Tokens gesamt</th><th>Neustart</th></tr></thead>
<tbody>{''.join(body_rows) if body_rows else '<tr><td colspan="10">Keine SLM-Aufrufe im Report.</td></tr>'}</tbody></table></div>
<footer>Erzeugt aus dem Runtime-JSON. Format-Reparaturbatches: {batch_repairs}.</footer>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_json", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate_html(report), encoding="utf-8")
    print(f"Visualization created: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
