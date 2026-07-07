from pathlib import Path
import re
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
REPORT_FILE = BASE_DIR / "output_slm" / "slm_runtime_report.txt"
SCENARIOS_DIR = BASE_DIR / "scenarios"

OUTPUT_PLOT_1 = BASE_DIR / "plot_zeit_zu_woertern.png"
OUTPUT_PLOT_2 = BASE_DIR / "plot_datei_zu_zeit.png"


def parse_duration_to_seconds(text: str) -> float:
    text = text.strip().replace(",", ".")

    hours = 0.0
    minutes = 0.0
    seconds = 0.0

    h_match = re.search(r"(\d+(?:\.\d+)?)\s*Std", text)
    m_match = re.search(r"(\d+(?:\.\d+)?)\s*Min", text)
    s_match = re.search(r"(\d+(?:\.\d+)?)\s*Sek", text)

    if h_match:
        hours = float(h_match.group(1))
    if m_match:
        minutes = float(m_match.group(1))
    if s_match:
        seconds = float(s_match.group(1))

    return hours * 3600 + minutes * 60 + seconds


def count_words(file_path: Path) -> int:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    return len(words)


def get_filename(path_text: str) -> str:
    return path_text.replace("\\", "/").split("/")[-1]


def extract_entries(report_text: str):
    pattern = re.compile(
        r"(?P<number>\d+)\.\s+(?P<title>.+?\.txt)\n"
        r"-+\n"
        r"Status:\s*(?P<status>.+?)\n"
        r"SLM-Output-Datei:\s*(?P<slm_output>.+?)\n"
        r"ADL-Output-Datei:\s*(?P<adl_output>.+?)\n"
        r"SLM-Output-Zeichen:\s*(?P<chars>.+?)\n"
        r"SLM-Zeit:\s*(?P<slm_time>.+?)\n",
        re.DOTALL
    )

    entries = []

    for match in pattern.finditer(report_text):
        slm_output_path = match.group("slm_output").strip()
        slm_output_name = get_filename(slm_output_path)

        scenario_name = slm_output_name.replace("__slm", "")
        slm_time_text = match.group("slm_time").strip()
        slm_seconds = parse_duration_to_seconds(slm_time_text)

        entries.append({
            "file_name": scenario_name,
            "status": match.group("status").strip(),
            "slm_time_text": slm_time_text,
            "slm_seconds": slm_seconds,
            "slm_minutes": slm_seconds / 60.0,
        })

    return entries


def shorten_filename(file_name: str) -> str:
    """
    Kürzt lange Dateinamen für die Plot-Beschriftung.
    Den eigentlichen Dateinamen verändert das nicht.
    """
    name = file_name.replace(".txt", "")
    name = name.replace("controlled_scenario_business_process_", "controlled_")
    name = name.replace("organized_scenario_business_process_", "organized_")
    return name


def main():
    print("Starte Auswertung...")

    if not REPORT_FILE.exists():
        print(f"FEHLER: Report-Datei nicht gefunden: {REPORT_FILE}")
        return

    if not SCENARIOS_DIR.exists():
        print(f"FEHLER: Scenarios-Ordner nicht gefunden: {SCENARIOS_DIR}")
        return

    report_text = REPORT_FILE.read_text(encoding="utf-8", errors="ignore")
    entries = extract_entries(report_text)

    print(f"Gefundene Dateien im Report: {len(entries)}")

    if not entries:
        print("FEHLER: Keine Einträge im Report gefunden.")
        return

    # ------------------------------------------------------------
    # Plot 1: Zeit zu Wörtern
    # X-Achse: SLM-Zeit
    # Y-Achse: Wörter
    # Label: Dateiname
    # ------------------------------------------------------------
    word_plot_data = []

    for entry in entries:
        scenario_file = SCENARIOS_DIR / entry["file_name"]

        if not scenario_file.exists():
            print(f"WARNUNG: Szenario-Datei nicht gefunden: {scenario_file}")
            continue

        word_count = count_words(scenario_file)

        word_plot_data.append({
            "file_name": entry["file_name"],
            "label": shorten_filename(entry["file_name"]),
            "slm_minutes": entry["slm_minutes"],
            "word_count": word_count,
        })

    if word_plot_data:
        x = [item["slm_minutes"] for item in word_plot_data]
        y = [item["word_count"] for item in word_plot_data]

        plt.figure(figsize=(12, 7))
        plt.scatter(x, y)

        for item in word_plot_data:
            plt.annotate(
                item["label"],
                (item["slm_minutes"], item["word_count"]),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points"
            )

        plt.xlabel("SLM-Zeit in Minuten")
        plt.ylabel("Anzahl Wörter")
        plt.title("Zeit zu Wörtern über alle Dateien")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_1, dpi=300)
        plt.close()

        print(f"Plot 1 gespeichert: {OUTPUT_PLOT_1}")
    else:
        print("WARNUNG: Plot 1 wurde nicht erstellt, weil keine Szenario-Dateien gefunden wurden.")

    # ------------------------------------------------------------
    # Plot 2: Datei zu Zeit
    # X-Achse: Dateiname
    # Y-Achse: SLM-Zeit
    # ------------------------------------------------------------
    file_labels = [shorten_filename(entry["file_name"]) for entry in entries]
    run_times = [entry["slm_minutes"] for entry in entries]

    plt.figure(figsize=(14, 7))
    plt.plot(file_labels, run_times, marker="o")

    for file_label, run_time in zip(file_labels, run_times):
        plt.annotate(
            f"{run_time:.2f}",
            (file_label, run_time),
            fontsize=8,
            xytext=(0, 5),
            textcoords="offset points",
            ha="center"
        )

    plt.xlabel("Datei")
    plt.ylabel("SLM-Zeit in Minuten")
    plt.title("SLM-Zeit pro Datei")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_2, dpi=300)
    plt.close()

    print(f"Plot 2 gespeichert: {OUTPUT_PLOT_2}")
    print("Fertig.")


if __name__ == "__main__":
    main()