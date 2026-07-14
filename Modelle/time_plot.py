from pathlib import Path
import re
import math
import matplotlib.pyplot as plt
import numpy as np


def zeit_zu_sekunden(text: str) -> float:
    """
    Wandelt deutsche Zeitangaben in Sekunden um.

    Beispiele:
    - "1 Std 34 Min 29.82 Sek"
    - "6 Min 44.99 Sek"
    - "3.70 Sekunden"
    """
    text = text.strip().replace(",", ".")

    stunden = 0.0
    minuten = 0.0
    sekunden = 0.0

    match_std = re.search(r"(\d+(?:\.\d+)?)\s*Std", text)
    match_min = re.search(r"(\d+(?:\.\d+)?)\s*Min", text)
    match_sek = re.search(r"(\d+(?:\.\d+)?)\s*Sek", text)

    if match_std:
        stunden = float(match_std.group(1))
    if match_min:
        minuten = float(match_min.group(1))
    if match_sek:
        sekunden = float(match_sek.group(1))

    return stunden * 3600 + minuten * 60 + sekunden


def lese_zeiten(report_datei: Path):
    """
    Liest Ø SLM-Zeit pro Datei und SLM-Gesamtzeit aus dem Report.
    Gibt beide Zeiten in Minuten zurück.
    """
    text = report_datei.read_text(encoding="utf-8", errors="replace")

    durchschnitt = None
    gesamt = None

    match_durchschnitt = re.search(
        r"Ø\s*SLM-Zeit\s*pro\s*Datei:\s*(.+)",
        text
    )

    match_gesamt = re.search(
        r"SLM-Gesamtzeit:\s*(.+)",
        text
    )

    if match_durchschnitt:
        durchschnitt = zeit_zu_sekunden(match_durchschnitt.group(1)) / 60

    if match_gesamt:
        gesamt = zeit_zu_sekunden(match_gesamt.group(1)) / 60

    return durchschnitt, gesamt


def main():
    skript_ordner = Path(__file__).resolve().parent

    ordnernamen = []
    durchschnittswerte = []
    gesamtwerte = []

    for unterordner in sorted(skript_ordner.iterdir()):
        if not unterordner.is_dir():
            continue

        report_datei = unterordner / "slm_runtime_report.txt"

        ordnernamen.append(unterordner.name)

        if report_datei.exists():
            durchschnitt, gesamt = lese_zeiten(report_datei)
            durchschnittswerte.append(durchschnitt)
            gesamtwerte.append(gesamt)
        else:
            # NaN erzeugt keinen Balken, der Ordner bleibt aber auf der x-Achse.
            durchschnittswerte.append(math.nan)
            gesamtwerte.append(math.nan)

    if not ordnernamen:
        print("Keine Unterordner gefunden.")
        return

    x = np.arange(len(ordnernamen))
    breite = 0.38

    plt.figure(figsize=(max(10, len(ordnernamen) * 1.2), 6))

    plt.bar(
        x - breite / 2,
        durchschnittswerte,
        width=breite,
        label="Ø SLM-Zeit pro Datei"
    )

    plt.bar(
        x + breite / 2,
        gesamtwerte,
        width=breite,
        label="SLM-Gesamtzeit"
    )

    plt.xlabel("Ordnername")
    plt.ylabel("Zeit in Minuten")
    plt.title("SLM-Zeiten pro Ordner")
    plt.xticks(x, ordnernamen, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    ausgabe_datei = skript_ordner / "slm_zeiten_balkendiagramm.png"
    plt.savefig(ausgabe_datei, dpi=300)
    plt.show()

    print(f"Diagramm gespeichert unter: {ausgabe_datei}")


if __name__ == "__main__":
    main()