# Evaluation

Dieser Ordner enthält die Rohdaten und Auswertungsdateien der Evaluation der Modellgenerierung und der Intermodellbeziehungen sowie die zugehörigen Laufzeitauswertungen.

## Ordnerstruktur

```text
Evaluation/
├── end_pipeline/
│   ├── model_generierung/
│   │   ├── runs/                         Rohdaten der Modellgenerierung
│   │   ├── Robustheitsauswertung_...csv  Zusammengefasste Robustheitsdaten
│   │   └── *.xlsx                        Manuelle Ergebnisbewertungen
│   └── intermodellbeziehungen/
│       ├── runs/                         Rohdaten der Intermodellgenerierung
│       └── auswertung_dreiklassen/       Detail- und Aggregatdaten der inhaltlichen Bewertung
└── laufzeitauswertung/
    ├── Schrittprotokolle_...csv          Vollständige Laufzeit-Rohdaten, das heißt aggregierte Berichte der SLM-Runs
    ├── modellgenerierung/                Normalisierung der Modellgenerierungslaufzeiten
    ├── intermodellbeziehungen/           Normalisierung der Intermodelllaufzeiten
    └── datenbasis_alle_laufzeiten/       CSV-Grundlagen der Tabellen und Abbildungen im Bericht
```

## Aufbau der Run-Ordner

Die Läufe unter `end_pipeline/*/runs/` sind grundsätzlich nach folgendem Schema abgelegt:

```text
<Modellfamilie>/<Modellgröße>/<Run>/<Szenariotyp>_phase<1-4>/
```

Die Szenariotypen sind `controlled` und `organized`; die Phasen entsprechen den Komplexitätsstufen S1 bis S4. In den jeweiligen Szenarioordnern liegen die Eingaben, erzeugten Modelle beziehungsweise Intermodellbeziehungen sowie Protokolle und Reparaturausgaben. Bei abgebrochenen oder unvollständigen Sonderläufen kann die Struktur abweichen.

Die CSV-Dateien unter `laufzeitauswertung/datenbasis_alle_laufzeiten/` bilden die im Bericht verwendeten Laufzeittabellen und -abbildungen ab. Die fachliche Auswertung der Intermodellbeziehungen befindet sich dagegen unter `end_pipeline/intermodellbeziehungen/auswertung_dreiklassen/`.
