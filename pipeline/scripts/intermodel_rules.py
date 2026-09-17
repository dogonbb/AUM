"""Canonical 4EM intermodel connector and reference rules.

The values are taken from ``AllInterModelConnections.adl`` and the allowed
target classes from ``All_Ref_objects.txt``.  Connector spelling, whitespace,
and capitalization are significant ADL enumeration values.
"""

from __future__ import annotations

import json
from pathlib import Path


EMPTY_CONNECTOR_TOKEN = "empty"

MODEL_TYPE_ALIASES = {
    "Product / Service Model": "Product-Service-Model",
}

MODEL_TYPE_BY_ID = {
    "GoalModel": "Goal Model",
    "BusinessProcessModel": "Business Process Model",
    "ActorsResourcesModel": "Actors and Resources Model",
    "ConceptsModel": "Concepts Model",
    "TechnicalComponentsRequirementsModel": "Technical Components and Requirements Model",
    "ProductServiceModel": "Product-Service-Model",
}

PROMPT_RULES_ROOT = (
    Path(__file__).resolve().parents[1] / "prompts" / "inter_model_generation"
)


def _load_connectors() -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    directory = PROMPT_RULES_ROOT / "source_connection_rules"
    for model_id, model_type in MODEL_TYPE_BY_ID.items():
        text = (directory / f"{model_id}.txt").read_text(encoding="utf-8")
        enumeration = text.split("Connector meanings", 1)[0]
        result[model_type] = tuple(
            line[2:].strip()
            for line in enumeration.splitlines()
            if line.startswith("- ")
        )
    return result


def _load_reference_classes() -> dict[tuple[str, str], tuple[str, ...]]:
    path = Path(__file__).with_name("intermodel_reference_classes.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        (MODEL_TYPE_BY_ID[source_id], MODEL_TYPE_BY_ID[target_id]): tuple(classes)
        for pair, classes in raw.items()
        for source_id, target_id in [pair.split("__", 1)]
    }


CONNECTORS_BY_SOURCE_MODEL = _load_connectors()
REFERENCE_CLASSES = _load_reference_classes()

def adl_connector_value(token: str) -> str:
    return "" if token == EMPTY_CONNECTOR_TOKEN else token


def canonical_model_type(model_type: str) -> str:
    return MODEL_TYPE_ALIASES.get(model_type, model_type)
