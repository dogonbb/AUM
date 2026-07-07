#!/usr/bin/env python3
# tcrm_to_adl.py
#
# Wandelt die LLM-Notation aus dem Prompt fuer das 4EM
# Technical Components and Requirements Model in eine ADL-Datei um.
#
# Erwarteter Input:
#   ELEMENTS
#   Goal <name>
#   Problem <name>
#   IS Technical Component <name>
#   IS Requirement <name>
#
#   CONNECTIONS
#   A supports B
#   A hinders B
#   A contradicts B
#   A has goal B
#   A has requirement B
#   A motivates B
#   A, B, C AND D
#   A, B, C OR D
#   A, B, C AND/OR D
#   A, B, C Partial-PartOF D
#   A, B, C Total-PartOF D
#
# Aufruf:
#   python tcrm_to_adl.py input.txt output.adl --model-name "My TCRM Model"

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ELEMENT_TYPES = {"Goal", "Problem", "IS Technical Component", "IS Requirement"}
DIRECT_CONNECTION_TYPES = {
    "supports",
    "hinders",
    "contradicts",
    "has goal",
    "has requirement",
    "motivates",
}
CONNECTOR_CONNECTION_TYPES = {"AND", "OR", "AND/OR", "Partial-PartOF", "Total-PartOF"}
CONNECTION_TYPES = DIRECT_CONNECTION_TYPES | CONNECTOR_CONNECTION_TYPES

# Laengere Tokens muessen vor kuerzeren Tokens stehen, weil Elementnamen
# Leerzeichen enthalten duerfen und z.B. "AND/OR" vor "AND" erkannt werden muss.
CONNECTION_TOKEN_RE = re.compile(
    r"\s("
    r"Partial-PartOF"
    r"|Total-PartOF"
    r"|AND/OR"
    r"|has\s+requirement"
    r"|has\s+goal"
    r"|contradicts"
    r"|supports"
    r"|motivates"
    r"|hinders"
    r"|AND"
    r"|OR"
    r")\s",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Element:
    type: str
    name: str


@dataclass(frozen=True)
class Connection:
    sources: List[str]
    kind: str
    target: str


def esc(value: str) -> str:
    """Escaping fuer ADL-Werte in spitzen Klammern und Quotes."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("<", "(").replace(">", ")")


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def split_sections(text: str) -> Tuple[List[str], List[str]]:
    section = None
    elements: List[str] = []
    connections: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        upper = line.upper()
        if upper in {"ELEMENTS", "ELEMENTE"}:
            section = "elements"
            continue
        if upper in {"CONNECTIONS", "VERBINDUNGEN"}:
            section = "connections"
            continue

        if section == "elements":
            elements.append(line)
        elif section == "connections":
            connections.append(line)

    return elements, connections


def parse_element_line(line: str) -> Element:
    """
    Elementtypen koennen aus mehreren Woertern bestehen, z.B.
    "IS Technical Component". Deshalb wird gegen bekannte Praefixe geparst.
    """
    normalized = normalize_spaces(line)
    for etype in sorted(ELEMENT_TYPES, key=len, reverse=True):
        prefix = etype + " "
        if normalized.startswith(prefix):
            name = normalize_spaces(normalized[len(prefix):])
            if not name:
                raise ValueError(f"Elementname fehlt in Zeile: {line!r}")
            return Element(etype, name)

    raise ValueError(
        f"Unbekannter Elementtyp in Zeile {line!r}. "
        f"Erlaubt: {', '.join(sorted(ELEMENT_TYPES))}"
    )


def normalize_connection_kind(raw_kind: str) -> str:
    compact = normalize_spaces(raw_kind)
    lower = compact.lower()

    if lower == "supports":
        return "supports"
    if lower == "hinders":
        return "hinders"
    if lower == "contradicts":
        return "contradicts"
    if lower == "has goal":
        return "has goal"
    if lower == "has requirement":
        return "has requirement"
    if lower == "motivates":
        return "motivates"
    if lower == "and":
        return "AND"
    if lower == "or":
        return "OR"
    if lower == "and/or":
        return "AND/OR"
    if lower == "partial-partof":
        return "Partial-PartOF"
    if lower == "total-partof":
        return "Total-PartOF"

    raise ValueError(f"Unbekannte Verbindungsart: {raw_kind!r}")


def validate_allowed_pattern(kind: str, source_type: str, target_type: str, line: str) -> None:
    """Validiert die im Prompt erlaubten Quell-/Ziel-Typen."""
    if kind == "supports":
        allowed = (source_type, target_type) in {
            ("Goal", "Goal"),
            ("IS Technical Component", "IS Technical Component"),
        }
    elif kind == "hinders":
        allowed = (source_type, target_type) in {
            ("Goal", "Goal"),
            ("Problem", "Goal"),
            ("IS Technical Component", "IS Technical Component"),
        }
    elif kind == "contradicts":
        allowed = (source_type, target_type) == ("Goal", "Goal")
    elif kind == "has goal":
        allowed = (source_type, target_type) == ("IS Technical Component", "Goal")
    elif kind == "has requirement":
        allowed = (source_type, target_type) in {
            ("Goal", "IS Technical Component"),
            ("Goal", "IS Requirement"),
            ("IS Technical Component", "IS Requirement"),
        }
    elif kind == "motivates":
        allowed = (source_type, target_type) == ("Goal", "IS Technical Component")
    elif kind in {"AND", "OR", "AND/OR"}:
        allowed = source_type in ELEMENT_TYPES and target_type in ELEMENT_TYPES
    elif kind in {"Partial-PartOF", "Total-PartOF"}:
        allowed = (source_type, target_type) == (
            "IS Technical Component",
            "IS Technical Component",
        )
    else:
        allowed = False

    if not allowed:
        raise ValueError(
            f"Nicht erlaubtes Verbindungsmuster in {line!r}: "
            f"{source_type} {kind} {target_type}"
        )


def parse_connection_line(line: str, elements_by_name: Dict[str, Element]) -> Connection:
    """
    Parst Verbindungen anhand bekannter Verbindungstokens, weil Elementnamen
    Leerzeichen enthalten duerfen.
    """
    match = CONNECTION_TOKEN_RE.search(line)
    if not match:
        allowed = ", ".join(sorted(CONNECTION_TYPES))
        raise ValueError(f"Keine gueltige Verbindungsart gefunden in: {line!r}. Erlaubt: {allowed}")

    kind = normalize_connection_kind(match.group(1))
    left = normalize_spaces(line[:match.start()])
    target = normalize_spaces(line[match.end():])

    sources = [normalize_spaces(part) for part in left.split(",") if normalize_spaces(part)]
    if not sources:
        raise ValueError(f"Keine Quelle in Verbindung: {line!r}")

    for source in sources:
        if source not in elements_by_name:
            raise ValueError(f"Quelle {source!r} wurde nicht unter ELEMENTS definiert.")
    if target not in elements_by_name:
        raise ValueError(f"Ziel {target!r} wurde nicht unter ELEMENTS definiert.")

    if kind in DIRECT_CONNECTION_TYPES and len(sources) != 1:
        raise ValueError(f"Direkte Verbindung {kind!r} darf genau eine Quelle haben: {line!r}")
    if kind in CONNECTOR_CONNECTION_TYPES and len(sources) < 2:
        raise ValueError(f"Connector-Verbindung {kind!r} sollte mindestens zwei Quellen haben: {line!r}")

    for source in sources:
        validate_allowed_pattern(kind, elements_by_name[source].type, elements_by_name[target].type, line)

    return Connection(sources, kind, target)


def parse_notation(text: str) -> Tuple[List[Element], List[Connection]]:
    element_lines, connection_lines = split_sections(text)
    if not element_lines:
        raise ValueError("Keine ELEMENTS-Sektion oder keine Elemente gefunden.")

    elements = [parse_element_line(line) for line in element_lines]
    names = [element.name for element in elements]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Doppelte Elementnamen gefunden: {', '.join(duplicates)}")

    elements_by_name = {element.name: element for element in elements}
    connections = [parse_connection_line(line, elements_by_name) for line in connection_lines]
    return elements, connections


def adl_attributes_for(element: Element) -> str:
    """ADL-Attribute fuer Technical Components and Requirements gemaess ADL-Beispielen."""
    if element.type == "Goal":
        return '''
\tATTRIBUTE <External tool coupling>
\tVALUE ""

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Criticality>
\tVALUE "Low"

\tATTRIBUTE <Priority>
\tVALUE "Low"

\tATTRIBUTE <Intermodel-Relations>
\tVALUE

\tATTRIBUTE <Decomposition>
\tVALUE ""

\tATTRIBUTE <Defined by>
\tVALUE ""

\tATTRIBUTE <Attributes>
\tVALUE
'''

    if element.type == "Problem":
        return '''
\tATTRIBUTE <External tool coupling>
\tVALUE ""

\tATTRIBUTE <Intermodel-Relations>
\tVALUE

\tATTRIBUTE <Decomposition>
\tVALUE ""

\tATTRIBUTE <Attributes>
\tVALUE

\tATTRIBUTE <Defined by>
\tVALUE ""

\tATTRIBUTE <Priority>
\tVALUE "Low"

\tATTRIBUTE <Criticality>
\tVALUE "Low"

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <type>
\tVALUE "Problem"
'''

    if element.type == "IS Technical Component":
        return '''
\tATTRIBUTE <External tool coupling>
\tVALUE ""

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Intermodel-Relations>
\tVALUE

\tATTRIBUTE <Decomposition>
\tVALUE ""

\tATTRIBUTE <Location>
\tVALUE ""

\tATTRIBUTE <Quantity>
\tVALUE 0

\tATTRIBUTE <Attributes>
\tVALUE
'''

    if element.type == "IS Requirement":
        return '''
\tATTRIBUTE <External tool coupling>
\tVALUE ""

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Type>
\tVALUE "Functional"

\tATTRIBUTE <Intermodel-Relations>
\tVALUE

\tATTRIBUTE <Decomposition>
\tVALUE ""

\tATTRIBUTE <Attributes>
\tVALUE
'''

    raise ValueError(f"Nicht unterstuetzter Elementtyp: {element.type}")


def node_size_for(element_type: str) -> Tuple[float, float]:
    if element_type in {"IS Technical Component", "IS Requirement"}:
        return 4.0, 2.0
    return 4.0, 1.5


def fallback_position(index: int) -> Tuple[float, float]:
    col = index // 10
    row = index % 10
    return 2.5 + col * 9.5, 3.0 + row * 2.4


def compute_layout(elements: List[Element], connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
    """Ein einfaches, reproduzierbares Layout mit Einflussquellen links und Zielen rechts."""
    input_order = {element.name: index for index, element in enumerate(elements)}
    base_layer = {
        "Problem": 0,
        "Goal": 1,
        "IS Technical Component": 2,
        "IS Requirement": 3,
    }
    layer: Dict[str, int] = {element.name: base_layer[element.type] for element in elements}

    for _ in range(len(elements)):
        changed = False
        for conn in connections:
            if conn.kind in {"Partial-PartOF", "Total-PartOF"}:
                # Part-of-Beziehungen sollen Komponenten in der Naehe halten.
                continue
            wanted = max(layer[source] for source in conn.sources) + 1
            if layer[conn.target] < wanted:
                layer[conn.target] = wanted
                changed = True
        if not changed:
            break

    min_layer = min(layer.values()) if layer else 0
    layer = {name: value - min_layer for name, value in layer.items()}

    layers: Dict[int, List[str]] = {}
    for element in elements:
        layers.setdefault(layer[element.name], []).append(element.name)

    for names in layers.values():
        names.sort(key=lambda name: input_order[name])

    positions: Dict[str, Tuple[float, float]] = {}
    for layer_no, names in sorted(layers.items()):
        for row, name in enumerate(names):
            positions[name] = (2.5 + layer_no * 8.5, 3.0 + row * 2.7)
    return positions


def junction_position(
    sources: List[str],
    target: str,
    positions: Dict[str, Tuple[float, float]],
    fallback_index: int,
) -> Tuple[float, float]:
    source_points = [positions[name] for name in sources if name in positions]
    target_point = positions.get(target)
    if not source_points or target_point is None:
        return fallback_position(fallback_index)

    sx = sum(point[0] for point in source_points) / len(source_points)
    sy = sum(point[1] for point in source_points) / len(source_points)
    tx, ty = target_point
    return tx + (sx - tx) * 0.55, ty + (sy - ty) * 0.55


def make_instance(
    name: str,
    adl_class: str,
    attrs: str,
    index: int,
    x: float,
    y: float,
    w: float | None = None,
    h: float | None = None,
) -> str:
    if w is None or h is None:
        pos = f'NODE x:{x:g}cm y:{y:g}cm index:{index}'
    else:
        pos = f'NODE x:{x:g}cm y:{y:g}cm w:{w:g}cm h:{h:g}cm index:{index}'

    return f'''INSTANCE <{esc(name)}> : <{esc(adl_class)}>

\tATTRIBUTE <Position>
\tVALUE "{pos}"{attrs}
'''


def make_relation(
    src_name: str,
    src_class: str,
    dst_name: str,
    dst_class: str,
    edge_index: int,
    rel_type: str = "",
) -> str:
    return f'''RELATION <4EM_Relation>
\tFROM <{esc(src_name)}> : <{esc(src_class)}>
\tTO <{esc(dst_name)}> : <{esc(dst_class)}>

\tATTRIBUTE <Positions>
\tVALUE "EDGE 0 index:{edge_index}"

\tATTRIBUTE <Type>
\tVALUE "{esc(rel_type)}"

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <IR>
\tVALUE "False"

'''


def relation_type_for(kind: str, source_type: str, target_type: str) -> str:
    """Relationstexte so nah wie moeglich an den ADL-Beispielen."""
    if kind == "supports":
        if (source_type, target_type) == ("IS Technical Component", "IS Technical Component"):
            return "supports"
        return "Supports"
    if kind == "hinders":
        if (source_type, target_type) == ("IS Technical Component", "IS Technical Component"):
            return "hinders"
        return "Hinders"
    if kind == "contradicts":
        return "Contradicts"
    if kind == "has goal":
        return "has goal"
    if kind == "has requirement":
        return "has requirement"
    if kind == "motivates":
        return "motivates"
    raise ValueError(f"Keine direkte Relation fuer {kind!r}")


def connector_class_for(kind: str) -> str:
    if kind not in CONNECTOR_CONNECTION_TYPES:
        raise ValueError(f"{kind!r} ist kein Connector-Typ.")
    return kind


def connector_name_for(kind: str, index: int) -> str:
    safe = kind.replace("/", "_")
    return f"{safe}-AUTO{index}"


def generate_adl(elements: List[Element], connections: List[Connection], model_name: str) -> str:
    now = datetime.now()
    created = now.strftime("%d.%m.%Y, %H:%M")
    changed = now.strftime("%d.%m.%Y, %H:%M:%S")

    positions = compute_layout(elements, connections)
    class_by_name: Dict[str, str] = {}
    instance_blocks: List[str] = []

    node_index = 1
    for index, element in enumerate(elements):
        adl_class = element.type
        class_by_name[element.name] = adl_class
        x, y = positions.get(element.name, fallback_position(index))
        w, h = node_size_for(element.type)
        instance_blocks.append(
            make_instance(element.name, adl_class, adl_attributes_for(element), node_index, x, y, w, h)
        )
        node_index += 1

    relation_blocks: List[str] = []
    edge_index = 1

    for c_idx, conn in enumerate(connections, start=1):
        if conn.kind in CONNECTOR_CONNECTION_TYPES:
            junction_class = connector_class_for(conn.kind)
            junction_name = connector_name_for(conn.kind, c_idx)
            x, y = junction_position(conn.sources, conn.target, positions, len(elements) + c_idx)
            instance_blocks.append(
                make_instance(
                    junction_name,
                    junction_class,
                    '\n\tATTRIBUTE <External tool coupling>\n\tVALUE ""\n',
                    node_index,
                    x,
                    y,
                )
            )
            class_by_name[junction_name] = junction_class
            node_index += 1

            for source in conn.sources:
                relation_blocks.append(
                    make_relation(source, class_by_name[source], junction_name, junction_class, edge_index, "")
                )
                edge_index += 1

            relation_blocks.append(
                make_relation(junction_name, junction_class, conn.target, class_by_name[conn.target], edge_index, "")
            )
            edge_index += 1
            continue

        for source in conn.sources:
            rel_type = relation_type_for(
                conn.kind,
                source_type=class_by_name[source],
                target_type=class_by_name[conn.target],
            )
            relation_blocks.append(
                make_relation(source, class_by_name[source], conn.target, class_by_name[conn.target], edge_index, rel_type)
            )
            edge_index += 1

    type_counts = {etype: 0 for etype in ELEMENT_TYPES}
    for element in elements:
        type_counts[element.type] += 1

    direct_counts = {kind: 0 for kind in DIRECT_CONNECTION_TYPES}
    for conn in connections:
        if conn.kind in DIRECT_CONNECTION_TYPES:
            direct_counts[conn.kind] += len(conn.sources)

    connector_counts = {kind: 0 for kind in CONNECTOR_CONNECTION_TYPES}
    for conn in connections:
        if conn.kind in CONNECTOR_CONNECTION_TYPES:
            connector_counts[conn.kind] += 1

    relation_count = sum(
        len(conn.sources) + 1 if conn.kind in CONNECTOR_CONNECTION_TYPES else len(conn.sources)
        for conn in connections
    )

    header = f'''///////////////////////////////////////////////////////////////
//
// Date: {now.strftime("%d.%m.%Y  %H:%M")}
//
// Generated by tcrm_to_adl.py
// Data version 4.0
//
///////////////////////////////////////////////////////////////
//
// The file contains the following models:
//
// {esc(model_name)} (Technical Components and Requirements Model)
//
//////////////////////////////////////////////////////////////

VERSION <4.0>


BUSINESS PROCESS MODEL <{esc(model_name)}> : <4EM current>
VERSION <>
TYPE <Technical Components and Requirements Model>

\tATTRIBUTE <Author>
\tVALUE "Admin"

\tATTRIBUTE <Creation date>
\tVALUE "{created}"

\tATTRIBUTE <Date last changed>
\tVALUE "{changed}"

\tATTRIBUTE <Last user>
\tVALUE "Admin"

\tATTRIBUTE <Keywords>
\tVALUE ""

\tATTRIBUTE <Comment>
\tVALUE ""

\tATTRIBUTE <Model type>
\tVALUE "Current model"

\tATTRIBUTE <State>
\tVALUE "In process"

\tATTRIBUTE <Reviewed on>
\tVALUE ""

\tATTRIBUTE <Reviewed by>
\tVALUE ""

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <World area>
\tVALUE "w:80cm h:80cm minw:5cm minh:5cm"

\tATTRIBUTE <Grid>
\tVALUE ""

\tATTRIBUTE <Zoom>
\tVALUE 100

\tATTRIBUTE <Viewable area>
\tVALUE "VIEW representation:graphic
GRAPHIC x:0 y:0 w:1376 h:828 scale:1
TABLE
"

\tATTRIBUTE <Current mode>
\tVALUE ""

\tATTRIBUTE <Access state>
\tVALUE "write"

\tATTRIBUTE <Current page layout>
\tVALUE ""

\tATTRIBUTE <Connector marks>
\tVALUE ""

\tATTRIBUTE <Change counter>
\tVALUE 1

\tATTRIBUTE <Font size>
\tVALUE 0

\tATTRIBUTE <Context of version>
\tVALUE ""

\tATTRIBUTE <Position>
\tVALUE ""

\tATTRIBUTE <External tool coupling>
\tVALUE ""

\tATTRIBUTE <IteratorModeltype>
\tVALUE ""

\tATTRIBUTE <Iterator: Problem>
\tVALUE {type_counts["Problem"]}

\tATTRIBUTE <Iterator: Goal>
\tVALUE {type_counts["Goal"]}

\tATTRIBUTE <Iterator: IS Technical Component>
\tVALUE {type_counts["IS Technical Component"]}

\tATTRIBUTE <Iterator: IS Requirement>
\tVALUE {type_counts["IS Requirement"]}

\tATTRIBUTE <Iterator: Relation>
\tVALUE {relation_count}

\tATTRIBUTE <Iterator: Hinders>
\tVALUE {direct_counts["hinders"]}

\tATTRIBUTE <Iterator: Supports>
\tVALUE {direct_counts["supports"]}

\tATTRIBUTE <Iterator: Contradicts>
\tVALUE {direct_counts["contradicts"]}

\tATTRIBUTE <Iterator: Has goal>
\tVALUE {direct_counts["has goal"]}

\tATTRIBUTE <Iterator: Has requirement>
\tVALUE {direct_counts["has requirement"]}

\tATTRIBUTE <Iterator: Motivates>
\tVALUE {direct_counts["motivates"]}

\tATTRIBUTE <Iterator: AND>
\tVALUE {connector_counts["AND"]}

\tATTRIBUTE <Iterator: OR>
\tVALUE {connector_counts["OR"]}

\tATTRIBUTE <Iterator: AND/OR>
\tVALUE {connector_counts["AND/OR"]}

\tATTRIBUTE <Iterator: Partial-PartOF>
\tVALUE {connector_counts["Partial-PartOF"]}

\tATTRIBUTE <Iterator: Total-PartOF>
\tVALUE {connector_counts["Total-PartOF"]}

'''

    return header + "\n".join(instance_blocks) + "\n" + "\n".join(relation_blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Konvertiert die Prompt-Ausgabe fuer ein 4EM Technical Components "
            "and Requirements Model in eine ADL-Datei."
        )
    )
    parser.add_argument("input", help="Textdatei mit ELEMENTS und CONNECTIONS")
    parser.add_argument("output", help="Ausgabedatei .adl")
    parser.add_argument("--model-name", default="Generated Technical Components and Requirements Model")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    elements, connections = parse_notation(text)
    adl = generate_adl(elements, connections, args.model_name)
    output_path.write_text(adl, encoding="utf-8")

    print(f"OK: {len(elements)} Elemente und {len(connections)} Notations-Verbindungen gelesen.")
    print(f"ADL geschrieben nach: {output_path}")


if __name__ == "__main__":
    main()
