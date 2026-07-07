#!/usr/bin/env python3
# concept_to_adl.py
#
# Wandelt die LLM-Notation aus dem Concepts-Model-Prompt in eine ADL-Datei
# fuer ein 4EM Concepts Model um.
#
# Unterstuetzte direkte Verbindungen:
#   A 1:1 B
#   A 1:n B
#   A n:m B
#   A has attribute B
#
# Unterstuetzte Connector-Verbindungen:
#   A, B, C Partial-ISA D
#   A, B, C Total-ISA D
#   A, B, C Partial-PartOF D
#   A, B, C Total-PartOF D
#
# Aufruf:
# python concept_to_adl.py input.txt output.adl --model-name "ConceptModel"

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ELEMENT_TYPES = {"Concept", "Attribute"}
DIRECT_CONNECTION_TYPES = {"1:1", "1:n", "n:m", "has attribute"}
CONNECTOR_CONNECTION_TYPES = {"Partial-ISA", "Total-ISA", "Partial-PartOF", "Total-PartOF"}
CONNECTION_TYPES = DIRECT_CONNECTION_TYPES | CONNECTOR_CONNECTION_TYPES

# Laengere Tokens muessen vor kuerzeren Tokens gesucht werden,
# weil Elementnamen Leerzeichen enthalten duerfen.
CONNECTION_TOKEN_RE = re.compile(
    r"\s("
    r"has\s+attribute"
    r"|Partial-PartOF"
    r"|Total-PartOF"
    r"|Partial-ISA"
    r"|Total-ISA"
    r"|1\s*:\s*1"
    r"|1\s*:\s*n"
    r"|n\s*:\s*m"
    r")\s",
    re.IGNORECASE,
)


@dataclass
class Element:
    type: str
    name: str


@dataclass
class Connection:
    sources: List[str]
    kind: str
    target: str


def esc(value: str) -> str:
    """Escaping fuer ADL-Werte in spitzen Klammern und Quotes."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("<", "(").replace(">", ")")


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


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
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Ungueltige Element-Zeile: {line!r}")

    etype, name = parts[0], normalize_spaces(parts[1])
    if etype not in ELEMENT_TYPES:
        raise ValueError(
            f"Unbekannter Elementtyp {etype!r} in Zeile {line!r}. "
            f"Erlaubt: {', '.join(sorted(ELEMENT_TYPES))}"
        )
    return Element(etype, name)


def normalize_connection_kind(raw_kind: str) -> str:
    """Normalisiert Schreibvarianten der Verbindungstypen aus dem Prompt."""
    compact = normalize_spaces(raw_kind)
    lower = compact.lower().replace(" ", "")

    if lower == "1:1":
        return "1:1"
    if lower == "1:n":
        return "1:n"
    if lower == "n:m":
        return "n:m"
    if lower == "hasattribute":
        return "has attribute"
    if lower == "partial-isa":
        return "Partial-ISA"
    if lower == "total-isa":
        return "Total-ISA"
    if lower == "partial-partof":
        return "Partial-PartOF"
    if lower == "total-partof":
        return "Total-PartOF"

    raise ValueError(f"Unbekannte Verbindungsart: {raw_kind!r}")


def validate_allowed_pattern(kind: str, source_type: str, target_type: str, line: str) -> None:
    """Validiert die im Prompt erlaubten Quell-/Ziel-Typen."""
    if kind in {"1:1", "1:n", "n:m"}:
        allowed = (source_type, target_type) == ("Concept", "Concept")
    elif kind == "has attribute":
        allowed = (source_type, target_type) == ("Concept", "Attribute")
    elif kind in CONNECTOR_CONNECTION_TYPES:
        allowed = source_type in ELEMENT_TYPES and target_type in ELEMENT_TYPES
    else:
        allowed = False

    if not allowed:
        raise ValueError(
            f"Nicht erlaubtes Verbindungsmuster in {line!r}: "
            f"{source_type} {kind} {target_type}"
        )


def parse_connection_line(line: str, elements_by_name: Dict[str, Element]) -> Connection:
    """
    Erwartete Syntax:
      A 1:1 B
      A 1:n B
      A n:m B
      A has attribute B
      A, B, C Partial-ISA D
      A, B, C Total-ISA D
      A, B, C Partial-PartOF D
      A, B, C Total-PartOF D

    Da Elementnamen Leerzeichen enthalten duerfen, wird anhand der bekannten
    Verbindungstokens geparst.
    """
    match = CONNECTION_TOKEN_RE.search(line)
    if not match:
        allowed = ", ".join(sorted(CONNECTION_TYPES))
        raise ValueError(f"Keine gueltige Verbindungsart gefunden in: {line!r}. Erlaubt: {allowed}")

    kind = normalize_connection_kind(match.group(1))
    left = normalize_spaces(line[:match.start()])
    target = normalize_spaces(line[match.end():])

    sources = [normalize_spaces(x) for x in left.split(",") if normalize_spaces(x)]
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
    """ADL-Attribute fuer Concepts-Model-Elemente gemaess den Beispieldateien."""
    if element.type == "Concept":
        return '''
\tATTRIBUTE <External tool coupling>
\tVALUE ""

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Decomposition>
\tVALUE ""

\tATTRIBUTE <Complexity>
\tVALUE 0

\tATTRIBUTE <Execution Time>
\tVALUE 0

\tATTRIBUTE <Attributes>
\tVALUE

\tATTRIBUTE <Intermodel-Relations>
\tVALUE
'''

    if element.type == "Attribute":
        return '''
\tATTRIBUTE <External tool coupling>
\tVALUE ""

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Intermodel-Relations>
\tVALUE

\tATTRIBUTE <Decomposition>
\tVALUE ""

\tATTRIBUTE <Attributes>
\tVALUE

\tATTRIBUTE <Data Type>
\tVALUE "String"

\tATTRIBUTE <Value Range>
\tVALUE ""
'''

    raise ValueError(f"Nicht unterstuetzter Elementtyp: {element.type}")


def layout_position(index: int) -> Tuple[float, float]:
    """Fallback-Layout in Spalten."""
    col = index // 10
    row = index % 10
    x = 3.0 + col * 9.5
    y = 2.5 + row * 2.2
    return x, y


def compute_layout(elements: List[Element], connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
    """Berechnet ein einfaches, strukturorientiertes Layout."""
    if not elements:
        return {}

    input_order = {element.name: i for i, element in enumerate(elements)}
    layer: Dict[str, int] = {element.name: 0 for element in elements}

    for _ in range(len(elements)):
        changed = False
        for conn in connections:
            # Quellen werden links vom Ziel platziert.
            wanted_target = max(layer[source] + 1 for source in conn.sources)
            if layer[conn.target] < wanted_target:
                layer[conn.target] = wanted_target
                changed = True
        if not changed:
            break

    min_layer = min(layer.values())
    if min_layer != 0:
        layer = {name: value - min_layer for name, value in layer.items()}

    adjacency: Dict[str, Dict[str, float]] = {element.name: {} for element in elements}

    def add_edge(a: str, b: str, weight: float) -> None:
        adjacency[a][b] = adjacency[a].get(b, 0.0) + weight
        adjacency[b][a] = adjacency[b].get(a, 0.0) + weight

    for conn in connections:
        weight = 4.0 if conn.kind in CONNECTOR_CONNECTION_TYPES else 2.5
        for source in conn.sources:
            add_edge(source, conn.target, weight)
        if len(conn.sources) > 1:
            for i, a in enumerate(conn.sources):
                for b in conn.sources[i + 1:]:
                    add_edge(a, b, weight * 0.75)

    component_id: Dict[str, int] = {}
    next_component_id = 0
    for element in elements:
        if element.name in component_id:
            continue
        stack = [element.name]
        component_id[element.name] = next_component_id
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in component_id:
                    component_id[neighbor] = next_component_id
                    stack.append(neighbor)
        next_component_id += 1

    layers: Dict[int, List[str]] = {}
    for element in elements:
        layers.setdefault(layer[element.name], []).append(element.name)

    for names in layers.values():
        names.sort(key=lambda name: (component_id[name], input_order[name]))

    order_value: Dict[str, float] = {}
    for names in layers.values():
        for row, name in enumerate(names):
            order_value[name] = float(row)

    for _ in range(10):
        for layer_no in sorted(layers):
            names = layers[layer_no]

            def sort_key(name: str) -> Tuple[int, float, int]:
                weighted_sum = 0.0
                total_weight = 0.0
                for neighbor, weight in adjacency[name].items():
                    factor = 1.0 if layer[neighbor] != layer[name] else 0.35
                    weighted_sum += order_value.get(neighbor, 0.0) * weight * factor
                    total_weight += weight * factor

                barycenter = weighted_sum / total_weight if total_weight else order_value[name]
                return (component_id[name], barycenter, input_order[name])

            names.sort(key=sort_key)
            for row, name in enumerate(names):
                order_value[name] = float(row)

    x_start = 3.0
    y_start = 2.5
    x_gap = 10.5
    y_gap = 2.4

    positions: Dict[str, Tuple[float, float]] = {}
    for layer_no, names in layers.items():
        for row, name in enumerate(names):
            positions[name] = (x_start + layer_no * x_gap, y_start + row * y_gap)

    return positions


def junction_position(
    sources: List[str],
    target: str,
    positions: Dict[str, Tuple[float, float]],
    fallback_index: int,
) -> Tuple[float, float]:
    """Positioniert Connector-Hilfsknoten sichtbar zwischen Quellen und Ziel."""
    source_points = [positions[name] for name in sources if name in positions]
    target_point = positions.get(target)
    if not source_points or target_point is None:
        points = source_points + ([target_point] if target_point is not None else [])
        if not points:
            return layout_position(fallback_index)
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    sx = sum(point[0] for point in source_points) / len(source_points)
    sy = sum(point[1] for point in source_points) / len(source_points)
    tx, ty = target_point

    x = tx + (sx - tx) * 0.55
    y = ty + (sy - ty) * 0.55
    return x, y


def node_size_for(element_type: str) -> Tuple[float, float]:
    if element_type == "Attribute":
        return 5.0, 1.2
    return 5.0, 1.8


def is_connector_connection(conn: Connection) -> bool:
    return conn.kind in CONNECTOR_CONNECTION_TYPES


def connector_class_for(conn: Connection) -> str:
    if conn.kind in CONNECTOR_CONNECTION_TYPES:
        return conn.kind
    raise ValueError(f"Verbindung {conn.kind!r} ist kein Connector.")


def relation_type_for(kind: str) -> str:
    mapping = {
        "1:1": "1:1",
        "1:n": "1:n",
        "n:m": "n:m",
        # Im Concepts-Model-Beispiel ist die Concept-Attribute-Kante ohne Type-Wert.
        "has attribute": "",
    }
    return mapping[kind]


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


def generate_adl(elements: List[Element], connections: List[Connection], model_name: str) -> str:
    now = datetime.now()
    created = now.strftime("%d.%m.%Y, %H:%M")
    changed = now.strftime("%d.%m.%Y, %H:%M:%S")

    class_by_name: Dict[str, str] = {}
    instance_blocks: List[str] = []
    positions = compute_layout(elements, connections)

    node_index = 1
    for index, element in enumerate(elements):
        adl_class = element.type
        attrs = adl_attributes_for(element)
        class_by_name[element.name] = adl_class

        x, y = positions.get(element.name, layout_position(index))
        w, h = node_size_for(element.type)
        instance_blocks.append(make_instance(element.name, adl_class, attrs, node_index, x, y, w, h))
        node_index += 1

    relation_blocks: List[str] = []
    edge_index = 1

    for c_idx, conn in enumerate(connections, start=1):
        if is_connector_connection(conn):
            # Connector-Zeilen werden als echte 4EM-Connector-Knoten erzeugt:
            # A, B, C Partial-ISA D -> A -> Partial-ISA, B -> Partial-ISA, C -> Partial-ISA, Partial-ISA -> D
            junction_class = connector_class_for(conn)
            junction_name = f"{junction_class}-AUTO{c_idx}"
            x, y = junction_position(conn.sources, conn.target, positions, len(elements) + c_idx)
            instance_blocks.append(
                make_instance(
                    junction_name,
                    junction_class,
                    "\n\tATTRIBUTE <External tool coupling>\n\tVALUE \"\"\n",
                    node_index,
                    x,
                    y,
                )
            )
            node_index += 1
            class_by_name[junction_name] = junction_class

            for source in conn.sources:
                relation_blocks.append(
                    make_relation(source, class_by_name[source], junction_name, junction_class, edge_index, "")
                )
                edge_index += 1

            relation_blocks.append(
                make_relation(junction_name, junction_class, conn.target, class_by_name[conn.target], edge_index, "")
            )
            edge_index += 1

        elif conn.kind in DIRECT_CONNECTION_TYPES:
            rel_type = relation_type_for(conn.kind)
            for source in conn.sources:
                relation_blocks.append(
                    make_relation(source, class_by_name[source], conn.target, class_by_name[conn.target], edge_index, rel_type)
                )
                edge_index += 1
        else:
            raise ValueError(f"Nicht unterstuetzte Verbindung: {conn.kind}")

    direct_counts = {kind: 0 for kind in DIRECT_CONNECTION_TYPES}
    for conn in connections:
        if conn.kind in DIRECT_CONNECTION_TYPES:
            direct_counts[conn.kind] += len(conn.sources)

    relation_count = sum(
        len(conn.sources) + 1 if conn.kind in CONNECTOR_CONNECTION_TYPES else len(conn.sources)
        for conn in connections
    )
    others_count = direct_counts["has attribute"] + sum(
        len(conn.sources) + 1 for conn in connections if conn.kind in CONNECTOR_CONNECTION_TYPES
    )

    header = f'''///////////////////////////////////////////////////////////////
//
// Date: {now.strftime("%d.%m.%Y  %H:%M")}
//
// Generated by concept_to_adl.py
// Data version 4.0
//
///////////////////////////////////////////////////////////////
//
// The file contains the following models:
//
// {esc(model_name)} (Concepts Model)
//
//////////////////////////////////////////////////////////////

VERSION <4.0>


BUSINESS PROCESS MODEL <{esc(model_name)}> : <4EM current>
VERSION <>
TYPE <Concepts Model>

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
\tVALUE 0

\tATTRIBUTE <Iterator: Goal>
\tVALUE 0

\tATTRIBUTE <Iterator: Cause>
\tVALUE 0

\tATTRIBUTE <Iterator: Constraint>
\tVALUE 0

\tATTRIBUTE <Iterator: Opportunity>
\tVALUE 0

\tATTRIBUTE <Iterator: Relation>
\tVALUE {relation_count}

\tATTRIBUTE <Iterator: Hinders>
\tVALUE 0

\tATTRIBUTE <Iterator: Supports>
\tVALUE 0

\tATTRIBUTE <Iterator: Contradicts>
\tVALUE 0

\tATTRIBUTE <Iterator: Causes>
\tVALUE 0

\tATTRIBUTE <Iterator: Others>
\tVALUE {others_count}

\tATTRIBUTE <Iterator: 1:1>
\tVALUE {direct_counts["1:1"]}

\tATTRIBUTE <Iterator: 1:n>
\tVALUE {direct_counts["1:n"]}

\tATTRIBUTE <Iterator: n:m>
\tVALUE {direct_counts["n:m"]}

\tATTRIBUTE <Iterator: Input>
\tVALUE 0

\tATTRIBUTE <Iterator: Output>
\tVALUE 0

\tATTRIBUTE <Iterator: Motivates>
\tVALUE 0

\tATTRIBUTE <Iterator: Requires>
\tVALUE 0

'''

    return header + "\n".join(instance_blocks) + "\n" + "\n".join(relation_blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Konvertiert Concepts-Model-Notation mit direkten Verbindungen und ISA/PartOF-Connectoren in eine 4EM-ADL-Datei."
    )
    parser.add_argument("input", help="Textdatei mit ELEMENTS und CONNECTIONS")
    parser.add_argument("output", help="Ausgabedatei .adl")
    parser.add_argument("--model-name", default="Generated Concepts Model")
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
