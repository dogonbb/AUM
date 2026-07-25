#!/usr/bin/env python3
# actors_resources_to_adl.py
#
# Wandelt die LLM-Notation aus dem Actors-and-Resources-Model-Prompt
# in eine ADL-Datei fuer ein 4EM Actors and Resources Model um.
#
# Unterstuetzte direkte Verbindungen:
#   A relation B
#   A plays B
#   A works in B
#   A works at B
#   A supplies B
#   A interacts with B
#   A navigates B
#   A responsible for B
#   A belongs to B
#   A maintains B
#
# Unterstuetzte Connector-Verbindungen:
#   A, B, C Partial-ISA D
#   A, B, C Total-ISA D
#   A, B, C Partial-PartOF D
#   A, B, C Total-PartOF D
#
# Aufruf:
# python actors_resources_to_adl.py input.txt output.adl --model-name "ActorsAndResourcesModel"

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ELEMENT_TYPES = {"Individual", "Role", "Resource", "Organizational Unit"}
DIRECT_CONNECTION_TYPES = {
    "relation",
    "plays",
    "works in",
    "works at",
    "supplies",
    "interacts with",
    "navigates",
    "responsible for",
    "belongs to",
    "maintains",
}
CONNECTOR_CONNECTION_TYPES = {"Partial-ISA", "Total-ISA", "Partial-PartOF", "Total-PartOF"}
CONNECTION_TYPES = DIRECT_CONNECTION_TYPES | CONNECTOR_CONNECTION_TYPES

# Laengere Tokens muessen zuerst stehen, weil Elementnamen Leerzeichen enthalten duerfen.
CONNECTION_TOKEN_RE = re.compile(
    r"\s("
    r"responsible\s+for"
    r"|interacts\s+with"
    r"|Partial-PartOF"
    r"|Total-PartOF"
    r"|Partial-ISA"
    r"|Total-ISA"
    r"|belongs\s+to"
    r"|works\s+in"
    r"|works\s+at"
    r"|relation"
    r"|supplies"
    r"|navigates"
    r"|maintains"
    r"|plays"
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
    # Wichtig: "Organizational Unit" ist ein Elementtyp mit Leerzeichen.
    for etype in sorted(ELEMENT_TYPES, key=len, reverse=True):
        prefix = etype + " "
        if line == etype or line.startswith(prefix):
            name = normalize_spaces(line[len(etype):])
            if not name:
                raise ValueError(f"Element ohne Namen: {line!r}")
            return Element(etype, name)

    raise ValueError(
        f"Unbekannter Elementtyp in Zeile {line!r}. "
        f"Erlaubt: {', '.join(sorted(ELEMENT_TYPES))}"
    )


def normalize_connection_kind(raw_kind: str) -> str:
    compact = normalize_spaces(raw_kind)
    lower = compact.lower()

    mapping = {
        "relation": "relation",
        "plays": "plays",
        "works in": "works in",
        "works at": "works at",
        "supplies": "supplies",
        "interacts with": "interacts with",
        "navigates": "navigates",
        "responsible for": "responsible for",
        "belongs to": "belongs to",
        "maintains": "maintains",
        "partial-isa": "Partial-ISA",
        "total-isa": "Total-ISA",
        "partial-partof": "Partial-PartOF",
        "total-partof": "Total-PartOF",
    }
    if lower in mapping:
        return mapping[lower]
    raise ValueError(f"Unbekannte Verbindungsart: {raw_kind!r}")


def validate_allowed_pattern(kind: str, source_type: str, target_type: str, line: str) -> None:
    """Validiert die im Prompt erlaubten Quell-/Ziel-Typen."""
    if kind == "relation":
        allowed = (source_type, target_type) in {
            ("Individual", "Individual"),
            ("Individual", "Role"),
            ("Individual", "Resource"),
            ("Individual", "Organizational Unit"),
            ("Role", "Resource"),
            ("Role", "Organizational Unit"),
            ("Resource", "Resource"),
            ("Resource", "Organizational Unit"),
            ("Resource", "Role"),
            ("Organizational Unit", "Organizational Unit"),
            ("Organizational Unit", "Resource"),
        }
    elif kind == "plays":
        allowed = (source_type, target_type) == ("Individual", "Role")
    elif kind == "works in":
        allowed = (source_type, target_type) == ("Role", "Organizational Unit")
    elif kind == "works at":
        allowed = (source_type, target_type) == ("Role", "Organizational Unit")
    elif kind == "supplies":
        allowed = (source_type, target_type) == ("Role", "Organizational Unit")
    elif kind == "interacts with":
        allowed = (source_type, target_type) == ("Resource", "Resource")
    elif kind == "navigates":
        allowed = (source_type, target_type) == ("Organizational Unit", "Resource")
    elif kind == "responsible for":
        allowed = (source_type, target_type) in {
            ("Role", "Resource"),
            ("Organizational Unit", "Resource"),
        }
    elif kind == "belongs to":
        allowed = (source_type, target_type) == ("Resource", "Organizational Unit")
    elif kind == "maintains":
        allowed = (source_type, target_type) == ("Role", "Resource")
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
    for source in sources:
        validate_allowed_pattern(kind, elements_by_name[source].type, elements_by_name[target].type, line)

    return Connection(sources, kind, target)


def parse_notation(text: str) -> Tuple[List[Element], List[Connection]]:
    element_lines, connection_lines = split_sections(text)
    elements = [parse_element_line(line) for line in element_lines]

    names = [element.name for element in elements]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Doppelte Elementnamen gefunden: {', '.join(duplicates)}")

    elements_by_name = {element.name: element for element in elements}
    connections = [parse_connection_line(line, elements_by_name) for line in connection_lines]
    return elements, connections


def adl_attributes_for(element: Element) -> str:
    """ADL-Attribute fuer Actors-and-Resources-Model-Elemente gemaess Beispieldateien."""
    if element.type == "Individual":
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
'''

    if element.type == "Role":
        return '''
\tATTRIBUTE <External tool coupling>
\tVALUE ""

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Intermodel-Relations>
\tVALUE

\tATTRIBUTE <Decomposition>
\tVALUE ""

\tATTRIBUTE <Qualification>
\tVALUE ""

\tATTRIBUTE <Number of Employees with this Role>
\tVALUE 0

\tATTRIBUTE <Attributes>
\tVALUE
'''

    if element.type == "Resource":
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

    if element.type == "Organizational Unit":
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

\tATTRIBUTE <Attributes>
\tVALUE
'''

    raise ValueError(f"Nicht unterstuetzter Elementtyp: {element.type}")


def layout_position(index: int) -> Tuple[float, float]:
    col = index // 10
    row = index % 10
    return 3.5 + col * 8.0, 3.5 + row * 2.5


def compute_layout(elements: List[Element], connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
    """Einfaches Layout: Quellen links, Ziele schrittweise rechts; nach Typ sortiert."""
    if not elements:
        return {}

    input_order = {element.name: i for i, element in enumerate(elements)}
    base_layer_by_type = {
        "Individual": 0,
        "Role": 1,
        "Resource": 2,
        "Organizational Unit": 3,
    }
    layer: Dict[str, int] = {element.name: base_layer_by_type[element.type] for element in elements}

    for _ in range(len(elements)):
        changed = False
        for conn in connections:
            for source in conn.sources:
                wanted = layer[source] + 1
                if layer[conn.target] < wanted:
                    layer[conn.target] = wanted
                    changed = True
        if not changed:
            break

    min_layer = min(layer.values())
    layer = {name: value - min_layer for name, value in layer.items()}

    layers: Dict[int, List[str]] = {}
    for element in elements:
        layers.setdefault(layer[element.name], []).append(element.name)

    for names in layers.values():
        names.sort(key=lambda name: (elements_by_type_order(elements, name), input_order[name]))

    positions: Dict[str, Tuple[float, float]] = {}
    for layer_no, names in layers.items():
        for row, name in enumerate(names):
            positions[name] = (3.5 + layer_no * 8.0, 3.5 + row * 2.5)

    return positions


def elements_by_type_order(elements: List[Element], name: str) -> int:
    order = {"Individual": 0, "Role": 1, "Resource": 2, "Organizational Unit": 3}
    for element in elements:
        if element.name == name:
            return order[element.type]
    return 99


def junction_position(
    sources: List[str],
    target: str,
    positions: Dict[str, Tuple[float, float]],
    fallback_index: int,
) -> Tuple[float, float]:
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
    return tx + (sx - tx) * 0.55, ty + (sy - ty) * 0.55


def node_size_for(element_type: str) -> Tuple[float, float]:
    return 4.0, 2.0


def connector_class_for(conn: Connection) -> str:
    if conn.kind in CONNECTOR_CONNECTION_TYPES:
        return conn.kind
    raise ValueError(f"Verbindung {conn.kind!r} ist kein Connector.")


def relation_type_for(kind: str) -> str:
    # In 4EM-ADL ist die generische Beziehung "relation" als leerer Type gespeichert.
    if kind == "relation":
        return ""
    return kind


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
        if conn.kind in CONNECTOR_CONNECTION_TYPES:
            # A, B, C Partial-ISA D -> A -> Connector, B -> Connector, C -> Connector, Connector -> D
            junction_class = connector_class_for(conn)
            junction_name = f"{junction_class}-AUTO{c_idx}"
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

    type_counts = {element_type: 0 for element_type in ELEMENT_TYPES}
    for element in elements:
        type_counts[element.type] += 1

    direct_counts = {kind: 0 for kind in DIRECT_CONNECTION_TYPES}
    for conn in connections:
        if conn.kind in DIRECT_CONNECTION_TYPES:
            direct_counts[conn.kind] += len(conn.sources)

    relation_count = sum(
        len(conn.sources) + 1 if conn.kind in CONNECTOR_CONNECTION_TYPES else len(conn.sources)
        for conn in connections
    )
    other_count = direct_counts["relation"] + sum(
        len(conn.sources) + 1 for conn in connections if conn.kind in CONNECTOR_CONNECTION_TYPES
    )

    header = f'''///////////////////////////////////////////////////////////////
//
// Date: {now.strftime("%d.%m.%Y  %H:%M")}
//
// Generated by actors_resources_to_adl.py
// Data version 4.0
//
///////////////////////////////////////////////////////////////
//
// The file contains the following models:
//
// {esc(model_name)} (Actors and Resources Model)
//
//////////////////////////////////////////////////////////////

VERSION <4.0>


BUSINESS PROCESS MODEL <{esc(model_name)}> : <4EM current>
VERSION <>
TYPE <Actors and Resources Model>

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

\tATTRIBUTE <Iterator: Individual>
\tVALUE {type_counts["Individual"]}

\tATTRIBUTE <Iterator: Role>
\tVALUE {type_counts["Role"]}

\tATTRIBUTE <Iterator: Resource>
\tVALUE {type_counts["Resource"]}

\tATTRIBUTE <Iterator: Organizational Unit>
\tVALUE {type_counts["Organizational Unit"]}

\tATTRIBUTE <Iterator: Relation>
\tVALUE {relation_count}

\tATTRIBUTE <Iterator: Others>
\tVALUE {other_count}

\tATTRIBUTE <Iterator: Plays>
\tVALUE {direct_counts["plays"]}

\tATTRIBUTE <Iterator: works in>
\tVALUE {direct_counts["works in"]}

\tATTRIBUTE <Iterator: works at>
\tVALUE {direct_counts["works at"]}

\tATTRIBUTE <Iterator: Supplies>
\tVALUE {direct_counts["supplies"]}

\tATTRIBUTE <Iterator: interacts with>
\tVALUE {direct_counts["interacts with"]}

\tATTRIBUTE <Iterator: Navigates>
\tVALUE {direct_counts["navigates"]}

\tATTRIBUTE <Iterator: responsible for>
\tVALUE {direct_counts["responsible for"]}

\tATTRIBUTE <Iterator: Maintains>
\tVALUE {direct_counts["maintains"]}

\tATTRIBUTE <Iterator: belongs to>
\tVALUE {direct_counts["belongs to"]}

\tATTRIBUTE <Iterator: has requirement>
\tVALUE 0

\tATTRIBUTE <Iterator: has goal>
\tVALUE 0

'''

    return header + "\n".join(instance_blocks) + "\n" + "\n".join(relation_blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Konvertiert Actors-and-Resources-Model-Notation mit direkten Verbindungen und 4EM-Connectoren in eine ADL-Datei."
    )
    parser.add_argument("input", help="Textdatei mit ELEMENTS und CONNECTIONS")
    parser.add_argument("output", help="Ausgabedatei .adl")
    parser.add_argument("--model-name", default="Generated Actors and Resources Model")
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