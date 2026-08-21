#!/usr/bin/env python3
# notation_to_adl_for_new_prompt.py
#
# Wandelt die LLM-Notation aus dem neuen Prompt in eine ADL-Datei
# fuer ein 4EM Product-Service-Model um.
#
# Unterstuetzte direkte Verbindungen:
#   A part_of B
#   A is_a B
#   A requires B
#
# Unterstuetzte Connector-Verbindungen:
#   A, B PartOF (AND) C
#   A, B PartOF (OR) C
#   A, B PartOF (XOR) C
#   A, B Total-ISA C
#   A, B Partial-ISA C
#
# Aufruf:
# python notation_to_adl_for_new_prompt.py input.txt output.adl --model-name "HandsOn2"

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


# Der neue Prompt erlaubt Product, Service, Component und Feature.
# ProductService bleibt als Abwaertskompatibilitaet zum alten Prompt erhalten.
ELEMENT_TYPES = {"Product", "Service", "Component", "Feature", "ProductService"}

DIRECT_CONNECTION_TYPES = {"part_of", "requires", "is_a"}
PARTOF_CONNECTOR_TYPES = {"PartOF (AND)", "PartOF (OR)", "PartOF (XOR)"}
ISA_CONNECTOR_TYPES = {"Total-ISA", "Partial-ISA"}
CONNECTOR_CONNECTION_TYPES = PARTOF_CONNECTOR_TYPES | ISA_CONNECTOR_TYPES
CONNECTION_TYPES = DIRECT_CONNECTION_TYPES | CONNECTOR_CONNECTION_TYPES

# Laengere Connector-Tokens muessen vor den kurzen direkten Tokens gesucht werden,
# weil Elementnamen Leerzeichen enthalten duerfen.
CONNECTION_TOKEN_RE = re.compile(
    r"\s("
    r"PartOF\s*\(\s*(?:AND|OR|XOR)\s*\)"
    r"|Total-ISA"
    r"|Partial-ISA"
    r"|part_of"
    r"|requires"
    r"|is_a"
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
        raise ValueError(f"Invalid element line: {line!r}")

    etype, name = parts[0], normalize_spaces(parts[1])
    if etype not in ELEMENT_TYPES:
        raise ValueError(
            f"Unknown element type {etype!r} in line {line!r}. "
            f"Allowed: {', '.join(sorted(ELEMENT_TYPES))}"
        )
    return Element(etype, name)


def normalize_connection_kind(raw_kind: str) -> str:
    """Normalisiert Schreibvarianten der Verbindungstypen aus dem Prompt."""
    compact = normalize_spaces(raw_kind)
    lower = compact.lower()

    if lower == "part_of":
        return "part_of"
    if lower == "requires":
        return "requires"
    if lower == "is_a":
        return "is_a"
    if lower == "total-isa":
        return "Total-ISA"
    if lower == "partial-isa":
        return "Partial-ISA"

    partof_match = re.fullmatch(r"partof\s*\(\s*(and|or|xor)\s*\)", compact, re.IGNORECASE)
    if partof_match:
        return f"PartOF ({partof_match.group(1).upper()})"

    raise ValueError(f"Unknown connection type: {raw_kind!r}")


def parse_connection_line(line: str, element_names: List[str]) -> Connection:
    """
    Erwartete Syntax:
      A part_of C
      A requires C
      A is_a C
      A, B PartOF (AND) C
      A, B PartOF (OR) C
      A, B PartOF (XOR) C
      A, B Total-ISA C
      A, B Partial-ISA C

    Da Elementnamen Leerzeichen enthalten duerfen, wird anhand der bekannten
    Verbindungstokens geparst.
    """
    match = CONNECTION_TOKEN_RE.search(line)
    if not match:
        allowed = ", ".join(sorted(CONNECTION_TYPES))
        raise ValueError(f"No valid connection type found in: {line!r}. Allowed: {allowed}")

    kind = normalize_connection_kind(match.group(1))
    left = normalize_spaces(line[:match.start()])
    target = normalize_spaces(line[match.end():])

    sources = [normalize_spaces(x) for x in left.split(",") if normalize_spaces(x)]
    if not sources:
        raise ValueError(f"Connection has no source: {line!r}")

    known = set(element_names)
    for source in sources:
        if source not in known:
            raise ValueError(f"Source {source!r} is not defined under ELEMENTS.")
    if target not in known:
        raise ValueError(f"Target {target!r} is not defined under ELEMENTS.")

    if kind not in CONNECTION_TYPES:
        raise ValueError(f"Unknown connection type: {kind}")

    return Connection(sources, kind, target)


def parse_notation(text: str) -> Tuple[List[Element], List[Connection]]:
    element_lines, connection_lines = split_sections(text)
    elements = [parse_element_line(line) for line in element_lines]

    names = [element.name for element in elements]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate element names found: {', '.join(duplicates)}")

    connections = [parse_connection_line(line, names) for line in connection_lines]
    return elements, connections


def adl_class_and_attributes(element: Element) -> Tuple[str, str]:
    """
    Mapping der einfachen Elementtypen auf ADL-Klassen.
    Product/ProductService/Service nutzen in den Beispieldateien dieselbe ADL-Klasse:
    <Unspecific/Product/Service>. Der konkrete Typ steht im Attribut <Specification>.
    """
    if element.type in {"ProductService", "Product", "Service"}:
        # 4EM 2.6 akzeptiert bei <Specification> nur Enumerationswerte
        # wie "Product" und "Service". ProductService wird daher als Product gespeichert.
        spec = "Product" if element.type == "ProductService" else element.type
        attrs = f'''
\tATTRIBUTE <Specification>
\tVALUE "{esc(spec)}"

\tATTRIBUTE <Attribute>
\tVALUE

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Intermodel-Relations>
\tVALUE

\tATTRIBUTE <Decomposition>
\tVALUE ""
'''
        return "Unspecific/Product/Service", attrs

    if element.type == "Feature":
        attrs = '''
\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Attribute>
\tVALUE

\tATTRIBUTE <Intermodel-Relations>
\tVALUE

\tATTRIBUTE <Decomposition>
\tVALUE ""
'''
        return "Feature", attrs

    if element.type == "Component":
        attrs = '''
\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Decomposition>
\tVALUE ""

\tATTRIBUTE <Quantity>
\tVALUE 0

\tATTRIBUTE <Location>
\tVALUE ""

\tATTRIBUTE <Intermodel-Relations>
\tVALUE
'''
        return "Component", attrs

    raise ValueError(f"Unsupported element type: {element.type}")


def layout_position(index: int) -> Tuple[float, float]:
    """Fallback-Layout in Spalten."""
    col = index // 10
    row = index % 10
    x = 4.0 + col * 8.0
    y = 3.0 + row * 4.0
    return x, y


def compute_layout(elements: List[Element], connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
    """
    Berechnet ein einfaches, strukturorientiertes Layout.

    Direkte part_of-Verbindungen, PartOF-Connectoren und ISA-Connectoren werden
    als Hierarchie interpretiert: Quellen/Spezialisierungen liegen rechts vom Ziel.
    requires-Beziehungen verschieben benoetigte Elemente vorsichtig nach rechts.
    """
    if not elements:
        return {}

    input_order = {element.name: i for i, element in enumerate(elements)}

    base_layer_by_type = {
        "ProductService": 0,
        "Product": 1,
        "Service": 1,
        "Feature": 2,
        "Component": 3,
    }
    layer: Dict[str, int] = {
        element.name: base_layer_by_type.get(element.type, 2)
        for element in elements
    }

    hierarchical_kinds = {"part_of", "is_a"} | PARTOF_CONNECTOR_TYPES | ISA_CONNECTOR_TYPES

    for _ in range(len(elements)):
        changed = False
        for conn in connections:
            if conn.kind not in hierarchical_kinds:
                continue
            target_layer = layer[conn.target]
            for source in conn.sources:
                wanted = target_layer + 1
                if layer[source] < wanted:
                    layer[source] = wanted
                    changed = True
        if not changed:
            break

    for conn in connections:
        if conn.kind != "requires":
            continue
        for source in conn.sources:
            if layer[conn.target] <= layer[source]:
                layer[conn.target] = layer[source] + 1

    min_layer = min(layer.values())
    if min_layer != 0:
        layer = {name: value - min_layer for name, value in layer.items()}

    adjacency: Dict[str, Dict[str, float]] = {element.name: {} for element in elements}

    def add_edge(a: str, b: str, weight: float) -> None:
        adjacency[a][b] = adjacency[a].get(b, 0.0) + weight
        adjacency[b][a] = adjacency[b].get(a, 0.0) + weight

    for conn in connections:
        if conn.kind in ({"part_of"} | PARTOF_CONNECTOR_TYPES):
            weight = 4.0
        elif conn.kind == "requires":
            weight = 2.5
        else:  # is_a, Total-ISA, Partial-ISA
            weight = 2.0

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

    x_start = 4.0
    y_start = 3.0
    x_gap = 9.0
    y_gap = 4.2

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

    x = tx + (sx - tx) * 0.62
    y = ty + (sy - ty) * 0.62
    y += 1.3
    return x, y


def node_size_for(element_type: str) -> Tuple[float, float]:
    if element_type == "Feature":
        return 3.0, 1.86
    if element_type == "Component":
        return 4.15, 2.15
    return 3.92, 2.45


def connector_attributes(connector_class: str) -> str:
    """ADL-Attribute fuer Connector-Knoten."""
    if connector_class in PARTOF_CONNECTOR_TYPES:
        alternatives = [kind for kind in ["PartOF (AND)", "PartOF (OR)", "PartOF (XOR)"] if kind != connector_class]
        conversion_lines = "\n".join(f'CLASS \\"{alternative}\\"' for alternative in alternatives)
        return (
            '\n\tATTRIBUTE <__Conversion__>\n'
            f'\tVALUE "{conversion_lines}"\n'
        )

    # Total-ISA und Partial-ISA haben in den Beispieldateien keine zusaetzlichen Attribute.
    return ""


def is_connector_connection(conn: Connection) -> bool:
    """Explizite Connector-Verbindungen oder alte Mehrfach-part_of-Zeilen."""
    return conn.kind in CONNECTOR_CONNECTION_TYPES or (conn.kind == "part_of" and len(conn.sources) > 1)


def connector_class_for(conn: Connection) -> str:
    if conn.kind in CONNECTOR_CONNECTION_TYPES:
        return conn.kind
    if conn.kind == "part_of" and len(conn.sources) > 1:
        return "PartOF (AND)"
    raise ValueError(f"Connection {conn.kind!r} is not a connector.")


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
\tVALUE "{pos}"

\tATTRIBUTE <External tool coupling>
\tVALUE ""{attrs}
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
        adl_class, attrs = adl_class_and_attributes(element)
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
            # A, B PartOF (XOR) C  ->  A -> PartOF(XOR), B -> PartOF(XOR), PartOF(XOR) -> C
            # A, B Total-ISA C     ->  A -> Total-ISA,  B -> Total-ISA,  Total-ISA  -> C
            junction_class = connector_class_for(conn)
            junction_name = f"{junction_class}-AUTO{c_idx}"
            x, y = junction_position(conn.sources, conn.target, positions, len(elements) + c_idx)
            junction_attrs = connector_attributes(junction_class)
            instance_blocks.append(make_instance(junction_name, junction_class, junction_attrs, node_index, x, y))
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

        elif conn.kind == "part_of":
            # Direkte part_of-Verbindung: einfache Kante ohne Connector.
            # 4EM speichert part_of/is_a in den Beispielen mit leerem Relationstyp.
            for source in conn.sources:
                relation_blocks.append(
                    make_relation(source, class_by_name[source], conn.target, class_by_name[conn.target], edge_index, "")
                )
                edge_index += 1

        elif conn.kind == "requires":
            for source in conn.sources:
                relation_blocks.append(
                    make_relation(source, class_by_name[source], conn.target, class_by_name[conn.target], edge_index, "requires")
                )
                edge_index += 1

        elif conn.kind == "is_a":
            # 4EM 2.6 akzeptiert "is_a" nicht als Enumerationswert fuer ATTRIBUTE <Type>.
            # Deshalb wird die Kante mit leerem Relationstyp geschrieben.
            for source in conn.sources:
                relation_blocks.append(
                    make_relation(source, class_by_name[source], conn.target, class_by_name[conn.target], edge_index, "")
                )
                edge_index += 1
        else:
            raise ValueError(f"Unsupported connection: {conn.kind}")

    component_count = sum(1 for element in elements if element.type == "Component")
    feature_count = sum(1 for element in elements if element.type == "Feature")
    ps_count = sum(1 for element in elements if element.type in {"ProductService", "Product", "Service"})

    header = f'''///////////////////////////////////////////////////////////////
//
// Date: {now.strftime("%d.%m.%Y  %H:%M")}
//
// Generated by notation_to_adl_for_new_prompt.py
// Data version 4.0
//
///////////////////////////////////////////////////////////////
//
// The file contains the following models:
//
// {esc(model_name)} (Product-Service-Model)
//
//////////////////////////////////////////////////////////////

VERSION <4.0>


BUSINESS PROCESS MODEL <{esc(model_name)}> : <4EM current>
VERSION <>
TYPE <Product-Service-Model>

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

\tATTRIBUTE <Iterator: Component>
\tVALUE {component_count}

\tATTRIBUTE <Iterator: Feature>
\tVALUE {feature_count}

\tATTRIBUTE <Iterator: Unspecific/Product/Service>
\tVALUE {ps_count}

\tATTRIBUTE <Eventlog>
\tVALUE ""

'''

    return header + "\n".join(instance_blocks) + "\n" + "\n".join(relation_blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Product/Service notation with direct connections and connectors to a 4EM ADL file."
    )
    parser.add_argument("input", help="Text file containing ELEMENTS and CONNECTIONS")
    parser.add_argument("output", help="Output .adl file")
    parser.add_argument("--model-name", default="Generated Product Service Model")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    elements, connections = parse_notation(text)
    adl = generate_adl(elements, connections, args.model_name)
    output_path.write_text(adl, encoding="utf-8")

    print(f"OK: read {len(elements)} elements and {len(connections)} notation connections.")
    print(f"ADL written to: {output_path}")


if __name__ == "__main__":
    main()
