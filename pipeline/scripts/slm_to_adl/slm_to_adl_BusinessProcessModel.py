#!/usr/bin/env python3
# bpm_to_adl.py
#
# Wandelt die LLM-Notation aus dem Business-Process-Model-Prompt in eine ADL-Datei
# fuer ein 4EM Business Process Model um.
#
# Unterstuetzte direkte Verbindungen:
#   A relation B
#   A Input B
#   A Output B
#
# Unterstuetzte Connector-Verbindungen:
#   A Split (AND) B, C, D
#   A Split (OR) B, C, D
#   A, B, C Join (AND) D
#   A, B, C Join (OR) D
#
# Zusaetzlich koennen Connectoren auch als normale Elemente unter ELEMENTS
# definiert und mit `relation` verbunden werden:
#   Split (AND) My split
#   Process A relation My split
#   My split relation Process B
#
# Aufruf:
# python bpm_to_adl.py input.txt output.adl --model-name "BusinessProcessModel"

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


BPM_ELEMENT_TYPES = {"Process", "External Process", "Information Set"}
CONNECTOR_ELEMENT_TYPES = {"Split (AND)", "Join (AND)", "Split (OR)", "Join (OR)"}
ELEMENT_TYPES = BPM_ELEMENT_TYPES | CONNECTOR_ELEMENT_TYPES

DIRECT_CONNECTION_TYPES = {"relation", "Input", "Output"}
CONNECTOR_CONNECTION_TYPES = {"Split (AND)", "Join (AND)", "Split (OR)", "Join (OR)"}
CONNECTION_TYPES = DIRECT_CONNECTION_TYPES | CONNECTOR_CONNECTION_TYPES

# Laengere Tokens muessen vor kuerzeren gesucht werden, weil Elementnamen
# Leerzeichen enthalten duerfen und Elementtypen wie "External Process" aus
# mehreren Woertern bestehen.
ELEMENT_TYPE_PREFIXES = sorted(ELEMENT_TYPES, key=len, reverse=True)
CONNECTION_TOKEN_RE = re.compile(
    r"\s("
    r"Split\s*\(\s*AND\s*\)"
    r"|Join\s*\(\s*AND\s*\)"
    r"|Split\s*\(\s*OR\s*\)"
    r"|Join\s*\(\s*OR\s*\)"
    r"|relation"
    r"|Input"
    r"|Output"
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
    targets: List[str]
    original_line: str


def esc(value: str) -> str:
    """Escaping fuer ADL-Werte in spitzen Klammern und Quotes."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("<", "(").replace(">", ")")


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def split_csv_names(s: str) -> List[str]:
    return [normalize_spaces(x) for x in s.split(",") if normalize_spaces(x)]


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
    compact = normalize_spaces(line)
    for etype in ELEMENT_TYPE_PREFIXES:
        prefix = etype + " "
        if compact == etype:
            raise ValueError(f"Element name is missing in line: {line!r}")
        if compact.startswith(prefix):
            name = normalize_spaces(compact[len(prefix):])
            if not name:
                raise ValueError(f"Element name is missing in line: {line!r}")
            return Element(etype, name)

    raise ValueError(
        f"Unknown element type in line {line!r}. "
        f"Allowed: {', '.join(sorted(ELEMENT_TYPES))}"
    )


def normalize_connection_kind(raw_kind: str) -> str:
    compact = normalize_spaces(raw_kind)
    lower = compact.lower().replace(" ", "")

    if lower == "relation":
        return "relation"
    if lower == "input":
        return "Input"
    if lower == "output":
        return "Output"
    if lower == "split(and)":
        return "Split (AND)"
    if lower == "join(and)":
        return "Join (AND)"
    if lower == "split(or)":
        return "Split (OR)"
    if lower == "join(or)":
        return "Join (OR)"

    raise ValueError(f"Unknown connection type: {raw_kind!r}")


def is_bpm_node(element_type: str) -> bool:
    return element_type in BPM_ELEMENT_TYPES


def is_connector_node(element_type: str) -> bool:
    return element_type in CONNECTOR_ELEMENT_TYPES


def validate_allowed_pattern(kind: str, source_type: str, target_type: str, line: str) -> None:
    """Validiert die im Business-Process-Model-Prompt erlaubten Muster."""
    if kind == "relation":
        # `relation` ist die unlabeled 4EM-Relation. Sie wird im Prompt nur zwischen
        # Process/External Process/Information Set und Connectoren erlaubt, nicht
        # als semantischer Informationsfluss zwischen beliebigen BPM-Elementen.
        allowed = (
            (is_bpm_node(source_type) and is_connector_node(target_type))
            or (is_connector_node(source_type) and is_bpm_node(target_type))
            or (source_type, target_type) in {
                ("Process", "External Process"),
                ("External Process", "Process"),
            }
        )
    elif kind == "Input":
        allowed = source_type == "Information Set" and target_type in {"Process", "External Process"}
    elif kind == "Output":
        allowed = source_type in {"Process", "External Process"} and target_type == "Information Set"
    elif kind in {"Split (AND)", "Split (OR)"}:
        allowed = is_bpm_node(source_type) and is_bpm_node(target_type)
    elif kind in {"Join (AND)", "Join (OR)"}:
        allowed = is_bpm_node(source_type) and is_bpm_node(target_type)
    else:
        allowed = False

    if not allowed:
        raise ValueError(
            f"Disallowed connection pattern in {line!r}: "
            f"{source_type} {kind} {target_type}"
        )


def parse_connection_line(line: str, elements_by_name: Dict[str, Element]) -> Connection:
    """
    Erwartete Syntax:
      A relation B
      A Input B
      A Output B
      A Split (AND) B, C, D
      A Split (OR) B, C, D
      A, B, C Join (AND) D
      A, B, C Join (OR) D

    Da Elementnamen Leerzeichen enthalten duerfen, wird anhand der bekannten
    Verbindungstokens geparst.
    """
    match = CONNECTION_TOKEN_RE.search(line)
    if not match:
        allowed = ", ".join(sorted(CONNECTION_TYPES))
        raise ValueError(f"No valid connection type found in: {line!r}. Allowed: {allowed}")

    kind = normalize_connection_kind(match.group(1))
    left = normalize_spaces(line[:match.start()])
    right = normalize_spaces(line[match.end():])

    sources = split_csv_names(left)
    targets = split_csv_names(right)
    if not sources:
        raise ValueError(f"Connection has no source: {line!r}")
    if not targets:
        raise ValueError(f"Connection has no target: {line!r}")

    for source in sources:
        if source not in elements_by_name:
            raise ValueError(f"Source {source!r} is not defined under ELEMENTS.")
    for target in targets:
        if target not in elements_by_name:
            raise ValueError(f"Target {target!r} is not defined under ELEMENTS.")

    if kind in DIRECT_CONNECTION_TYPES:
        if len(sources) != 1 or len(targets) != 1:
            raise ValueError(f"Direct connection {kind!r} requires exactly one source and one target: {line!r}")
    elif kind.startswith("Split"):
        if len(sources) != 1 or len(targets) < 2:
            raise ValueError(f"{kind!r} requires exactly one source and at least two targets: {line!r}")
    elif kind.startswith("Join"):
        if len(sources) < 2 or len(targets) != 1:
            raise ValueError(f"{kind!r} requires at least two sources and exactly one target: {line!r}")

    for source in sources:
        for target in targets:
            validate_allowed_pattern(kind, elements_by_name[source].type, elements_by_name[target].type, line)

    return Connection(sources=sources, kind=kind, targets=targets, original_line=line)


def parse_notation(text: str) -> Tuple[List[Element], List[Connection]]:
    element_lines, connection_lines = split_sections(text)
    if not element_lines:
        raise ValueError("No ELEMENTS section or no elements found.")

    elements = [parse_element_line(line) for line in element_lines]
    names = [element.name for element in elements]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate element names found: {', '.join(duplicates)}")

    elements_by_name = {element.name: element for element in elements}
    connections = [parse_connection_line(line, elements_by_name) for line in connection_lines]
    return elements, connections


def adl_attributes_for(element_type: str) -> str:
    """ADL-Attribute fuer Business-Process-Model-Elemente gemaess Beispieldateien."""
    if element_type == "Process":
        return '''
\tATTRIBUTE <External tool coupling>
\tVALUE ""

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Decomposed Process>
\tVALUE ""

\tATTRIBUTE <Intermodel-Relations>
\tVALUE

\tATTRIBUTE <Decomposition>
\tVALUE ""

\tATTRIBUTE <Execution Time>
\tVALUE 0

\tATTRIBUTE <Complexity>
\tVALUE 0

\tATTRIBUTE <Type>
\tVALUE ""

\tATTRIBUTE <Attributes>
\tVALUE
'''

    if element_type == "External Process":
        return '''
\tATTRIBUTE <External tool coupling>
\tVALUE ""

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Intermodel-Relations>
\tVALUE

\tATTRIBUTE <Decomposition>
\tVALUE ""

\tATTRIBUTE <Execution Time>
\tVALUE 0

\tATTRIBUTE <Complexity>
\tVALUE 0

\tATTRIBUTE <Type>
\tVALUE ""

\tATTRIBUTE <Attributes>
\tVALUE
'''

    if element_type == "Information Set":
        return '''
\tATTRIBUTE <External tool coupling>
\tVALUE ""

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <Type>
\tVALUE "Information Set"

\tATTRIBUTE <Intermodel-Relations>
\tVALUE

\tATTRIBUTE <Decomposition>
\tVALUE ""

\tATTRIBUTE <Attributes>
\tVALUE
'''

    if element_type in CONNECTOR_ELEMENT_TYPES:
        return '''
\tATTRIBUTE <External tool coupling>
\tVALUE ""
'''

    raise ValueError(f"Unsupported element type: {element_type}")


def node_size_for(element_type: str) -> Tuple[float | None, float | None]:
    # Groessen aus den Business-Process-Beispieldateien.
    if element_type == "Process":
        return 4.4, 1.2
    if element_type == "External Process":
        return 5.3, 2.1
    if element_type == "Information Set":
        return 6.0, 1.6
    if element_type in CONNECTOR_ELEMENT_TYPES:
        return None, None
    return 4.4, 1.2


def layout_position(index: int) -> Tuple[float, float]:
    col = index // 10
    row = index % 10
    return 3.0 + col * 10.0, 2.5 + row * 2.5


def expand_graph_edges(connections: Iterable[Connection]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for conn in connections:
        if conn.kind in DIRECT_CONNECTION_TYPES:
            edges.append((conn.sources[0], conn.targets[0]))
        elif conn.kind.startswith("Split"):
            for target in conn.targets:
                edges.append((conn.sources[0], target))
        elif conn.kind.startswith("Join"):
            for source in conn.sources:
                edges.append((source, conn.targets[0]))
    return edges


def compute_layout(elements: List[Element], connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
    """Einfaches Layout: Quellen links, Ergebnisse rechts, zusammenhaengende Knoten nah beieinander."""
    if not elements:
        return {}

    input_order = {element.name: i for i, element in enumerate(elements)}
    base_layer_by_type = {
        "External Process": 0,
        "Process": 1,
        "Information Set": 2,
        "Split (AND)": 1,
        "Join (AND)": 1,
        "Split (OR)": 1,
        "Join (OR)": 1,
    }
    layer: Dict[str, int] = {element.name: base_layer_by_type.get(element.type, 1) for element in elements}

    for _ in range(len(elements)):
        changed = False
        for source, target in expand_graph_edges(connections):
            wanted = layer[source] + 1
            if layer[target] < wanted:
                layer[target] = wanted
                changed = True
        if not changed:
            break

    min_layer = min(layer.values())
    layer = {name: value - min_layer for name, value in layer.items()}

    adjacency: Dict[str, Dict[str, float]] = {element.name: {} for element in elements}

    def add_edge(a: str, b: str, weight: float) -> None:
        adjacency[a][b] = adjacency[a].get(b, 0.0) + weight
        adjacency[b][a] = adjacency[b].get(a, 0.0) + weight

    for source, target in expand_graph_edges(connections):
        add_edge(source, target, 2.0)

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

    order_value: Dict[str, float] = {}
    for names in layers.values():
        names.sort(key=lambda name: (component_id[name], input_order[name]))
        for row, name in enumerate(names):
            order_value[name] = float(row)

    for _ in range(8):
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

    positions: Dict[str, Tuple[float, float]] = {}
    for layer_no, names in layers.items():
        for row, name in enumerate(names):
            positions[name] = (3.0 + layer_no * 9.0, 2.5 + row * 2.7)
    return positions


def junction_position(
    sources: List[str],
    targets: List[str],
    positions: Dict[str, Tuple[float, float]],
    fallback_index: int,
) -> Tuple[float, float]:
    points = [positions[name] for name in sources + targets if name in positions]
    if not points:
        return layout_position(fallback_index)
    return (sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points))


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


def make_relation(src_name: str, src_class: str, dst_name: str, dst_class: str, edge_index: int, rel_type: str = "") -> str:
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


def make_connector_name(kind: str, c_idx: int) -> str:
    # Die 4EM-Exportdateien nutzen Namen wie "Split (AND)-75640". Hier wird ein
    # stabiler AUTO-Name benutzt, damit keine internen IDs erfunden werden.
    return f"{kind}-AUTO{c_idx}"


def generate_adl(elements: List[Element], connections: List[Connection], model_name: str) -> str:
    now = datetime.now()
    created = now.strftime("%d.%m.%Y, %H:%M")
    changed = now.strftime("%d.%m.%Y, %H:%M:%S")

    positions = compute_layout(elements, connections)
    class_by_name: Dict[str, str] = {}
    instance_blocks: List[str] = []

    node_index = 1
    for idx, element in enumerate(elements):
        adl_class = element.type
        class_by_name[element.name] = adl_class
        x, y = positions.get(element.name, layout_position(idx))
        w, h = node_size_for(element.type)
        instance_blocks.append(make_instance(element.name, adl_class, adl_attributes_for(element.type), node_index, x, y, w, h))
        node_index += 1

    relation_blocks: List[str] = []
    edge_index = 1

    for c_idx, conn in enumerate(connections, start=1):
        if conn.kind in DIRECT_CONNECTION_TYPES:
            source = conn.sources[0]
            target = conn.targets[0]
            rel_type = "" if conn.kind == "relation" else conn.kind
            relation_blocks.append(make_relation(source, class_by_name[source], target, class_by_name[target], edge_index, rel_type))
            edge_index += 1
            continue

        junction_class = conn.kind
        junction_name = make_connector_name(junction_class, c_idx)
        x, y = junction_position(conn.sources, conn.targets, positions, len(elements) + c_idx)
        instance_blocks.append(make_instance(junction_name, junction_class, adl_attributes_for(junction_class), node_index, x, y))
        class_by_name[junction_name] = junction_class
        node_index += 1

        if conn.kind.startswith("Split"):
            source = conn.sources[0]
            relation_blocks.append(make_relation(source, class_by_name[source], junction_name, junction_class, edge_index, ""))
            edge_index += 1
            for target in conn.targets:
                relation_blocks.append(make_relation(junction_name, junction_class, target, class_by_name[target], edge_index, ""))
                edge_index += 1
        elif conn.kind.startswith("Join"):
            target = conn.targets[0]
            for source in conn.sources:
                relation_blocks.append(make_relation(source, class_by_name[source], junction_name, junction_class, edge_index, ""))
                edge_index += 1
            relation_blocks.append(make_relation(junction_name, junction_class, target, class_by_name[target], edge_index, ""))
            edge_index += 1

    type_counts = {element_type: 0 for element_type in BPM_ELEMENT_TYPES}
    for element in elements:
        if element.type in type_counts:
            type_counts[element.type] += 1

    relation_count = len(relation_blocks)
    input_count = sum(1 for conn in connections if conn.kind == "Input")
    output_count = sum(1 for conn in connections if conn.kind == "Output")
    other_count = relation_count - input_count - output_count

    header = f'''///////////////////////////////////////////////////////////////
//
// Date: {now.strftime("%d.%m.%Y  %H:%M")}
//
// Generated by bpm_to_adl.py
// Data version 4.0
//
///////////////////////////////////////////////////////////////
//
// The file contains the following models:
//
// {esc(model_name)} (Business Process Model)
//
//////////////////////////////////////////////////////////////

VERSION <4.0>


BUSINESS PROCESS MODEL <{esc(model_name)}> : <4EM current>
VERSION <>
TYPE <Business Process Model>

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

\tATTRIBUTE <Iterator: Process>
\tVALUE {type_counts["Process"]}

\tATTRIBUTE <Iterator: External Process>
\tVALUE {type_counts["External Process"]}

\tATTRIBUTE <Iterator: Information Set>
\tVALUE {type_counts["Information Set"]}

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
\tVALUE {other_count}

\tATTRIBUTE <Iterator: 1:1>
\tVALUE 0

\tATTRIBUTE <Iterator: 1:n>
\tVALUE 0

\tATTRIBUTE <Iterator: n:m>
\tVALUE 0

\tATTRIBUTE <Iterator: Input>
\tVALUE {input_count}

\tATTRIBUTE <Iterator: Output>
\tVALUE {output_count}

\tATTRIBUTE <Iterator: Motivates>
\tVALUE 0

\tATTRIBUTE <Iterator: Requires>
\tVALUE 0

\tATTRIBUTE <Iterator: Triggers>
\tVALUE 0

\tATTRIBUTE <Iterator: Plays>
\tVALUE 0

\tATTRIBUTE <Iterator: works in>
\tVALUE 0

\tATTRIBUTE <Iterator: works at>
\tVALUE 0

'''

    return header + "\n".join(instance_blocks) + "\n" + "\n".join(relation_blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Business Process Model notation with relation/Input/Output and split/join connectors to a 4EM ADL file."
    )
    parser.add_argument("input", help="Text file containing ELEMENTS and CONNECTIONS")
    parser.add_argument("output", help="Output .adl file")
    parser.add_argument("--model-name", default="Generated Business Process Model")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    elements, connections = parse_notation(text)
    adl = generate_adl(elements, connections, args.model_name)
    output_path.write_text(adl, encoding="utf-8")

    generated_relations = adl.count("RELATION <4EM_Relation>")
    print(f"OK: read {len(elements)} elements and {len(connections)} notation connections.")
    print(f"ADL written to: {output_path}")
    print(f"Generierte ADL-Relationen: {generated_relations}")


if __name__ == "__main__":
    main()
