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
from typing import Dict, List, Set, Tuple
try:
    from scripts.slm_to_adl.hierarchical_layout import LayoutEdge, LayoutNode, UniformLayoutConfig, compute_uniform_layout
    from scripts.slm_to_adl.validation import parse_lines_collect, raise_validation_errors, validate_references
except ModuleNotFoundError:
    from hierarchical_layout import LayoutEdge, LayoutNode, UniformLayoutConfig, compute_uniform_layout
    from validation import parse_lines_collect, raise_validation_errors, validate_references


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
                raise ValueError(f"Element has no name: {line!r}")
            return Element(etype, name)

    raise ValueError(
        f"Unknown element type in line {line!r}. "
        f"Allowed: {', '.join(sorted(ELEMENT_TYPES))}"
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
    raise ValueError(f"Unknown connection type: {raw_kind!r}")


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
            f"Disallowed connection pattern in {line!r}: "
            f"{source_type} {kind} {target_type}"
        )


def parse_connection_line(line: str, elements_by_name: Dict[str, Element]) -> Connection:
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

    validate_references(sources, [target], elements_by_name)

    if kind in DIRECT_CONNECTION_TYPES and len(sources) != 1:
        raise ValueError(f"Direct connection {kind!r} must have exactly one source: {line!r}")
    if kind in CONNECTOR_CONNECTION_TYPES and len(sources) < 2:
        raise ValueError(f"Connector connection {kind!r} must have at least two sources: {line!r}")
    for source in sources:
        validate_allowed_pattern(kind, elements_by_name[source].type, elements_by_name[target].type, line)

    return Connection(sources, kind, target)


def parse_notation(text: str) -> Tuple[List[Element], List[Connection]]:
    element_lines, connection_lines = split_sections(text)
    elements, errors = parse_lines_collect(element_lines, parse_element_line, "ELEMENTS")

    names = [element.name for element in elements]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        errors.append(f"Duplicate element names found: {', '.join(duplicates)}")

    elements_by_name = {element.name: element for element in elements}
    connections, connection_errors = parse_lines_collect(
        connection_lines,
        lambda line: parse_connection_line(line, elements_by_name),
        "CONNECTIONS",
    )
    errors.extend(connection_errors)
    raise_validation_errors(errors)
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

    raise ValueError(f"Unsupported element type: {element.type}")


def layout_position(index: int) -> Tuple[float, float]:
    col = index // 10
    row = index % 10
    return 3.5 + col * 8.0, 3.5 + row * 2.5


def connector_layout_name(connection_index: int, connector_class: str) -> str:
    return f"{connector_class}-AUTO{connection_index}"


def layout_components(elements: List[Element], connections: List[Connection]) -> List[List[str]]:
    """Return weakly connected ARM components including connector nodes."""
    nodes = [element.name for element in elements]
    adjacency: Dict[str, Set[str]] = {name: set() for name in nodes}

    def link(left: str, right: str) -> None:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    for index, connection in enumerate(connections, start=1):
        if connection.kind in CONNECTOR_CONNECTION_TYPES:
            connector = connector_layout_name(index, connection.kind)
            nodes.append(connector)
            adjacency[connector] = set()
            link(connection.target, connector)
            for source in connection.sources:
                link(connector, source)
        else:
            for source in connection.sources:
                link(connection.target, source)

    components: List[List[str]] = []
    visited: Set[str] = set()
    for root in nodes:
        if root in visited:
            continue
        stack = [root]
        visited.add(root)
        component: List[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in reversed(nodes):
                if neighbor in adjacency[current] and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def compute_layout(elements: List[Element], connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
    """Create a top-down forest with connectors on intermediate half-levels."""
    if not elements:
        return {}
    nodes = [LayoutNode(e.name, *node_size_for(e.type)) for e in elements]
    edges: List[LayoutEdge] = []
    branches: List[str] = []
    for index, connection in enumerate(connections, start=1):
        if connection.kind in CONNECTOR_CONNECTION_TYPES:
            connector = connector_layout_name(index, connection.kind)
            nodes.append(LayoutNode(connector, 1.0, 1.0, True))
            branches.append(connector)
            edges.extend(LayoutEdge(source, connector, 1) for source in connection.sources)
            edges.append(LayoutEdge(connector, connection.target, 1))
        else:
            edges.extend(LayoutEdge(source, connection.target, 2) for source in connection.sources)
    return compute_uniform_layout(nodes, edges, branch_nodes=branches,
        config=UniformLayoutConfig(orientation="bottom_up", x_start=3.5,
                                   y_start=3.5, node_gap=6.5, level_gap=3.2))

    node_order = {element.name: index for index, element in enumerate(elements)}
    all_nodes = [element.name for element in elements]
    raw_edges: List[Tuple[str, str, int]] = []
    for index, connection in enumerate(connections, start=1):
        if connection.kind in CONNECTOR_CONNECTION_TYPES:
            connector = connector_layout_name(index, connection.kind)
            node_order[connector] = len(node_order)
            all_nodes.append(connector)
            raw_edges.append((connection.target, connector, 1))
            raw_edges.extend((connector, source, 1) for source in connection.sources)
        else:
            raw_edges.extend((connection.target, source, 2) for source in connection.sources)

    accepted: List[Tuple[str, str, int]] = []
    outgoing: Dict[str, Set[str]] = {name: set() for name in all_nodes}

    def reaches(start: str, wanted: str) -> bool:
        stack = [start]
        visited: Set[str] = set()
        while stack:
            current = stack.pop()
            if current == wanted:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(outgoing[current])
        return False

    for source, target, distance in raw_edges:
        if source == target or reaches(target, source):
            continue
        if target not in outgoing[source]:
            outgoing[source].add(target)
            accepted.append((source, target, distance))

    positions: Dict[str, Tuple[float, float]] = {}
    component_x = 3.5
    half_level_gap = 3.2
    node_gap = 6.5
    for component_index, component in enumerate(layout_components(elements, connections)):
        component_set = set(component)
        edges = [edge for edge in accepted if edge[0] in component_set and edge[1] in component_set]
        rank = {name: 0 for name in component}
        for _ in component:
            changed = False
            for source, target, distance in edges:
                wanted = rank[source] + distance
                if rank[target] < wanted:
                    rank[target] = wanted
                    changed = True
            if not changed:
                break

        connector_groups = []
        for connection_index, connection in enumerate(connections, start=1):
            if connection.kind not in CONNECTOR_CONNECTION_TYPES:
                continue
            connector = connector_layout_name(connection_index, connection.kind)
            if connector in component_set:
                connector_groups.append((connector, connection.sources))

        # A secondary relation may push one connector child deeper. Move the
        # complete connector block so all grouped children remain on one row and
        # the connector remains exactly one half-level above them.
        for _ in range(max(1, len(component) * 2)):
            changed = False
            for source, target, distance in edges:
                wanted = rank[source] + distance
                if rank[target] < wanted:
                    rank[target] = wanted
                    changed = True
            for connector, children in connector_groups:
                child_level = max(rank[child] for child in children)
                if rank[connector] < child_level - 1:
                    rank[connector] = child_level - 1
                    changed = True
                aligned_level = rank[connector] + 1
                for child in children:
                    if rank[child] < aligned_level:
                        rank[child] = aligned_level
                        changed = True
            if not changed:
                break

        layout_nodes = list(component)
        virtual_nodes: Set[str] = set()
        layout_edges: List[Tuple[str, str]] = []
        for edge_index, (source, target, _) in enumerate(edges):
            previous = source
            for virtual_rank in range(rank[source] + 1, rank[target]):
                virtual = f"__ARM_ROUTE_{component_index}_{edge_index}_{virtual_rank}"
                layout_nodes.append(virtual)
                virtual_nodes.add(virtual)
                rank[virtual] = virtual_rank
                node_order[virtual] = len(node_order)
                layout_edges.append((previous, virtual))
                previous = virtual
            layout_edges.append((previous, target))

        adjacency: Dict[str, Set[str]] = {name: set() for name in layout_nodes}
        for source, target in layout_edges:
            adjacency[source].add(target)
            adjacency[target].add(source)
        rows: Dict[int, List[str]] = {}
        for name in layout_nodes:
            rows.setdefault(rank[name], []).append(name)
        for names in rows.values():
            names.sort(key=lambda name: node_order[name])

        order = {name: float(index) for names in rows.values() for index, name in enumerate(names)}
        for _ in range(8):
            for level in sorted(rows):
                rows[level].sort(key=lambda name: (
                    sum(order[n] for n in adjacency[name]) / len(adjacency[name])
                    if adjacency[name] else order[name],
                    node_order[name],
                ))
                for index, name in enumerate(rows[level]):
                    order[name] = float(index)
            for level in sorted(rows, reverse=True):
                rows[level].sort(key=lambda name: (
                    sum(order[n] for n in adjacency[name]) / len(adjacency[name])
                    if adjacency[name] else order[name],
                    node_order[name],
                ))
                for index, name in enumerate(rows[level]):
                    order[name] = float(index)

        def crossing_count() -> int:
            row_order = {
                name: index
                for names in rows.values()
                for index, name in enumerate(names)
            }
            count = 0
            for edge_index, (source_a, target_a) in enumerate(layout_edges):
                for source_b, target_b in layout_edges[edge_index + 1:]:
                    if source_a == source_b or target_a == target_b:
                        continue
                    if rank[source_a] != rank[source_b] or rank[target_a] != rank[target_b]:
                        continue
                    if ((row_order[source_a] - row_order[source_b])
                            * (row_order[target_a] - row_order[target_b]) < 0):
                        count += 1
            return count

        # Barycentric sorting can retain a crossing after a connector block was
        # moved. Adjacent exchanges now explicitly minimise crossings between
        # every pair of neighbouring tree rows, including virtual routing nodes.
        def row_signature() -> Tuple[Tuple[str, ...], ...]:
            return tuple(tuple(rows[level]) for level in sorted(rows))

        visited_orders = {row_signature()}
        improved = True
        while improved:
            improved = False
            for level in sorted(rows):
                names = rows[level]
                for index in range(len(names) - 1):
                    before = crossing_count()
                    names[index], names[index + 1] = names[index + 1], names[index]
                    after = crossing_count()
                    signature = row_signature()
                    if after <= before and signature not in visited_orders:
                        visited_orders.add(signature)
                        improved = True
                    else:
                        names[index], names[index + 1] = names[index + 1], names[index]

        widest = max((len(names) for names in rows.values()), default=1)
        component_width = max(7.0, (widest - 1) * node_gap + 5.0)
        center_x = component_x + component_width / 2.0
        for level, names in rows.items():
            start_x = center_x - (len(names) - 1) * node_gap / 2.0
            for index, name in enumerate(names):
                if name not in virtual_nodes:
                    positions[name] = (start_x + index * node_gap, 3.5 + level * half_level_gap)

        component_x += component_width + 7.0

    minimum_x = min((point[0] - 2.0 for point in positions.values()), default=0.0)
    shift_x = max(0.0, 3.5 - minimum_x)
    if shift_x:
        positions = {
            name: (point[0] + shift_x, point[1])
            for name, point in positions.items()
        }
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
    raise ValueError(f"Connection {conn.kind!r} is not a connector.")


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
    world_width = max(80, int(max((x for x, _ in positions.values()), default=70.0) + 10.0))
    world_height = max(80, int(max((y for _, y in positions.values()), default=70.0) + 10.0))

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
            junction_name = connector_layout_name(c_idx, junction_class)
            x, y = positions[junction_name]
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
            raise ValueError(f"Unsupported connection: {conn.kind}")

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
\tVALUE "w:{world_width}cm h:{world_height}cm minw:5cm minh:5cm"

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
        description="Convert Actors and Resources Model notation with direct connections and 4EM connectors to an ADL file."
    )
    parser.add_argument("input", help="Text file containing ELEMENTS and CONNECTIONS")
    parser.add_argument("output", help="Output .adl file")
    parser.add_argument("--model-name", default="Generated Actors and Resources Model")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    elements, connections = parse_notation(text)
    components = layout_components(elements, connections)
    element_names = {element.name for element in elements}
    sizes = ", ".join(str(sum(name in element_names for name in component)) for component in components)
    print(f"Graph analysis: {len(components)} connected component(s) (element counts: {sizes}).")
    adl = generate_adl(elements, connections, args.model_name)
    output_path.write_text(adl, encoding="utf-8")

    print(f"OK: read {len(elements)} elements and {len(connections)} notation connections.")
    print(f"ADL written to: {output_path}")


if __name__ == "__main__":
    main()
