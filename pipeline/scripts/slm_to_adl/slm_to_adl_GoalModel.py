#!/usr/bin/env python3
# goal_to_adl.py
#
# Wandelt die LLM-Notation aus dem Goal-Model-Prompt in eine ADL-Datei
# fuer ein 4EM Goal Model um.
#
# Unterstuetzte direkte Verbindungen:
#   A supports B
#   A hinders B
#   A contradicts B
#   A causes B
#
# Unterstuetzte Connector-Verbindungen:
#   A, B, C AND D
#   A, B, C OR D
#   A, B, C AND/OR D
#
# Aufruf:
# python goal_to_adl.py input.txt output.adl --model-name "GoalModel"

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
try:
    from scripts.slm_to_adl.hierarchical_layout import LayoutEdge, LayoutNode, UniformLayoutConfig, compute_uniform_layout
    from scripts.slm_to_adl.validation import parse_lines_collect, raise_validation_errors, validate_references
except ModuleNotFoundError:
    from hierarchical_layout import LayoutEdge, LayoutNode, UniformLayoutConfig, compute_uniform_layout
    from validation import parse_lines_collect, raise_validation_errors, validate_references


ELEMENT_TYPES = {"Goal", "Problem", "Cause", "Constraint", "Opportunity"}
DIRECT_CONNECTION_TYPES = {"supports", "hinders", "contradicts", "causes"}
CONNECTOR_CONNECTION_TYPES = {"AND", "OR", "AND/OR"}
CONNECTION_TYPES = DIRECT_CONNECTION_TYPES | CONNECTOR_CONNECTION_TYPES

# Laengere Connector-Tokens muessen vor kuerzeren Tokens gesucht werden,
# weil Elementnamen Leerzeichen enthalten duerfen.
CONNECTION_TOKEN_RE = re.compile(
    r"\s("
    r"AND/OR"
    r"|supports"
    r"|hinders"
    r"|contradicts"
    r"|causes"
    r"|AND"
    r"|OR"
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

    if lower == "supports":
        return "supports"
    if lower == "hinders":
        return "hinders"
    if lower == "contradicts":
        return "contradicts"
    if lower == "causes":
        return "causes"
    if lower == "and":
        return "AND"
    if lower == "or":
        return "OR"
    if lower == "and/or":
        return "AND/OR"

    raise ValueError(f"Unknown connection type: {raw_kind!r}")


def validate_allowed_pattern(kind: str, source_type: str, target_type: str, line: str) -> None:
    """Validiert die im Prompt erlaubten Quell-/Ziel-Typen."""
    if kind == "supports":
        allowed = (source_type, target_type) in {("Goal", "Goal"), ("Opportunity", "Goal")}
    elif kind == "hinders":
        allowed = (source_type, target_type) in {
            ("Goal", "Goal"),
            ("Problem", "Goal"),
            ("Constraint", "Goal"),
        }
    elif kind == "contradicts":
        allowed = (source_type, target_type) == ("Goal", "Goal")
    elif kind == "causes":
        allowed = source_type == "Cause" and target_type in {"Problem", "Opportunity"}
    elif kind in CONNECTOR_CONNECTION_TYPES:
        allowed = source_type in {"Goal", "Problem", "Opportunity"} and target_type in {
            "Goal",
            "Problem",
            "Opportunity",
        }
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
      A supports B
      A hinders B
      A contradicts B
      A causes B
      A, B, C AND D
      A, B, C OR D
      A, B, C AND/OR D

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
    """ADL-Attribute fuer Goal-Model-Elemente gemaess den Beispieldateien."""
    common = '''
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

\tATTRIBUTE <Defined by>
\tVALUE ""
'''

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

    if element.type in {"Cause", "Constraint", "Opportunity"}:
        return common

    raise ValueError(f"Unsupported element type: {element.type}")


def layout_position(index: int) -> Tuple[float, float]:
    """Fallback-Layout in Spalten."""
    col = index // 10
    row = index % 10
    x = 3.0 + col * 9.5
    y = 2.5 + row * 2.2
    return x, y


def connector_layout_name(connection_index: int, kind: str) -> str:
    return f"{kind}-AUTO{connection_index}"


def segment_intersects_node(
    start: Tuple[float, float],
    end: Tuple[float, float],
    node: Tuple[float, float],
    width: float = 4.8,
    height: float = 2.1,
) -> bool:
    """Return whether a line segment crosses an expanded node rectangle."""
    x0, y0 = start
    x1, y1 = end
    cx, cy = node
    left, right = cx - width / 2.0, cx + width / 2.0
    top, bottom = cy - height / 2.0, cy + height / 2.0
    dx, dy = x1 - x0, y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - left, right - x0, y0 - top, bottom - y0)
    lower, upper = 0.0, 1.0
    for direction, distance in zip(p, q):
        if abs(direction) < 1e-9:
            if distance < 0:
                return False
            continue
        ratio = distance / direction
        if direction < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return False
    return True


def reduce_edge_node_collisions(
    positions: Dict[str, Tuple[float, float]],
    visual_nodes: set[str],
    connector_names: set[str],
    edges: List[Tuple[str, str, int]],
    x_gap: float,
) -> None:
    """Move blockers horizontally while preserving their hierarchy row."""
    simple_edges = [(source, target) for source, target, _ in edges]

    def collision_count(name: str, point: Tuple[float, float]) -> int:
        width = 1.0 if name in connector_names else 4.8
        height = 1.0 if name in connector_names else 2.1
        count = 0
        for source, target in simple_edges:
            if name in {source, target}:
                continue
            if segment_intersects_node(
                positions[source], positions[target], point, width, height
            ):
                count += 1
        for other in visual_nodes:
            if other == name or abs(positions[other][1] - point[1]) > 0.01:
                continue
            minimum_gap = 2.0 if name in connector_names or other in connector_names else 5.0
            if abs(positions[other][0] - point[0]) < minimum_gap:
                count += 10
        return count

    # Re-evaluate after each move because moving an endpoint changes its edges.
    for _ in range(6):
        changed = False
        for name in sorted(visual_nodes, key=lambda item: (positions[item][1], positions[item][0])):
            original = positions[name]
            original_score = collision_count(name, original)
            if original_score == 0:
                continue
            candidates = [original]
            for step in range(1, 21):
                candidates.append((original[0] - step * x_gap, original[1]))
                candidates.append((original[0] + step * x_gap, original[1]))
            candidates = [point for point in candidates if point[0] >= 3.0]
            best = min(
                candidates,
                key=lambda point: (
                    collision_count(name, point),
                    abs(point[0] - original[0]),
                    point[0],
                ),
            )
            if collision_count(name, best) < original_score:
                positions[name] = best
                changed = True
        if not changed:
            break


def align_one_to_one_leaf_edges(
    positions: Dict[str, Tuple[float, float]],
    visual_nodes: set[str],
    connector_names: set[str],
    edges: List[Tuple[str, str, int]],
) -> None:
    """Place unambiguous leaf-to-parent pairs on the same vertical axis."""
    incoming: Dict[str, List[str]] = {name: [] for name in visual_nodes}
    outgoing: Dict[str, List[str]] = {name: [] for name in visual_nodes}
    for source, target, _ in edges:
        if source in visual_nodes and target in visual_nodes:
            outgoing[source].append(target)
            incoming[target].append(source)

    for source, targets in outgoing.items():
        if (
            source in connector_names
            or len(targets) != 1
            or incoming[source]
        ):
            continue
        target = targets[0]
        if target in connector_names or len(incoming[target]) != 1:
            continue
        candidate = (positions[target][0], positions[source][1])
        # Do not trade vertical alignment for a node overlap on the source row.
        if any(
            other != source
            and abs(positions[other][1] - candidate[1]) < 0.01
            and abs(positions[other][0] - candidate[0]) < 5.0
            for other in visual_nodes
        ):
            continue
        # The newly vertical direct edge must not cross another visible node.
        if any(
            other not in {source, target}
            and segment_intersects_node(
                candidate,
                positions[target],
                positions[other],
                1.0 if other in connector_names else 4.8,
                1.0 if other in connector_names else 2.1,
            )
            for other in visual_nodes
        ):
            continue
        positions[source] = candidate


def compute_layout(elements: List[Element], connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
    """Create a bottom-up hierarchical layout for all Goal Model elements.

    All semantic connection kinds are treated as identical directed layout
    edges. Real elements occupy full hierarchy levels (even half-level ranks),
    while explicit AND/OR/AND-OR connector nodes occupy intermediate ranks.
    Strongly connected nodes are kept on the same row. Long edges are split by
    virtual layout-only nodes so skipped rows reserve a routing corridor.
    """
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
        config=UniformLayoutConfig(orientation="bottom_up", x_start=4.0,
                                   y_start=2.5, node_gap=6.5, level_gap=3.2))

    real_names = [element.name for element in elements]
    real_set = set(real_names)
    connector_names: set[str] = set()
    nodes = list(real_names)
    # Edge tuple: source, target, minimum distance in half-levels.
    edges: List[Tuple[str, str, int]] = []
    input_order: Dict[str, int] = {name: i for i, name in enumerate(real_names)}

    for connection_index, connection in enumerate(connections, start=1):
        if connection.kind in CONNECTOR_CONNECTION_TYPES:
            connector = connector_layout_name(connection_index, connection.kind)
            connector_names.add(connector)
            nodes.append(connector)
            input_order[connector] = len(input_order)
            for source in connection.sources:
                edges.append((source, connector, 1))
            edges.append((connector, connection.target, 1))
        else:
            for source in connection.sources:
                edges.append((source, connection.target, 2))

    outgoing: Dict[str, List[str]] = {name: [] for name in nodes}
    for source, target, _ in edges:
        outgoing[source].append(target)

    # Tarjan SCC: a directed cycle cannot be drawn as a strict hierarchy, so
    # all members of the cycle deliberately share one hierarchy row.
    index = 0
    stack: List[str] = []
    on_stack: set[str] = set()
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    components: List[List[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in outgoing[node]:
            if target not in indices:
                strongconnect(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif target in on_stack:
                lowlink[node] = min(lowlink[node], indices[target])
        if lowlink[node] == indices[node]:
            component: List[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(component)

    for node in nodes:
        if node not in indices:
            strongconnect(node)

    component_of = {
        node: component_index
        for component_index, component in enumerate(components)
        for node in component
    }
    component_edges: Dict[int, Dict[int, int]] = {
        component_index: {} for component_index in range(len(components))
    }
    for source, target, distance in edges:
        source_component = component_of[source]
        target_component = component_of[target]
        if source_component != target_component:
            component_edges[source_component][target_component] = max(
                distance,
                component_edges[source_component].get(target_component, 0),
            )

    component_rank: Dict[int, int] = {}

    def rank_component(component_index: int) -> int:
        if component_index in component_rank:
            return component_rank[component_index]
        minimum = 0
        for target_component, distance in component_edges[component_index].items():
            minimum = max(
                minimum,
                rank_component(target_component) + distance,
            )
        members = components[component_index]
        # Normal elements live on even ranks; connector-only SCCs live on odd
        # ranks. Mixed cyclic SCCs use a normal element row.
        desired_parity = 1 if all(name in connector_names for name in members) else 0
        if minimum % 2 != desired_parity:
            minimum += 1
        component_rank[component_index] = minimum
        return minimum

    rank = {node: rank_component(component_of[node]) for node in nodes}

    # Split long edges into unit half-level segments for ordering only. Virtual
    # nodes are never emitted to ADL, but occupy horizontal slots on skipped
    # rows and therefore keep direct long edges away from real elements.
    routed_edges: List[Tuple[str, str]] = []
    virtual_nodes: set[str] = set()
    virtual_index = 0
    for source, target, _ in edges:
        source_rank = rank[source]
        target_rank = rank[target]
        if component_of[source] == component_of[target] or source_rank <= target_rank + 1:
            routed_edges.append((source, target))
            continue
        previous = source
        for intermediate_rank in range(source_rank - 1, target_rank, -1):
            virtual_index += 1
            virtual = f"__LAYOUT_ROUTE_{virtual_index}"
            virtual_nodes.add(virtual)
            nodes.append(virtual)
            rank[virtual] = intermediate_rank
            input_order[virtual] = len(input_order)
            routed_edges.append((previous, virtual))
            previous = virtual
        routed_edges.append((previous, target))

    adjacency: Dict[str, List[str]] = {name: [] for name in nodes}
    for source, target in routed_edges:
        adjacency[source].append(target)
        adjacency[target].append(source)

    rows: Dict[int, List[str]] = {}
    for node in nodes:
        rows.setdefault(rank[node], []).append(node)
    for row_nodes in rows.values():
        row_nodes.sort(key=lambda name: input_order[name])

    order_value: Dict[str, float] = {}

    def refresh_order() -> None:
        for row_nodes in rows.values():
            for position, node in enumerate(row_nodes):
                order_value[node] = float(position)

    refresh_order()
    row_numbers = sorted(rows)
    # Alternating barycentric sweeps are the standard crossing-reduction step
    # of a Sugiyama-style hierarchical layout.
    for _ in range(12):
        for sweep in (row_numbers, list(reversed(row_numbers))):
            for row_number in sweep:
                row_nodes = rows[row_number]

                def barycenter(node: str) -> Tuple[float, int]:
                    neighbors = adjacency[node]
                    value = (
                        sum(order_value[neighbor] for neighbor in neighbors) / len(neighbors)
                        if neighbors
                        else order_value[node]
                    )
                    return value, input_order[node]

                row_nodes.sort(key=barycenter)
                refresh_order()

    x_start = 3.0
    y_start = 2.5
    x_gap = 6.5
    half_level_gap = 3.2
    max_slots = max(len(row_nodes) for row_nodes in rows.values())

    positions: Dict[str, Tuple[float, float]] = {}
    for row_number, row_nodes in rows.items():
        # Center narrower hierarchy rows over the widest row.
        offset = (max_slots - len(row_nodes)) * x_gap / 2.0
        for column, node in enumerate(row_nodes):
            positions[node] = (
                x_start + offset + column * x_gap,
                y_start + row_number * half_level_gap,
            )

    visual_nodes = real_set | connector_names
    reduce_edge_node_collisions(
        positions,
        visual_nodes,
        connector_names,
        edges,
        x_gap,
    )
    align_one_to_one_leaf_edges(
        positions,
        visual_nodes,
        connector_names,
        edges,
    )

    return {
        name: point
        for name, point in positions.items()
        if name in real_set or name in connector_names
    }


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
    return 4.0, 1.5


def is_connector_connection(conn: Connection) -> bool:
    return conn.kind in CONNECTOR_CONNECTION_TYPES


def connector_class_for(conn: Connection) -> str:
    if conn.kind in CONNECTOR_CONNECTION_TYPES:
        return conn.kind
    raise ValueError(f"Connection {conn.kind!r} is not a connector.")


def relation_type_for(kind: str) -> str:
    mapping = {
        "supports": "Supports",
        "hinders": "Hinders",
        "contradicts": "Contradicts",
        "causes": "Causes",
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
            # A, B, C AND D -> A -> AND, B -> AND, C -> AND, AND -> D
            junction_class = connector_class_for(conn)
            junction_name = connector_layout_name(c_idx, junction_class)
            x, y = positions.get(
                junction_name,
                junction_position(conn.sources, conn.target, positions, len(elements) + c_idx),
            )
            instance_blocks.append(make_instance(junction_name, junction_class, "\n\tATTRIBUTE <External tool coupling>\n\tVALUE \"\"\n", node_index, x, y))
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

    relation_count = sum(len(conn.sources) + 1 if conn.kind in CONNECTOR_CONNECTION_TYPES else len(conn.sources) for conn in connections)

    world_width = max(
        80,
        math.ceil(max((x for x, _ in positions.values()), default=70.0) + 8.0),
    )
    world_height = max(
        80,
        math.ceil(max((y for _, y in positions.values()), default=70.0) + 8.0),
    )

    header = f'''///////////////////////////////////////////////////////////////
//
// Date: {now.strftime("%d.%m.%Y  %H:%M")}
//
// Generated by goal_to_adl.py
// Data version 4.0
//
///////////////////////////////////////////////////////////////
//
// The file contains the following models:
//
// {esc(model_name)} (Goal Model)
//
//////////////////////////////////////////////////////////////

VERSION <4.0>


BUSINESS PROCESS MODEL <{esc(model_name)}> : <4EM current>
VERSION <>
TYPE <Goal Model>

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

\tATTRIBUTE <Iterator: Problem>
\tVALUE {type_counts["Problem"]}

\tATTRIBUTE <Iterator: Goal>
\tVALUE {type_counts["Goal"]}

\tATTRIBUTE <Iterator: Cause>
\tVALUE {type_counts["Cause"]}

\tATTRIBUTE <Iterator: Constraint>
\tVALUE {type_counts["Constraint"]}

\tATTRIBUTE <Iterator: Opportunity>
\tVALUE {type_counts["Opportunity"]}

\tATTRIBUTE <Iterator: Relation>
\tVALUE {relation_count}

\tATTRIBUTE <Iterator: Hinders>
\tVALUE {direct_counts["hinders"]}

\tATTRIBUTE <Iterator: Supports>
\tVALUE {direct_counts["supports"]}

\tATTRIBUTE <Iterator: Contradicts>
\tVALUE {direct_counts["contradicts"]}

\tATTRIBUTE <Iterator: Causes>
\tVALUE {direct_counts["causes"]}

'''

    return header + "\n".join(instance_blocks) + "\n" + "\n".join(relation_blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Goal Model notation with direct connections and AND/OR connectors to a 4EM ADL file."
    )
    parser.add_argument("input", help="Text file containing ELEMENTS and CONNECTIONS")
    parser.add_argument("output", help="Output .adl file")
    parser.add_argument("--model-name", default="Generated Goal Model")
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
