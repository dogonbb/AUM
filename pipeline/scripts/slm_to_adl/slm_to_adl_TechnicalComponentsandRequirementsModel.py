#!/usr/bin/env python3
"""Convert the LLM text notation for a 4EM Technical Components and
Requirements Model into an ADL file.

Usage:
  python technical_components_requirements_to_adl.py input.txt output.adl \
      --model-name "Generated Technical Components and Requirements Model"

Accepted element declarations (aliases are accepted):
  Information System Goal <name>
  Information System Problem <name>
  Information System Requirement <name>
  Information System Functional Requirement <name>
  Information System Non-Functional Requirement <name>
  Technical Component <name>

The generated ADL classes follow the supplied 4EM exports:
  Goal, Problem, IS Requirement, IS Technical Component.
"""
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

# Longest aliases first.
ELEMENT_ALIASES = {
    "Information System Non-Functional Requirement": ("IS Requirement", "Non-Functional"),
    "Information System Functional Requirement": ("IS Requirement", "Functional"),
    "Information System Requirement": ("IS Requirement", "Functional"),
    "Information System Problem": ("Problem", None),
    "Information System Goal": ("Goal", None),
    "IS Non-Functional Requirement": ("IS Requirement", "Non-Functional"),
    "IS Functional Requirement": ("IS Requirement", "Functional"),
    "IS Technical Component": ("IS Technical Component", None),
    "IS Requirement": ("IS Requirement", "Functional"),
    "Technical Component": ("IS Technical Component", None),
    "Problem": ("Problem", None),
    "Goal": ("Goal", None),
}

DIRECT_KINDS = {
    "supports", "hinders", "contradicts", "motivates", "has requirement", "has goal",
}
CONNECTOR_KINDS = {"AND", "OR", "AND/OR", "Partial-PartOF", "Total-PartOF"}

TOKEN_RE = re.compile(
    r"\s(AND/OR|Partial-PartOF|Total-PartOF|has requirement|has goal|contradicts|"
    r"motivates|supports|hinders|AND|OR)\s",
    re.IGNORECASE,
)

@dataclass
class Element:
    input_type: str
    name: str
    adl_class: str
    requirement_type: str | None = None

@dataclass
class Connection:
    sources: List[str]
    kind: str
    target: str


def esc(value: str) -> str:
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
    if not elements:
        raise ValueError("No elements found. An ELEMENTS section is required.")
    return elements, connections


def parse_element_line(line: str) -> Element:
    normalized = normalize_spaces(line)
    for alias in sorted(ELEMENT_ALIASES, key=len, reverse=True):
        prefix = alias + " "
        if normalized.lower().startswith(prefix.lower()):
            name = normalize_spaces(normalized[len(prefix):])
            if not name:
                raise ValueError(f"Element name is missing in line: {line!r}")
            adl_class, req_type = ELEMENT_ALIASES[alias]
            return Element(alias, name, adl_class, req_type)
    raise ValueError(
        f"Unknown element type in {line!r}. Allowed: " + ", ".join(ELEMENT_ALIASES)
    )


def normalize_kind(raw: str) -> str:
    value = normalize_spaces(raw)
    low = value.lower()
    mapping = {
        "and": "AND", "or": "OR", "and/or": "AND/OR",
        "partial-partof": "Partial-PartOF", "total-partof": "Total-PartOF",
    }
    return mapping.get(low, low)


def validate_pattern(conn: Connection, elements: Dict[str, Element], line: str) -> None:
    source_classes = [elements[name].adl_class for name in conn.sources]
    target_class = elements[conn.target].adl_class
    kind = conn.kind

    if kind in CONNECTOR_KINDS:
        if len(conn.sources) < 2:
            raise ValueError(f"{kind} requires at least two sources: {line!r}")
        if kind in {"Partial-PartOF", "Total-PartOF"}:
            if any(c != "IS Technical Component" for c in source_classes) or target_class != "IS Technical Component":
                raise ValueError(f"{kind} is allowed only between technical components: {line!r}")
        return

    if len(conn.sources) != 1:
        raise ValueError(f"A direct connection requires exactly one source: {line!r}")
    s = source_classes[0]
    t = target_class

    allowed = False
    if kind == "supports":
        allowed = (s, t) in {("Goal", "Goal"), ("IS Technical Component", "IS Technical Component")}
    elif kind == "hinders":
        allowed = (s, t) in {
            ("Goal", "Goal"), ("Problem", "Goal"),
            ("IS Technical Component", "IS Technical Component"),
        }
    elif kind == "contradicts":
        allowed = s == "Goal" and t == "Goal"
    elif kind == "motivates":
        allowed = s == "Goal" and t in {"IS Technical Component", "IS Requirement"}
    elif kind == "has requirement":
        allowed = (
            (s == "Goal" and t in {"IS Technical Component", "IS Requirement"})
            or (s == "IS Technical Component" and t == "IS Requirement")
        )
    elif kind == "has goal":
        allowed = s == "IS Technical Component" and t == "Goal"

    if not allowed:
        raise ValueError(f"Disallowed connection pattern in {line!r}: {s} {kind} {t}")


def parse_connection_line(line: str, elements: Dict[str, Element]) -> Connection:
    match = TOKEN_RE.search(line)
    if not match:
        raise ValueError(f"No valid connection type found in: {line!r}")
    kind = normalize_kind(match.group(1))
    left = normalize_spaces(line[:match.start()])
    target = normalize_spaces(line[match.end():])
    sources = [normalize_spaces(x) for x in left.split(",") if normalize_spaces(x)]
    if not sources or not target:
        raise ValueError(f"Incomplete connection: {line!r}")
    validate_references(sources, [target], elements)
    conn = Connection(sources, kind, target)
    validate_pattern(conn, elements, line)
    return conn


def parse_notation(text: str) -> Tuple[List[Element], List[Connection]]:
    element_lines, connection_lines = split_sections(text)
    elements, errors = parse_lines_collect(element_lines, parse_element_line, "ELEMENTS")
    by_name: Dict[str, Element] = {}
    for element in elements:
        if element.name in by_name:
            errors.append(f"Duplicate element name: {element.name!r}")
        by_name[element.name] = element
    indexed_connections = []
    for line_number, line in enumerate(connection_lines, start=1):
        try:
            indexed_connections.append((line_number, line, parse_connection_line(line, by_name)))
        except ValueError as exc:
            errors.append(f"CONNECTIONS line {line_number}: {exc}")
    connections = [connection for _, _, connection in indexed_connections]
    seen_pairs: Dict[frozenset[str], Tuple[int, str]] = {}
    for line_number, line, connection in indexed_connections:
        for source in connection.sources:
            pair = frozenset((source, connection.target))
            previous = seen_pairs.get(pair)
            if previous is not None:
                previous_line, previous_kind = previous
                errors.append(
                    "More than one connection between the same two elements: "
                    f"{source!r} and {connection.target!r}. "
                    f"Earlier CONNECTIONS line {previous_line} uses {previous_kind!r}; "
                    f"line {line_number} uses {connection.kind!r}: {line!r}"
                )
            seen_pairs[pair] = (line_number, connection.kind)
    raise_validation_errors(errors)
    return elements, connections


def attrs_for(element: Element) -> str:
    if element.adl_class == "Goal":
        return '''
\tATTRIBUTE <External tool coupling>\n\tVALUE ""
\n\tATTRIBUTE <Description>\n\tVALUE ""
\n\tATTRIBUTE <Criticality>\n\tVALUE "Low"
\n\tATTRIBUTE <Priority>\n\tVALUE "Low"
\n\tATTRIBUTE <Intermodel-Relations>\n\tVALUE
\n\tATTRIBUTE <Decomposition>\n\tVALUE ""
\n\tATTRIBUTE <Defined by>\n\tVALUE ""
\n\tATTRIBUTE <Attributes>\n\tVALUE
'''
    if element.adl_class == "Problem":
        return '''
\tATTRIBUTE <External tool coupling>\n\tVALUE ""
\n\tATTRIBUTE <Intermodel-Relations>\n\tVALUE
\n\tATTRIBUTE <Decomposition>\n\tVALUE ""
\n\tATTRIBUTE <Attributes>\n\tVALUE
\n\tATTRIBUTE <Defined by>\n\tVALUE ""
\n\tATTRIBUTE <Priority>\n\tVALUE "Low"
\n\tATTRIBUTE <Criticality>\n\tVALUE "Low"
\n\tATTRIBUTE <Description>\n\tVALUE ""
\n\tATTRIBUTE <type>\n\tVALUE "Problem"
'''
    if element.adl_class == "IS Technical Component":
        return '''
\tATTRIBUTE <External tool coupling>\n\tVALUE ""
\n\tATTRIBUTE <Description>\n\tVALUE ""
\n\tATTRIBUTE <Intermodel-Relations>\n\tVALUE
\n\tATTRIBUTE <Decomposition>\n\tVALUE ""
\n\tATTRIBUTE <Location>\n\tVALUE ""
\n\tATTRIBUTE <Quantity>\n\tVALUE 0
\n\tATTRIBUTE <Attributes>\n\tVALUE
'''
    if element.adl_class == "IS Requirement":
        req_type = element.requirement_type or "Functional"
        # The 4EM reference exports prove "Functional" as an enumeration value.
        # "Non-Functional" is not part of the importer enumeration and causes
        # "Wrong enumeration value"; an empty enumeration is rejected as well.
        # Omit the optional attribute instead of misclassifying the requirement.
        type_attribute = (
            '\n\tATTRIBUTE <Type>\n\tVALUE "Functional"\n'
            if req_type == "Functional" else ""
        )
        return f'''
\tATTRIBUTE <External tool coupling>\n\tVALUE ""
\n\tATTRIBUTE <Description>\n\tVALUE ""
{type_attribute}
\n\tATTRIBUTE <Intermodel-Relations>\n\tVALUE
\n\tATTRIBUTE <Decomposition>\n\tVALUE ""
\n\tATTRIBUTE <Attributes>\n\tVALUE
'''
    raise ValueError(f"Unsupported ADL class: {element.adl_class}")


def size_for(adl_class: str) -> Tuple[float, float]:
    return (4.0, 2.0) if adl_class in {"IS Requirement", "IS Technical Component"} else (4.0, 1.5)


def connector_layout_name(connection_index: int, kind: str) -> str:
    return f"{kind}-AUTO{connection_index}"


def layout_components(elements: List[Element], connections: List[Connection]) -> List[List[str]]:
    """Analyse weakly connected components including visible connector nodes."""
    nodes = [element.name for element in elements]
    adjacency: Dict[str, Set[str]] = {name: set() for name in nodes}

    def link(left: str, right: str) -> None:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    for index, connection in enumerate(connections, start=1):
        if connection.kind in CONNECTOR_KINDS:
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


def move_long_edges_around_nodes(
    positions: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str, int]],
    rank: Dict[str, int],
    real_nodes: Set[str],
) -> None:
    """Shift long-edge targets so straight ADL edges do not cross nodes."""
    outgoing: Dict[str, Set[str]] = {}
    for source, target, _ in edges:
        outgoing.setdefault(source, set()).add(target)

    def descendants(start: str) -> Set[str]:
        found = {start}
        stack = list(outgoing.get(start, set()))
        while stack:
            node = stack.pop()
            if node in found:
                continue
            found.add(node)
            stack.extend(outgoing.get(node, set()))
        return {node for node in found if node in positions}

    for _ in range(4):
        changed = False
        for source, target, _ in edges:
            if source not in positions or target not in positions:
                continue
            if rank[target] - rank[source] <= 2:
                continue
            source_x, source_y = positions[source]
            target_x, target_y = positions[target]
            left_limit = target_x
            right_limit = target_x
            collision = False
            for blocker in real_nodes:
                if (
                    blocker in {source, target}
                    or blocker not in positions
                    or not rank[source] < rank[blocker] < rank[target]
                ):
                    continue
                blocker_x, blocker_y = positions[blocker]
                fraction = (blocker_y - source_y) / (target_y - source_y)
                if not 0.0 < fraction < 1.0:
                    continue
                edge_x = source_x + (target_x - source_x) * fraction
                if abs(edge_x - blocker_x) > 3.0:
                    continue
                collision = True
                left_limit = min(
                    left_limit,
                    source_x + (blocker_x - 3.5 - source_x) / fraction,
                )
                right_limit = max(
                    right_limit,
                    source_x + (blocker_x + 3.5 - source_x) / fraction,
                )
            if not collision:
                continue
            shift_left = left_limit - target_x
            shift_right = right_limit - target_x
            shift = shift_left if abs(shift_left) <= abs(shift_right) else shift_right
            for node in descendants(target):
                x, y = positions[node]
                positions[node] = (x + shift, y)
            changed = True
        if not changed:
            break


def compute_layout(elements: List[Element], connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
    """Create a top-down forest with connectors on intermediate half-levels."""
    if not elements:
        return {}

    layout_nodes = [LayoutNode(element.name, *size_for(element.adl_class)) for element in elements]
    layout_edges: List[LayoutEdge] = []
    branches: List[str] = []
    for index, connection in enumerate(connections, start=1):
        if connection.kind in CONNECTOR_KINDS:
            connector = connector_layout_name(index, connection.kind)
            layout_nodes.append(LayoutNode(connector, 1.0, 1.0, True))
            branches.append(connector)
            layout_edges.extend(LayoutEdge(source, connector, 1) for source in connection.sources)
            layout_edges.append(LayoutEdge(connector, connection.target, 1))
        else:
            layout_edges.extend(
                LayoutEdge(source, connection.target, 2) for source in connection.sources
            )
    return compute_uniform_layout(
        layout_nodes, layout_edges, branch_nodes=branches,
        config=UniformLayoutConfig(orientation="bottom_up", x_start=3.0,
                                   y_start=2.5, node_gap=6.5, level_gap=3.2),
    )

    node_order = {element.name: index for index, element in enumerate(elements)}
    all_nodes = [element.name for element in elements]
    raw_edges: List[Tuple[str, str, int]] = []
    for index, connection in enumerate(connections, start=1):
        if connection.kind in CONNECTOR_KINDS:
            connector = connector_layout_name(index, connection.kind)
            node_order[connector] = len(node_order)
            all_nodes.append(connector)
            raw_edges.append((connection.target, connector, 1))
            raw_edges.extend((connector, source, 1) for source in connection.sources)
        else:
            raw_edges.extend((connection.target, source, 2) for source in connection.sources)

    # Cyclic/backward relations remain in ADL but cannot inflate the hierarchy.
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
    component_x = 3.0
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

        layout_nodes = list(component)
        virtual_nodes: Set[str] = set()
        layout_edges: List[Tuple[str, str]] = []
        for edge_index, (source, target, _) in enumerate(edges):
            previous = source
            for virtual_rank in range(rank[source] + 1, rank[target]):
                virtual = f"__TM_ROUTE_{component_index}_{edge_index}_{virtual_rank}"
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

        widest = max((len(names) for names in rows.values()), default=1)
        component_width = max(7.0, (widest - 1) * node_gap + 5.0)
        center_x = component_x + component_width / 2.0
        for level, names in rows.items():
            start_x = center_x - (len(names) - 1) * node_gap / 2.0
            for index, name in enumerate(names):
                if name not in virtual_nodes:
                    positions[name] = (start_x + index * node_gap, 2.5 + level * half_level_gap)
        real_component_nodes = {name for name in component if name in positions}
        move_long_edges_around_nodes(positions, edges, rank, real_component_nodes)
        minimum_x = min((positions[name][0] for name in real_component_nodes), default=component_x)
        if minimum_x < component_x:
            correction = component_x - minimum_x
            for name in real_component_nodes:
                x, y = positions[name]
                positions[name] = (x + correction, y)
        component_x = max(
            (positions[name][0] for name in real_component_nodes),
            default=component_x + component_width,
        ) + 7.0
    return positions


def make_instance(name: str, cls: str, attrs: str, index: int, x: float, y: float,
                  w: float | None = None, h: float | None = None) -> str:
    pos = f"NODE x:{x:g}cm y:{y:g}cm index:{index}"
    if w is not None and h is not None:
        pos = f"NODE x:{x:g}cm y:{y:g}cm w:{w:g}cm h:{h:g}cm index:{index}"
    return f'''INSTANCE <{esc(name)}> : <{esc(cls)}>

\tATTRIBUTE <Position>
\tVALUE "{pos}"{attrs}
'''


def make_relation(src: str, src_cls: str, dst: str, dst_cls: str, index: int, rel_type: str) -> str:
    return f'''RELATION <4EM_Relation>
\tFROM <{esc(src)}> : <{esc(src_cls)}>
\tTO <{esc(dst)}> : <{esc(dst_cls)}>

\tATTRIBUTE <Positions>
\tVALUE "EDGE 0 index:{index}"

\tATTRIBUTE <Type>
\tVALUE "{esc(rel_type)}"

\tATTRIBUTE <Description>
\tVALUE ""

\tATTRIBUTE <IR>
\tVALUE "False"

'''


def relation_type(conn: Connection, classes: Dict[str, str]) -> str:
    kind = conn.kind
    s = classes[conn.sources[0]]
    t = classes[conn.target]
    if kind == "supports":
        return "supports" if s == "IS Technical Component" and t == "IS Technical Component" else "Supports"
    if kind == "hinders":
        return "hinders" if s == "IS Technical Component" and t == "IS Technical Component" else "Hinders"
    if kind == "contradicts":
        return "Contradicts"
    return kind


def generate_adl(elements: List[Element], connections: List[Connection], model_name: str) -> str:
    now = datetime.now()
    created = now.strftime("%d.%m.%Y, %H:%M")
    changed = now.strftime("%d.%m.%Y, %H:%M:%S")
    positions = compute_layout(elements, connections)
    world_width = max(80, int(max((x for x, _ in positions.values()), default=70.0) + 10.0))
    world_height = max(80, int(max((y for _, y in positions.values()), default=70.0) + 10.0))
    classes = {e.name: e.adl_class for e in elements}
    instances: List[str] = []
    relations: List[str] = []
    node_index = 1
    for e in elements:
        x, y = positions[e.name]
        w, h = size_for(e.adl_class)
        instances.append(make_instance(e.name, e.adl_class, attrs_for(e), node_index, x, y, w, h))
        node_index += 1

    edge_index = 1
    for c_idx, conn in enumerate(connections, start=1):
        if conn.kind in CONNECTOR_KINDS:
            cls = conn.kind
            junction = connector_layout_name(c_idx, cls)
            x, y = positions[junction]
            instances.append(make_instance(junction, cls, '\n\tATTRIBUTE <External tool coupling>\n\tVALUE ""\n', node_index, x, y))
            node_index += 1
            classes[junction] = cls
            for source in conn.sources:
                relations.append(make_relation(source, classes[source], junction, cls, edge_index, ""))
                edge_index += 1
            relations.append(make_relation(junction, cls, conn.target, classes[conn.target], edge_index, ""))
            edge_index += 1
        else:
            relations.append(make_relation(conn.sources[0], classes[conn.sources[0]], conn.target,
                                           classes[conn.target], edge_index, relation_type(conn, classes)))
            edge_index += 1

    counts = {"Goal": 0, "Problem": 0, "IS Technical Component": 0, "IS Requirement": 0}
    for e in elements:
        counts[e.adl_class] += 1
    rel_counts = {"Hinders": 0, "Supports": 0, "Contradicts": 0, "Motivates": 0,
                  "has requirement": 0, "has goal": 0, "Others": 0}
    relation_total = 0
    for c in connections:
        if c.kind in CONNECTOR_KINDS:
            relation_total += len(c.sources) + 1
            rel_counts["Others"] += len(c.sources) + 1
        else:
            relation_total += 1
            typ = relation_type(c, classes)
            if typ in {"Hinders", "hinders"}: rel_counts["Hinders"] += 1
            elif typ in {"Supports", "supports"}: rel_counts["Supports"] += 1
            elif typ == "Contradicts": rel_counts["Contradicts"] += 1
            elif typ == "motivates": rel_counts["Motivates"] += 1
            elif typ == "has requirement": rel_counts["has requirement"] += 1
            elif typ == "has goal": rel_counts["has goal"] += 1
            else: rel_counts["Others"] += 1

    header = f'''///////////////////////////////////////////////////////////////
//
// Date: {now.strftime("%d.%m.%Y  %H:%M")}
//
// Generated by technical_components_requirements_to_adl.py
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

\tATTRIBUTE <Author>\n\tVALUE "Admin"

\tATTRIBUTE <Creation date>\n\tVALUE "{created}"

\tATTRIBUTE <Date last changed>\n\tVALUE "{changed}"

\tATTRIBUTE <Last user>\n\tVALUE "Admin"

\tATTRIBUTE <Keywords>\n\tVALUE ""

\tATTRIBUTE <Comment>\n\tVALUE ""

\tATTRIBUTE <Model type>\n\tVALUE "Current model"

\tATTRIBUTE <State>\n\tVALUE "In process"

\tATTRIBUTE <Reviewed on>\n\tVALUE ""

\tATTRIBUTE <Reviewed by>\n\tVALUE ""

\tATTRIBUTE <Description>\n\tVALUE ""

\tATTRIBUTE <World area>\n\tVALUE "w:{world_width}cm h:{world_height}cm minw:5cm minh:5cm"

\tATTRIBUTE <Grid>\n\tVALUE ""

\tATTRIBUTE <Zoom>\n\tVALUE 100

\tATTRIBUTE <Viewable area>\n\tVALUE "VIEW representation:graphic
GRAPHIC x:0 y:0 w:1376 h:828 scale:1
TABLE
"

\tATTRIBUTE <Current mode>\n\tVALUE ""

\tATTRIBUTE <Access state>\n\tVALUE "write"

\tATTRIBUTE <Current page layout>\n\tVALUE ""

\tATTRIBUTE <Connector marks>\n\tVALUE ""

\tATTRIBUTE <Change counter>\n\tVALUE 1

\tATTRIBUTE <Font size>\n\tVALUE 0

\tATTRIBUTE <Context of version>\n\tVALUE ""

\tATTRIBUTE <Position>\n\tVALUE ""

\tATTRIBUTE <External tool coupling>\n\tVALUE ""

\tATTRIBUTE <IteratorModeltype>\n\tVALUE ""

\tATTRIBUTE <Iterator: Problem>\n\tVALUE {counts["Problem"]}

\tATTRIBUTE <Iterator: Goal>\n\tVALUE {counts["Goal"]}

\tATTRIBUTE <Iterator: Cause>\n\tVALUE 0

\tATTRIBUTE <Iterator: Constraint>\n\tVALUE 0

\tATTRIBUTE <Iterator: Opportunity>\n\tVALUE 0

\tATTRIBUTE <Iterator: IS Technical Component>\n\tVALUE {counts["IS Technical Component"]}

\tATTRIBUTE <Iterator: IS Requirement>\n\tVALUE {counts["IS Requirement"]}

\tATTRIBUTE <Iterator: Component>\n\tVALUE 0

\tATTRIBUTE <Iterator: Feature>\n\tVALUE 0

\tATTRIBUTE <Iterator: Unspecific/Product/Service>\n\tVALUE 0

\tATTRIBUTE <Iterator: Relation>\n\tVALUE {relation_total}

\tATTRIBUTE <Iterator: Hinders>\n\tVALUE {rel_counts["Hinders"]}

\tATTRIBUTE <Iterator: Supports>\n\tVALUE {rel_counts["Supports"]}

\tATTRIBUTE <Iterator: Contradicts>\n\tVALUE {rel_counts["Contradicts"]}

\tATTRIBUTE <Iterator: Causes>\n\tVALUE 0

\tATTRIBUTE <Iterator: Others>\n\tVALUE {rel_counts["Others"]}

\tATTRIBUTE <Iterator: Motivates>\n\tVALUE {rel_counts["Motivates"]}

\tATTRIBUTE <Iterator: Requires>\n\tVALUE 0

\tATTRIBUTE <Iterator: has requirement>\n\tVALUE {rel_counts["has requirement"]}

\tATTRIBUTE <Iterator: has goal>\n\tVALUE {rel_counts["has goal"]}

'''
    return header + "\n".join(instances) + "\n" + "\n".join(relations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 4EM Technical Components and Requirements Model LLM notation to ADL.")
    parser.add_argument("input", help="Text file containing ELEMENTS and CONNECTIONS")
    parser.add_argument("output", help="Output .adl file")
    parser.add_argument("--model-name", default="Generated Technical Components and Requirements Model")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    elements, connections = parse_notation(input_path.read_text(encoding="utf-8"))
    components = layout_components(elements, connections)
    element_names = {element.name for element in elements}
    sizes = ", ".join(str(sum(name in element_names for name in component)) for component in components)
    print(f"Graph analysis: {len(components)} connected component(s) (element counts: {sizes}).")
    output_path.write_text(generate_adl(elements, connections, args.model_name), encoding="utf-8")
    print(f"OK: read {len(elements)} elements and {len(connections)} notation connections.")
    print(f"ADL written to: {output_path}")

if __name__ == "__main__":
    main()
