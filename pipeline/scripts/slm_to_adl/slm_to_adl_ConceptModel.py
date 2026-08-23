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
from typing import Dict, List, Set, Tuple
try:
    from scripts.slm_to_adl.hierarchical_layout import LayoutEdge, LayoutNode, UniformLayoutConfig, compute_uniform_layout
    from scripts.slm_to_adl.validation import parse_lines_collect, raise_validation_errors, validate_references
except ModuleNotFoundError:
    from hierarchical_layout import LayoutEdge, LayoutNode, UniformLayoutConfig, compute_uniform_layout
    from validation import parse_lines_collect, raise_validation_errors, validate_references


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

    raise ValueError(f"Unknown connection type: {raw_kind!r}")


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
            f"Disallowed connection pattern in {line!r}: "
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
        raise ValueError(f"No valid connection type found in: {line!r}. Allowed: {allowed}")

    kind = normalize_connection_kind(match.group(1))
    left = normalize_spaces(line[:match.start()])
    target = normalize_spaces(line[match.end():])

    sources = [normalize_spaces(x) for x in left.split(",") if normalize_spaces(x)]
    if not sources:
        raise ValueError(f"Connection has no source: {line!r}")

    validate_references(sources, [target], elements_by_name)
    if target in sources:
        raise ValueError(
            f"Connection target {target!r} must not also occur in its source list: {line!r}"
        )

    if kind in DIRECT_CONNECTION_TYPES and len(sources) != 1:
        raise ValueError(f"Direct connection {kind!r} must have exactly one source: {line!r}")
    if len(set(sources)) != len(sources):
        raise ValueError(f"Connection contains duplicate source names: {line!r}")

    for source in sources:
        validate_allowed_pattern(kind, elements_by_name[source].type, elements_by_name[target].type, line)

    return Connection(sources, kind, target)


def parse_notation(text: str) -> Tuple[List[Element], List[Connection]]:
    element_lines, connection_lines = split_sections(text)
    errors = []
    if not element_lines:
        errors.append("No ELEMENTS section or no elements found.")
    elements, element_errors = parse_lines_collect(element_lines, parse_element_line, "ELEMENTS")
    errors.extend(element_errors)

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

    raise ValueError(f"Unsupported element type: {element.type}")


def layout_position(index: int) -> Tuple[float, float]:
    """Fallback-Layout in Spalten."""
    col = index // 10
    row = index % 10
    x = 3.0 + col * 9.5
    y = 2.5 + row * 2.2
    return x, y


def concept_components(
    elements: List[Element], connections: List[Connection]
) -> List[List[str]]:
    """Return weakly connected concept components in deterministic order.

    Attributes do not connect concept components.  A ``has attribute`` edge only
    assigns an attribute to the visual block of its owning concept.
    """
    concepts = [element.name for element in elements if element.type == "Concept"]
    concept_set = set(concepts)
    adjacency: Dict[str, Set[str]] = {name: set() for name in concepts}

    for conn in connections:
        if conn.kind == "has attribute" or conn.target not in concept_set:
            continue
        for source in conn.sources:
            if source in concept_set:
                adjacency[source].add(conn.target)
                adjacency[conn.target].add(source)

    components: List[List[str]] = []
    visited: Set[str] = set()
    for root in concepts:
        if root in visited:
            continue
        component: List[str] = []
        stack = [root]
        visited.add(root)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in reversed(concepts):
                if neighbor in adjacency[current] and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _forward_concept_edges(
    concept_names: List[str], connections: List[Connection]
) -> List[Tuple[str, str]]:
    """Keep a stable acyclic main direction for hierarchical placement."""
    concept_set = set(concept_names)
    accepted: List[Tuple[str, str]] = []
    outgoing: Dict[str, Set[str]] = {name: set() for name in concept_names}

    def reaches(start: str, wanted: str) -> bool:
        stack = [start]
        seen: Set[str] = set()
        while stack:
            current = stack.pop()
            if current == wanted:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(outgoing[current])
        return False

    for conn in connections:
        if conn.kind == "has attribute" or conn.target not in concept_set:
            continue
        for source in conn.sources:
            if source not in concept_set or source == conn.target:
                continue
            # Feedback relations remain in the ADL, but do not invert or inflate
            # the visual hierarchy.
            if reaches(conn.target, source):
                continue
            if conn.target not in outgoing[source]:
                outgoing[source].add(conn.target)
                accepted.append((source, conn.target))
    return accepted


def _segment_intersects_box(
    start: Tuple[float, float],
    end: Tuple[float, float],
    center: Tuple[float, float],
    width: float,
    height: float,
    margin: float = 0.35,
) -> bool:
    """Return whether a line segment enters an expanded axis-aligned node box."""
    left = center[0] - width / 2.0 - margin
    right = center[0] + width / 2.0 + margin
    top = center[1] - height / 2.0 - margin
    bottom = center[1] + height / 2.0 + margin
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    lower, upper = 0.0, 1.0
    for p, q in ((-dx, start[0] - left), (dx, right - start[0]),
                 (-dy, start[1] - top), (dy, bottom - start[1])):
        if abs(p) < 1e-9:
            if q < 0:
                return False
            continue
        ratio = q / p
        if p < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return False
    return True


def compute_layout(elements: List[Element], connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
    """Lay out concept components top-down and attach attributes below owners.

    The graph is analysed as weakly connected components before placement.  Only
    concept-to-concept relations define hierarchy levels.  Attributes are placed
    in compact rows below their concept, with a clear vertical connection corridor
    through the centre of the concept block.
    """
    if not elements:
        return {}

    # Concepts and attributes use the same shared hierarchy rules as every
    # other model. ``has attribute`` is a normal parent-to-child edge; only
    # parsing, element sizes and ADL classes remain concept-specific.
    normal_nodes = [LayoutNode(element.name, *node_size_for(element.type)) for element in elements]
    normal_edges: List[LayoutEdge] = []
    normal_branches: List[str] = []
    for index, connection in enumerate(connections, start=1):
        if connection.kind in CONNECTOR_CONNECTION_TYPES:
            connector = f"{connection.kind}-AUTO{index}"
            normal_nodes.append(LayoutNode(connector, 1.0, 1.0, True))
            normal_branches.append(connector)
            normal_edges.extend(
                LayoutEdge(source, connector, 1) for source in connection.sources
            )
            normal_edges.append(LayoutEdge(connector, connection.target, 1))
        else:
            normal_edges.extend(
                LayoutEdge(source, connection.target, 2) for source in connection.sources
            )
    return compute_uniform_layout(
        normal_nodes,
        normal_edges,
        branch_nodes=normal_branches,
        config=UniformLayoutConfig(
            orientation="top_down",
            x_start=3.0,
            y_start=2.5,
            node_gap=9.0,
            level_gap=4.5,
        ),
    )

    attributes = {e.name for e in elements if e.type == "Attribute"}
    concepts = [e for e in elements if e.type != "Attribute"]
    nodes = [LayoutNode(e.name, *node_size_for(e.type)) for e in concepts]
    edges: List[LayoutEdge] = []
    branches: List[str] = []
    owned_attributes: Dict[str, List[str]] = {}
    owned: Set[str] = set()
    for index, connection in enumerate(connections, start=1):
        if connection.kind == "has attribute":
            if connection.target in attributes and connection.sources:
                owned_attributes.setdefault(connection.sources[0], []).append(connection.target)
                owned.add(connection.target)
            continue
        if connection.kind in CONNECTOR_CONNECTION_TYPES:
            connector = f"{connection.kind}-AUTO{index}"
            nodes.append(LayoutNode(connector, 1.0, 1.0, True))
            branches.append(connector)
            edges.append(LayoutEdge(connection.target, connector, 1))
            edges.extend(LayoutEdge(connector, source, 1) for source in connection.sources)
        else:
            edges.extend(LayoutEdge(source, connection.target, 2) for source in connection.sources)
    positions = compute_hierarchical_layout(nodes, edges, branch_nodes=branches,
        options=LayoutOptions(
            x_start=3.0, y_start=2.5, node_gap=9.0, half_level_gap=4.5,
            component_bottom_reserve=3.5,
        ))

    # Attributes remain attachments, not hierarchy nodes. Place them on the
    # less occupied side of their owner and outside all concept-edge corridors.
    for owner, attrs in owned_attributes.items():
        if owner not in positions:
            continue
        owner_x, owner_y = positions[owner]
        attribute_y = owner_y + 2.6
        crossings: List[float] = []
        for edge in edges:
            if edge.source not in positions or edge.target not in positions:
                continue
            start, end = positions[edge.source], positions[edge.target]
            if abs(end[1] - start[1]) < 1e-9:
                continue
            ratio = (attribute_y - start[1]) / (end[1] - start[1])
            if 0.0 <= ratio <= 1.0:
                crossings.append(start[0] + ratio * (end[0] - start[0]))
        left = max(4.0, owner_x - min(crossings, default=owner_x) + 3.8)
        right = max(4.0, max(crossings, default=owner_x) - owner_x + 3.8)
        direction = -1.0 if left <= right else 1.0
        distance = min(left, right)
        for attribute in attrs:
            positions[attribute] = (owner_x + direction * distance, attribute_y)
            distance = (distance + 3.2) / ((2.6 - 0.85) / 2.6)

    # Re-pack complete concept trees by their actual rectangles. Attribute
    # attachments may extend far beyond a concept centre, so packing only the
    # hierarchy nodes is insufficient to guarantee separation.
    element_by_name = {element.name: element for element in elements}
    connector_names = {
        f"{connection.kind}-AUTO{index}"
        for index, connection in enumerate(connections, start=1)
        if connection.kind in CONNECTOR_CONNECTION_TYPES
    }
    packed_groups: List[Set[str]] = []
    for component in concept_components(elements, connections):
        group = set(component)
        for owner in component:
            group.update(owned_attributes.get(owner, []))
        for connector in connector_names:
            if connector in positions:
                # A connector belongs to the component containing any endpoint.
                connection_index = int(connector.rsplit("AUTO", 1)[1]) - 1
                connection = connections[connection_index]
                if connection.target in group or any(source in group for source in connection.sources):
                    group.add(connector)
        packed_groups.append({name for name in group if name in positions})

    pack_x = 3.0
    pack_y = 2.5
    row_bottom = pack_y
    for group_index, group in enumerate(packed_groups):
        def size(name: str) -> Tuple[float, float]:
            if name in connector_names:
                return 1.0, 1.0
            return node_size_for(element_by_name[name].type)

        left = min(positions[name][0] - size(name)[0] / 2 for name in group)
        top = min(positions[name][1] - size(name)[1] / 2 for name in group)
        shift_x = pack_x - left
        shift_y = pack_y - top
        for name in group:
            x, y = positions[name]
            positions[name] = (x + shift_x, y + shift_y)
        right = max(positions[name][0] + size(name)[0] / 2 for name in group)
        bottom = max(positions[name][1] + size(name)[1] / 2 for name in group)
        row_bottom = max(row_bottom, bottom)
        if (group_index + 1) % 3 == 0:
            pack_x = 3.0
            pack_y = row_bottom + 8.0
            row_bottom = pack_y
        else:
            pack_x = right + 8.0

    orphan_x = max((x for x, _ in positions.values()), default=3.0) + 8.0
    for index, element in enumerate(e for e in elements if e.name in attributes - owned):
        positions[element.name] = (orphan_x + (index % 4) * 5.8, 2.5 + (index // 4) * 1.55)
    minimum_x = min((positions[e.name][0] - node_size_for(e.type)[0] / 2 for e in elements), default=2.0)
    if minimum_x < 2.0:
        shift = 2.0 - minimum_x
        positions = {name: (x + shift, y) for name, (x, y) in positions.items()}
    return positions

    input_order = {element.name: i for i, element in enumerate(elements)}
    positions: Dict[str, Tuple[float, float]] = {}

    attributes = {element.name for element in elements if element.type == "Attribute"}
    owned_attributes: Dict[str, List[str]] = {}
    attribute_owner: Dict[str, str] = {}
    for conn in connections:
        if conn.kind != "has attribute" or conn.target not in attributes:
            continue
        owner = conn.sources[0]
        if conn.target not in attribute_owner:
            attribute_owner[conn.target] = owner
            owned_attributes.setdefault(owner, []).append(conn.target)

    components = concept_components(elements, connections)
    component_x = 3.0
    concept_y = 2.5
    concept_gap = 9.0
    attribute_y_gap = 1.55
    attribute_x_gap = 5.8
    component_gap = 8.0

    for component in components:
        forward_edges = _forward_concept_edges(component, connections)
        layer = {name: 0 for name in component}
        for _ in component:
            changed = False
            for source, target in forward_edges:
                wanted = layer[source] + 1
                if layer[target] < wanted:
                    layer[target] = wanted
                    changed = True
            if not changed:
                break

        predecessors: Dict[str, List[str]] = {name: [] for name in component}
        successors: Dict[str, List[str]] = {name: [] for name in component}
        for source, target in forward_edges:
            predecessors[target].append(source)
            successors[source].append(target)

        rows: Dict[int, List[str]] = {}
        for name in component:
            rows.setdefault(layer[name], []).append(name)

        order = {name: float(input_order[name]) for name in component}
        # Alternating barycentric sweeps reduce crossings between adjacent rows.
        for _ in range(8):
            for level in sorted(rows):
                rows[level].sort(
                    key=lambda name: (
                        sum(order[p] for p in predecessors[name]) / len(predecessors[name])
                        if predecessors[name] else order[name],
                        input_order[name],
                    )
                )
                for index, name in enumerate(rows[level]):
                    order[name] = float(index)
            for level in sorted(rows, reverse=True):
                rows[level].sort(
                    key=lambda name: (
                        sum(order[s] for s in successors[name]) / len(successors[name])
                        if successors[name] else order[name],
                        input_order[name],
                    )
                )
                for index, name in enumerate(rows[level]):
                    order[name] = float(index)

        max_row_width = 7.0
        for names in rows.values():
            widths = []
            for name in names:
                count = len(owned_attributes.get(name, []))
                columns = min(4, max(1, count))
                widths.append(max(7.0, columns * attribute_x_gap + 2.0 if count else 7.0))
            max_row_width = max(max_row_width, sum(widths) + 3.0 * max(0, len(widths) - 1))

        component_center = component_x + max_row_width / 2.0
        for level, names in rows.items():
            block_widths: List[float] = []
            for name in names:
                count = len(owned_attributes.get(name, []))
                columns = min(4, max(1, count))
                block_widths.append(max(7.0, columns * attribute_x_gap + 2.0 if count else 7.0))
            row_width = sum(block_widths) + 3.0 * max(0, len(names) - 1)
            cursor = component_center - row_width / 2.0
            for name, block_width in zip(names, block_widths):
                center_x = cursor + block_width / 2.0
                positions[name] = (center_x, concept_y + level * concept_gap)
                cursor += block_width + 3.0
        component_x += max_row_width + component_gap

    concept_set = {element.name for element in elements if element.type == "Concept"}
    concept_edges = [
        (source, conn.target)
        for conn in connections
        if conn.kind != "has attribute" and conn.target in concept_set
        for source in conn.sources
        if source in concept_set and source != conn.target
    ]

    # Attributes return to a simple row below their concept. The whole row is put
    # outside the horizontal corridor occupied by tree edges at that height. On
    # one side, distances grow sufficiently fast that an edge to an outer
    # attribute cannot pass through an inner attribute box.
    for owner, attrs in owned_attributes.items():
        if owner not in positions:
            continue
        owner_x, owner_y = positions[owner]
        attribute_y = owner_y + 2.6
        edge_x_values: List[float] = []
        for source, target in concept_edges:
            start = positions[source]
            end = positions[target]
            if abs(end[1] - start[1]) < 1e-9:
                continue
            for sample_y in (attribute_y - 0.95, attribute_y, attribute_y + 0.95):
                ratio = (sample_y - start[1]) / (end[1] - start[1])
                if 0.0 <= ratio <= 1.0:
                    edge_x_values.append(start[0] + ratio * (end[0] - start[0]))

        left_clearance = max(
            4.0,
            owner_x - min(edge_x_values, default=owner_x) + 3.8,
        )
        right_clearance = max(
            4.0,
            max(edge_x_values, default=owner_x) - owner_x + 3.8,
        )
        direction = -1.0 if left_clearance <= right_clearance else 1.0
        distance = min(left_clearance, right_clearance)
        # At the upper edge of an attribute node, an attribute line has travelled
        # this fraction of the way from its owner. This recurrence guarantees the
        # next outer line clears the previous attribute rectangle.
        arrival_fraction = (2.6 - 0.85) / 2.6
        for attribute in attrs:
            positions[attribute] = (owner_x + direction * distance, attribute_y)
            distance = (distance + 3.2) / arrival_fraction

    # Orphan attributes form compact blocks after all concept components.
    orphan_attributes = [
        element.name
        for element in elements
        if element.type == "Attribute" and element.name not in attribute_owner
    ]
    for index, name in enumerate(orphan_attributes):
        positions[name] = (
            component_x + (index % 4) * attribute_x_gap,
            concept_y + (index // 4) * attribute_y_gap,
        )

    # Attribute rows may extend beyond the initial canvas origin. Translate
    # the complete drawing together so every node remains importable and visible.
    minimum_x = min((point[0] - node_size_for(next(
        element.type for element in elements if element.name == name
    ))[0] / 2.0 for name, point in positions.items()), default=0.0)
    minimum_y = min((point[1] - node_size_for(next(
        element.type for element in elements if element.name == name
    ))[1] / 2.0 for name, point in positions.items()), default=0.0)
    shift_x = max(0.0, 2.0 - minimum_x)
    shift_y = max(0.0, 2.0 - minimum_y)
    if shift_x or shift_y:
        positions = {
            name: (point[0] + shift_x, point[1] + shift_y)
            for name, point in positions.items()
        }

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
    raise ValueError(f"Connection {conn.kind!r} is not a connector.")


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
        if is_connector_connection(conn):
            # Connector-Zeilen werden als echte 4EM-Connector-Knoten erzeugt:
            # A, B, C Partial-ISA D -> A -> Partial-ISA, B -> Partial-ISA, C -> Partial-ISA, Partial-ISA -> D
            junction_class = connector_class_for(conn)
            junction_name = f"{junction_class}-AUTO{c_idx}"
            x, y = positions.get(
                junction_name,
                junction_position(conn.sources, conn.target, positions, len(elements) + c_idx),
            )
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
            raise ValueError(f"Unsupported connection: {conn.kind}")

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
        description="Convert Concept Model notation with direct connections and ISA/PartOF connectors to a 4EM ADL file."
    )
    parser.add_argument("input", help="Text file containing ELEMENTS and CONNECTIONS")
    parser.add_argument("output", help="Output .adl file")
    parser.add_argument("--model-name", default="Generated Concepts Model")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    elements, connections = parse_notation(text)
    components = concept_components(elements, connections)
    if len(components) == 1:
        print(f"Graph analysis: one connected concept component ({len(components[0])} concepts).")
    else:
        sizes = ", ".join(str(len(component)) for component in components) or "none"
        print(f"Graph analysis: {len(components)} connected concept components (sizes: {sizes}).")
    adl = generate_adl(elements, connections, args.model_name)
    output_path.write_text(adl, encoding="utf-8")

    print(f"OK: read {len(elements)} elements and {len(connections)} notation connections.")
    print(f"ADL written to: {output_path}")


if __name__ == "__main__":
    main()
