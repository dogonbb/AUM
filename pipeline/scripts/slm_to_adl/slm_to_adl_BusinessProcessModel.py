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
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from scripts.slm_to_adl.hierarchical_layout import LayoutEdge, LayoutNode, LayoutOptions, compute_hierarchical_layout
    from scripts.slm_to_adl.validation import parse_lines_collect, raise_validation_errors, validate_references
except ModuleNotFoundError:
    from hierarchical_layout import LayoutEdge, LayoutNode, LayoutOptions, compute_hierarchical_layout
    from validation import parse_lines_collect, raise_validation_errors, validate_references


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

    validate_references(sources, targets, elements_by_name)

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


def dominant_forward_edges(
    nodes: List[str],
    edges: List[Tuple[str, str, int]],
    input_order: Dict[str, int],
) -> List[Tuple[str, str, int]]:
    """Build the main acyclic flow and omit only cycle-closing feedback arcs.

    SLM process connections are emitted in narrative process order. Keeping
    that order makes a final loop-back edge lose against the already assembled
    continuous main path, instead of accidentally discarding a Split or Join.
    """
    del input_order  # retained in the signature for deterministic API symmetry
    accepted: List[Tuple[str, str, int]] = []
    outgoing: Dict[str, List[str]] = {node: [] for node in nodes}

    def reaches(start: str, wanted: str) -> bool:
        stack = [start]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node == wanted:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(outgoing[node])
        return False

    for edge in edges:
        source, target, _ = edge
        if source == target or reaches(target, source):
            continue
        accepted.append(edge)
        outgoing[source].append(target)
    return accepted


def separate_split_branch_lanes(
    positions: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str, int]],
    split_connectors: List[str],
    input_order: Dict[str, int],
) -> None:
    """Move exclusive split subtrees into non-overlapping horizontal lanes.

    Crossing reduction alone may swap a short terminal branch with a large
    sibling subtree on lower hierarchy rows.  Since ADL writes straight edges,
    that makes the terminal edge run through the sibling subtree.  Processing
    inner splits first and then moving every exclusive descendant set as one
    block preserves each branch as a separate lane.
    """
    outgoing: Dict[str, List[str]] = {}
    for source, target, _ in edges:
        outgoing.setdefault(source, []).append(target)

    descendants_cache: Dict[str, set[str]] = {}

    def descendants(node: str) -> set[str]:
        if node in descendants_cache:
            return set(descendants_cache[node])
        found: set[str] = {node}
        stack = list(outgoing.get(node, []))
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            stack.extend(outgoing.get(current, []))
        descendants_cache[node] = found
        return set(found)

    # Later connectors are normally nested below earlier connectors. Moving
    # them first lets the outer split treat the nested layout as one block.
    for connector in reversed(split_connectors):
        targets = outgoing.get(connector, [])
        if len(targets) < 2 or connector not in positions:
            continue

        reachable = [descendants(target) for target in targets]
        exclusive_groups: List[Tuple[int, int, set[str]]] = []
        for index, group in enumerate(reachable):
            siblings = set().union(
                *(other for other_index, other in enumerate(reachable) if other_index != index)
            )
            exclusive = {node for node in group - siblings if node in positions}
            if exclusive:
                exclusive_groups.append((index, len(exclusive), exclusive))

        if len(exclusive_groups) < 2:
            continue

        # Keep the substantial/main continuation on the left and terminal
        # alternatives on the right. Input order is the deterministic tie-break.
        exclusive_groups.sort(
            key=lambda item: (-item[1], input_order.get(targets[item[0]], item[0]))
        )
        bounds = []
        for _, _, group in exclusive_groups:
            xs = [positions[node][0] for node in group]
            bounds.append((min(xs), max(xs)))

        lane_gap = 8.0
        widths = [maximum - minimum for minimum, maximum in bounds]
        total_width = sum(widths) + lane_gap * (len(widths) - 1)
        cursor = positions[connector][0] - total_width / 2.0
        for (_, _, group), (minimum, _), width in zip(exclusive_groups, bounds, widths):
            shift = cursor - minimum
            for node in group:
                x, y = positions[node]
                positions[node] = (x + shift, y)
            cursor += width + lane_gap


def move_long_edge_targets_around_nodes(
    positions: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str, int]],
    rank: Dict[str, int],
    real_nodes: set[str],
) -> None:
    """Keep straight ADL edges from crossing nodes on skipped hierarchy rows."""
    outgoing: Dict[str, List[str]] = {}
    for source, target, _ in edges:
        outgoing.setdefault(source, []).append(target)

    def descendants(start: str) -> set[str]:
        found = {start}
        stack = list(outgoing.get(start, []))
        while stack:
            node = stack.pop()
            if node in found:
                continue
            found.add(node)
            stack.extend(outgoing.get(node, []))
        return {node for node in found if node in positions}

    # Re-evaluate after every shift because moving one continuation also moves
    # its downstream nodes. A few passes are enough for nested process graphs.
    for _ in range(4):
        changed = False
        for source, target, _ in edges:
            if source not in positions or target not in positions:
                continue
            rank_span = rank[target] - rank[source]
            if rank_span <= 2:
                continue

            source_x, source_y = positions[source]
            target_x, target_y = positions[target]
            blockers = [
                node
                for node in real_nodes
                if node not in {source, target}
                and rank[source] < rank.get(node, -1) < rank[target]
                and node in positions
            ]
            required_target_x = target_x
            for blocker in blockers:
                blocker_x, blocker_y = positions[blocker]
                vertical_fraction = (blocker_y - source_y) / (target_y - source_y)
                if not 0.0 < vertical_fraction < 1.0:
                    continue
                edge_x = source_x + (target_x - source_x) * vertical_fraction
                # BPM nodes are at most 6 cm wide. Four centimetres from the
                # centre includes half the node width plus visible clearance.
                if abs(edge_x - blocker_x) > 4.0:
                    continue
                clear_x = blocker_x + 4.5
                required_target_x = max(
                    required_target_x,
                    source_x + (clear_x - source_x) / vertical_fraction,
                )

            if required_target_x > target_x + 0.01:
                shift = required_target_x - target_x
                for node in descendants(target):
                    x, y = positions[node]
                    positions[node] = (x + shift, y)
                changed = True
        if not changed:
            break


def align_clear_one_to_one_edges(
    positions: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str, int]],
    rank: Dict[str, int],
) -> None:
    """Place unambiguous 1:1 successors directly below their predecessors."""
    outgoing_count: Dict[str, int] = {}
    incoming_count: Dict[str, int] = {}
    for source, target, _ in edges:
        outgoing_count[source] = outgoing_count.get(source, 0) + 1
        incoming_count[target] = incoming_count.get(target, 0) + 1

    for source, target, _ in sorted(edges, key=lambda edge: rank.get(edge[1], 0)):
        if (
            source not in positions
            or target not in positions
            or outgoing_count.get(source) != 1
            or incoming_count.get(target) != 1
        ):
            continue
        desired_x = positions[source][0]
        # Do not create an overlap with a sibling already occupying this row.
        occupied = any(
            node != target
            and rank.get(node) == rank.get(target)
            and abs(point[0] - desired_x) < 7.5
            for node, point in positions.items()
        )
        if not occupied:
            _, target_y = positions[target]
            positions[target] = (desired_x, target_y)


def compute_layout(elements: List[Element], connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
    """Lay out the dominant continuous process flow from top to bottom."""
    if not elements:
        return {}

    layout_nodes = []
    for element in elements:
        width, height = node_size_for(element.type)
        layout_nodes.append(LayoutNode(
            element.name, width or 1.0, height or 1.0,
            element.type in CONNECTOR_ELEMENT_TYPES,
        ))
    layout_edges: List[LayoutEdge] = []
    branches: List[str] = []
    for index, connection in enumerate(connections, start=1):
        if connection.kind in DIRECT_CONNECTION_TYPES:
            layout_edges.append(LayoutEdge(connection.sources[0], connection.targets[0], 2))
            continue
        connector = make_connector_name(connection.kind, index)
        layout_nodes.append(LayoutNode(connector, 1.0, 1.0, True))
        if connection.kind.startswith("Split"):
            branches.append(connector)
            layout_edges.append(LayoutEdge(connection.sources[0], connector, 1))
            layout_edges.extend(LayoutEdge(connector, target, 1) for target in connection.targets)
        else:
            layout_edges.extend(LayoutEdge(source, connector, 1) for source in connection.sources)
            layout_edges.append(LayoutEdge(connector, connection.targets[0], 1))
    return compute_hierarchical_layout(
        layout_nodes, layout_edges, branch_nodes=branches,
        options=LayoutOptions(x_start=4.0, y_start=2.5, node_gap=8.0, half_level_gap=3.2),
    )

    real_names = [element.name for element in elements]
    real_set = set(real_names)
    nodes = list(real_names)
    connector_names = {
        element.name for element in elements if element.type in CONNECTOR_ELEMENT_TYPES
    }
    input_order: Dict[str, int] = {name: index for index, name in enumerate(nodes)}
    # Distances are expressed in half-levels: a normal direct transition spans
    # a full level; a source->connector or connector->target edge spans half.
    edges: List[Tuple[str, str, int]] = []
    synthetic_connectors: set[str] = set()
    split_connectors: List[str] = []
    for connection_index, connection in enumerate(connections, start=1):
        if connection.kind in DIRECT_CONNECTION_TYPES:
            edges.append((connection.sources[0], connection.targets[0], 2))
            continue
        connector = make_connector_name(connection.kind, connection_index)
        synthetic_connectors.add(connector)
        connector_names.add(connector)
        if connection.kind.startswith("Split"):
            split_connectors.append(connector)
        nodes.append(connector)
        input_order[connector] = len(input_order)
        if connection.kind.startswith("Split"):
            edges.append((connection.sources[0], connector, 1))
            edges.extend((connector, target, 1) for target in connection.targets)
        else:
            edges.extend((source, connector, 1) for source in connection.sources)
            edges.append((connector, connection.targets[0], 1))

    forward_edges = dominant_forward_edges(nodes, edges, input_order)
    incoming: Dict[str, List[Tuple[str, int]]] = {node: [] for node in nodes}
    for source, target, distance in forward_edges:
        incoming[target].append((source, distance))

    rank: Dict[str, int] = {}

    def node_rank(node: str) -> int:
        if node in rank:
            return rank[node]
        minimum = max(
            (node_rank(source) + distance for source, distance in incoming[node]),
            default=0,
        )
        desired_parity = 1 if node in connector_names else 0
        if minimum % 2 != desired_parity:
            minimum += 1
        rank[node] = minimum
        return minimum

    for node in nodes:
        node_rank(node)

    # Virtual nodes split long forward edges for crossing reduction and reserve
    # free slots on every skipped hierarchy row. They are layout-only objects.
    routed_edges: List[Tuple[str, str]] = []
    virtual_nodes: set[str] = set()
    virtual_index = 0
    for source, target, _ in forward_edges:
        previous = source
        for intermediate_rank in range(rank[source] + 1, rank[target]):
            virtual_index += 1
            virtual = f"__BPM_LAYOUT_ROUTE_{virtual_index}"
            virtual_nodes.add(virtual)
            nodes.append(virtual)
            rank[virtual] = intermediate_rank
            input_order[virtual] = len(input_order)
            routed_edges.append((previous, virtual))
            previous = virtual
        routed_edges.append((previous, target))

    adjacency: Dict[str, List[str]] = {node: [] for node in nodes}
    for source, target in routed_edges:
        adjacency[source].append(target)
        adjacency[target].append(source)

    rows: Dict[int, List[str]] = {}
    for node in nodes:
        rows.setdefault(rank[node], []).append(node)
    for row_nodes in rows.values():
        row_nodes.sort(key=lambda node: input_order[node])

    order_value: Dict[str, float] = {}

    def refresh_order() -> None:
        for row_nodes in rows.values():
            for position, node in enumerate(row_nodes):
                order_value[node] = float(position)

    refresh_order()
    row_numbers = sorted(rows)
    for _ in range(12):
        for sweep in (row_numbers, list(reversed(row_numbers))):
            for row_number in sweep:
                def key(node: str) -> Tuple[float, int]:
                    neighbors = adjacency[node]
                    center = (
                        sum(order_value[neighbor] for neighbor in neighbors) / len(neighbors)
                        if neighbors else order_value[node]
                    )
                    return center, input_order[node]
                rows[row_number].sort(key=key)
                refresh_order()

    x_start = 4.0
    y_start = 2.5
    x_gap = 8.0
    half_level_gap = 3.2
    max_slots = max(len(row_nodes) for row_nodes in rows.values())
    positions: Dict[str, Tuple[float, float]] = {}
    for row_number, row_nodes in rows.items():
        offset = (max_slots - len(row_nodes)) * x_gap / 2.0
        for column, node in enumerate(row_nodes):
            positions[node] = (
                x_start + offset + column * x_gap,
                y_start + row_number * half_level_gap,
            )

    separate_split_branch_lanes(positions, forward_edges, split_connectors, input_order)
    align_clear_one_to_one_edges(positions, forward_edges, rank)
    move_long_edge_targets_around_nodes(positions, forward_edges, rank, real_set)

    return {
        node: point
        for node, point in positions.items()
        if node in real_set or node in synthetic_connectors
    }


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
        x, y = positions.get(
            junction_name,
            junction_position(conn.sources, conn.targets, positions, len(elements) + c_idx),
        )
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

    world_width = max(
        80,
        math.ceil(max((x for x, _ in positions.values()), default=70.0) + 10.0),
    )
    world_height = max(
        80,
        math.ceil(max((y for _, y in positions.values()), default=70.0) + 10.0),
    )

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
    print(f"Generated ADL relationships: {generated_relations}")


if __name__ == "__main__":
    main()
