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
from typing import Dict, List, Set, Tuple
try:
    from scripts.slm_to_adl.hierarchical_layout import LayoutEdge, LayoutNode, UniformLayoutConfig, compute_uniform_layout
    from scripts.slm_to_adl.validation import parse_lines_collect, raise_validation_errors, validate_references
except ModuleNotFoundError:
    from hierarchical_layout import LayoutEdge, LayoutNode, UniformLayoutConfig, compute_uniform_layout
    from validation import parse_lines_collect, raise_validation_errors, validate_references


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


def parse_connection_line(line: str, elements_by_name: Dict[str, Element]) -> Connection:
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

    known = set(elements_by_name)
    validate_references(sources, [target], known)

    if kind not in CONNECTION_TYPES:
        raise ValueError(f"Unknown connection type: {kind}")
    if kind in DIRECT_CONNECTION_TYPES and len(sources) != 1:
        raise ValueError(f"Direct connection {kind!r} requires exactly one source: {line!r}")
    if kind in DIRECT_CONNECTION_TYPES:
        source_type = elements_by_name[sources[0]].type
        target_type = elements_by_name[target].type
        product_service_types = {"Product", "Service", "ProductService"}
        if kind == "part_of":
            allowed = source_type == "Component" and target_type in product_service_types
        elif kind == "is_a":
            allowed = source_type in product_service_types and target_type in product_service_types
        else:  # requires
            allowed = source_type == "Feature" and target_type in product_service_types | {"Component"}
        if not allowed:
            raise ValueError(
                f"Disallowed connection pattern in {line!r}: "
                f"{source_type} {kind} {target_type}"
            )

    return Connection(sources, kind, target)


def parse_notation(text: str) -> Tuple[List[Element], List[Connection]]:
    element_lines, connection_lines = split_sections(text)
    elements, errors = parse_lines_collect(element_lines, parse_element_line, "ELEMENTS")

    names = [element.name for element in elements]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        errors.append(f"Duplicate element names found: {', '.join(duplicates)}")

    connections, connection_errors = parse_lines_collect(
        connection_lines,
        lambda line: parse_connection_line(line, {element.name: element for element in elements}),
        "CONNECTIONS",
    )
    errors.extend(connection_errors)
    raise_validation_errors(errors)
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


def connector_layout_name(connection_index: int, connector_class: str) -> str:
    return f"{connector_class}-AUTO{connection_index}"


def segment_intersects_node(
    start: Tuple[float, float],
    end: Tuple[float, float],
    center: Tuple[float, float],
    width: float,
    height: float,
    margin: float = 0.2,
) -> bool:
    """Check a straight ADL edge against an expanded node rectangle."""
    left, right = center[0] - width / 2 - margin, center[0] + width / 2 + margin
    top, bottom = center[1] - height / 2 - margin, center[1] + height / 2 + margin
    dx, dy = end[0] - start[0], end[1] - start[1]
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


def layout_components(
    elements: List[Element], connections: List[Connection]
) -> List[List[str]]:
    """Return weakly connected PM components, including connector layout nodes."""
    nodes = [element.name for element in elements]
    adjacency: Dict[str, Set[str]] = {name: set() for name in nodes}

    def link(left: str, right: str) -> None:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    for index, conn in enumerate(connections, start=1):
        if is_connector_connection(conn):
            connector = connector_layout_name(index, connector_class_for(conn))
            if connector not in adjacency:
                nodes.append(connector)
                adjacency[connector] = set()
            link(conn.target, connector)
            for source in conn.sources:
                link(connector, source)
        else:
            for source in conn.sources:
                link(source, conn.target)

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
    """Create a component-aware top-down tree with connector intermediate rows."""
    if not elements:
        return {}
    nodes = [LayoutNode(e.name, *node_size_for(e.type)) for e in elements]
    edges: List[LayoutEdge] = []
    branches: List[str] = []
    for index, connection in enumerate(connections, start=1):
        if is_connector_connection(connection):
            connector = connector_layout_name(index, connector_class_for(connection))
            nodes.append(LayoutNode(connector, 1.0, 1.0, True))
            branches.append(connector)
            edges.extend(LayoutEdge(source, connector, 1) for source in connection.sources)
            edges.append(LayoutEdge(connector, connection.target, 1))
        else:
            edges.extend(LayoutEdge(source, connection.target, 2) for source in connection.sources)
    return compute_uniform_layout(nodes, edges, branch_nodes=branches,
        config=UniformLayoutConfig(orientation="bottom_up", x_start=4.0,
                                   y_start=3.0, node_gap=7.0, level_gap=3.4))

    input_order = {element.name: i for i, element in enumerate(elements)}
    node_order = dict(input_order)
    directed_edges: List[Tuple[str, str, int]] = []
    all_nodes = [element.name for element in elements]
    for index, conn in enumerate(connections, start=1):
        if is_connector_connection(conn):
            connector = connector_layout_name(index, connector_class_for(conn))
            node_order[connector] = len(node_order)
            all_nodes.append(connector)
            directed_edges.append((conn.target, connector, 1))
            directed_edges.extend((connector, source, 1) for source in conn.sources)
        elif conn.kind in {"part_of", "is_a"}:
            directed_edges.extend((conn.target, source, 2) for source in conn.sources)
        else:
            # The required element must visually precede the feature that needs
            # it: "Digital Payment requires Payment Adapter" means Adapter first.
            directed_edges.extend((conn.target, source, 2) for source in conn.sources)

    # Preserve input direction but omit feedback edges only from ranking.
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

    for source, target, distance in directed_edges:
        if source == target or reaches(target, source):
            continue
        if target not in outgoing[source]:
            outgoing[source].add(target)
            accepted.append((source, target, distance))

    positions: Dict[str, Tuple[float, float]] = {}
    element_type = {element.name: element.type for element in elements}
    component_x = 4.0
    half_level_gap = 3.4
    node_gap = 7.0
    for component_index, component in enumerate(layout_components(elements, connections)):
        component_set = set(component)
        component_edges = [edge for edge in accepted if edge[0] in component_set and edge[1] in component_set]
        rank = {name: 0 for name in component}
        for _ in component:
            changed = False
            for source, target, distance in component_edges:
                wanted = rank[source] + distance
                if rank[target] < wanted:
                    rank[target] = wanted
                    changed = True
            if not changed:
                break

        connector_groups = []
        for connection_index, conn in enumerate(connections, start=1):
            if not is_connector_connection(conn):
                continue
            connector = connector_layout_name(connection_index, connector_class_for(conn))
            if connector in component_set:
                connector_groups.append((connector, conn.sources))

        # If one child of a connector is pushed down by a dependency, move the
        # complete connector block with it. Siblings remain on one row and the
        # connector remains exactly one half-level above them.
        for _ in range(max(1, len(component) * 2)):
            changed = False
            for source, target, distance in component_edges:
                wanted = rank[source] + distance
                if rank[target] < wanted:
                    rank[target] = wanted
                    changed = True
            for connector, children in connector_groups:
                child_level = max(rank[child] for child in children)
                wanted_connector = child_level - 1
                if rank[connector] < wanted_connector:
                    rank[connector] = wanted_connector
                    changed = True
                aligned_child_level = rank[connector] + 1
                for child in children:
                    if rank[child] < aligned_child_level:
                        rank[child] = aligned_child_level
                        changed = True
            if not changed:
                break

        # Split every long edge into one-rank segments using invisible nodes.
        # These nodes reserve an empty straight corridor on every skipped row;
        # only the real endpoints are emitted to ADL.
        layout_nodes = list(component)
        virtual_nodes: Set[str] = set()
        layout_edges: List[Tuple[str, str]] = []
        for edge_index, (source, target, _) in enumerate(component_edges):
            previous = source
            for virtual_rank in range(rank[source] + 1, rank[target]):
                virtual = f"__PM_ROUTE_{component_index}_{edge_index}_{virtual_rank}"
                layout_nodes.append(virtual)
                virtual_nodes.add(virtual)
                rank[virtual] = virtual_rank
                node_order[virtual] = len(node_order)
                layout_edges.append((previous, virtual))
                previous = virtual
            layout_edges.append((previous, target))

        layout_adjacency: Dict[str, Set[str]] = {name: set() for name in layout_nodes}
        for source, target in layout_edges:
            layout_adjacency[source].add(target)
            layout_adjacency[target].add(source)

        rows: Dict[int, List[str]] = {}
        for name in layout_nodes:
            rows.setdefault(rank[name], []).append(name)
        for names in rows.values():
            names.sort(key=lambda name: node_order[name])

        order = {name: float(index) for names in rows.values() for index, name in enumerate(names)}
        for _ in range(8):
            for level in sorted(rows):
                names = rows[level]
                names.sort(key=lambda name: (
                    sum(order[n] for n in layout_adjacency[name]) / len(layout_adjacency[name])
                    if layout_adjacency[name] else order[name],
                    node_order[name],
                ))
                for index, name in enumerate(names):
                    order[name] = float(index)

        widest_row = max((len(names) for names in rows.values()), default=1)
        component_width = max(8.0, (widest_row - 1) * node_gap + 5.0)
        center_x = component_x + component_width / 2.0
        for level, names in rows.items():
            row_width = (len(names) - 1) * node_gap
            start_x = center_x - row_width / 2.0
            for index, name in enumerate(names):
                if name not in virtual_nodes:
                    positions[name] = (start_x + index * node_gap, 3.0 + level * half_level_gap)

        def visual_size(name: str) -> Tuple[float, float]:
            return node_size_for(element_type[name]) if name in element_type else (1.4, 1.4)

        # A semantic edge may skip rows because another relation placed its
        # endpoint deeper. Move that endpoint into a free side corridor when the
        # resulting straight ADL edge would otherwise pass through a real node.
        for source, target, _ in component_edges:
            if rank[target] - rank[source] <= 2:
                continue

            def blocked(candidate: Tuple[float, float]) -> bool:
                return any(
                    name not in {source, target}
                    and segment_intersects_node(
                        positions[source], candidate, positions[name], *visual_size(name)
                    )
                    for name in component
                    if name in positions
                )

            original = positions[target]
            if not blocked(original):
                continue
            for step in range(1, len(component) + 3):
                candidates = [
                    (original[0] + step * node_gap, original[1]),
                    (original[0] - step * node_gap, original[1]),
                ]
                chosen = next((candidate for candidate in candidates if not blocked(candidate)), None)
                if chosen is not None:
                    positions[target] = chosen
                    break
        component_x += component_width + 7.0

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
    world_width = max(80, int(max((x for x, _ in positions.values()), default=70.0) + 10.0))
    world_height = max(80, int(max((y for _, y in positions.values()), default=70.0) + 10.0))

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
            junction_name = connector_layout_name(c_idx, junction_class)
            x, y = positions.get(
                junction_name,
                junction_position(conn.sources, conn.target, positions, len(elements) + c_idx),
            )
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
    components = layout_components(elements, connections)
    sizes = ", ".join(str(sum(name in {element.name for element in elements} for name in component))
                      for component in components)
    print(f"Graph analysis: {len(components)} connected component(s) (element counts: {sizes}).")
    adl = generate_adl(elements, connections, args.model_name)
    output_path.write_text(adl, encoding="utf-8")

    print(f"OK: read {len(elements)} elements and {len(connections)} notation connections.")
    print(f"ADL written to: {output_path}")


if __name__ == "__main__":
    main()
