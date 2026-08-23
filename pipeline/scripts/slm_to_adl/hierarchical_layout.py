"""Shared hierarchical graph layout used by all 4EM SLM-to-ADL converters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class LayoutNode:
    name: str
    width: float = 4.0
    height: float = 2.0
    connector: bool = False


@dataclass(frozen=True)
class LayoutEdge:
    source: str
    target: str
    distance: int = 2
    primary: bool = True


@dataclass(frozen=True)
class LayoutOptions:
    x_start: float = 3.0
    y_start: float = 2.5
    node_gap: float = 7.0
    half_level_gap: float = 3.2
    component_gap: float = 8.0
    collision_margin: float = 1.0
    align_one_to_one: bool = True
    separate_branches: bool = True
    enforce_final_branch_grouping: bool = False
    center_shared_children: bool = False
    collision_side: str = "nearest"
    max_components_per_row: int = 3
    component_row_gap: float = 8.0
    component_bottom_reserve: float = 0.0


@dataclass(frozen=True)
class UniformLayoutConfig:
    """The complete set of model-dependent inputs accepted by the layout.

    Relation names deliberately do not occur here.  Once a converter has
    normalized its notation to nodes and edges, every edge is treated alike.
    """

    orientation: str = "top_down"
    x_start: float = 4.0
    y_start: float = 2.5
    node_gap: float = 8.0
    level_gap: float = 3.2
    component_gap: float = 8.0
    collision_margin: float = 1.0
    spacing_scale: float = 1.2


def compute_uniform_layout(
    nodes: Iterable[LayoutNode],
    edges: Iterable[LayoutEdge],
    *,
    branch_nodes: Iterable[str] = (),
    config: UniformLayoutConfig = UniformLayoutConfig(),
) -> Dict[str, Tuple[float, float]]:
    """Apply one relation-agnostic layout algorithm to any 4EM model.

    ``orientation`` only changes the visual direction.  It never reverses or
    prioritizes individual edges based on their label.  Connector nodes are
    ordinary graph nodes with a smaller caller-supplied geometry.
    """
    if config.orientation not in {"top_down", "bottom_up"}:
        raise ValueError("orientation must be 'top_down' or 'bottom_up'")
    if config.spacing_scale <= 0:
        raise ValueError("spacing_scale must be greater than zero")

    # Geometry is the only node-specific input. Even connector nodes take part
    # in exactly the same ordering and collision rules as normal elements.
    node_list = [LayoutNode(node.name, node.width, node.height, False) for node in nodes]
    # Remove both the primary/secondary distinction and caller-supplied edge
    # distances. Every graph edge advances exactly one common hierarchy level.
    edge_list = [LayoutEdge(edge.source, edge.target, 1, True) for edge in edges]
    positions = compute_hierarchical_layout(
        node_list,
        edge_list,
        branch_nodes=(),
        options=LayoutOptions(
            x_start=config.x_start,
            y_start=config.y_start,
            node_gap=config.node_gap * config.spacing_scale,
            half_level_gap=config.level_gap * config.spacing_scale,
            component_gap=config.component_gap * config.spacing_scale,
            collision_margin=config.collision_margin,
        ),
    )
    if config.orientation == "bottom_up" and positions:
        minimum_y = min(y for _, y in positions.values())
        maximum_y = max(y for _, y in positions.values())
        positions = {
            name: (x, minimum_y + maximum_y - y)
            for name, (x, y) in positions.items()
        }
    return positions


def _components(names: Sequence[str], edges: Sequence[LayoutEdge]) -> List[List[str]]:
    adjacency = {name: set() for name in names}
    for edge in edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    result: List[List[str]] = []
    visited: Set[str] = set()
    for root in names:
        if root in visited:
            continue
        visited.add(root)
        stack = [root]
        component: List[str] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in reversed(names):
                if neighbor in adjacency[node] and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def _segments_cross(
    first_start: Tuple[float, float],
    first_end: Tuple[float, float],
    second_start: Tuple[float, float],
    second_end: Tuple[float, float],
) -> bool:
    """Return whether two straight edge segments cross away from endpoints."""
    def orientation(
        start: Tuple[float, float],
        end: Tuple[float, float],
        point: Tuple[float, float],
    ) -> float:
        return ((end[0] - start[0]) * (point[1] - start[1])
                - (end[1] - start[1]) * (point[0] - start[0]))

    return (
        orientation(first_start, first_end, second_start)
        * orientation(first_start, first_end, second_end) < -1e-9
        and orientation(second_start, second_end, first_start)
        * orientation(second_start, second_end, first_end) < -1e-9
    )


def _edge_crossing_count(
    positions: Dict[str, Tuple[float, float]],
    edges: Sequence[LayoutEdge],
) -> int:
    visible_edges = [
        edge for edge in edges
        if edge.source in positions and edge.target in positions
    ]
    crossings = 0
    for index, first in enumerate(visible_edges):
        for second in visible_edges[index + 1:]:
            if {first.source, first.target} & {second.source, second.target}:
                continue
            if _segments_cross(
                positions[first.source], positions[first.target],
                positions[second.source], positions[second.target],
            ):
                crossings += 1
    return crossings


def _reduce_final_edge_crossings(
    positions: Dict[str, Tuple[float, float]],
    edges: Sequence[LayoutEdge],
) -> None:
    """Swap adjacent nodes on a row when that strictly reduces crossings."""
    rows: Dict[float, List[str]] = {}
    for name, (_, y) in positions.items():
        rows.setdefault(y, []).append(name)
    for row in rows.values():
        row.sort(key=lambda name: positions[name][0])

    for _ in range(8):
        changed = False
        baseline = _edge_crossing_count(positions, edges)
        for row in rows.values():
            for index in range(len(row) - 1):
                left, right = row[index:index + 2]
                left_point, right_point = positions[left], positions[right]
                positions[left] = (right_point[0], left_point[1])
                positions[right] = (left_point[0], right_point[1])
                candidate = _edge_crossing_count(positions, edges)
                # Preserve established grouping for cosmetic one-crossing
                # changes. A swap is worthwhile only when it removes a real
                # crossing cluster (as in long edges spanning several levels).
                if candidate <= baseline - 2:
                    row[index:index + 2] = [right, left]
                    baseline = candidate
                    changed = True
                else:
                    positions[left], positions[right] = left_point, right_point
        if not changed:
            break


def _separate_final_row_overlaps(
    positions: Dict[str, Tuple[float, float]],
    nodes: Dict[str, LayoutNode],
    margin: float,
) -> None:
    """Guarantee that final centering passes cannot leave same-row overlaps."""
    rows: Dict[float, List[str]] = {}
    for name, (_, y) in positions.items():
        rows.setdefault(y, []).append(name)
    for row in rows.values():
        row.sort(key=lambda name: positions[name][0])
        for index in range(1, len(row)):
            previous, current = row[index - 1], row[index]
            required = ((nodes[previous].width + nodes[current].width) / 2.0
                        + margin)
            minimum_x = positions[previous][0] + required
            if positions[current][0] < minimum_x:
                positions[current] = (minimum_x, positions[current][1])


def _enforce_final_one_to_one_alignment(
    positions: Dict[str, Tuple[float, float]],
    edges: Sequence[LayoutEdge],
    nodes: Dict[str, LayoutNode],
    rank: Dict[str, int],
    margin: float,
) -> None:
    """Put the predecessor of every pure 1:1 edge on the target's x-axis.

    This deliberately runs after all cosmetic layout passes. Other nodes on
    the predecessor's row are packed to either side of the fixed predecessor,
    so preserving the vertical line cannot introduce a node overlap.
    """
    incoming: Dict[str, Set[str]] = {}
    outgoing: Dict[str, Set[str]] = {}
    for edge in edges:
        outgoing.setdefault(edge.source, set()).add(edge.target)
        incoming.setdefault(edge.target, set()).add(edge.source)

    candidates = [
        edge for edge in edges
        if len(outgoing.get(edge.source, set())) == 1
        and len(incoming.get(edge.target, set())) == 1
        and rank.get(edge.source) != rank.get(edge.target)
    ]
    # Work from the target end of a chain backwards. This makes A->B->C share
    # one x-coordinate rather than preserving only the last processed pair.
    candidates.sort(key=lambda edge: rank[edge.target], reverse=True)
    for edge in candidates:
        if edge.source not in positions or edge.target not in positions:
            continue
        fixed_x = positions[edge.target][0]
        source_y = positions[edge.source][1]
        positions[edge.source] = (fixed_x, source_y)

        row = [
            name for name, (_, y) in positions.items()
            if name != edge.source and abs(y - source_y) < 1e-9
        ]
        left = sorted(
            (name for name in row if positions[name][0] < fixed_x),
            key=lambda name: positions[name][0],
            reverse=True,
        )
        cursor = fixed_x - nodes[edge.source].width / 2.0 - margin
        for name in left:
            x = min(positions[name][0], cursor - nodes[name].width / 2.0)
            positions[name] = (x, source_y)
            cursor = x - nodes[name].width / 2.0 - margin

        right = sorted(
            (name for name in row if positions[name][0] >= fixed_x),
            key=lambda name: positions[name][0],
        )
        cursor = fixed_x + nodes[edge.source].width / 2.0 + margin
        for name in right:
            x = max(positions[name][0], cursor + nodes[name].width / 2.0)
            positions[name] = (x, source_y)
            cursor = x + nodes[name].width / 2.0 + margin


def _acyclic_edges(names: Sequence[str], edges: Sequence[LayoutEdge]) -> List[LayoutEdge]:
    """Keep stable primary direction; feedback edges remain semantic ADL edges."""
    accepted: List[LayoutEdge] = []
    outgoing: Dict[str, Set[str]] = {name: set() for name in names}

    def reaches(start: str, wanted: str) -> bool:
        stack = [start]
        seen: Set[str] = set()
        while stack:
            node = stack.pop()
            if node == wanted:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(outgoing[node])
        return False

    # Primary hierarchy wins over secondary cross-relations when selecting the
    # stable acyclic skeleton. Original order remains the deterministic tie-break.
    indexed = list(enumerate(edges))
    indexed.sort(key=lambda item: (not item[1].primary, item[0]))
    for _, edge in indexed:
        if edge.source == edge.target or reaches(edge.target, edge.source):
            continue
        if edge.target not in outgoing[edge.source]:
            outgoing[edge.source].add(edge.target)
            accepted.append(edge)
    return accepted


def _strong_components(names: Sequence[str], edges: Sequence[LayoutEdge]) -> List[List[str]]:
    outgoing: Dict[str, List[str]] = {name: [] for name in names}
    for edge in edges:
        outgoing[edge.source].append(edge.target)
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    low: Dict[str, int] = {}
    result: List[List[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in outgoing[node]:
            if target not in indices:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            members: List[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                members.append(member)
                if member == node:
                    break
            result.append(members)

    for name in names:
        if name not in indices:
            visit(name)
    return result


def _separate_branch_lanes(
    positions: Dict[str, Tuple[float, float]],
    edges: Sequence[LayoutEdge],
    branch_nodes: Sequence[str],
    order: Dict[str, int],
    gap: float,
) -> None:
    outgoing: Dict[str, List[str]] = {}
    for edge in edges:
        outgoing.setdefault(edge.source, []).append(edge.target)

    def descendants(start: str) -> Set[str]:
        found = {start}
        stack = list(outgoing.get(start, []))
        while stack:
            node = stack.pop()
            if node in found:
                continue
            found.add(node)
            stack.extend(outgoing.get(node, []))
        return {node for node in found if node in positions}

    for connector in reversed(branch_nodes):
        targets = outgoing.get(connector, [])
        if len(targets) < 2 or connector not in positions:
            continue
        reachable = [descendants(target) for target in targets]
        groups = []
        for index, group in enumerate(reachable):
            siblings = set().union(*(g for i, g in enumerate(reachable) if i != index))
            exclusive = group - siblings
            if exclusive:
                groups.append((index, exclusive))
        if len(groups) < 2:
            continue
        groups.sort(key=lambda item: (
            -len(item[1]),
            sum(positions[node][0] for node in item[1]) / len(item[1]),
            order.get(targets[item[0]], item[0]),
        ))
        bounds = [(min(positions[n][0] for n in group), max(positions[n][0] for n in group))
                  for _, group in groups]
        widths = [right - left for left, right in bounds]
        cursor = positions[connector][0] - (sum(widths) + gap * (len(groups) - 1)) / 2.0
        for (_, group), (left, _), width in zip(groups, bounds, widths):
            shift = cursor - left
            for node in group:
                x, y = positions[node]
                positions[node] = (x + shift, y)
            cursor += width + gap

        # The immediate children of one connector must form a contiguous band.
        # A direct sibling on the same row must not sit between connector
        # branches, otherwise the parent-to-sibling edge crosses a branch edge.
        child_y_values = {positions[target][1] for target in targets if target in positions}
        if len(child_y_values) == 1:
            child_y = next(iter(child_y_values))
            connector_descendants = descendants(connector)
            branch_left = min(positions[target][0] for target in targets if target in positions)
            branch_right = max(positions[target][0] for target in targets if target in positions)
            intruders = [
                node for node, (x, y) in positions.items()
                if node not in connector_descendants
                and abs(y - child_y) < 1e-9
                and branch_left < x < branch_right
            ]
            cursor_right = branch_right + gap
            for intruder in sorted(intruders, key=lambda node: positions[node][0]):
                shift = cursor_right - positions[intruder][0]
                moved = descendants(intruder)
                for node in moved:
                    x, y = positions[node]
                    positions[node] = (x + shift, y)
                cursor_right = max(positions[node][0] for node in moved) + gap


def _align_one_to_one(
    positions: Dict[str, Tuple[float, float]], edges: Sequence[LayoutEdge], rank: Dict[str, int], gap: float
) -> None:
    incoming: Dict[str, int] = {}
    outgoing: Dict[str, int] = {}
    for edge in edges:
        incoming[edge.target] = incoming.get(edge.target, 0) + 1
        outgoing[edge.source] = outgoing.get(edge.source, 0) + 1
    for edge in sorted(edges, key=lambda item: rank[item.target]):
        if outgoing.get(edge.source) != 1 or incoming.get(edge.target) != 1:
            continue
        wanted = positions[edge.source][0]
        occupied = any(
            node != edge.target and rank.get(node) == rank[edge.target]
            and abs(point[0] - wanted) < gap * 0.9
            for node, point in positions.items()
        )
        if not occupied:
            positions[edge.target] = (wanted, positions[edge.target][1])


def _center_branch_parents(
    positions: Dict[str, Tuple[float, float]], edges: Sequence[LayoutEdge]
) -> None:
    """Center every branching parent above children sharing one hierarchy row."""
    outgoing: Dict[str, List[str]] = {}
    for edge in edges:
        outgoing.setdefault(edge.source, []).append(edge.target)
    for source, targets in outgoing.items():
        unique_targets = list(dict.fromkeys(targets))
        if source not in positions or len(unique_targets) < 2:
            continue
        child_rows = {positions[target][1] for target in unique_targets}
        if len(child_rows) != 1:
            continue
        center_x = sum(positions[target][0] for target in unique_targets) / len(unique_targets)
        positions[source] = (center_x, positions[source][1])


def _center_shared_children(
    positions: Dict[str, Tuple[float, float]],
    edges: Sequence[LayoutEdge],
    nodes: Dict[str, LayoutNode],
) -> None:
    """Move real children with multiple parents into the middle slot of a row.

    Shared children are visual bridges between parent subtrees. Leaving such a
    node at an outer input-order position creates crossing fans and can pull its
    parents onto the same coordinates when they are subsequently centered.
    Existing row slots are reused, so no new node overlap is introduced.
    """
    incoming: Dict[str, Set[str]] = {}
    for edge in edges:
        incoming.setdefault(edge.target, set()).add(edge.source)
    rows: Dict[float, List[str]] = {}
    for name, (_, y) in positions.items():
        if name in nodes and not nodes[name].connector:
            rows.setdefault(y, []).append(name)
    for row in rows.values():
        row.sort(key=lambda name: positions[name][0])
        shared = [name for name in row if len(incoming.get(name, set())) > 1]
        if not shared:
            continue
        slots = sorted(positions[name][0] for name in row)
        middle = (len(row) - 1) // 2
        reordered = list(row)
        for offset, name in enumerate(shared):
            current = reordered.index(name)
            reordered.pop(current)
            target = min(len(reordered), middle + offset)
            reordered.insert(target, name)
        for name, x in zip(reordered, slots):
            positions[name] = (x, positions[name][1])


def _align_parallel_connectors(
    positions: Dict[str, Tuple[float, float]],
    edges: Sequence[LayoutEdge],
    nodes: Dict[str, LayoutNode],
) -> None:
    """Center a connector on a parallel direct source-to-target relation."""
    direct_pairs = {(edge.source, edge.target) for edge in edges if edge.distance >= 2}
    incoming: Dict[str, List[str]] = {}
    outgoing: Dict[str, List[str]] = {}
    for edge in edges:
        incoming.setdefault(edge.target, []).append(edge.source)
        outgoing.setdefault(edge.source, []).append(edge.target)
    for name, node in nodes.items():
        if not node.connector or name not in positions:
            continue
        sources = incoming.get(name, [])
        targets = outgoing.get(name, [])
        if len(sources) != 1 or len(targets) != 1:
            continue
        source, target = sources[0], targets[0]
        if (source, target) not in direct_pairs:
            continue
        source_x = positions[source][0]
        target_x = positions[target][0]
        positions[name] = ((source_x + target_x) / 2.0, positions[name][1])


def _balance_parallel_branch_parents(
    positions: Dict[str, Tuple[float, float]],
    edges: Sequence[LayoutEdge],
    nodes: Dict[str, LayoutNode],
    gap: float,
) -> None:
    """Place parents with the same child set evenly around their child band.

    This commonly occurs in process models where an information set and a
    split connector both lead to the same activities. Centering each parent
    independently would stack them on one axis and make the connector edge run
    through the information set.
    """
    outgoing: Dict[str, List[str]] = {}
    for edge in edges:
        outgoing.setdefault(edge.source, []).append(edge.target)

    groups: Dict[frozenset[str], List[str]] = {}
    for source, targets in outgoing.items():
        unique_targets = frozenset(targets)
        if len(unique_targets) >= 2:
            groups.setdefault(unique_targets, []).append(source)

    for targets, parents in groups.items():
        if len(parents) < 2 or any(name not in positions for name in targets):
            continue
        # Keep semantic nodes on the left and routing connectors on the right.
        parents = sorted(parents, key=lambda name: (nodes[name].connector, positions[name][0]))
        center = sum(positions[name][0] for name in targets) / len(targets)
        start = center - (len(parents) - 1) * gap / 2.0
        for index, parent in enumerate(parents):
            positions[parent] = (start + index * gap, positions[parent][1])


def _avoid_node_crossings(
    positions: Dict[str, Tuple[float, float]],
    edges: Sequence[LayoutEdge],
    rank: Dict[str, int],
    nodes: Dict[str, LayoutNode],
    margin: float,
    collision_side: str,
    connectors_only: bool = False,
) -> None:
    outgoing: Dict[str, Set[str]] = {}
    for edge in edges:
        outgoing.setdefault(edge.source, set()).add(edge.target)

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

    real = {name for name, node in nodes.items() if not node.connector}
    for _ in range(6):
        changed = False
        for edge in edges:
            if connectors_only and not nodes[edge.target].connector:
                continue
            if rank[edge.target] - rank[edge.source] <= 2:
                continue
            sx, sy = positions[edge.source]
            tx, ty = positions[edge.target]
            force_right = collision_side == "right" or (
                collision_side == "right_non_connector"
                and not nodes[edge.target].connector
            )
            left, right = tx, tx
            hit = False
            for blocker in real - {edge.source, edge.target}:
                if blocker not in rank or not rank[edge.source] < rank[blocker] < rank[edge.target]:
                    continue
                bx, by = positions[blocker]
                fraction = (by - sy) / (ty - sy)
                if not 0.0 < fraction < 1.0:
                    continue
                edge_x = sx + (tx - sx) * fraction
                clearance = nodes[blocker].width / 2.0 + margin
                if force_right:
                    # Process shortcut edges deliberately use an outer lane,
                    # even when the current diagonal only narrowly misses a node.
                    hit = True
                    right = max(right, sx + (bx + clearance + 0.5 - sx) / fraction)
                    continue
                if abs(edge_x - bx) > clearance:
                    continue
                hit = True
                left = min(left, sx + (bx - clearance - 0.5 - sx) / fraction)
                right = max(right, sx + (bx + clearance + 0.5 - sx) / fraction)
            if hit:
                shift_left, shift_right = left - tx, right - tx
                shift = (
                    shift_right if force_right
                    else shift_left if collision_side == "left"
                    else shift_left if abs(shift_left) <= abs(shift_right) else shift_right
                )
                if abs(shift) > 0.01:
                    moved_nodes = (
                        {edge.target}
                        if nodes[edge.target].connector
                        else descendants(edge.target)
                    )
                    for name in moved_nodes:
                        x, y = positions[name]
                        positions[name] = (x + shift, y)
                    changed = True
        if not changed:
            break


def compute_hierarchical_layout(
    nodes: Iterable[LayoutNode],
    edges: Iterable[LayoutEdge],
    *,
    branch_nodes: Iterable[str] = (),
    options: LayoutOptions = LayoutOptions(),
) -> Dict[str, Tuple[float, float]]:
    """Lay out an already normalized top-to-bottom directed graph."""
    node_list = list(nodes)
    if not node_list:
        return {}
    node_by_name = {node.name: node for node in node_list}
    names = [node.name for node in node_list]
    order = {name: index for index, name in enumerate(names)}
    edge_list = [edge for edge in edges if edge.source in node_by_name and edge.target in node_by_name]
    accepted = _acyclic_edges(names, edge_list)
    branch_set = list(branch_nodes)
    positions: Dict[str, Tuple[float, float]] = {}
    component_x = options.x_start
    component_y = options.y_start
    components_in_row = 0
    row_bottom = component_y

    for component_index, component in enumerate(_components(names, edge_list)):
        component_set = set(component)
        component_edges = [e for e in accepted if e.source in component_set and e.target in component_set]
        rank = {name: 0 for name in component}
        for _ in component:
            changed = False
            for edge in component_edges:
                wanted = rank[edge.source] + edge.distance
                if rank[edge.target] < wanted:
                    rank[edge.target] = wanted
                    changed = True
            if not changed:
                break

        # Use tight layering for every edge, independent of its original
        # relation name. A root that directly points to a deep node is moved
        # onto the level immediately before that node instead of producing a
        # long edge through unrelated intermediate rows.
        outgoing_edges: Dict[str, List[LayoutEdge]] = {}
        for edge in component_edges:
            outgoing_edges.setdefault(edge.source, []).append(edge)
        for _ in component:
            changed = False
            for name in sorted(component, key=lambda item: rank[item], reverse=True):
                outgoing = outgoing_edges.get(name, [])
                if not outgoing:
                    continue
                upper = min(rank[edge.target] - edge.distance for edge in outgoing)
                lower = max(
                    (rank[source] + edge.distance
                     for edge in component_edges
                     for source in [edge.source]
                     if edge.target == name),
                    default=0,
                )
                wanted = max(lower, upper)
                if wanted > rank[name]:
                    rank[name] = wanted
                    changed = True
            if not changed:
                break

        # Connector children describe one grouped decomposition and must share
        # a row. Raise the connector directly above its deepest child and then
        # align all children below it; repeat with edge relaxation until stable.
        component_branches = [name for name in branch_set if name in component_set]
        outgoing_targets: Dict[str, List[str]] = {}
        for edge in component_edges:
            outgoing_targets.setdefault(edge.source, []).append(edge.target)
        for _ in range(max(1, len(component) * 2)):
            changed = False
            for edge in component_edges:
                wanted = rank[edge.source] + edge.distance
                if rank[edge.target] < wanted:
                    rank[edge.target] = wanted
                    changed = True
            for connector in component_branches:
                children = outgoing_targets.get(connector, [])
                if not children:
                    continue
                connector_rank = max(rank[child] for child in children) - 1
                if rank[connector] < connector_rank:
                    rank[connector] = connector_rank
                    changed = True
                child_rank = rank[connector] + 1
                for child in children:
                    if rank[child] < child_rank:
                        rank[child] = child_rank
                        changed = True
            if not changed:
                break

        # Cyclic semantic relations cannot form strict hierarchy. Keep every
        # member of such a cycle deliberately on one shared level.
        original_component_edges = [
            edge for edge in edge_list
            if edge.source in component_set and edge.target in component_set
        ]
        for members in _strong_components(component, original_component_edges):
            if len(members) > 1:
                shared_rank = max(rank[member] for member in members)
                for member in members:
                    rank[member] = shared_rank

        layout_names = list(component)
        virtual: Set[str] = set()
        routed: List[Tuple[str, str]] = []
        for edge_index, edge in enumerate(component_edges):
            previous = edge.source
            for level in range(rank[edge.source] + 1, rank[edge.target]):
                name = f"__LAYOUT_ROUTE_{component_index}_{edge_index}_{level}"
                virtual.add(name)
                layout_names.append(name)
                rank[name] = level
                order[name] = len(order)
                routed.append((previous, name))
                previous = name
            routed.append((previous, edge.target))

        adjacency = {name: [] for name in layout_names}
        for source, target in routed:
            adjacency[source].append(target)
            adjacency[target].append(source)
        rows: Dict[int, List[str]] = {}
        for name in layout_names:
            rows.setdefault(rank[name], []).append(name)
        for row in rows.values():
            row.sort(key=lambda name: order[name])
        current_order: Dict[str, float] = {}

        def refresh() -> None:
            for row in rows.values():
                for index, name in enumerate(row):
                    current_order[name] = float(index)

        refresh()
        levels = sorted(rows)
        for _ in range(12):
            for sweep in (levels, list(reversed(levels))):
                for level in sweep:
                    rows[level].sort(key=lambda name: (
                        sum(current_order[n] for n in adjacency[name]) / len(adjacency[name])
                        if adjacency[name] else current_order[name], order[name]
                    ))
                    refresh()

        widest = max(len(row) for row in rows.values())
        width = max(options.node_gap, (widest - 1) * options.node_gap + 5.0)
        center = component_x + width / 2.0
        for level, row in rows.items():
            start = center - (len(row) - 1) * options.node_gap / 2.0
            for index, name in enumerate(row):
                if name not in virtual:
                    positions[name] = (start + index * options.node_gap,
                                       component_y + level * options.half_level_gap)

        if options.separate_branches:
            _separate_branch_lanes(positions, component_edges,
                                   [n for n in branch_set if n in component_set], order,
                                   options.node_gap)
        if options.center_shared_children:
            _center_shared_children(positions, component_edges, node_by_name)
        _center_branch_parents(positions, component_edges)
        if options.align_one_to_one:
            _align_one_to_one(positions, component_edges, rank, options.node_gap)
        _align_parallel_connectors(positions, component_edges, node_by_name)
        _avoid_node_crossings(positions, component_edges, rank, node_by_name,
                              options.collision_margin, options.collision_side)
        # Collision routing can move a complete child subtree. Re-center its
        # branching parents once more over the final child coordinates.
        _center_branch_parents(positions, component_edges)
        _align_parallel_connectors(positions, component_edges, node_by_name)
        _balance_parallel_branch_parents(
            positions, component_edges, node_by_name, options.node_gap
        )
        _avoid_node_crossings(
            positions, component_edges, rank, node_by_name,
            options.collision_margin, options.collision_side,
            connectors_only=True,
        )
        # Later centering rules (especially for a direct edge parallel to a
        # connector path) may pull an unrelated sibling back between the
        # connector's children. Connector members are a visual group, so make
        # their contiguous band a final invariant as well.
        if options.separate_branches and options.enforce_final_branch_grouping:
            _separate_branch_lanes(
                positions,
                component_edges,
                [name for name in branch_set if name in component_set],
                order,
                options.node_gap,
            )
            _center_branch_parents(positions, component_edges)

        # Every earlier collision pass can be invalidated by the final
        # centering/grouping operations above. Optimize the visible order once
        # more, then enforce node-size-aware spacing as the last layout rule.
        _reduce_final_edge_crossings(positions, component_edges)
        _separate_final_row_overlaps(
            positions, node_by_name, options.collision_margin
        )
        _enforce_final_one_to_one_alignment(
            positions,
            component_edges,
            node_by_name,
            rank,
            options.collision_margin,
        )

        real_component = [name for name in component if name in positions]
        minimum = min(positions[name][0] - node_by_name[name].width / 2 for name in real_component)
        if minimum < component_x:
            correction = component_x - minimum
            for name in real_component:
                x, y = positions[name]
                positions[name] = (x + correction, y)
        component_right = max(
            positions[name][0] + node_by_name[name].width / 2 for name in real_component
        )
        component_bottom = max(
            positions[name][1] + node_by_name[name].height / 2 for name in real_component
        ) + options.component_bottom_reserve
        row_bottom = max(row_bottom, component_bottom)
        components_in_row += 1
        if components_in_row >= options.max_components_per_row:
            component_x = options.x_start
            component_y = row_bottom + options.component_row_gap
            row_bottom = component_y
            components_in_row = 0
        else:
            component_x = component_right + options.component_gap

    return positions
