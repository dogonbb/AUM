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
from typing import Dict, List, Tuple

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
    "supports", "hinders", "contradicts", "motivates",
    "has requirement", "has goal", "communicates", "relates_to",
    "weakly conflicts", "moderately conflicts", "strongly conflicts",
}
CONNECTOR_KINDS = {"AND", "OR", "AND/OR", "Partial-PartOF", "Total-PartOF"}

TOKEN_RE = re.compile(
    r"\s(AND/OR|Partial-PartOF|Total-PartOF|moderately conflicts|strongly conflicts|"
    r"weakly conflicts|has requirement|has goal|communicates|relates_to|contradicts|"
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
        raise ValueError("Keine Elemente gefunden. Erwartet wird ein Abschnitt ELEMENTS.")
    return elements, connections


def parse_element_line(line: str) -> Element:
    normalized = normalize_spaces(line)
    for alias in sorted(ELEMENT_ALIASES, key=len, reverse=True):
        prefix = alias + " "
        if normalized.lower().startswith(prefix.lower()):
            name = normalize_spaces(normalized[len(prefix):])
            if not name:
                raise ValueError(f"Elementname fehlt in Zeile: {line!r}")
            adl_class, req_type = ELEMENT_ALIASES[alias]
            return Element(alias, name, adl_class, req_type)
    raise ValueError(
        f"Unbekannter Elementtyp in {line!r}. Erlaubt: " + ", ".join(ELEMENT_ALIASES)
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
            raise ValueError(f"{kind} benötigt mindestens zwei Quellen: {line!r}")
        if kind in {"Partial-PartOF", "Total-PartOF"}:
            if any(c != "IS Technical Component" for c in source_classes) or target_class != "IS Technical Component":
                raise ValueError(f"{kind} ist nur zwischen technischen Komponenten erlaubt: {line!r}")
        return

    if len(conn.sources) != 1:
        raise ValueError(f"Direkte Verbindung benötigt genau eine Quelle: {line!r}")
    s = source_classes[0]
    t = target_class

    allowed = False
    if kind in {"supports", "hinders"}:
        allowed = ((s in {"Goal", "Problem", "IS Requirement", "IS Technical Component"} and t == "Goal") or
                   (s == "IS Technical Component" and t in {"IS Technical Component", "IS Requirement"}))
    elif kind in {"contradicts", "weakly conflicts", "moderately conflicts", "strongly conflicts"}:
        allowed = s == "Goal" and t == "Goal"
    elif kind == "motivates":
        allowed = s == "Goal" and t in {"IS Technical Component", "IS Requirement"}
    elif kind == "has requirement":
        allowed = s in {"Goal", "IS Technical Component"} and t in {"IS Technical Component", "IS Requirement"}
    elif kind == "has goal":
        allowed = s == "IS Technical Component" and t == "Goal"
    elif kind == "communicates":
        allowed = s == "IS Technical Component" and t == "IS Technical Component"
    elif kind == "relates_to":
        allowed = s == "IS Technical Component" and t in {"Goal", "Problem", "IS Requirement"}

    if not allowed:
        raise ValueError(f"Nicht erlaubtes Verbindungsmuster in {line!r}: {s} {kind} {t}")


def parse_connection_line(line: str, elements: Dict[str, Element]) -> Connection:
    match = TOKEN_RE.search(line)
    if not match:
        raise ValueError(f"Keine gültige Verbindungsart gefunden in: {line!r}")
    kind = normalize_kind(match.group(1))
    left = normalize_spaces(line[:match.start()])
    target = normalize_spaces(line[match.end():])
    sources = [normalize_spaces(x) for x in left.split(",") if normalize_spaces(x)]
    if not sources or not target:
        raise ValueError(f"Unvollständige Verbindung: {line!r}")
    for source in sources:
        if source not in elements:
            raise ValueError(f"Quelle {source!r} wurde nicht unter ELEMENTS definiert.")
    if target not in elements:
        raise ValueError(f"Ziel {target!r} wurde nicht unter ELEMENTS definiert.")
    conn = Connection(sources, kind, target)
    validate_pattern(conn, elements, line)
    return conn


def parse_notation(text: str) -> Tuple[List[Element], List[Connection]]:
    element_lines, connection_lines = split_sections(text)
    elements = [parse_element_line(line) for line in element_lines]
    by_name: Dict[str, Element] = {}
    for element in elements:
        if element.name in by_name:
            raise ValueError(f"Doppelter Elementname: {element.name!r}")
        by_name[element.name] = element
    connections = [parse_connection_line(line, by_name) for line in connection_lines]
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
        return f'''
\tATTRIBUTE <External tool coupling>\n\tVALUE ""
\n\tATTRIBUTE <Description>\n\tVALUE ""
\n\tATTRIBUTE <Type>\n\tVALUE "{esc(req_type)}"
\n\tATTRIBUTE <Intermodel-Relations>\n\tVALUE
\n\tATTRIBUTE <Decomposition>\n\tVALUE ""
\n\tATTRIBUTE <Attributes>\n\tVALUE
'''
    raise ValueError(f"Nicht unterstützte ADL-Klasse: {element.adl_class}")


def size_for(adl_class: str) -> Tuple[float, float]:
    return (4.0, 2.0) if adl_class in {"IS Requirement", "IS Technical Component"} else (4.0, 1.5)


def compute_layout(elements: List[Element], connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
    base = {"Problem": 0, "Goal": 1, "IS Requirement": 2, "IS Technical Component": 3}
    layer = {e.name: base.get(e.adl_class, 1) for e in elements}
    for _ in range(len(elements)):
        changed = False
        for c in connections:
            if c.kind in CONNECTOR_KINDS:
                continue
            for s in c.sources:
                wanted = layer[s] + 1
                if layer[c.target] < wanted:
                    layer[c.target] = wanted
                    changed = True
        if not changed:
            break
    minimum = min(layer.values(), default=0)
    layer = {k: v - minimum for k, v in layer.items()}
    grouped: Dict[int, List[str]] = {}
    for e in elements:
        grouped.setdefault(layer[e.name], []).append(e.name)
    positions: Dict[str, Tuple[float, float]] = {}
    for col, names in grouped.items():
        for row, name in enumerate(names):
            positions[name] = (3.0 + col * 8.5, 2.5 + row * 2.8)
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
    if kind in {"contradicts", "weakly conflicts", "moderately conflicts", "strongly conflicts"}:
        return "Contradicts"
    return kind


def generate_adl(elements: List[Element], connections: List[Connection], model_name: str) -> str:
    now = datetime.now()
    created = now.strftime("%d.%m.%Y, %H:%M")
    changed = now.strftime("%d.%m.%Y, %H:%M:%S")
    positions = compute_layout(elements, connections)
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
            junction = f"{cls}-AUTO{c_idx}"
            pts = [positions[s] for s in conn.sources] + [positions[conn.target]]
            x = sum(p[0] for p in pts) / len(pts)
            y = sum(p[1] for p in pts) / len(pts)
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

\tATTRIBUTE <World area>\n\tVALUE "w:80cm h:80cm minw:5cm minh:5cm"

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
    parser = argparse.ArgumentParser(description="Konvertiert die LLM-Notation eines 4EM Technical Components and Requirements Model in ADL.")
    parser.add_argument("input", help="Textdatei mit ELEMENTS und CONNECTIONS")
    parser.add_argument("output", help="Ausgabedatei .adl")
    parser.add_argument("--model-name", default="Generated Technical Components and Requirements Model")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    elements, connections = parse_notation(input_path.read_text(encoding="utf-8"))
    output_path.write_text(generate_adl(elements, connections, args.model_name), encoding="utf-8")
    print(f"OK: {len(elements)} Elemente und {len(connections)} Notations-Verbindungen gelesen.")
    print(f"ADL geschrieben nach: {output_path}")

if __name__ == "__main__":
    main()
