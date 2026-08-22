"""
Merge one 4EM ADL file with a text file containing intermodel relationships.

Element resolution order:
1. exact normalized INSTANCE name
2. unique tolerant INSTANCE-name match (redundant prefix / shortened name)
3. exact normalized Description value

The input ADL is never modified.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SUPPORTED_CONNECTORS = {
    "play",
    "plays",
    "supports",
    "empty",
    "defines",
    "is_responsible_for",
    "performs",
}

MODEL_ALIASES = {
    "GoalModel": {"goalmodel", "goal", "goalandproblemmodel", "goalproblemmodel"},
    "BusinessRuleModel": {"businessrulemodel", "businessrulesmodel", "businessrule"},
    "ConceptsModel": {"conceptmodel", "conceptsmodel", "concept", "concepts"},
    "BusinessProcessModel": {"businessprocessmodel", "businessprocess", "processmodel"},
    "ActorsandResourcesModel": {
        "actorandresourcemodel", "actorsandresourcesmodel",
        "actorresourcemodel", "actorsresourcesmodel",
    },
    "TechnicalComponentsandRequirementsModel": {
        "technicalcomponentandrequirementmodel",
        "technicalcomponentsandrequirementsmodel",
        "technicalcomponentrequirementsmodel",
        "technicalrequirementsmodel",
        "tcrmodel",
    },
    "ProductServiceModel": {
        "productservicemodel", "productservice", "productmodel",
    },
}

DESCRIPTION_TYPE_PREFIXES = {
    "Goal",
    "Problem",
    "Cause",
    "Constraint",
    "Opportunity",
    "Individual",
    "Role",
    "Resource",
    "Organizational Unit",
    "Process",
    "External Process",
    "Information Set",
    "Concept",
    "Attribute",
    "Information System Goal",
    "Information System Problem",
    "Information System Requirement",
    "Information System Functional Requirement",
    "Information System Non-Functional Requirement",
    "Technical Component",
    "IS Technical Component",
    "IS Requirement",
    "Product",
    "Service",
    "Component",
    "Feature",
    "ProductService",
    "Unspecific/Product/Service",
}


class MergeError(RuntimeError):
    pass


@dataclass
class Instance:
    name: str
    element_type: str
    description: str
    start: int
    end: int


@dataclass
class Model:
    internal_name: str
    model_type: str
    start: int
    end: int
    instances: dict[str, Instance] = field(default_factory=dict)
    descriptions: dict[str, list[Instance]] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    source_model: str
    target_model: str
    source_element: str
    connector: str
    target_element: str
    line_number: int


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def unescape_adl_string(value: str) -> str:
    return value.replace(r"\"", '"').replace(r"\\", "\\")


def read_text_preserving_encoding(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    # Only select utf-8-sig when the source actually contains a BOM.
    # Decoding plain UTF-8/ASCII with utf-8-sig also succeeds, but writing it
    # back would introduce a new BOM that 4EM rejects as a line-0 syntax error.
    encodings = (
        ("utf-8-sig", "utf-8", "cp1252", "latin-1")
        if raw.startswith(b"\xef\xbb\xbf")
        else ("utf-8", "cp1252", "latin-1")
    )
    for encoding in encodings:
        try:
            output_encoding = "utf-8" if encoding == "utf-8-sig" else encoding
            return raw.decode(encoding), output_encoding
        except UnicodeDecodeError:
            continue
    raise MergeError(f"Could not decode file: {path}")


def extract_description(instance_block: str) -> str:
    match = re.search(
        r'(?ms)^\tATTRIBUTE <Description>\s*\r?\n'
        r'\tVALUE(?P<body>.*?)(?=^\tATTRIBUTE <|\Z)',
        instance_block,
    )
    if not match:
        return ""
    pieces = re.findall(r'"((?:\\.|[^"\\])*)"', match.group("body"))
    return "".join(unescape_adl_string(piece) for piece in pieces).strip()


def parse_adl(text: str) -> dict[str, Model]:
    model_header = re.compile(
        r"^BUSINESS PROCESS MODEL <(?P<name>[^>]+)>[^\n]*\n"
        r"(?:.*?\n)*?^TYPE <(?P<type>[^>]+)>\s*$",
        re.MULTILINE,
    )
    matches = list(model_header.finditer(text))
    if not matches:
        raise MergeError("No 'BUSINESS PROCESS MODEL' sections were found in the ADL file.")

    models: dict[str, Model] = {}

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        internal_name = match.group("name").strip()
        model_type = match.group("type").strip()
        model = Model(internal_name, model_type, start, end)

        section = text[start:end]
        instance_pattern = re.compile(
            r"^INSTANCE <(?P<name>[^>]+)> : <(?P<type>[^>]+)>\s*$",
            re.MULTILINE,
        )
        instance_matches = list(instance_pattern.finditer(section))

        for instance_index, instance_match in enumerate(instance_matches):
            instance_start = start + instance_match.start()
            instance_end = (
                start + instance_matches[instance_index + 1].start()
                if instance_index + 1 < len(instance_matches)
                else end
            )
            instance_block = text[instance_start:instance_end]
            instance = Instance(
                name=instance_match.group("name").strip(),
                element_type=instance_match.group("type").strip(),
                description=extract_description(instance_block),
                start=instance_start,
                end=instance_end,
            )

            name_key = normalize(instance.name)
            if name_key in model.instances:
                raise MergeError(
                    f"Duplicate normalized element name '{instance.name}' "
                    f"in model '{internal_name}'."
                )
            model.instances[name_key] = instance

            description_key = normalize(instance.description)
            if description_key:
                model.descriptions.setdefault(description_key, []).append(instance)

        models[internal_name] = model

    return models


def build_model_lookup(models: dict[str, Model]) -> dict[str, str]:
    lookup: dict[str, str] = {}

    def add(alias: str, internal_name: str) -> None:
        key = normalize(alias)
        if key and key not in lookup:
            lookup[key] = internal_name

    for internal_name, model in models.items():
        add(internal_name, internal_name)
        add(model.model_type, internal_name)

    for canonical_name, aliases in MODEL_ALIASES.items():
        target_aliases = {normalize(canonical_name), *(normalize(x) for x in aliases)}
        actual = None
        for internal_name, model in models.items():
            candidates = {normalize(internal_name), normalize(model.model_type)}
            if candidates & target_aliases:
                actual = internal_name
                break
        if actual:
            add(canonical_name, actual)
            for alias in aliases:
                add(alias, actual)

    return lookup


def resolve_model(raw_name: str, model_lookup: dict[str, str], line_number: int) -> str:
    model = model_lookup.get(normalize(raw_name))
    if model:
        return model
    known = ", ".join(sorted(set(model_lookup.values())))
    raise MergeError(
        f"Relationship line {line_number}: unknown model '{raw_name}'. "
        f"Models found in ADL: {known}"
    )


def connector_pattern() -> str:
    return "|".join(
        re.escape(connector)
        for connector in sorted(SUPPORTED_CONNECTORS, key=len, reverse=True)
    )


def parse_relationship_text(text: str, model_lookup: dict[str, str]) -> list[Relation]:
    relations: list[Relation] = []
    current_pair: tuple[str, str] | None = None
    connectors = connector_pattern()

    section_pattern = re.compile(
        r"^\[\s*(?P<source>.+?)\s*(?:->|=>|\bto\b)\s*(?P<target>.+?)\s*\]$",
        re.IGNORECASE,
    )
    pipe_pattern = re.compile(
        rf"^(?P<source_model>[^|]+?)\s*\|\s*"
        rf"(?P<source_element>[^|]+?)\s*\|\s*"
        rf"(?P<connector>{connectors})\s*\|\s*"
        rf"(?P<target_model>[^|]+?)\s*\|\s*"
        rf"(?P<target_element>.+?)$"
    )
    qualified_pattern = re.compile(
        rf"^(?P<source_model>.+?)::(?P<source_element>.+?)\s+"
        rf"(?P<connector>{connectors})\s+"
        rf"(?P<target_model>.+?)::(?P<target_element>.+?)$"
    )
    simple_pattern = re.compile(
        rf"^(?P<source_element>.+?)\s+"
        rf"(?P<connector>{connectors})\s+"
        rf"(?P<target_element>.+?)$"
    )

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if (
            not line
            or line.startswith("#")
            or line.startswith("//")
            or line == "NO_INTERMODEL_RELATIONSHIPS"
        ):
            continue

        section_match = section_pattern.fullmatch(line)
        if section_match:
            current_pair = (
                resolve_model(section_match.group("source"), model_lookup, line_number),
                resolve_model(section_match.group("target"), model_lookup, line_number),
            )
            continue

        pipe_match = pipe_pattern.fullmatch(line)
        if pipe_match:
            relations.append(Relation(
                resolve_model(pipe_match.group("source_model"), model_lookup, line_number),
                resolve_model(pipe_match.group("target_model"), model_lookup, line_number),
                pipe_match.group("source_element").strip(),
                pipe_match.group("connector"),
                pipe_match.group("target_element").strip(),
                line_number,
            ))
            continue

        qualified_match = qualified_pattern.fullmatch(line)
        if qualified_match:
            relations.append(Relation(
                resolve_model(qualified_match.group("source_model"), model_lookup, line_number),
                resolve_model(qualified_match.group("target_model"), model_lookup, line_number),
                qualified_match.group("source_element").strip(),
                qualified_match.group("connector"),
                qualified_match.group("target_element").strip(),
                line_number,
            ))
            continue

        simple_match = simple_pattern.fullmatch(line)
        if simple_match and current_pair:
            relations.append(Relation(
                current_pair[0],
                current_pair[1],
                simple_match.group("source_element").strip(),
                simple_match.group("connector"),
                simple_match.group("target_element").strip(),
                line_number,
            ))
            continue

        if simple_match:
            raise MergeError(
                f"Relationship line {line_number}: relationship has no model information."
            )
        raise MergeError(
            f"Relationship line {line_number}: unsupported syntax: {line!r}"
        )

    unique: list[Relation] = []
    seen: dict[frozenset[tuple[str, str]], Relation] = {}
    for relation in relations:
        key = frozenset((
            (relation.source_model, normalize(relation.source_element)),
            (relation.target_model, normalize(relation.target_element)),
        ))
        previous = seen.get(key)
        if previous is not None:
            raise MergeError(
                "More than one intermodel relationship between the same two elements: "
                f"lines {previous.line_number} and {relation.line_number} connect "
                f"'{relation.source_element}' and '{relation.target_element}' "
                f"using '{previous.connector}' and '{relation.connector}'."
            )
        seen[key] = relation
        unique.append(relation)
    return unique


def resolve_instance(
    model: Model,
    value: str,
    relation: Relation,
    role: str,
) -> tuple[Instance, str]:
    """
    Resolution order:
    1. exact name
    2. unique tolerant name match
    3. exact description
    4. description with a redundant 4EM element-type prefix
    """
    key = normalize(value)

    # 1. Exact INSTANCE name.
    exact = model.instances.get(key)
    if exact:
        return exact, "name_exact"

    # 2a. Redundant type/name prefix, e.g. "Goal Goal - 1".
    suffix_matches = [
        item for name_key, item in model.instances.items()
        if name_key and key.endswith(name_key)
    ]
    if suffix_matches:
        longest_length = max(len(normalize(item.name)) for item in suffix_matches)
        longest = [
            item for item in suffix_matches
            if len(normalize(item.name)) == longest_length
        ]
        if len(longest) == 1:
            return longest[0], "name_suffix"

    # 2b. Unique shortened name, e.g. "Goal" for "Goal - 1".
    prefix_matches = [
        item for name_key, item in model.instances.items()
        if key and name_key.startswith(key)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0], "name_prefix"

    # 3. Exact Description value.
    description_matches = model.descriptions.get(key, [])
    if len(description_matches) == 1:
        return description_matches[0], "description_exact"
    if len(description_matches) > 1:
        # A few exported ADL models contain duplicate INSTANCE objects with the
        # same Description. The reconstructed SLM files intentionally collapse
        # those duplicate names. Resolve them stably in ADL order so a later
        # intermodel_only run remains executable.
        first = min(description_matches, key=lambda item: item.start)
        return first, "description_exact_duplicate_first"

    # 4. Description with a redundant element-type prefix. Some SLM answers
    # return "Attribute Energy-efficiency" although the prompt requires only
    # "Energy-efficiency". Accept the prefix only when it is a known 4EM type,
    # then resolve the remaining text through the Description index.
    type_prefixes = {
        normalize(prefix) for prefix in DESCRIPTION_TYPE_PREFIXES
    }
    type_prefixes.update(
        normalize(item.element_type) for item in model.instances.values()
    )
    prefixed_description_matches: list[Instance] = []
    for prefix in sorted(type_prefixes, key=len, reverse=True):
        if not prefix or not key.startswith(prefix):
            continue
        remainder = key[len(prefix):]
        if not remainder:
            continue
        prefixed_description_matches.extend(
            model.descriptions.get(remainder, [])
        )

    unique_prefixed_matches = {
        normalize(item.name): item for item in prefixed_description_matches
    }
    if len(unique_prefixed_matches) == 1:
        return next(iter(unique_prefixed_matches.values())), "description_type_prefix"
    if len(unique_prefixed_matches) > 1:
        first = min(
            unique_prefixed_matches.values(),
            key=lambda item: item.start,
        )
        return first, "description_type_prefix_duplicate_first"

    available_names = sorted(item.name for item in model.instances.values())
    available_descriptions = sorted(
        item.description for item in model.instances.values() if item.description
    )
    name_preview = ", ".join(available_names[:10]) or "<none>"
    desc_preview = " | ".join(available_descriptions[:8]) or "<none>"

    raise MergeError(
        f"Relationship line {relation.line_number}: {role} element '{value}' was not "
        f"found by INSTANCE name or Description in model '{model.internal_name}' "
        f"({model.model_type}). Names: {name_preview}. Descriptions: {desc_preview}"
    )


def escape_adl_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def four_em_record_type(
    source_model: Model,
    source: Instance,
    target_model: Model,
    target: Instance,
) -> tuple[str, bool] | None:
    """Return the 4EM record Type and whether Type precedes interref.

    The rules mirror records exported by the confirmed-valid Controlled_S1.adl.
    SLM connector words are not copied into ADL because the record enumeration
    is defined by the 4EM source model, source class and target model.
    """
    source_model_type = normalize(source_model.model_type)
    target_model_type = normalize(target_model.model_type)

    if (
        source_model_type == normalize("Actors and Resources Model")
        and source.element_type == "Role"
    ):
        if (
            target_model_type == normalize("Goal Model")
            and target.element_type == "Goal"
        ):
            return "is responsible for", True
        if (
            target_model_type == normalize("Business Process Model")
            and target.element_type == "Process"
        ):
            if "overview" in normalize(target_model.internal_name):
                return "is responsible for", True
            return "performs", True

    if (
        source_model_type == normalize("Business Process Model")
        and source.element_type == "Information Set"
        and target_model_type == normalize("Concepts Model")
        and target.element_type == "Concept"
    ):
        return "Output", True

    if (
        source_model_type
        == normalize("Technical Components and Requirements Model")
        and source.element_type == "IS Technical Component"
        and target_model_type == normalize("Goal Model")
        and target.element_type == "Goal"
    ):
        return "supports", True

    if (
        source_model_type == normalize("Product-Service-Model")
        and source.element_type == "Unspecific/Product/Service"
        and target_model_type == normalize("Concepts Model")
        and target.element_type == "Concept"
    ):
        return "relates to", False

    return None


def potentially_supported_model_pair(
    source_model: Model,
    target_model: Model,
) -> bool:
    pair = (
        normalize(source_model.model_type),
        normalize(target_model.model_type),
    )
    return pair in {
        (
            normalize("Actors and Resources Model"),
            normalize("Goal Model"),
        ),
        (
            normalize("Actors and Resources Model"),
            normalize("Business Process Model"),
        ),
        (
            normalize("Business Process Model"),
            normalize("Concepts Model"),
        ),
        (
            normalize("Technical Components and Requirements Model"),
            normalize("Goal Model"),
        ),
        (
            normalize("Product-Service-Model"),
            normalize("Concepts Model"),
        ),
    }


def create_record(
    targets: list[tuple[Model, Instance]],
    adl_type: str,
    type_first: bool,
    newline: str,
) -> str:
    references = "".join(
        (
            f'REF mt:"{target_model.model_type}" '
            f'm:"{target_model.internal_name}" '
            f'c:"{target.element_type}" '
            f'i:"{target.name}"{newline}'
        )
        for target_model, target in targets
    )
    type_block = (
        f"\t\t\tATTRIBUTE <Type>{newline}"
        f'\t\t\tVALUE "{escape_adl_string(adl_type)}"{newline}'
    )
    interref_block = (
        f"\t\t\tATTRIBUTE <interref>{newline}"
        f'\t\t\tVALUE "{escape_adl_string(references)}"{newline}'
    )
    blocks = (
        (type_block, interref_block)
        if type_first
        else (interref_block, type_block)
    )
    return (
        f"{newline}\t\tRECORD{newline}"
        f"{blocks[0]}{newline}{blocks[1]}"
        f"\t\tEND{newline}"
    )


def find_intermodel_value_insertion(text: str, instance: Instance) -> int:
    block = text[instance.start:instance.end]
    match = re.search(
        r"(?m)^\tATTRIBUTE <Intermodel-Relations>\s*\r?\n"
        r"\tVALUE[ \t]*(?=\r?\n)",
        block,
    )
    if not match:
        raise MergeError(
            f"Element '{instance.name}' has no writable 'Intermodel-Relations' attribute."
        )
    return instance.start + match.end()


def existing_relation_keys(text: str, instance: Instance) -> set[tuple[str, str, str]]:
    block = text[instance.start:instance.end]
    attr_match = re.search(
        r"(?ms)^\tATTRIBUTE <Intermodel-Relations>\s*\r?\n"
        r"\tVALUE(?P<body>.*?)(?=^\tATTRIBUTE <|\Z)",
        block,
    )
    if not attr_match:
        return set()

    body = attr_match.group("body")
    reference_pattern = re.compile(
        r"REF mt:\\?\"(?P<mt>[^\"]+)\\?\"\s+"
        r"m:\\?\"(?P<m>[^\"]+)\\?\"\s+"
        r"c:\\?\"(?P<c>[^\"]+)\\?\"\s+"
        r"i:\\?\"(?P<i>[^\"]+)\\?\"",
    )
    return {
        (m.group("m"), m.group("c"), normalize(m.group("i")))
        for m in reference_pattern.finditer(body)
    }


def apply_relations(
    text: str,
    models: dict[str, Model],
    relations: Iterable[Relation],
) -> tuple[str, int, int, dict[str, int]]:
    newline = "\r\n" if "\r\n" in text else "\n"
    grouped: dict[
        tuple[int, str, bool, str],
        list[tuple[Model, Instance]],
    ] = {}
    added = 0
    skipped_existing = 0
    resolution_counts: dict[str, int] = {}
    existing_cache: dict[tuple[str, str], set[tuple[str, str, str]]] = {}

    for relation in relations:
        source_model = models[relation.source_model]
        target_model = models[relation.target_model]

        if not potentially_supported_model_pair(source_model, target_model):
            skipped_existing += 1
            resolution_counts["unsupported_4em_relation_skipped"] = (
                resolution_counts.get("unsupported_4em_relation_skipped", 0) + 1
            )
            continue

        try:
            source, source_method = resolve_instance(
                source_model, relation.source_element, relation, "Source"
            )
            target, target_method = resolve_instance(
                target_model, relation.target_element, relation, "Target"
            )
        except MergeError:
            skipped_existing += 1
            resolution_counts["unresolved_supported_relation_skipped"] = (
                resolution_counts.get("unresolved_supported_relation_skipped", 0) + 1
            )
            continue
        resolution_counts[f"source_{source_method}"] = (
            resolution_counts.get(f"source_{source_method}", 0) + 1
        )
        resolution_counts[f"target_{target_method}"] = (
            resolution_counts.get(f"target_{target_method}", 0) + 1
        )

        four_em_rule = four_em_record_type(
            source_model, source, target_model, target
        )
        if four_em_rule is None:
            skipped_existing += 1
            resolution_counts["unsupported_4em_relation_skipped"] = (
                resolution_counts.get("unsupported_4em_relation_skipped", 0) + 1
            )
            continue
        adl_type, type_first = four_em_rule

        cache_key = (source_model.internal_name, normalize(source.name))
        if cache_key not in existing_cache:
            existing_cache[cache_key] = existing_relation_keys(text, source)

        relation_key = (
            target_model.internal_name,
            target.element_type,
            normalize(target.name),
        )
        if relation_key in existing_cache[cache_key]:
            skipped_existing += 1
            continue

        position = find_intermodel_value_insertion(text, source)
        # The valid 4EM reference groups multiple targets only when they belong
        # to the same target model and share the same record Type.
        group_key = (
            position,
            adl_type,
            type_first,
            target_model.internal_name,
        )
        grouped.setdefault(group_key, []).append((target_model, target))
        existing_cache[cache_key].add(relation_key)
        added += 1

    insertions: dict[int, list[str]] = {}
    for (position, adl_type, type_first, _target_model), targets in grouped.items():
        insertions.setdefault(position, []).append(
            create_record(targets, adl_type, type_first, newline)
        )

    result = text
    for position in sorted(insertions, reverse=True):
        result = result[:position] + "".join(insertions[position]) + result[position:]
    return result, added, skipped_existing, resolution_counts


def choose_output_path(input_adl: Path, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output
    return input_adl.with_name(
        f"{input_adl.stem}_with_intermodel_relations{input_adl.suffix}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_adl", type=Path)
    parser.add_argument("relations_txt", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        input_adl = args.input_adl.resolve()
        relations_txt = args.relations_txt.resolve()
        output_path = choose_output_path(input_adl, args.output).resolve()

        if not input_adl.is_file():
            raise MergeError(f"Input ADL file does not exist: {input_adl}")
        if not relations_txt.is_file():
            raise MergeError(f"Relationship text file does not exist: {relations_txt}")
        if relations_txt.suffix.casefold() not in {".txt", ".slm"}:
            raise MergeError("Relationship file must have .txt or .slm extension.")
        if output_path == input_adl:
            raise MergeError("The output path must differ from the input ADL path.")
        if output_path.exists() and not args.overwrite:
            raise MergeError(f"Output exists: {output_path}. Use --overwrite.")

        adl_text, adl_encoding = read_text_preserving_encoding(input_adl)
        relationship_text, _ = read_text_preserving_encoding(relations_txt)

        models = parse_adl(adl_text)
        model_lookup = build_model_lookup(models)
        relations = parse_relationship_text(relationship_text, model_lookup)
        result, added, skipped, resolution_counts = apply_relations(
            adl_text, models, relations
        )
        # The confirmed-valid 4EM exports use CRLF throughout.
        result = result.replace("\r\n", "\n").replace("\n", "\r\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_bytes(result.encode(adl_encoding))
        temporary.replace(output_path)

        print(f"Created: {output_path}")
        print(f"Relationships read: {len(relations)}")
        print(f"Relationships added: {added}")
        print(f"Skipped (already present or unsupported target): {skipped}")
        print("Resolution counts:")
        for key in sorted(resolution_counts):
            print(f"  {key}: {resolution_counts[key]}")
        return 0
    except (MergeError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
