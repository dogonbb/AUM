"""
Merge one 4EM ADL file with one text file containing intermodel relationships.

Usage:
    python add_intermodel_relations_txt.py <input_adl> <relations_txt>
    python add_intermodel_relations_txt.py <input_adl> <relations_txt> --output <output_adl>

By default, the result is written next to the input ADL file as:
    <input_stem>_with_intermodel_relations.adl

Supported connectors:
    play
    empty
    defines
    is_responsible_for
    performs

Supported relationship-text formats
==================================

Format A: pipe-separated, one relationship per line

    SourceModel | SourceElement | connector | TargetModel | TargetElement

Example:

    ActorsandResourcesModel | Customer Service Agent | performs | BusinessProcessModel | Handle Customer Request


Format B: qualified element names

    SourceModel::SourceElement connector TargetModel::TargetElement

Example:

    ActorsandResourcesModel::Customer Service Agent performs BusinessProcessModel::Handle Customer Request


Format C: model-pair sections followed by simple relationship lines

    [ActorsandResourcesModel -> BusinessProcessModel]
    Customer Service Agent performs Handle Customer Request
    Customer Service Manager is_responsible_for Handle Customer Request

    [GoalModel -> ConceptsModel]
    Improve customer satisfaction empty Customer


Blank lines and lines beginning with # or // are ignored.
The exact line NO_INTERMODEL_RELATIONSHIPS is also ignored.

Accepted input extensions for the relationship file:
    .txt
    .slm

The extension does not affect parsing; the file is read as plain text.

The script does not modify the input ADL file.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CONNECTOR_TO_ADL = {
    "play": "play",
    "plays": "play",
    "empty": "",
    "defines": "defines",
    "is_responsible_for": "is responsible for",
    "performs": "performs",
}

MODEL_ALIASES = {
    "GoalModel": {
        "goalmodel",
        "goal",
        "goalandproblemmodel",
        "goalproblemmodel",
    },
    "BusinessRuleModel": {
        "businessrulemodel",
        "businessrulesmodel",
        "businessrule",
    },
    "ConceptsModel": {
        "conceptmodel",
        "conceptsmodel",
        "concept",
        "concepts",
    },
    "BusinessProcessModel": {
        "businessprocessmodel",
        "businessprocess",
        "processmodel",
    },
    "ActorsandResourcesModel": {
        "actorandresourcemodel",
        "actorsandresourcesmodel",
        "actorresourcemodel",
        "actorsresourcesmodel",
    },
    "TechnicalComponentsandRequirementsModel": {
        "technicalcomponentandrequirementmodel",
        "technicalcomponentsandrequirementsmodel",
        "technicalcomponentrequirementsmodel",
        "technicalrequirementsmodel",
        "tcrmodel",
    },
    "ProductServiceModel": {
        "productservicemodel",
        "productservice",
        "productmodel",
    },
}


class MergeError(RuntimeError):
    """Raised when ADL or relationship text cannot be parsed or merged safely."""


@dataclass
class Instance:
    name: str
    element_type: str
    start: int
    end: int


@dataclass
class Model:
    internal_name: str
    model_type: str
    start: int
    end: int
    instances: dict[str, Instance] = field(default_factory=dict)


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


def read_text_preserving_encoding(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise MergeError(f"Could not decode file: {path}")


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
            instance = Instance(
                name=instance_match.group("name").strip(),
                element_type=instance_match.group("type").strip(),
                start=instance_start,
                end=instance_end,
            )
            key = normalize(instance.name)
            if key in model.instances:
                raise MergeError(
                    f"Duplicate normalized element name '{instance.name}' "
                    f"in model '{internal_name}'."
                )
            model.instances[key] = instance

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
            candidates = {
                normalize(internal_name),
                normalize(model.model_type),
            }
            if candidates & target_aliases:
                actual = internal_name
                break

        if actual:
            add(canonical_name, actual)
            for alias in aliases:
                add(alias, actual)

    return lookup


def resolve_model(
    raw_name: str,
    model_lookup: dict[str, str],
    line_number: int,
) -> str:
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
        for connector in sorted(CONNECTOR_TO_ADL, key=len, reverse=True)
    )


def parse_relationship_text(text: str, model_lookup: dict[str, str]) -> list[Relation]:
    """
    Parse all supported relationship-text formats.

    See the module-level documentation for examples.
    """
    relations: list[Relation] = []
    current_pair: tuple[str, str] | None = None
    connectors = connector_pattern()

    section_pattern = re.compile(
        r"^\[\s*(?P<source>.+?)\s*(?:->|=>|to)\s*(?P<target>.+?)\s*\]$",
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
            source_model = resolve_model(
                section_match.group("source"), model_lookup, line_number
            )
            target_model = resolve_model(
                section_match.group("target"), model_lookup, line_number
            )
            current_pair = (source_model, target_model)
            continue

        pipe_match = pipe_pattern.fullmatch(line)
        if pipe_match:
            relations.append(
                Relation(
                    source_model=resolve_model(
                        pipe_match.group("source_model"),
                        model_lookup,
                        line_number,
                    ),
                    target_model=resolve_model(
                        pipe_match.group("target_model"),
                        model_lookup,
                        line_number,
                    ),
                    source_element=pipe_match.group("source_element").strip(),
                    connector=pipe_match.group("connector"),
                    target_element=pipe_match.group("target_element").strip(),
                    line_number=line_number,
                )
            )
            continue

        qualified_match = qualified_pattern.fullmatch(line)
        if qualified_match:
            relations.append(
                Relation(
                    source_model=resolve_model(
                        qualified_match.group("source_model"),
                        model_lookup,
                        line_number,
                    ),
                    target_model=resolve_model(
                        qualified_match.group("target_model"),
                        model_lookup,
                        line_number,
                    ),
                    source_element=qualified_match.group("source_element").strip(),
                    connector=qualified_match.group("connector"),
                    target_element=qualified_match.group("target_element").strip(),
                    line_number=line_number,
                )
            )
            continue

        simple_match = simple_pattern.fullmatch(line)
        if simple_match and current_pair:
            relations.append(
                Relation(
                    source_model=current_pair[0],
                    target_model=current_pair[1],
                    source_element=simple_match.group("source_element").strip(),
                    connector=simple_match.group("connector"),
                    target_element=simple_match.group("target_element").strip(),
                    line_number=line_number,
                )
            )
            continue

        if simple_match and not current_pair:
            raise MergeError(
                f"Relationship line {line_number}: relationship has no model information. "
                "Use a [SourceModel -> TargetModel] section, the pipe-separated "
                "format, or qualified Model::Element names."
            )

        raise MergeError(
            f"Relationship line {line_number}: unsupported syntax: {line!r}"
        )

    unique: list[Relation] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for relation in relations:
        key = (
            relation.source_model,
            normalize(relation.source_element),
            relation.connector,
            relation.target_model,
            normalize(relation.target_element),
        )
        if key not in seen:
            seen.add(key)
            unique.append(relation)

    return unique


def resolve_instance(
    model: Model,
    name: str,
    relation: Relation,
    role: str,
) -> Instance:
    """Resolve an element name, including harmless type-name prefixes.

    SLM output sometimes repeats the element type, for example
    ``Goal Goal - 1`` instead of ``Goal - 1`` or
    ``Process Process - 1`` instead of ``Process - 1``. An exact normalized
    match is preferred. Otherwise a unique longest suffix match is accepted.
    """
    normalized_name = normalize(name)
    instance = model.instances.get(normalized_name)
    if instance:
        return instance

    suffix_matches = [
        item
        for key, item in model.instances.items()
        if key and normalized_name.endswith(key)
    ]
    if suffix_matches:
        longest_length = max(len(normalize(item.name)) for item in suffix_matches)
        longest = [
            item
            for item in suffix_matches
            if len(normalize(item.name)) == longest_length
        ]
        if len(longest) == 1:
            return longest[0]

    # Some generated relations shorten a uniquely identifiable instance,
    # for example ``Goal`` for ``Goal - 1`` or ``Problem`` for
    # ``Problem - 1``. Accept this only when the prefix identifies exactly
    # one instance in the selected model.
    prefix_matches = [
        item
        for key, item in model.instances.items()
        if normalized_name and key.startswith(normalized_name)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    available = sorted(item.name for item in model.instances.values())
    preview = ", ".join(available[:15])
    if len(available) > 15:
        preview += ", ..."

    raise MergeError(
        f"Relationship line {relation.line_number}: {role} element '{name}' was not "
        f"found unambiguously in model '{model.internal_name}' ({model.model_type}). "
        f"Available elements: {preview or '<none>'}"
    )


def escape_adl_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def create_record(target_model: Model, target: Instance, connector: str) -> str:
    adl_type = CONNECTOR_TO_ADL[connector]
    reference = (
        f'REF mt:"{escape_adl_string(target_model.model_type)}" '
        f'm:"{escape_adl_string(target_model.internal_name)}" '
        f'c:"{escape_adl_string(target.element_type)}" '
        f'i:"{escape_adl_string(target.name)}"\n'
    )
    return (
        "\n\t\tRECORD\n"
        "\t\t\tATTRIBUTE <Type>\n"
        f'\t\t\tVALUE "{escape_adl_string(adl_type)}"\n'
        "\n"
        "\t\t\tATTRIBUTE <interref>\n"
        f'\t\t\tVALUE "{reference}"\n'
        "\t\tEND\n"
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
            f"Element '{instance.name}' has no writable "
            "'Intermodel-Relations' attribute."
        )
    return instance.start + match.end()


def existing_relation_keys(
    text: str,
    instance: Instance,
) -> set[tuple[str, str, str, str]]:
    block = text[instance.start:instance.end]
    attr_match = re.search(
        r"(?ms)^\tATTRIBUTE <Intermodel-Relations>\s*\r?\n"
        r"\tVALUE(?P<body>.*?)(?=^\tATTRIBUTE <|\Z)",
        block,
    )
    if not attr_match:
        return set()

    body = attr_match.group("body")
    record_pattern = re.compile(
        r"(?ms)\bRECORD\b.*?"
        r"ATTRIBUTE <Type>\s*\r?\n\s*VALUE \"(?P<type>[^\"]*)\".*?"
        r"ATTRIBUTE <interref>\s*\r?\n\s*VALUE \""
        r"REF mt:\\?\"(?P<mt>[^\"]+)\\?\"\s+"
        r"m:\\?\"(?P<m>[^\"]+)\\?\"\s+"
        r"c:\\?\"(?P<c>[^\"]+)\\?\"\s+"
        r"i:\\?\"(?P<i>[^\"]+)\\?\"",
    )

    return {
        (
            match.group("type"),
            match.group("m"),
            match.group("c"),
            normalize(match.group("i")),
        )
        for match in record_pattern.finditer(body)
    }


def apply_relations(
    text: str,
    models: dict[str, Model],
    relations: Iterable[Relation],
) -> tuple[str, int, int]:
    insertions: dict[int, list[str]] = {}
    added = 0
    skipped_existing = 0
    existing_cache: dict[tuple[str, str], set[tuple[str, str, str, str]]] = {}

    for relation in relations:
        source_model = models[relation.source_model]
        target_model = models[relation.target_model]

        source = resolve_instance(
            source_model, relation.source_element, relation, "Source"
        )
        target = resolve_instance(
            target_model, relation.target_element, relation, "Target"
        )

        cache_key = (source_model.internal_name, normalize(source.name))
        if cache_key not in existing_cache:
            existing_cache[cache_key] = existing_relation_keys(text, source)

        adl_type = CONNECTOR_TO_ADL[relation.connector]
        relation_key = (
            adl_type,
            target_model.internal_name,
            target.element_type,
            normalize(target.name),
        )

        if relation_key in existing_cache[cache_key]:
            skipped_existing += 1
            continue

        position = find_intermodel_value_insertion(text, source)
        insertions.setdefault(position, []).append(
            create_record(target_model, target, relation.connector)
        )
        existing_cache[cache_key].add(relation_key)
        added += 1

    result = text
    for position in sorted(insertions, reverse=True):
        result = result[:position] + "".join(insertions[position]) + result[position:]

    return result, added, skipped_existing


def choose_output_path(input_adl: Path, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output
    return input_adl.with_name(
        f"{input_adl.stem}_with_intermodel_relations{input_adl.suffix}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a 4EM ADL file with intermodel relationships from one TXT or SLM text file."
    )
    parser.add_argument(
        "input_adl",
        type=Path,
        help="Path to the complete ADL file.",
    )
    parser.add_argument(
        "relations_txt",
        type=Path,
        help="Path to the TXT or SLM text file containing intermodel relationships.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output ADL path. Default: next to the input ADL file with suffix "
            "'_with_intermodel_relations.adl'."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
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
            raise MergeError(
                f"Relationship text file does not exist: {relations_txt}"
            )
        if relations_txt.suffix.casefold() not in {".txt", ".slm"}:
            raise MergeError(
                "Expected a plain-text relationship file with the extension "
                f".txt or .slm, received: {relations_txt.name}"
            )
        if output_path == input_adl:
            raise MergeError("The output path must differ from the input ADL path.")
        if output_path.exists() and not args.overwrite:
            raise MergeError(
                f"Output file already exists: {output_path}. "
                "Use --overwrite to replace it."
            )

        adl_text, adl_encoding = read_text_preserving_encoding(input_adl)
        relationship_text, _ = read_text_preserving_encoding(relations_txt)

        models = parse_adl(adl_text)
        model_lookup = build_model_lookup(models)
        relations = parse_relationship_text(relationship_text, model_lookup)

        result, added, skipped_existing = apply_relations(
            adl_text, models, relations
        )

        # An empty relationship file is valid. In that case the unchanged
        # input ADL is still written to the requested output path.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(result, encoding=adl_encoding, newline="")
        temporary.replace(output_path)

        print(f"Created: {output_path}")
        print(f"Relationships read: {len(relations)}")
        print(f"Relationships added: {added}")
        print(f"Already present and skipped: {skipped_existing}")
        return 0

    except (MergeError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())