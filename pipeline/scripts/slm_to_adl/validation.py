"""Helpers for reporting every independently detectable SLM validation error."""

from __future__ import annotations

from typing import Callable, Collection, Iterable, List, Sequence, Tuple, TypeVar


T = TypeVar("T")


def parse_lines_collect(
    lines: Iterable[str], parser: Callable[[str], T], section: str
) -> Tuple[List[T], List[str]]:
    results: List[T] = []
    errors: List[str] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            results.append(parser(line))
        except ValueError as exc:
            errors.append(f"{section} line {line_number}: {exc}")
    return results, errors


def raise_validation_errors(errors: Iterable[str]) -> None:
    unique = list(dict.fromkeys(error for error in errors if error))
    if not unique:
        return
    if len(unique) == 1:
        raise ValueError(unique[0])
    details = "\n".join(f"  {index}. {error}" for index, error in enumerate(unique, start=1))
    raise ValueError(f"Multiple validation errors ({len(unique)}):\n{details}")


def validate_references(
    sources: Sequence[str], targets: Sequence[str], known: Collection[str]
) -> None:
    errors = [
        f"Source {source!r} is not defined under ELEMENTS."
        for source in sources if source not in known
    ]
    errors.extend(
        f"Target {target!r} is not defined under ELEMENTS."
        for target in targets if target not in known
    )
    raise_validation_errors(errors)
