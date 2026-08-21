"""Create an ADL copy with one model name replaced consistently."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MODEL_HEADER = re.compile(r"(?m)^BUSINESS PROCESS MODEL <([^>\r\n]+)>")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_adl", type=Path)
    parser.add_argument("output_adl", type=Path)
    parser.add_argument("--old-name", required=True)
    parser.add_argument("--new-name", required=True)
    parser.add_argument("--max-length", type=int, default=63)
    args = parser.parse_args()

    if not args.input_adl.is_file():
        parser.error(f"Input file does not exist: {args.input_adl}")
    if args.input_adl.resolve() == args.output_adl.resolve():
        parser.error("Input and output paths must differ.")
    if not args.new_name.strip():
        parser.error("The new model name must not be empty.")
    if len(args.new_name) > args.max_length:
        parser.error(
            f"The new model name has {len(args.new_name)} characters; "
            f"the configured maximum is {args.max_length}."
        )

    text = args.input_adl.read_text(encoding="utf-8-sig")
    headers_before = MODEL_HEADER.findall(text)
    if args.old_name not in headers_before:
        parser.error(f"Model name was not found in an ADL header: {args.old_name}")
    if args.new_name in headers_before and args.new_name != args.old_name:
        parser.error(f"The new model name already exists: {args.new_name}")

    occurrence_count = text.count(args.old_name)
    result = text.replace(args.old_name, args.new_name)
    headers_after = MODEL_HEADER.findall(result)
    expected_headers = [
        args.new_name if name == args.old_name else name for name in headers_before
    ]
    if headers_after != expected_headers:
        raise RuntimeError("ADL model-header validation failed after renaming.")
    if args.old_name in result:
        raise RuntimeError("The old model name remains in the generated ADL file.")

    args.output_adl.parent.mkdir(parents=True, exist_ok=True)
    args.output_adl.write_text(result, encoding="utf-8", newline="\n")
    mapping_path = args.output_adl.with_name(
        args.output_adl.stem + "_model_name_mapping.json"
    )
    mapping_path.write_text(
        json.dumps(
            {
                "input_adl": str(args.input_adl.resolve()),
                "output_adl": str(args.output_adl.resolve()),
                "old_name": args.old_name,
                "old_name_length": len(args.old_name),
                "new_name": args.new_name,
                "new_name_length": len(args.new_name),
                "replaced_occurrences": occurrence_count,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Created: {args.output_adl.resolve()}")
    print(f"Mapping: {mapping_path.resolve()}")
    print(f"Replaced occurrences: {occurrence_count}")
    print(f"New model name length: {len(args.new_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
