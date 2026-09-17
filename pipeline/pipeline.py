#!/usr/bin/env python3
"""
pipeline.py

Configuration-driven 4EM pipeline for the folder structure that accompanies
this file. By default, the script reads ``parameter.json`` from the same
folder as ``pipeline.py``. All project paths, prompt names, converter names,
output names, model-folder mappings, SLM settings, retry limits, and
intermodel pairs are read from that JSON file.

Supported execution modes
-------------------------

``full``
    Generate all available models for every selected scenario, convert them
    to ADL, merge the ADL files, generate intermodel relationships, and add
    those relationships to the merged ADL.

``models_only``
    Process exactly one configured scenario subdirectory, generate its model
    SLM outputs, convert them to ADL, and merge the ADL files. The intermodel
    stage is skipped.

``intermodel_only``
    Skip model generation and process exactly one configured scenario. The
    matching entry from ``intermodel_only.jobs`` supplies its artifact
    manifest, merged ADL file, and output directory.

The script calls Ollama's native ``/api/generate`` endpoint. A generation is
hard-stopped after the configured total timeout or when streamed thinking text
repeats beyond the configured limit. The model request is then started again
when retry handling permits it. An optional service restart command can also
be configured.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import platform
import queue
import re
import secrets
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

from scripts.intermodel_rules import (
    CONNECTORS_BY_SOURCE_MODEL,
)


VALID_MODES = {"full", "models_only", "intermodel_only"}


class PipelineError(RuntimeError):
    """Raised when configuration or pipeline execution is invalid."""


class SLMTimeoutError(TimeoutError):
    """Raised when one SLM request exceeds its hard timeout."""


class SLMRepetitionError(PipelineError):
    """Raised when streamed SLM thinking is stuck in a repetition loop."""


@dataclass
class ExternalCommandResult:
    args: list[str]
    cwd: str
    started_at: str
    finished_at: str
    duration_seconds: float
    timeout_seconds: float | None
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.returncode == 0


@dataclass
class SLMResult:
    answer: str
    thinking: str
    started_at: str
    finished_at: str
    wall_seconds: float
    metadata: dict[str, Any]
    request_without_prompt: dict[str, Any]


@dataclass
class ModelDefinition:
    model_id: str
    scenario_folder: str
    display_name: str
    model_type: str
    prompt_path: Path
    description_path: Path
    format_prompt_path: Path
    converter_path: Path
    converter_command: list[str]
    converter_cwd: Path
    converter_timeout_seconds: float | None
    adl_model_name_template: str
    slm_filename_template: str
    adl_filename_template: str
    enabled: bool = True


@dataclass
class ModelArtifact:
    scenario_name: str
    model_id: str
    model_folder: str
    model_display_name: str
    model_type: str
    source_scenario_file: Path
    source_relative_path: Path
    source_stem: str
    adl_model_name: str
    slm_file: Path
    adl_file: Path
    status: str
    task_id: str


@dataclass
class ScenarioRun:
    scenario_name: str
    scenario_directory: Path
    run_directory: Path
    model_slm_directory: Path
    model_adl_directory: Path
    artifact_manifest: Path
    merged_adl: Path
    intermodel_slm_directory: Path
    intermodel_aggregate_slm: Path
    final_adl: Path


@dataclass
class RunContext:
    parameter_path: Path
    project_root: Path
    config: dict[str, Any]
    run_id: str
    timestamp: str
    started_at: str
    started_perf: float
    report_json: Path
    report_text: Path
    event_log: Path
    report: dict[str, Any] = field(default_factory=dict)

    def event(self, event_type: str, **payload: Any) -> None:
        record = {
            "timestamp": iso_now(),
            "run_id": self.run_id,
            "event_type": event_type,
            **to_json_safe(payload),
        }
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def to_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    return value


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise PipelineError(f"Could not decode text file: {path}")


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(
        path,
        json.dumps(to_json_safe(value), ensure_ascii=False, indent=2) + "\n",
    )


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "unnamed"


def normalized_extension(value: str) -> str:
    return value if value.startswith(".") else "." + value


def txt_output_path(path: Path) -> Path:
    """Force generated textual SLM responses to use the .txt suffix."""
    return path.with_suffix(".txt")


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "is_file": path.is_file(),
        "is_directory": path.is_dir(),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": file_sha256(path),
    }


def clean_slm_answer(text: str) -> str:
    text = re.sub(
        r"<(?:think|thinking)>.*?</(?:think|thinking)>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    fenced = re.fullmatch(
        r"```(?:text|txt|slm)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fenced.group(1).strip() if fenced else text


def dotted_get(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def require(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value = dotted_get(mapping, dotted_key, None)
    if value is None:
        raise PipelineError(f"Missing required parameter: {dotted_key}")
    return value


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def render_string(template: str, values: Mapping[str, Any]) -> str:
    class StrictDictionary(dict[str, str]):
        def __missing__(self, key: str) -> str:
            raise PipelineError(f"Unknown template placeholder: {{{key}}}")

    return template.format_map(
        StrictDictionary({key: str(value) for key, value in values.items()})
    )


def redact_secrets(value: Any) -> Any:
    secret_fragments = (
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
    )
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if any(fragment in str(key).casefold() for fragment in secret_fragments):
                result[str(key)] = "***REDACTED***"
            else:
                result[str(key)] = redact_secrets(item)
        return result
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_parameter_file(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise PipelineError(
            f"Invalid JSON in {path}, line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(config, dict):
        raise PipelineError("parameter.json must contain a JSON object.")

    mode = dotted_get(config, "execution.mode", "full")
    if mode not in VALID_MODES:
        raise PipelineError(
            f"execution.mode must be one of {sorted(VALID_MODES)}, got {mode!r}."
        )

    provider = dotted_get(config, "slm.provider", "ollama")
    if provider != "ollama":
        raise PipelineError("This pipeline currently supports slm.provider='ollama'.")

    timeout_seconds = dotted_get(config, "slm.timeout_seconds", 3600)
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise PipelineError("slm.timeout_seconds must be a positive number.")
    enabled_setting(
        dotted_get(config, "slm.thinking_repetition_enabled", True),
        "slm.thinking_repetition_enabled",
    )
    enabled_setting(
        dotted_get(config, "slm.log_thinking", False),
        "slm.log_thinking",
    )
    thinking_repetition_limit = dotted_get(
        config, "slm.thinking_repetition_limit", 5
    )
    if (
        not isinstance(thinking_repetition_limit, int)
        or isinstance(thinking_repetition_limit, bool)
        or thinking_repetition_limit < 2
    ):
        raise PipelineError(
            "slm.thinking_repetition_limit must be an integer of at least 2."
        )
    thinking_repetition_min_block_chars = dotted_get(
        config, "slm.thinking_repetition_min_block_chars", 0
    )
    if (
        not isinstance(thinking_repetition_min_block_chars, int)
        or isinstance(thinking_repetition_min_block_chars, bool)
        or thinking_repetition_min_block_chars < 0
    ):
        raise PipelineError(
            "slm.thinking_repetition_min_block_chars must be a non-negative integer."
        )
    configured_seed = dotted_get(config, "slm.options.seed")
    if configured_seed is not None and configured_seed is not False and (
        not isinstance(configured_seed, int) or isinstance(configured_seed, bool)
    ):
        raise PipelineError("slm.options.seed must be an integer or false.")
    max_restart = dotted_get(config, "retry.max_restart", 0)
    if not isinstance(max_restart, int) or isinstance(max_restart, bool) or max_restart < 0:
        raise PipelineError("retry.max_restart must be a non-negative integer.")
    seed_mode = str(dotted_get(config, "retry.seed_mode", "incremental")).casefold()
    if seed_mode not in {"incremental", "random"}:
        raise PipelineError("retry.seed_mode must be 'incremental' or 'random'.")
    seed_increment = dotted_get(config, "retry.seed_increment", 1)
    if (
        not isinstance(seed_increment, int)
        or isinstance(seed_increment, bool)
        or seed_increment <= 0
    ):
        raise PipelineError("retry.seed_increment must be a positive integer.")
    max_repairs = dotted_get(config, "format_repair.max_attempts", 0)
    if not isinstance(max_repairs, int) or isinstance(max_repairs, bool) or max_repairs < 0:
        raise PipelineError("format_repair.max_attempts must be a non-negative integer.")

    selected_scenario = dotted_get(config, "execution.selected_scenario")
    if not isinstance(selected_scenario, str) or not selected_scenario.strip():
        raise PipelineError(
            "execution.selected_scenario must contain the name or relative path "
            "of exactly one scenario subdirectory."
        )

    return config


def parse_model_definitions(ctx: RunContext) -> list[ModelDefinition]:
    raw_models = require(ctx.config, "models")
    if not isinstance(raw_models, list):
        raise PipelineError("models must be an array in parameter.json.")

    definitions: list[ModelDefinition] = []
    seen_ids: set[str] = set()
    seen_folders: set[str] = set()

    default_converter_timeout = dotted_get(
        ctx.config, "tools.default_python_timeout_seconds", 600
    )

    for index, raw in enumerate(raw_models):
        if not isinstance(raw, dict):
            raise PipelineError(f"models[{index}] must be an object.")

        model_id = str(require(raw, "id"))
        scenario_folder = str(require(raw, "scenario_folder"))
        if model_id in seen_ids:
            raise PipelineError(f"Duplicate model id: {model_id}")
        if scenario_folder in seen_folders:
            raise PipelineError(f"Duplicate model scenario_folder: {scenario_folder}")
        seen_ids.add(model_id)
        seen_folders.add(scenario_folder)

        converter = require(raw, "converter")
        if not isinstance(converter, dict):
            raise PipelineError(f"models[{index}].converter must be an object.")

        command = converter.get(
            "command",
            [
                "{python}",
                "{converter_path}",
                "{input_slm}",
                "{output_adl}",
                "--model-name",
                "{adl_model_name}",
            ],
        )
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            raise PipelineError(
                f"models[{index}].converter.command must be an array of strings."
            )

        definitions.append(
            ModelDefinition(
                model_id=model_id,
                scenario_folder=scenario_folder,
                display_name=str(raw.get("display_name", model_id)),
                model_type=str(raw.get("model_type", raw.get("display_name", model_id))),
                prompt_path=resolve_path(ctx.project_root, require(raw, "prompt_path")),
                description_path=resolve_path(
                    ctx.project_root, require(raw, "description_path")
                ),
                format_prompt_path=resolve_path(
                    ctx.project_root, require(raw, "format_prompt_path")
                ),
                converter_path=resolve_path(
                    ctx.project_root, require(converter, "path")
                ),
                converter_command=list(command),
                converter_cwd=resolve_path(
                    ctx.project_root, converter.get("cwd", ".")
                ),
                converter_timeout_seconds=(
                    float(converter["timeout_seconds"])
                    if converter.get("timeout_seconds") is not None
                    else (
                        float(default_converter_timeout)
                        if default_converter_timeout is not None
                        else None
                    )
                ),
                adl_model_name_template=str(
                    raw.get(
                        "adl_model_name_template",
                        "{scenario_name}_{model_id}_{source_relative_id}",
                    )
                ),
                slm_filename_template=str(
                    raw.get("slm_filename_template", "{source_stem}.txt")
                ),
                adl_filename_template=str(
                    raw.get("adl_filename_template", "{source_stem}.adl")
                ),
                enabled=bool(raw.get("enabled", True)),
            )
        )

    return definitions


def validate_static_paths(ctx: RunContext, models: Sequence[ModelDefinition]) -> None:
    paths_to_check: list[tuple[str, Path]] = []

    for model in models:
        if not model.enabled:
            continue
        paths_to_check.extend(
            [
                (f"Prompt for {model.model_id}", model.prompt_path),
                (f"Description for {model.model_id}", model.description_path),
                (f"Format prompt for {model.model_id}", model.format_prompt_path),
                (f"Converter for {model.model_id}", model.converter_path),
            ]
        )

    mode = dotted_get(ctx.config, "execution.mode", "full")
    if mode in {"full", "models_only"}:
        merger_path = resolve_path(
            ctx.project_root, require(ctx.config, "tools.adl_merger.path")
        )
        paths_to_check.append(("ADL merger", merger_path))

    if mode in {"full", "intermodel_only"} and dotted_get(
        ctx.config, "intermodel.enabled", True
    ):
        template = resolve_path(
            ctx.project_root, require(ctx.config, "prompts.intermodel_template")
        )
        integrator = resolve_path(
            ctx.project_root, require(ctx.config, "tools.intermodel_integrator.path")
        )
        paths_to_check.extend(
            [("Intermodel prompt template", template), ("Intermodel integrator", integrator)]
        )
        source_ids = {source_id for source_id, _ in intermodel_pairs(ctx)}
        for source_id in sorted(source_ids):
            paths_to_check.append(
                (
                    f"Intermodel source rules for {source_id}",
                    ctx.project_root
                    / "prompts"
                    / "inter_model_generation"
                    / "source_connection_rules"
                    / f"{source_id}.txt",
                )
            )
        for source_id, target_id in intermodel_pairs(ctx):
            paths_to_check.append(
                (
                    f"Intermodel target rules for {source_id} -> {target_id}",
                    ctx.project_root
                    / "prompts"
                    / "inter_model_generation"
                    / "target_reference_rules"
                    / f"{source_id}__{target_id}.txt",
                )
            )

    if bool(dotted_get(ctx.config, "format_repair.enabled", False)):
        paths_to_check.append(
            (
                "General format repair prompt",
                resolve_path(
                    ctx.project_root,
                    require(ctx.config, "format_repair.general_prompt_path"),
                ),
            )
        )
        if mode in {"full", "intermodel_only"}:
            paths_to_check.append(
                (
                    "Intermodel format prompt",
                    resolve_path(
                        ctx.project_root,
                        require(ctx.config, "intermodel.format_prompt_path"),
                    ),
                )
            )
            paths_to_check.append(
                (
                    "Intermodel repair prompt",
                    resolve_path(
                        ctx.project_root,
                        require(ctx.config, "intermodel.repair_prompt_path"),
                    ),
                )
            )

    if enabled_setting(
        dotted_get(ctx.config, "visualization.enabled", "off"),
        "visualization.enabled",
    ):
        paths_to_check.append(
            (
                "Runtime visualization script",
                resolve_path(
                    ctx.project_root,
                    require(ctx.config, "visualization.script_path"),
                ),
            )
        )

    missing = [f"{label}: {path}" for label, path in paths_to_check if not path.is_file()]
    if missing:
        raise PipelineError("Required files are missing:\n" + "\n".join(missing))


# ---------------------------------------------------------------------------
# External commands
# ---------------------------------------------------------------------------


def build_command(
    command_template: Sequence[str],
    values: Mapping[str, Any],
    list_values: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    list_values = list_values or {}
    result: list[str] = []
    for raw_part in command_template:
        exact = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", raw_part)
        if exact and exact.group(1) in list_values:
            result.extend(str(item) for item in list_values[exact.group(1)])
        else:
            result.append(render_string(raw_part, values))
    return result


def run_external_command(
    command_template: Sequence[str],
    values: Mapping[str, Any],
    cwd: Path,
    timeout_seconds: float | None,
    list_values: Mapping[str, Sequence[str]] | None = None,
) -> ExternalCommandResult:
    args = build_command(command_template, values, list_values)
    started_at = iso_now()
    started_perf = time.perf_counter()

    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        return ExternalCommandResult(
            args=args,
            cwd=str(cwd),
            started_at=started_at,
            finished_at=iso_now(),
            duration_seconds=time.perf_counter() - started_perf,
            timeout_seconds=timeout_seconds,
            returncode=completed.returncode,
            timed_out=False,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return ExternalCommandResult(
            args=args,
            cwd=str(cwd),
            started_at=started_at,
            finished_at=iso_now(),
            duration_seconds=time.perf_counter() - started_perf,
            timeout_seconds=timeout_seconds,
            returncode=None,
            timed_out=True,
            stdout=stdout,
            stderr=stderr,
        )


def command_report(ctx: RunContext, result: ExternalCommandResult) -> dict[str, Any]:
    include_output = bool(dotted_get(ctx.config, "report.include_command_output", True))
    data = asdict(result)
    data["ok"] = result.ok
    data["stdout_chars"] = len(result.stdout)
    data["stderr_chars"] = len(result.stderr)
    if not include_output:
        data["stdout"] = None
        data["stderr"] = None
    return data


# ---------------------------------------------------------------------------
# Ollama SLM execution with a hard timeout
# ---------------------------------------------------------------------------


def normalize_thinking(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    lowered = str(value).strip().casefold()
    if lowered in {"false", "off", "none", "0"}:
        return False
    if lowered in {"true", "on", "1"}:
        return True
    if lowered in {"low", "medium", "high"}:
        return lowered
    raise PipelineError("slm.thinking must be true, false, low, medium, or high.")


def enabled_setting(value: Any, name: str) -> bool:
    """Read a boolean switch that may also be written as 'on' or 'off'."""
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().casefold()
    if lowered in {"on", "true", "1"}:
        return True
    if lowered in {"off", "false", "0"}:
        return False
    raise PipelineError(f"{name} must be on, off, true, or false.")


def ollama_worker(
    result_queue: mp.Queue,
    endpoint: str,
    payload: dict[str, Any],
    socket_timeout_seconds: float,
) -> None:
    started_at = iso_now()
    started_perf = time.perf_counter()
    try:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response_parts: list[str] = []
        thinking_parts: list[str] = []
        response_json: dict[str, Any] | None = None
        with urllib.request.urlopen(request, timeout=socket_timeout_seconds) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                chunk = json.loads(raw_line.decode("utf-8"))
                if "error" in chunk:
                    raise RuntimeError(str(chunk["error"]))

                response_text = str(chunk.get("response", ""))
                thinking_text = str(chunk.get("thinking", ""))
                response_parts.append(response_text)
                thinking_parts.append(thinking_text)
                response_json = chunk

                # A non-empty response or thinking chunk represents token
                # activity and resets the idle timeout in the parent process.
                if response_text or thinking_text:
                    result_queue.put(
                        {
                            "type": "token",
                            "response": response_text,
                            "thinking": thinking_text,
                        }
                    )

        if response_json is None:
            raise RuntimeError("The Ollama response stream was empty.")
        if not response_json.get("done", False):
            raise RuntimeError("The Ollama response stream ended without done=true.")
        response_json["response"] = "".join(response_parts)
        response_json["thinking"] = "".join(thinking_parts)
        result_queue.put(
            {
                "type": "result",
                "ok": True,
                "started_at": started_at,
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - started_perf,
                "result": response_json,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "type": "result",
                "ok": False,
                "started_at": started_at,
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - started_perf,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def wait_for_slm_worker(
    result_queue: mp.Queue,
    process: mp.Process,
    timeout_seconds: float,
    thinking_repetition_limit: int,
    show_stream_output: bool = False,
    thinking_repetition_enabled: bool = True,
    thinking_log_handle: TextIO | None = None,
    thinking_repetition_min_block_chars: int = 0,
) -> dict[str, Any] | None:
    """Wait for Ollama while enforcing total time and thinking repetition."""
    started_perf = time.perf_counter()
    total_deadline = started_perf + timeout_seconds
    thinking_header_shown = False
    response_header_shown = False
    stream_text_shown = False
    thinking_line_buffer = ""
    thinking_block_lines: list[str] = []
    thinking_block_counts: dict[str, int] = {}

    def finish_stream_line() -> None:
        if show_stream_output and stream_text_shown:
            print(flush=True)

    def count_thinking_block() -> None:
        nonlocal thinking_block_lines
        if not thinking_block_lines:
            return
        block = "".join(thinking_block_lines)
        thinking_block_lines = []
        # Structured model-output sections may legitimately be emitted more
        # than once while the model assembles its answer. Ignore those blocks,
        # but keep monitoring all other thinking blocks for repetition loops.
        block_start = block.lstrip()
        if re.match(r"(?:ELEMENTS|CONNECTIONS)(?:\s|$)", block_start) is not None:
            return
        if len(block.strip()) < thinking_repetition_min_block_chars:
            return
        count = thinking_block_counts.get(block, 0) + 1
        thinking_block_counts[block] = count
        if count >= thinking_repetition_limit:
            finish_stream_line()
            excerpt = re.sub(r"\s+", " ", block).strip()[:160]
            raise SLMRepetitionError(
                "The same thinking block was generated "
                f"{count} times: {excerpt!r}"
            )

    def inspect_thinking_blocks(text: str) -> None:
        nonlocal thinking_line_buffer
        thinking_line_buffer += text
        lines = thinking_line_buffer.splitlines(keepends=True)
        thinking_line_buffer = ""
        for line in lines:
            if not line.endswith(("\n", "\r")):
                thinking_line_buffer = line
                continue
            if not line.strip():
                count_thinking_block()
                continue
            thinking_block_lines.append(line)

    def finish_thinking_blocks() -> None:
        nonlocal thinking_line_buffer
        if thinking_line_buffer:
            if thinking_line_buffer.strip():
                thinking_block_lines.append(thinking_line_buffer)
            thinking_line_buffer = ""
        count_thinking_block()

    while True:
        now = time.perf_counter()
        total_remaining = total_deadline - now
        if total_remaining <= 0:
            finish_stream_line()
            raise SLMTimeoutError(
                "The SLM request exceeded the configured total limit of "
                f"{timeout_seconds:.2f} seconds."
            )
        try:
            message = result_queue.get(timeout=min(0.25, total_remaining))
        except queue.Empty:
            if process.is_alive():
                continue
            try:
                message = result_queue.get_nowait()
            except queue.Empty:
                finish_stream_line()
                return None

        if message.get("type") == "token":
            thinking_text = str(message.get("thinking", ""))
            response_text = str(message.get("response", ""))
            if thinking_text and thinking_log_handle is not None:
                thinking_log_handle.write(thinking_text)
                thinking_log_handle.flush()
            if show_stream_output:
                if thinking_text:
                    if not thinking_header_shown:
                        print("\n[SLM THINKING]", flush=True)
                        thinking_header_shown = True
                    print(thinking_text, end="", flush=True)
                    stream_text_shown = True
                if response_text:
                    if not response_header_shown:
                        print("\n[SLM OUTPUT]", flush=True)
                        response_header_shown = True
                    print(response_text, end="", flush=True)
                    stream_text_shown = True
            if thinking_text and thinking_repetition_enabled:
                inspect_thinking_blocks(thinking_text)
            continue
        if thinking_repetition_enabled:
            finish_thinking_blocks()
        finish_stream_line()
        return message


def stop_process(process: mp.Process) -> None:
    """Stop a worker that is still waiting on an Ollama response."""
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def call_slm(
    ctx: RunContext,
    prompt: str,
    seed_override: int | None = None,
    thinking_log_path: Path | None = None,
) -> SLMResult:
    slm = require(ctx.config, "slm")
    if not isinstance(slm, dict):
        raise PipelineError("slm must be an object.")

    timeout_seconds = float(slm.get("timeout_seconds", 3600))
    thinking_repetition_enabled = enabled_setting(
        slm.get("thinking_repetition_enabled", True),
        "slm.thinking_repetition_enabled",
    )
    thinking_repetition_limit = int(
        slm.get("thinking_repetition_limit", 5)
    )
    thinking_repetition_min_block_chars = int(
        slm.get("thinking_repetition_min_block_chars", 0)
    )
    show_stream_output = enabled_setting(
        slm.get("show_stream_output", False), "slm.show_stream_output"
    )
    endpoint = str(require(slm, "base_url")).rstrip("/") + "/api/generate"

    payload: dict[str, Any] = {
        "model": str(require(slm, "model")),
        "prompt": prompt,
        "stream": True,
        "think": normalize_thinking(slm.get("thinking", False)),
    }
    options = copy.deepcopy(slm.get("options", {}))
    if options:
        if not isinstance(options, dict):
            raise PipelineError("slm.options must be an object.")
    # JSON false explicitly disables Ollama's seed option. This is distinct
    # from seed=0 and takes precedence over retry.seed_mode.
    if isinstance(options, dict) and options.get("seed") is False:
        options.pop("seed")
    if seed_override is not None:
        if not isinstance(options, dict):
            raise PipelineError("slm.options must be an object.")
        options["seed"] = seed_override
    if options:
        payload["options"] = options
    if slm.get("keep_alive") is not None:
        payload["keep_alive"] = slm["keep_alive"]

    request_without_prompt = copy.deepcopy(payload)
    request_without_prompt["prompt"] = {
        "characters": len(prompt),
        "utf8_bytes": len(prompt.encode("utf-8")),
        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }

    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue()
    process = context.Process(
        target=ollama_worker,
        args=(result_queue, endpoint, payload, timeout_seconds),
        daemon=True,
    )

    started_at = iso_now()
    started_perf = time.perf_counter()
    thinking_log_handle: TextIO | None = None
    process_started = False
    try:
        if thinking_log_path is not None:
            thinking_log_path.parent.mkdir(parents=True, exist_ok=True)
            thinking_log_handle = thinking_log_path.open(
                "w", encoding="utf-8", newline=""
            )
        process.start()
        process_started = True
        message = wait_for_slm_worker(
            result_queue,
            process,
            timeout_seconds,
            thinking_repetition_limit,
            show_stream_output,
            thinking_repetition_enabled,
            thinking_log_handle,
            thinking_repetition_min_block_chars,
        )
    except BaseException:
        if process_started:
            stop_process(process)
        raise
    finally:
        if thinking_log_handle is not None:
            thinking_log_handle.close()

    process.join(timeout=5)
    if message is None:
        raise PipelineError(
            f"The SLM worker exited without a result. Exit code: {process.exitcode}"
        )
    if not message.get("ok"):
        raise PipelineError(
            f"SLM request failed: {message.get('error_type')}: "
            f"{message.get('error_message')}\n{message.get('traceback', '')}"
        )

    result = message["result"]
    answer = clean_slm_answer(str(result.get("response", "")))
    thinking = str(result.get("thinking", "")).strip()
    metadata = {
        key: value
        for key, value in result.items()
        if key not in {"response", "thinking", "context"}
    }
    metadata.update(
        {
            "answer_characters": len(answer),
            "answer_utf8_bytes": len(answer.encode("utf-8")),
            "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            "thinking_characters": len(thinking),
        }
    )
    eval_count = metadata.get("eval_count")
    eval_duration = metadata.get("eval_duration")
    if isinstance(eval_count, (int, float)) and isinstance(eval_duration, (int, float)) and eval_duration:
        metadata["output_tokens_per_second"] = float(eval_count) / (
            float(eval_duration) / 1_000_000_000
        )

    return SLMResult(
        answer=answer,
        thinking=thinking,
        started_at=started_at,
        finished_at=iso_now(),
        wall_seconds=time.perf_counter() - started_perf,
        metadata=metadata,
        request_without_prompt=request_without_prompt,
    )


def slm_report(ctx: RunContext, result: SLMResult, prompt: str) -> dict[str, Any]:
    include_prompt = bool(dotted_get(ctx.config, "report.include_prompt_text", False))
    include_output = bool(dotted_get(ctx.config, "report.include_slm_output_text", False))
    return {
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "wall_seconds": result.wall_seconds,
        "request": result.request_without_prompt,
        "metadata": result.metadata,
        "prompt_text": prompt if include_prompt else None,
        "answer_text": result.answer if include_output else None,
        "thinking_text": result.thinking if include_output else None,
    }


def restart_slm_service(
    ctx: RunContext,
    reason: str,
    task_id: str,
) -> ExternalCommandResult | None:
    restart = dotted_get(ctx.config, "tools.slm_restart", {})
    if not isinstance(restart, dict) or not restart.get("enabled", False):
        ctx.event(
            "slm_service_restart_skipped",
            reason=reason,
            task_id=task_id,
            detail="No enabled restart command is configured; a fresh SLM request will still be created.",
        )
        return None

    command = restart.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise PipelineError("tools.slm_restart.command must be an array of strings.")

    python_executable = str(dotted_get(ctx.config, "tools.python_executable", "") or sys.executable)
    values = {
        "python": python_executable,
        "project_root": ctx.project_root,
        "reason": reason,
        "task_id": task_id,
        "model": str(require(ctx.config, "slm.model")),
        "base_url": str(require(ctx.config, "slm.base_url")),
    }
    result = run_external_command(
        command,
        values,
        resolve_path(ctx.project_root, restart.get("cwd", ".")),
        float(restart["timeout_seconds"])
        if restart.get("timeout_seconds") is not None
        else None,
    )
    ctx.event(
        "slm_service_restart_finished",
        reason=reason,
        task_id=task_id,
        result=command_report(ctx, result),
    )
    return result


def slm_thinking_log_path(
    ctx: RunContext,
    task_id: str,
    attempt_number: int,
    seed: int | None,
) -> Path | None:
    """Return the per-attempt streamed thinking log inside the current run."""
    if not enabled_setting(
        dotted_get(ctx.config, "slm.log_thinking", False),
        "slm.log_thinking",
    ):
        return None

    configured_directory = str(
        dotted_get(ctx.config, "paths.thinking_log_directory", "thinking_logs")
    ).strip()
    relative_directory = Path(configured_directory)
    if (
        not configured_directory
        or relative_directory.is_absolute()
        or ".." in relative_directory.parts
    ):
        raise PipelineError(
            "paths.thinking_log_directory must be a non-empty relative path "
            "without '..'."
        )

    task_slug = safe_name(task_id)
    if len(task_slug) > 100:
        task_hash = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
        task_slug = f"{task_slug[:80]}_{task_hash}"
    seed_label = str(seed) if seed is not None else "none"
    return (
        ctx.report_json.parent
        / relative_directory
        / task_slug
        / f"attempt_{attempt_number:02d}_seed_{seed_label}.thinking.txt"
    )


def call_slm_with_restart(
    ctx: RunContext,
    prompt: str,
    task_id: str,
) -> tuple[SLMResult, list[dict[str, Any]]]:
    """Call the SLM and restart it after a timeout or thinking loop."""
    max_restart = int(dotted_get(ctx.config, "retry.max_restart", 0))
    seed_mode = str(
        dotted_get(ctx.config, "retry.seed_mode", "incremental")
    ).casefold()
    seed_increment = int(dotted_get(ctx.config, "retry.seed_increment", 1))
    configured_seed = dotted_get(ctx.config, "slm.options.seed")
    seed_disabled = configured_seed is False
    base_seed = (
        int(configured_seed)
        if configured_seed is not None and not seed_disabled
        else None
    )
    attempts: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    for attempt_number in range(1, max_restart + 2):
        started = time.perf_counter()
        if seed_disabled:
            attempt_seed = None
        elif seed_mode == "random":
            attempt_seed = secrets.randbelow(2**31)
            while attempt_seed in used_seeds:
                attempt_seed = secrets.randbelow(2**31)
            used_seeds.add(attempt_seed)
        else:
            attempt_seed = (
                base_seed + (attempt_number - 1) * seed_increment
                if base_seed is not None
                else None
            )
        attempt: dict[str, Any] = {
            "attempt_number": attempt_number,
            "started_at": iso_now(),
            "seed": attempt_seed,
        }
        thinking_log_path = slm_thinking_log_path(
            ctx, task_id, attempt_number, attempt_seed
        )
        attempt["thinking_log"] = (
            str(thinking_log_path) if thinking_log_path is not None else None
        )
        attempts.append(attempt)
        ctx.event(
            "slm_request_started",
            task_id=task_id,
            attempt_number=attempt_number,
            seed=attempt_seed,
            thinking_log=thinking_log_path,
        )
        try:
            result = call_slm(
                ctx,
                prompt,
                seed_override=attempt_seed,
                thinking_log_path=thinking_log_path,
            )
            attempt.update(
                {
                    "status": "OK",
                    "slm": slm_report(ctx, result, prompt),
                    "thinking_log_file": (
                        file_info(thinking_log_path)
                        if thinking_log_path is not None
                        else None
                    ),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - started,
                }
            )
            return result, attempts
        except (SLMTimeoutError, SLMRepetitionError) as exc:
            repetition_detected = isinstance(exc, SLMRepetitionError)
            attempt.update(
                {
                    "status": "REPETITION" if repetition_detected else "TIMEOUT",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "thinking_log_file": (
                        file_info(thinking_log_path)
                        if thinking_log_path is not None
                        else None
                    ),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - started,
                }
            )
            restart_reason = "thinking_repetition" if repetition_detected else "timeout"
            ctx.event(
                "slm_attempt_aborted",
                reason=restart_reason,
                task_id=task_id,
                attempt_number=attempt_number,
                seed=attempt_seed,
                error_type=type(exc).__name__,
                error_message=str(exc),
                thinking_log=thinking_log_path,
                will_restart=attempt_number <= max_restart,
            )
            if attempt_number > max_restart:
                setattr(exc, "slm_attempts", attempts)
                raise
            restart_result = restart_slm_service(ctx, restart_reason, task_id)
            attempt["model_restart"] = (
                command_report(ctx, restart_result) if restart_result else None
            )
        except Exception as exc:
            attempt.update(
                {
                    "status": "FAILED_SLM",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "thinking_log_file": (
                        file_info(thinking_log_path)
                        if thinking_log_path is not None
                        else None
                    ),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - started,
                }
            )
            setattr(exc, "slm_attempts", attempts)
            raise
    raise AssertionError("unreachable")


def build_format_repair_prompt(
    ctx: RunContext,
    context_description: str,
    format_prompt_path: Path,
    malformed_output: str,
    python_error: str,
    model_description: str = "",
    model_generation_rules: str = "",
    general_prompt_path: Path | None = None,
    format_prompt_replacements: Mapping[str, str] | None = None,
) -> str:
    general_path = general_prompt_path or resolve_path(
        ctx.project_root, require(ctx.config, "format_repair.general_prompt_path")
    )
    prompt_replacements = format_prompt_replacements or {}
    general_prompt = read_text(general_path)
    format_prompt = read_text(format_prompt_path)
    structured_placeholder = "<ScenarioTexts>"
    if structured_placeholder in general_prompt:
        replacements = {
            "<ScenarioTexts>": context_description.strip(),
            "<OriginalIntermodelGenerationRules>": model_generation_rules.strip(),
            "<RequiredOutputFormat>": format_prompt.strip(),
            "<PythonError>": python_error.strip(),
            "<OutputToRepair>": malformed_output.strip(),
            **prompt_replacements,
        }
        result = general_prompt
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        unresolved = sorted(
            placeholder for placeholder in replacements if placeholder in result
        )
        if unresolved:
            raise PipelineError(
                "Unresolved repair prompt placeholders: " + ", ".join(unresolved)
            )
        return result.rstrip() + "\n"
    for placeholder, value in prompt_replacements.items():
        general_prompt = general_prompt.replace(placeholder, value)
    for placeholder, value in prompt_replacements.items():
        format_prompt = format_prompt.replace(placeholder, value)
    unresolved = []
    for placeholder in prompt_replacements:
        if placeholder in general_prompt or placeholder in format_prompt:
            unresolved.append(placeholder)
    required_placeholder = "<SourceIntermodelConnectionRules>"
    if (
        required_placeholder in general_prompt
        or required_placeholder in format_prompt
    ):
        unresolved.append(required_placeholder)
    if unresolved:
        raise PipelineError(
            "Unresolved repair prompt placeholders: "
            + ", ".join(sorted(set(unresolved)))
        )
    sections = [
        general_prompt.rstrip(),
        "## Scenario texts\n\n" + context_description.strip(),
    ]
    if model_description.strip():
        sections.append("## Model explanation\n\n" + model_description.strip())
    if model_generation_rules.strip():
        sections.append(
            "## Original model generation rules\n\n"
            + model_generation_rules.strip()
        )
    sections.extend(
        [
            "## Required output format\n\n" + format_prompt.strip(),
            "## Python error\n\n" + python_error.strip(),
            "## Output to repair\n\n" + malformed_output.strip(),
        ]
    )
    return "\n\n".join(sections) + "\n"


def concise_command_error(result: ExternalCommandResult, fallback: str) -> str:
    """Extract the final exception message without sending a traceback to repair."""
    output = "\n".join(part.strip() for part in (result.stderr, result.stdout) if part.strip())
    if not output:
        return fallback
    exception_matches = list(re.finditer(
        r"(?m)^(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception):[ \t]*",
        output,
    ))
    if exception_matches:
        return output[exception_matches[-1].end():].strip()
    error_matches = list(re.finditer(r"(?mi)^ERROR:[ \t]*", output))
    if error_matches:
        return output[error_matches[-1].end():].strip()
    # Non-Python tools sometimes emit only one useful diagnostic line.
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else fallback


def model_generation_rules(prompt_path: Path) -> str:
    """Return the reusable model rules without the original generation task suffix."""
    prompt = read_text(prompt_path).rstrip()
    return re.split(r"(?m)^Task:\s*$", prompt, maxsplit=1)[0].rstrip()


def build_model_repair_context(
    scenario: ScenarioRun,
    model: ModelDefinition,
    source_file: Path,
) -> str:
    return read_text(source_file).strip()


def build_intermodel_pair_repair_context(
    source: ModelArtifact,
    target: ModelArtifact,
) -> str:
    texts = [read_text(source.source_scenario_file).strip()]
    target_text = read_text(target.source_scenario_file).strip()
    if target_text not in texts:
        texts.append(target_text)
    return "\n\n---\n\n".join(texts)


def build_intermodel_pair_model_description(
    source: ModelArtifact,
    target: ModelArtifact,
    model_by_id: Mapping[str, ModelDefinition],
) -> str:
    source_type = model_by_id[source.model_id].display_name
    target_type = model_by_id[target.model_id].display_name
    descriptions = [
        "The output describes relationships from elements of "
        f"the {source_type} to elements of the {target_type}.",
        read_text(model_by_id[source.model_id].description_path).strip(),
    ]
    target_description = read_text(
        model_by_id[target.model_id].description_path
    ).strip()
    if target_description not in descriptions:
        descriptions.append(target_description)
    descriptions.extend(
        [
            "## Existing source model\n\n"
            f"The following is the source {source_type}:\n\n"
            + read_text(source.slm_file).strip(),
            "## Existing target model\n\n"
            f"The following is the target {target_type}:\n\n"
            + read_text(target.slm_file).strip(),
        ]
    )
    return "\n\n".join(descriptions)


def intermodel_generation_rules(rendered_prompt: str) -> str:
    marker = "## Task"
    index = rendered_prompt.find(marker)
    return rendered_prompt[index:].strip() if index >= 0 else rendered_prompt.strip()


def repair_format(
    ctx: RunContext,
    input_path: Path,
    format_prompt_path: Path,
    context_description: str,
    python_error: str,
    repair_directory: Path,
    attempt_number: int,
    task_id: str,
    model_description: str = "",
    model_generation_rules_text: str = "",
    general_prompt_path: Path | None = None,
    format_prompt_replacements: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Repair formatting first, otherwise validation content; preserve the input."""
    repair_directory.mkdir(parents=True, exist_ok=True)
    original = read_text(input_path)
    original_path = repair_directory / "original_output.txt"
    if not original_path.exists():
        write_text_atomic(original_path, original)
    error_path = repair_directory / f"attempt_{attempt_number}_python_error.txt"
    write_text_atomic(error_path, python_error)
    prompt = build_format_repair_prompt(
        ctx,
        context_description,
        format_prompt_path,
        original,
        python_error,
        model_description,
        model_generation_rules_text,
        general_prompt_path,
        format_prompt_replacements,
    )
    prompt_path = repair_directory / f"attempt_{attempt_number}_prompt.txt"
    write_text_atomic(prompt_path, prompt)
    result, slm_attempts = call_slm_with_restart(
        ctx, prompt, f"format_repair:{task_id}:attempt:{attempt_number}"
    )
    candidate = clean_slm_answer(result.answer).strip()
    candidate_path = repair_directory / f"attempt_{attempt_number}_repaired.txt"
    write_text_atomic(candidate_path, candidate + ("\n" if candidate else ""))
    report = {
        "attempt_number": attempt_number,
        "status": "GENERATED",
        "input": file_info(input_path),
        "preserved_original": file_info(original_path),
        "python_error": file_info(error_path),
        "prompt": file_info(prompt_path),
        "output": file_info(candidate_path),
        "slm_attempts": slm_attempts,
    }
    return candidate, report


# ---------------------------------------------------------------------------
# Scenario and artifact discovery
# ---------------------------------------------------------------------------


def resolve_selected_scenario_directory(ctx: RunContext) -> Path:
    """Resolve exactly one scenario subdirectory configured for this run."""
    scenarios_root = resolve_path(
        ctx.project_root,
        require(ctx.config, "paths.scenarios_root"),
    )
    if not scenarios_root.is_dir():
        raise PipelineError(f"Scenario root does not exist: {scenarios_root}")

    selected_raw = str(
        require(ctx.config, "execution.selected_scenario")
    ).strip()
    selected_path = Path(selected_raw)

    if selected_path.is_absolute():
        raise PipelineError(
            "execution.selected_scenario must be relative to paths.scenarios_root."
        )

    scenario_directory = (scenarios_root / selected_path).resolve()

    try:
        relative = scenario_directory.relative_to(scenarios_root.resolve())
    except ValueError as exc:
        raise PipelineError(
            "execution.selected_scenario must point to a directory inside "
            f"the scenario root: {scenarios_root}"
        ) from exc

    if relative == Path("."):
        raise PipelineError(
            "execution.selected_scenario must select one scenario subdirectory, "
            "not the complete scenarios root."
        )

    if not scenario_directory.is_dir():
        raise PipelineError(
            "The selected scenario directory does not exist: "
            f"{scenario_directory}"
        )

    return scenario_directory


def resolve_selected_intermodel_only_job(
    ctx: RunContext,
) -> Mapping[str, Any]:
    """Return the one enabled intermodel-only job for the selected scenario."""
    selected_scenario = str(
        require(ctx.config, "execution.selected_scenario")
    ).strip()

    jobs = dotted_get(ctx.config, "intermodel_only.jobs", [])
    if not isinstance(jobs, list) or not jobs:
        raise PipelineError(
            "execution.mode='intermodel_only' requires intermodel_only.jobs."
        )

    matches: list[Mapping[str, Any]] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise PipelineError(
                f"intermodel_only.jobs[{index}] must be an object."
            )
        if not job.get("enabled", True):
            continue

        job_scenario_name = str(job.get("scenario_name", "")).strip()
        job_scenario_directory = str(job.get("scenario_directory", "")).strip()
        job_directory_name = (
            Path(job_scenario_directory).name if job_scenario_directory else ""
        )

        if selected_scenario in {job_scenario_name, job_directory_name}:
            matches.append(job)

    if not matches:
        raise PipelineError(
            "No enabled intermodel_only.jobs entry matches "
            f"execution.selected_scenario={selected_scenario!r}."
        )
    if len(matches) > 1:
        raise PipelineError(
            "More than one enabled intermodel_only.jobs entry matches "
            f"execution.selected_scenario={selected_scenario!r}. "
            "Exactly one job is allowed per run."
        )

    return matches[0]


def create_scenario_run(ctx: RunContext, scenario_directory: Path) -> ScenarioRun:
    output_root = resolve_path(ctx.project_root, require(ctx.config, "paths.output_root"))
    run_template = str(require(ctx.config, "paths.scenario_run_directory_template"))
    run_name = render_string(
        run_template,
        {
            "scenario_name": scenario_directory.name,
            "timestamp": ctx.timestamp,
            "run_id": ctx.run_id,
        },
    )
    run_directory = output_root / run_name
    path_values = {
        "scenario_name": scenario_directory.name,
        "timestamp": ctx.timestamp,
        "run_id": ctx.run_id,
    }

    def relative_output(config_key: str) -> Path:
        return run_directory / render_string(
            str(require(ctx.config, config_key)), path_values
        )

    return ScenarioRun(
        scenario_name=scenario_directory.name,
        scenario_directory=scenario_directory,
        run_directory=run_directory,
        model_slm_directory=relative_output("paths.model_slm_directory"),
        model_adl_directory=relative_output("paths.model_adl_directory"),
        artifact_manifest=relative_output("paths.artifact_manifest_file"),
        merged_adl=relative_output("paths.merged_adl_file"),
        intermodel_slm_directory=relative_output("paths.intermodel_slm_directory"),
        intermodel_aggregate_slm=txt_output_path(
            relative_output("paths.intermodel_aggregate_slm_file")
        ),
        final_adl=relative_output("paths.final_adl_file"),
    )


def scenario_files(
    ctx: RunContext,
    scenario: ScenarioRun,
    model: ModelDefinition,
) -> list[Path]:
    model_directory = scenario.scenario_directory / model.scenario_folder
    if not model_directory.is_dir():
        ctx.event(
            "model_folder_missing",
            scenario=scenario.scenario_name,
            model_id=model.model_id,
            expected_directory=model_directory,
            interpretation="This scenario has no model of this type.",
        )
        return []

    extensions = {
        normalized_extension(str(item)).casefold()
        for item in dotted_get(ctx.config, "files.scenario_extensions", [".txt"])
    }
    return sorted(
        path
        for path in model_directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in extensions
    )


def artifact_paths_and_name(
    ctx: RunContext,
    scenario: ScenarioRun,
    model: ModelDefinition,
    source_file: Path,
) -> tuple[Path, Path, Path, str]:
    source_model_dir = scenario.scenario_directory / model.scenario_folder
    relative = source_file.relative_to(source_model_dir)
    relative_parent = relative.parent
    relative_id = safe_name(relative.with_suffix("").as_posix())

    values = {
        "scenario_name": safe_name(scenario.scenario_name),
        "model_id": safe_name(model.model_id),
        "model_folder": safe_name(model.scenario_folder),
        "source_stem": safe_name(source_file.stem),
        "source_name": safe_name(source_file.name),
        "source_relative_id": relative_id,
    }

    slm_filename = render_string(model.slm_filename_template, values)
    adl_filename = render_string(model.adl_filename_template, values)
    adl_model_name = render_string(model.adl_model_name_template, values)

    slm_file = txt_output_path(
        scenario.model_slm_directory
        / model.scenario_folder
        / relative_parent
        / slm_filename
    )
    adl_file = scenario.model_adl_directory / model.scenario_folder / relative_parent / adl_filename
    return relative, slm_file, adl_file, adl_model_name


def build_model_prompt(prompt_text: str, scenario_text: str, source_file: Path) -> str:
    return (
        prompt_text.rstrip()
        + "\n\n---\n\n"
        + f"Scenario file: {source_file.name}\n\n"
        + scenario_text.strip()
        + "\n"
    )


# ---------------------------------------------------------------------------
# Model generation and SLM-to-ADL conversion
# ---------------------------------------------------------------------------


def run_model_task(
    ctx: RunContext,
    scenario: ScenarioRun,
    model: ModelDefinition,
    source_file: Path,
) -> tuple[ModelArtifact, dict[str, Any]]:
    relative, slm_file, adl_file, adl_model_name = artifact_paths_and_name(
        ctx, scenario, model, source_file
    )
    task_id = f"model:{scenario.scenario_name}:{model.model_id}:{relative.as_posix()}"
    artifact = ModelArtifact(
        scenario_name=scenario.scenario_name,
        model_id=model.model_id,
        model_folder=model.scenario_folder,
        model_display_name=model.display_name,
        model_type=model.model_type,
        source_scenario_file=source_file,
        source_relative_path=relative,
        source_stem=source_file.stem,
        adl_model_name=adl_model_name,
        slm_file=slm_file,
        adl_file=adl_file,
        status="PENDING",
        task_id=task_id,
    )

    task_started_perf = time.perf_counter()
    task_report: dict[str, Any] = {
        "task_id": task_id,
        "scenario": scenario.scenario_name,
        "model_id": model.model_id,
        "model_folder": model.scenario_folder,
        "source_file": file_info(source_file),
        "source_relative_path": relative.as_posix(),
        "slm_file": str(slm_file),
        "adl_file": str(adl_file),
        "adl_model_name": adl_model_name,
        "started_at": iso_now(),
        "attempts": [],
        "format_repairs": [],
    }

    overwrite = bool(dotted_get(ctx.config, "execution.overwrite", False))
    if not overwrite and slm_file.is_file() and adl_file.is_file():
        artifact.status = "SKIPPED_EXISTING"
        task_report.update(
            {
                "status": artifact.status,
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - task_started_perf,
                "existing_slm": file_info(slm_file),
                "existing_adl": file_info(adl_file),
            }
        )
        return artifact, task_report

    prompt = build_model_prompt(
        read_text(model.prompt_path),
        read_text(source_file),
        source_file,
    )

    attempt_number = 0
    while True:
        attempt_number += 1
        attempt_started_perf = time.perf_counter()
        attempt: dict[str, Any] = {
            "attempt_number": attempt_number,
            "started_at": iso_now(),
            "failure_reason": None,
        }
        task_report["attempts"].append(attempt)
        ctx.event("model_attempt_started", task_id=task_id, attempt=attempt_number)

        try:
            slm_result, slm_attempts = call_slm_with_restart(ctx, prompt, task_id)
            attempt["slm_attempts"] = slm_attempts
            attempt["slm"] = slm_report(ctx, slm_result, prompt)
            write_text_atomic(slm_file, slm_result.answer)
            attempt["slm_output"] = file_info(slm_file)
        except SLMTimeoutError as exc:
            attempt["slm_attempts"] = getattr(exc, "slm_attempts", [])
            attempt.update(
                {
                    "failure_reason": "timeout",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - attempt_started_perf,
                }
            )
            artifact.status = "FAILED_TIMEOUT"
            task_report.update(
                {
                    "status": artifact.status,
                    "final_error": str(exc),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - task_started_perf,
                }
            )
            return artifact, task_report
        except Exception as exc:
            attempt["slm_attempts"] = getattr(exc, "slm_attempts", [])
            artifact.status = "FAILED_SLM"
            attempt.update(
                {
                    "failure_reason": "slm_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - attempt_started_perf,
                }
            )
            task_report.update(
                {
                    "status": artifact.status,
                    "final_error": str(exc),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - task_started_perf,
                }
            )
            return artifact, task_report

        python_executable = str(dotted_get(ctx.config, "tools.python_executable", "") or sys.executable)
        command_values = {
            "python": python_executable,
            "project_root": ctx.project_root,
            "converter_path": model.converter_path,
            "input_slm": slm_file,
            "output_adl": adl_file,
            "adl_model_name": adl_model_name,
            "model_id": model.model_id,
            "scenario_name": scenario.scenario_name,
            "scenario_file": source_file,
        }
        adl_file.parent.mkdir(parents=True, exist_ok=True)
        if adl_file.exists():
            adl_file.unlink()

        converter_result = run_external_command(
            model.converter_command,
            command_values,
            model.converter_cwd,
            model.converter_timeout_seconds,
        )
        attempt["converter"] = command_report(ctx, converter_result)

        if converter_result.ok and adl_file.is_file():
            artifact.status = "OK"
            attempt.update(
                {
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - attempt_started_perf,
                    "adl_output": file_info(adl_file),
                }
            )
            task_report.update(
                {
                    "status": artifact.status,
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - task_started_perf,
                    "slm_output": file_info(slm_file),
                    "adl_output": file_info(adl_file),
                }
            )
            return artifact, task_report

        error_message = (
            "The SLM-to-ADL converter failed."
            if not converter_result.ok
            else "The converter returned success but did not create the ADL output."
        )
        attempt.update(
            {
                "failure_reason": "python_error",
                "error_message": error_message,
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - attempt_started_perf,
            }
        )

        repair_enabled = bool(dotted_get(ctx.config, "format_repair.enabled", False))
        max_repairs = int(dotted_get(ctx.config, "format_repair.max_attempts", 0))
        repair_directory = slm_file.parent / "format_repair" / safe_name(slm_file.stem)
        for repair_number in range(1, max_repairs + 1 if repair_enabled else 1):
            python_error = concise_command_error(converter_result, error_message)
            try:
                repaired, repair_report = repair_format(
                    ctx,
                    slm_file,
                    model.format_prompt_path,
                    build_model_repair_context(scenario, model, source_file),
                    python_error,
                    repair_directory,
                    repair_number,
                    task_id,
                    read_text(model.description_path),
                    model_generation_rules(model.prompt_path),
                )
            except Exception as exc:
                task_report["format_repairs"].append(
                    {
                        "attempt_number": repair_number,
                        "status": "FAILED_SLM",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "slm_attempts": getattr(exc, "slm_attempts", []),
                    }
                )
                break
            write_text_atomic(slm_file, repaired + ("\n" if repaired else ""))
            if adl_file.exists():
                adl_file.unlink()
            converter_result = run_external_command(
                model.converter_command,
                command_values,
                model.converter_cwd,
                model.converter_timeout_seconds,
            )
            repair_report["converter"] = command_report(ctx, converter_result)
            task_report["format_repairs"].append(repair_report)
            if converter_result.ok and adl_file.is_file():
                repair_report["status"] = "OK"
                artifact.status = "OK_AFTER_FORMAT_REPAIR"
                task_report.update(
                    {
                        "status": artifact.status,
                        "finished_at": iso_now(),
                        "duration_seconds": time.perf_counter() - task_started_perf,
                        "slm_output": file_info(slm_file),
                        "adl_output": file_info(adl_file),
                    }
                )
                return artifact, task_report
            repair_report["status"] = "FAILED_PYTHON"
            error_message = (
                "The SLM-to-ADL converter failed after format repair."
                if not converter_result.ok
                else "The converter created no ADL output after format repair."
            )

        artifact.status = "FAILED_PYTHON"
        task_report.update(
            {
                "status": artifact.status,
                "final_error": error_message,
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - task_started_perf,
            }
        )
        return artifact, task_report


def write_artifact_manifest(scenario: ScenarioRun, artifacts: Sequence[ModelArtifact]) -> None:
    manifest = {
        "format": "4em-pipeline-artifact-manifest-v1",
        "scenario_name": scenario.scenario_name,
        "scenario_directory": str(scenario.scenario_directory),
        "run_directory": str(scenario.run_directory),
        "created_at": iso_now(),
        "artifacts": [
            {
                **to_json_safe(asdict(artifact)),
                "slm_file_info": file_info(artifact.slm_file),
                "adl_file_info": file_info(artifact.adl_file),
                "source_file_info": file_info(artifact.source_scenario_file),
            }
            for artifact in artifacts
        ],
    }
    write_json_atomic(scenario.artifact_manifest, manifest)


# ---------------------------------------------------------------------------
# ADL merger
# ---------------------------------------------------------------------------


def run_adl_merger(
    ctx: RunContext,
    scenario: ScenarioRun,
    artifacts: Sequence[ModelArtifact],
) -> dict[str, Any]:
    started_perf = time.perf_counter()
    report: dict[str, Any] = {
        "started_at": iso_now(),
        "output_file": str(scenario.merged_adl),
    }

    input_files = sorted(
        {
            artifact.adl_file.resolve()
            for artifact in artifacts
            if artifact.status in {"OK", "OK_AFTER_FORMAT_REPAIR", "SKIPPED_EXISTING", "EXISTING"}
            and artifact.adl_file.is_file()
        }
    )
    report["input_files"] = [file_info(path) for path in input_files]

    if not input_files:
        report.update(
            {
                "status": "SKIPPED_NO_ADL_INPUT",
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - started_perf,
            }
        )
        return report

    overwrite = bool(dotted_get(ctx.config, "execution.overwrite", False))
    if scenario.merged_adl.is_file() and not overwrite:
        report.update(
            {
                "status": "SKIPPED_EXISTING",
                "output": file_info(scenario.merged_adl),
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - started_perf,
            }
        )
        return report

    merger = require(ctx.config, "tools.adl_merger")
    if not isinstance(merger, dict):
        raise PipelineError("tools.adl_merger must be an object.")
    command = require(merger, "command")
    if not isinstance(command, list):
        raise PipelineError("tools.adl_merger.command must be an array.")

    merger_path = resolve_path(ctx.project_root, require(merger, "path"))
    python_executable = str(dotted_get(ctx.config, "tools.python_executable", "") or sys.executable)
    values = {
        "python": python_executable,
        "project_root": ctx.project_root,
        "merger_path": merger_path,
        "input_directory": scenario.model_adl_directory,
        "output_adl": scenario.merged_adl,
        "scenario_name": scenario.scenario_name,
    }
    scenario.merged_adl.parent.mkdir(parents=True, exist_ok=True)
    if scenario.merged_adl.exists():
        scenario.merged_adl.unlink()

    result = run_external_command(
        command,
        values,
        resolve_path(ctx.project_root, merger.get("cwd", ".")),
        float(merger["timeout_seconds"])
        if merger.get("timeout_seconds") is not None
        else float(dotted_get(ctx.config, "tools.default_python_timeout_seconds", 600)),
        list_values={"input_files": [str(path) for path in input_files]},
    )
    report["command"] = command_report(ctx, result)

    if result.ok and scenario.merged_adl.is_file():
        report["status"] = "OK"
        report["output"] = file_info(scenario.merged_adl)
    else:
        report["status"] = "FAILED"
        report["error"] = (
            "The ADL merger command failed."
            if not result.ok
            else "The ADL merger returned success but created no output file."
        )

    report.update(
        {
            "finished_at": iso_now(),
            "duration_seconds": time.perf_counter() - started_perf,
        }
    )
    return report


# ---------------------------------------------------------------------------
# Intermodel generation and integration
# ---------------------------------------------------------------------------


def render_intermodel_prompt(
    ctx: RunContext,
    template: str,
    source: ModelArtifact,
    target: ModelArtifact,
    model_by_id: Mapping[str, ModelDefinition],
) -> str:
    source_definition = model_by_id[source.model_id]
    target_definition = model_by_id[target.model_id]
    replacements = {
        "<Model1Descr>": read_text(source_definition.description_path),
        "<Model2Descr>": read_text(target_definition.description_path),
        "<Model1Elements>": read_text(source.slm_file),
        "<Model2Elements>": read_text(target.slm_file),
        "<Model1Description>": read_text(source.source_scenario_file),
        "<Model2Description>": read_text(target.source_scenario_file),
        "<SourceIntermodelConnectionRules>": read_text(
            ctx.project_root
            / "prompts"
            / "inter_model_generation"
            / "source_connection_rules"
            / f"{source.model_id}.txt"
        ).strip(),
        "<TargetIntermodelReferenceRules>": read_text(
            ctx.project_root
            / "prompts"
            / "inter_model_generation"
            / "target_reference_rules"
            / f"{source.model_id}__{target.model_id}.txt"
        ).strip(),
    }
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)

    unresolved = [
        placeholder
        for placeholder in replacements
        if placeholder in result
    ]
    if unresolved:
        raise PipelineError(
            "Unresolved intermodel prompt placeholders: " + ", ".join(unresolved)
        )
    return result


def intermodel_pairs(ctx: RunContext) -> list[tuple[str, str]]:
    raw_pairs = dotted_get(ctx.config, "intermodel.pairs", [])
    if not isinstance(raw_pairs, list):
        raise PipelineError("intermodel.pairs must be an array.")
    result: list[tuple[str, str]] = []
    for index, raw in enumerate(raw_pairs):
        if isinstance(raw, list) and len(raw) == 2:
            result.append((str(raw[0]), str(raw[1])))
        elif isinstance(raw, dict):
            if not raw.get("enabled", True):
                continue
            result.append((str(require(raw, "source")), str(require(raw, "target"))))
        else:
            raise PipelineError(
                f"intermodel.pairs[{index}] must be [source, target] or an object."
            )
    return result


def intermodel_tasks(
    ctx: RunContext,
    artifacts: Sequence[ModelArtifact],
) -> list[tuple[ModelArtifact, ModelArtifact]]:
    excluded_names = dotted_get(
        ctx.config, "intermodel.excluded_adl_model_name_substrings", []
    )
    if not isinstance(excluded_names, list) or not all(
        isinstance(value, str) for value in excluded_names
    ):
        raise PipelineError(
            "intermodel.excluded_adl_model_name_substrings must be an array of strings."
        )
    excluded = tuple(value.casefold() for value in excluded_names if value)
    available = [
        artifact
        for artifact in artifacts
        if artifact.slm_file.is_file()
        and artifact.status in {"OK", "OK_AFTER_FORMAT_REPAIR", "SKIPPED_EXISTING", "EXISTING"}
        and not any(value in artifact.adl_model_name.casefold() for value in excluded)
    ]
    by_model: dict[str, list[ModelArtifact]] = {}
    for artifact in available:
        by_model.setdefault(artifact.model_id, []).append(artifact)

    tasks: list[tuple[ModelArtifact, ModelArtifact]] = []
    for source_id, target_id in intermodel_pairs(ctx):
        for source in by_model.get(source_id, []):
            for target in by_model.get(target_id, []):
                tasks.append((source, target))
    return tasks


def intermodel_output_file(
    ctx: RunContext,
    scenario: ScenarioRun,
    source: ModelArtifact,
    target: ModelArtifact,
) -> Path:
    template = str(
        dotted_get(
            ctx.config,
            "intermodel.output_filename_template",
            "intermodel_{source_adl_model_name}__{target_adl_model_name}.txt",
        )
    )
    filename = render_string(
        template,
        {
            "scenario_name": safe_name(scenario.scenario_name),
            "source_model_id": safe_name(source.model_id),
            "target_model_id": safe_name(target.model_id),
            "source_adl_model_name": safe_name(source.adl_model_name),
            "target_adl_model_name": safe_name(target.adl_model_name),
            "source_stem": safe_name(source.source_stem),
            "target_stem": safe_name(target.source_stem),
        },
    )
    return txt_output_path(scenario.intermodel_slm_directory / filename)


def run_intermodel_task(
    ctx: RunContext,
    scenario: ScenarioRun,
    source: ModelArtifact,
    target: ModelArtifact,
    prompt_template: str,
    model_by_id: Mapping[str, ModelDefinition],
    input_adl: Path,
    force_regenerate: bool,
) -> tuple[Path | None, dict[str, Any]]:
    output_file = intermodel_output_file(ctx, scenario, source, target)
    task_id = (
        f"intermodel:{scenario.scenario_name}:{source.adl_model_name}"
        f"->{target.adl_model_name}"
    )
    started_perf = time.perf_counter()
    report: dict[str, Any] = {
        "task_id": task_id,
        "source_model_id": source.model_id,
        "target_model_id": target.model_id,
        "source_adl_model_name": source.adl_model_name,
        "target_adl_model_name": target.adl_model_name,
        "source_slm": file_info(source.slm_file),
        "target_slm": file_info(target.slm_file),
        "output_file": str(output_file),
        "started_at": iso_now(),
        "attempts": [],
        "format_repairs": [],
    }

    overwrite = bool(dotted_get(ctx.config, "execution.overwrite", False))
    if output_file.is_file() and not overwrite and not force_regenerate:
        report.update(
            {
                "status": "SKIPPED_EXISTING",
                "output": file_info(output_file),
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - started_perf,
            }
        )
        return output_file, report

    prompt = render_intermodel_prompt(
        ctx, prompt_template, source, target, model_by_id
    )
    attempt_number = 0
    while True:
        attempt_number += 1
        attempt_started_perf = time.perf_counter()
        attempt: dict[str, Any] = {
            "attempt_number": attempt_number,
            "started_at": iso_now(),
            "failure_reason": None,
        }
        report["attempts"].append(attempt)
        try:
            result, slm_attempts = call_slm_with_restart(ctx, prompt, task_id)
            attempt["slm_attempts"] = slm_attempts
            attempt["slm"] = slm_report(ctx, result, prompt)
            write_text_atomic(output_file, result.answer)
        except SLMTimeoutError as exc:
            attempt["slm_attempts"] = getattr(exc, "slm_attempts", [])
            attempt.update(
                {
                    "failure_reason": "timeout",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - attempt_started_perf,
                }
            )
            report.update(
                {
                    "status": "FAILED_TIMEOUT",
                    "final_error": str(exc),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - started_perf,
                }
            )
            return None, report
        except Exception as exc:
            attempt["slm_attempts"] = getattr(exc, "slm_attempts", [])
            attempt.update(
                {
                    "failure_reason": "slm_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - attempt_started_perf,
                }
            )
            report.update(
                {
                    "status": "FAILED_SLM",
                    "final_error": str(exc),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - started_perf,
                }
            )
            return None, report

        validation_ok, validation_report, validation_error = validate_intermodel_output(
            ctx, input_adl, output_file, source, target
        )
        attempt["validation"] = validation_report
        attempt.update(
            {
                "output": file_info(output_file),
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - attempt_started_perf,
            }
        )
        if validation_ok:
            report.update(
                {
                    "status": "OK",
                    "output": file_info(output_file),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - started_perf,
                }
            )
            return output_file, report

        repair_enabled = bool(dotted_get(ctx.config, "format_repair.enabled", False))
        max_repairs = int(dotted_get(ctx.config, "format_repair.max_attempts", 0))
        repair_directory = (
            output_file.parent / "format_repair" / safe_name(output_file.stem)
        )
        format_prompt = resolve_path(
            ctx.project_root, require(ctx.config, "intermodel.format_prompt_path")
        )
        for repair_number in range(1, max_repairs + 1 if repair_enabled else 1):
            try:
                repaired, repair_report = repair_format(
                    ctx,
                    output_file,
                    format_prompt,
                    build_intermodel_pair_repair_context(source, target),
                    validation_error,
                    repair_directory,
                    repair_number,
                    task_id,
                    "",
                    intermodel_generation_rules(prompt),
                    resolve_path(
                        ctx.project_root,
                        require(ctx.config, "intermodel.repair_prompt_path"),
                    ),
                    {
                        "<SourceModelDescription>": read_text(
                            model_by_id[source.model_id].description_path
                        ).strip(),
                        "<TargetModelDescription>": read_text(
                            model_by_id[target.model_id].description_path
                        ).strip(),
                        "<ExistingSourceModel>": read_text(source.slm_file).strip(),
                        "<ExistingTargetModel>": read_text(target.slm_file).strip(),
                    },
                )
            except Exception as exc:
                report["format_repairs"].append(
                    {
                        "attempt_number": repair_number,
                        "status": "FAILED_SLM",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "slm_attempts": getattr(exc, "slm_attempts", []),
                    }
                )
                break
            write_text_atomic(output_file, repaired + ("\n" if repaired else ""))
            validation_ok, validation_report, validation_error = (
                validate_intermodel_output(
                    ctx, input_adl, output_file, source, target
                )
            )
            repair_report["validation"] = validation_report
            repair_report["status"] = (
                "OK" if validation_ok else "FAILED_PYTHON"
            )
            report["format_repairs"].append(repair_report)
            if validation_ok:
                report.update(
                    {
                        "status": "OK_AFTER_FORMAT_REPAIR",
                        "output": file_info(output_file),
                        "finished_at": iso_now(),
                        "duration_seconds": time.perf_counter() - started_perf,
                    }
                )
                return output_file, report

        report.update(
            {
                "status": "FAILED_INTERMODEL_VALIDATION",
                "final_error": validation_error,
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - started_perf,
            }
        )
        return None, report


def validate_intermodel_output(
    ctx: RunContext,
    input_adl: Path,
    output_file: Path,
    source: ModelArtifact,
    target: ModelArtifact,
) -> tuple[bool, dict[str, Any], str]:
    body = clean_slm_answer(read_text(output_file)).strip()
    no_marker = str(
        dotted_get(
            ctx.config,
            "intermodel.no_relationship_marker",
            "NO_INTERMODEL_RELATIONSHIPS",
        )
    )
    if body == no_marker:
        return True, {"status": "OK_NO_RELATIONSHIPS"}, ""
    if not body:
        error = (
            "The intermodel output is empty. Return valid relationship lines or "
            f"exactly {no_marker}."
        )
        return False, {"status": "FAILED", "error_message": error}, error

    integrator = require(ctx.config, "tools.intermodel_integrator")
    if not isinstance(integrator, dict):
        raise PipelineError("tools.intermodel_integrator must be an object.")
    command = require(integrator, "command")
    if not isinstance(command, list):
        raise PipelineError("tools.intermodel_integrator.command must be an array.")
    integrator_path = resolve_path(ctx.project_root, require(integrator, "path"))
    python_executable = str(
        dotted_get(ctx.config, "tools.python_executable", "") or sys.executable
    )
    wrapped = (
        f"[{source.adl_model_name} -> {target.adl_model_name}]\n{body}\n"
    )
    with tempfile.TemporaryDirectory(
        prefix="intermodel_validation_", dir=str(output_file.parent)
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        relations_file = temporary_root / "relations.txt"
        validation_adl = temporary_root / "validation.adl"
        write_text_atomic(relations_file, wrapped)
        values = {
            "python": python_executable,
            "project_root": ctx.project_root,
            "integrator_path": integrator_path,
            "input_adl": input_adl,
            "relations_slm": relations_file,
            "output_adl": validation_adl,
            "scenario_name": source.scenario_name,
        }
        result = run_external_command(
            command,
            values,
            resolve_path(ctx.project_root, integrator.get("cwd", ".")),
            float(integrator["timeout_seconds"])
            if integrator.get("timeout_seconds") is not None
            else float(
                dotted_get(ctx.config, "tools.default_python_timeout_seconds", 600)
            ),
        )
        report = command_report(ctx, result)
        ok = result.ok and validation_adl.is_file()
        if ok:
            return True, report, ""
        error = concise_command_error(
            result,
            "The individual intermodel output could not be integrated.",
        )
        return False, report, error


def aggregate_intermodel_slm(
    ctx: RunContext,
    scenario: ScenarioRun,
    outputs: Sequence[tuple[ModelArtifact, ModelArtifact, Path]],
) -> None:
    no_marker = str(
        dotted_get(
            ctx.config,
            "intermodel.no_relationship_marker",
            "NO_INTERMODEL_RELATIONSHIPS",
        )
    )
    blocks: list[str] = []
    for source, target, path in outputs:
        body = clean_slm_answer(read_text(path)).strip()
        if not body or body == no_marker:
            continue
        blocks.append(f"[{source.adl_model_name} -> {target.adl_model_name}]\n{body}")
    content = "\n\n".join(blocks)
    if content:
        content += "\n"
    write_text_atomic(scenario.intermodel_aggregate_slm, content)


def run_intermodel_stage(
    ctx: RunContext,
    scenario: ScenarioRun,
    artifacts: Sequence[ModelArtifact],
    model_definitions: Sequence[ModelDefinition],
    input_adl: Path,
) -> dict[str, Any]:
    started_perf = time.perf_counter()
    stage: dict[str, Any] = {
        "started_at": iso_now(),
        "input_adl": file_info(input_adl),
        "final_adl": str(scenario.final_adl),
        "batches": [],
    }

    if not bool(dotted_get(ctx.config, "intermodel.enabled", True)):
        stage.update(
            {
                "status": "DISABLED",
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - started_perf,
            }
        )
        return stage

    if not input_adl.is_file():
        stage.update(
            {
                "status": "SKIPPED_NO_INPUT_ADL",
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - started_perf,
            }
        )
        return stage

    overwrite = bool(dotted_get(ctx.config, "execution.overwrite", False))
    if scenario.final_adl.is_file() and not overwrite:
        stage.update(
            {
                "status": "SKIPPED_EXISTING",
                "output": file_info(scenario.final_adl),
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - started_perf,
            }
        )
        return stage

    prompt_template_path = resolve_path(
        ctx.project_root, require(ctx.config, "prompts.intermodel_template")
    )
    prompt_template = read_text(prompt_template_path)
    model_by_id = {model.model_id: model for model in model_definitions}

    unknown_pair_models = {
        model_id
        for pair in intermodel_pairs(ctx)
        for model_id in pair
        if model_id not in model_by_id
    }
    if unknown_pair_models:
        raise PipelineError(
            "Intermodel pairs reference unknown model ids: "
            + ", ".join(sorted(unknown_pair_models))
        )

    force_regenerate = False
    batch_number = 0

    while True:
        batch_number += 1
        batch_started_perf = time.perf_counter()
        task_reports: list[dict[str, Any]] = []
        successful: list[tuple[ModelArtifact, ModelArtifact, Path]] = []

        for source, target in intermodel_tasks(ctx, artifacts):
            output, task_report = run_intermodel_task(
                ctx,
                scenario,
                source,
                target,
                prompt_template,
                model_by_id,
                input_adl,
                force_regenerate,
            )
            task_reports.append(task_report)
            if output is not None:
                successful.append((source, target, output))

        aggregate_intermodel_slm(ctx, scenario, successful)

        integrator = require(ctx.config, "tools.intermodel_integrator")
        if not isinstance(integrator, dict):
            raise PipelineError("tools.intermodel_integrator must be an object.")
        command = require(integrator, "command")
        if not isinstance(command, list):
            raise PipelineError("tools.intermodel_integrator.command must be an array.")

        integrator_path = resolve_path(ctx.project_root, require(integrator, "path"))
        python_executable = str(dotted_get(ctx.config, "tools.python_executable", "") or sys.executable)
        values = {
            "python": python_executable,
            "project_root": ctx.project_root,
            "integrator_path": integrator_path,
            "input_adl": input_adl,
            "relations_slm": scenario.intermodel_aggregate_slm,
            "output_adl": scenario.final_adl,
            "scenario_name": scenario.scenario_name,
        }
        scenario.final_adl.parent.mkdir(parents=True, exist_ok=True)
        if scenario.final_adl.exists():
            scenario.final_adl.unlink()

        integration_result = run_external_command(
            command,
            values,
            resolve_path(ctx.project_root, integrator.get("cwd", ".")),
            float(integrator["timeout_seconds"])
            if integrator.get("timeout_seconds") is not None
            else float(dotted_get(ctx.config, "tools.default_python_timeout_seconds", 600)),
        )

        batch = {
            "batch_number": batch_number,
            "started_at": iso_now(),
            "task_reports": task_reports,
            "successful_intermodel_files": [
                {
                    "source": source.adl_model_name,
                    "target": target.adl_model_name,
                    "file": file_info(path),
                }
                for source, target, path in successful
            ],
            "aggregate_slm": file_info(scenario.intermodel_aggregate_slm),
            "integration_command": command_report(ctx, integration_result),
            "format_repairs": [],
            "duration_seconds": time.perf_counter() - batch_started_perf,
        }
        stage["batches"].append(batch)

        if integration_result.ok and scenario.final_adl.is_file():
            failed_tasks = sum(
                1
                for task in task_reports
                if str(task.get("status", "")).startswith("FAILED")
            )
            batch["status"] = "OK"
            stage.update(
                {
                    "status": "COMPLETED_WITH_ERRORS" if failed_tasks else "OK",
                    "output": file_info(scenario.final_adl),
                    "finished_at": iso_now(),
                    "duration_seconds": time.perf_counter() - started_perf,
                    "summary": summarize_attempts(task_reports),
                }
            )
            return stage

        batch["status"] = "FAILED_PYTHON"
        batch["error"] = (
            "The intermodel integration script failed."
            if not integration_result.ok
            else "The integration script returned success but created no final ADL."
        )

        stage.update(
            {
                "status": "FAILED_PYTHON",
                "final_error": batch["error"],
                "finished_at": iso_now(),
                "duration_seconds": time.perf_counter() - started_perf,
                "summary": summarize_attempts(
                    [task for item in stage["batches"] for task in item["task_reports"]]
                ),
            }
        )
        return stage


# ---------------------------------------------------------------------------
# Intermodel-only artifact loading
# ---------------------------------------------------------------------------


def artifact_from_manifest_item(item: Mapping[str, Any]) -> ModelArtifact:
    return ModelArtifact(
        scenario_name=str(item["scenario_name"]),
        model_id=str(item["model_id"]),
        model_folder=str(item["model_folder"]),
        model_display_name=str(item["model_display_name"]),
        model_type=str(item["model_type"]),
        source_scenario_file=Path(item["source_scenario_file"]).expanduser().resolve(),
        source_relative_path=Path(item["source_relative_path"]),
        source_stem=str(item["source_stem"]),
        adl_model_name=str(item["adl_model_name"]),
        slm_file=Path(item["slm_file"]).expanduser().resolve(),
        adl_file=Path(item["adl_file"]).expanduser().resolve(),
        status="EXISTING" if Path(item["slm_file"]).is_file() else "MISSING_SLM",
        task_id=str(item.get("task_id", "manifest-artifact")),
    )


def load_artifact_manifest(path: Path) -> list[ModelArtifact]:
    try:
        data = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid artifact manifest JSON: {path}: {exc}") from exc
    raw_artifacts = data.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise PipelineError(f"Artifact manifest has no valid artifacts array: {path}")
    return [artifact_from_manifest_item(item) for item in raw_artifacts]


def create_intermodel_only_scenario(
    ctx: RunContext,
    job: Mapping[str, Any],
) -> tuple[ScenarioRun, list[ModelArtifact], Path]:
    scenario_name = str(require(job, "scenario_name"))
    scenario_directory = resolve_path(ctx.project_root, require(job, "scenario_directory"))
    run_directory = resolve_path(ctx.project_root, require(job, "output_run_directory"))
    path_values = {
        "scenario_name": scenario_name,
        "timestamp": ctx.timestamp,
        "run_id": ctx.run_id,
    }

    def output_path(job_key: str, global_key: str) -> Path:
        raw = job.get(job_key)
        rendered = render_string(
            str(raw if raw is not None else require(ctx.config, global_key)),
            path_values,
        )
        return (
            resolve_path(ctx.project_root, rendered)
            if raw is not None
            else run_directory / rendered
        )

    scenario = ScenarioRun(
        scenario_name=scenario_name,
        scenario_directory=scenario_directory,
        run_directory=run_directory,
        model_slm_directory=output_path("model_slm_directory", "paths.model_slm_directory"),
        model_adl_directory=output_path("model_adl_directory", "paths.model_adl_directory"),
        artifact_manifest=resolve_path(
            ctx.project_root, require(job, "artifact_manifest")
        ),
        merged_adl=resolve_path(ctx.project_root, require(job, "input_adl")),
        intermodel_slm_directory=output_path(
            "intermodel_slm_directory", "paths.intermodel_slm_directory"
        ),
        intermodel_aggregate_slm=txt_output_path(
            output_path(
                "intermodel_aggregate_slm_file",
                "paths.intermodel_aggregate_slm_file",
            )
        ),
        final_adl=output_path("final_adl_file", "paths.final_adl_file"),
    )
    if not scenario.artifact_manifest.is_file():
        raise PipelineError(
            f"Intermodel-only artifact manifest does not exist: {scenario.artifact_manifest}"
        )
    artifacts = load_artifact_manifest(scenario.artifact_manifest)
    return scenario, artifacts, scenario.merged_adl


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize_attempts(task_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    model_restarts = 0
    format_repairs = 0
    attempts = 0
    prompt_tokens = 0
    output_tokens = 0
    slm_seconds = 0.0
    python_seconds = 0.0

    for task in task_reports:
        status = str(task.get("status", "UNKNOWN"))
        statuses[status] = statuses.get(status, 0) + 1
        for attempt in task.get("attempts", []):
            attempts += 1
            model_restarts += sum(
                1 for item in attempt.get("slm_attempts", []) if item.get("model_restart") is not None
            )
            slm = attempt.get("slm")
            if slm:
                metadata = slm.get("metadata", {})
                prompt_tokens += int(metadata.get("prompt_eval_count") or 0)
                output_tokens += int(metadata.get("eval_count") or 0)
                slm_seconds += float(slm.get("wall_seconds") or 0.0)
            converter = attempt.get("converter")
            if converter:
                python_seconds += float(converter.get("duration_seconds") or 0.0)
        for repair in task.get("format_repairs", []):
            format_repairs += 1
            model_restarts += sum(
                1 for item in repair.get("slm_attempts", []) if item.get("model_restart") is not None
            )
            for slm_attempt in repair.get("slm_attempts", []):
                slm = slm_attempt.get("slm")
                if slm:
                    metadata = slm.get("metadata", {})
                    prompt_tokens += int(metadata.get("prompt_eval_count") or 0)
                    output_tokens += int(metadata.get("eval_count") or 0)
                    slm_seconds += float(slm.get("wall_seconds") or 0.0)
            converter = repair.get("converter") or repair.get("integration_command")
            if converter:
                python_seconds += float(converter.get("duration_seconds") or 0.0)

    return {
        "statuses": statuses,
        "attempt_count": attempts,
        "model_restarts": model_restarts,
        "format_repair_attempts": format_repairs,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "slm_wall_seconds": slm_seconds,
        "python_command_seconds": python_seconds,
    }


def complete_run_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    model_tasks: list[Mapping[str, Any]] = []
    intermodel_tasks: list[Mapping[str, Any]] = []
    merger_seconds = 0.0
    intermodel_integration_seconds = 0.0
    intermodel_format_repairs = 0

    for scenario in report.get("scenarios", []):
        model_tasks.extend(scenario.get("model_generation", {}).get("tasks", []))
        merger_seconds += float(
            scenario.get("adl_merge", {}).get("duration_seconds") or 0.0
        )
        intermodel = scenario.get("intermodel", {})
        for batch in intermodel.get("batches", []):
            intermodel_tasks.extend(batch.get("task_reports", []))
            intermodel_integration_seconds += float(
                batch.get("integration_command", {}).get("duration_seconds") or 0.0
            )
            intermodel_format_repairs += len(batch.get("format_repairs", []))

    model_summary = summarize_attempts(model_tasks)
    intermodel_summary = summarize_attempts(intermodel_tasks)
    return {
        "scenario_count": len(report.get("scenarios", [])),
        "model_generation": model_summary,
        "intermodel_generation": intermodel_summary,
        "all_prompt_tokens": model_summary["prompt_tokens"]
        + intermodel_summary["prompt_tokens"],
        "all_output_tokens": model_summary["output_tokens"]
        + intermodel_summary["output_tokens"],
        "all_tokens": model_summary["total_tokens"]
        + intermodel_summary["total_tokens"],
        "all_model_restarts": model_summary["model_restarts"]
        + intermodel_summary["model_restarts"],
        "all_format_repair_attempts": model_summary["format_repair_attempts"]
        + intermodel_summary["format_repair_attempts"]
        + intermodel_format_repairs,
        "adl_merger_seconds": merger_seconds,
        "intermodel_integration_seconds": intermodel_integration_seconds,
    }


def write_text_report(path: Path, report: Mapping[str, Any]) -> None:
    summary = report.get("summary", {})
    lines = [
        "4EM PIPELINE RUNTIME REPORT",
        "=" * 100,
        "",
        f"Run ID:                         {report.get('run_id')}",
        f"Status:                         {report.get('status')}",
        f"Mode:                           {report.get('mode')}",
        f"Started:                        {report.get('started_at')}",
        f"Finished:                       {report.get('finished_at')}",
        f"Total runtime seconds:          {float(report.get('duration_seconds') or 0.0):.3f}",
        f"Parameter file:                  {report.get('parameter_file', {}).get('path')}",
        "",
        "SLM CONFIGURATION",
        "-" * 100,
        f"Provider:                       {dotted_get(report, 'configuration.slm.provider')}",
        f"Base URL:                       {dotted_get(report, 'configuration.slm.base_url')}",
        f"Model:                          {dotted_get(report, 'configuration.slm.model')}",
        f"Thinking:                       {dotted_get(report, 'configuration.slm.thinking')}",
        f"Timeout seconds:                {dotted_get(report, 'configuration.slm.timeout_seconds')}",
        f"Thinking repetition enabled:    {dotted_get(report, 'configuration.slm.thinking_repetition_enabled', True)}",
        f"Thinking repetition limit:      {dotted_get(report, 'configuration.slm.thinking_repetition_limit')}",
        f"Thinking minimum block chars:   {dotted_get(report, 'configuration.slm.thinking_repetition_min_block_chars', 0)}",
        f"Seed mode:                      {dotted_get(report, 'configuration.retry.seed_mode', 'incremental')}",
        f"Temperature:                    {dotted_get(report, 'configuration.slm.options.temperature')}",
        "",
        "TOTALS",
        "-" * 100,
        f"Scenarios:                      {summary.get('scenario_count', 0)}",
        f"Prompt tokens:                  {summary.get('all_prompt_tokens', 0)}",
        f"Output tokens:                  {summary.get('all_output_tokens', 0)}",
        f"Total tokens:                   {summary.get('all_tokens', 0)}",
        f"Model restarts:                 {summary.get('all_model_restarts', 0)}",
        f"Format repair attempts:         {summary.get('all_format_repair_attempts', 0)}",
        f"ADL merger seconds:             {float(summary.get('adl_merger_seconds') or 0.0):.3f}",
        f"Intermodel integration seconds: {float(summary.get('intermodel_integration_seconds') or 0.0):.3f}",
        f"Model statuses:                 {dotted_get(summary, 'model_generation.statuses', {})}",
        f"Intermodel statuses:            {dotted_get(summary, 'intermodel_generation.statuses', {})}",
        "",
        "SCENARIOS",
        "=" * 100,
    ]

    def append_slm_attempts(container: Mapping[str, Any], indent: str) -> None:
        for slm_attempt in container.get("slm_attempts", []):
            error = re.sub(
                r"\s+", " ", str(slm_attempt.get("error_message") or "")
            ).strip()
            lines.append(
                f"{indent}slm_attempt={slm_attempt.get('attempt_number')} "
                f"status={slm_attempt.get('status')} "
                f"seed={slm_attempt.get('seed')} "
                f"seconds={float(slm_attempt.get('duration_seconds') or 0.0):.3f} "
                f"restart={'yes' if 'model_restart' in slm_attempt else 'no'}"
                + (f" error={error}" if error else "")
            )

    for scenario in report.get("scenarios", []):
        lines.extend(
            [
                "",
                f"Scenario: {scenario.get('scenario_name')}",
                "-" * 100,
                f"Status:                  {scenario.get('status')}",
                f"Scenario directory:      {scenario.get('scenario_directory')}",
                f"Run directory:           {scenario.get('run_directory')}",
                f"Duration seconds:         {float(scenario.get('duration_seconds') or 0.0):.3f}",
                f"Model stage status:       {dotted_get(scenario, 'model_generation.status')}",
                f"Merge stage status:       {dotted_get(scenario, 'adl_merge.status')}",
                f"Intermodel stage status:  {dotted_get(scenario, 'intermodel.status')}",
            ]
        )
        for task in scenario.get("model_generation", {}).get("tasks", []):
            lines.append(
                f"  MODEL {task.get('task_id')} | status={task.get('status')} | "
                f"seconds={float(task.get('duration_seconds') or 0.0):.3f} | "
                f"format_repairs={len(task.get('format_repairs', []))}"
            )
            for attempt in task.get("attempts", []):
                metadata = dotted_get(attempt, "slm.metadata", {}) or {}
                lines.append(
                    f"    attempt={attempt.get('attempt_number')} "
                    f"reason={attempt.get('failure_reason')} "
                    f"seconds={float(attempt.get('duration_seconds') or 0.0):.3f} "
                    f"prompt_tokens={metadata.get('prompt_eval_count', 0)} "
                    f"output_tokens={metadata.get('eval_count', 0)}"
                )
                append_slm_attempts(attempt, "      ")
            for repair in task.get("format_repairs", []):
                lines.append(
                    f"    FORMAT REPAIR attempt={repair.get('attempt_number')} "
                    f"status={repair.get('status')}"
                )
                append_slm_attempts(repair, "      ")
        for batch in scenario.get("intermodel", {}).get("batches", []):
            lines.append(
                f"  INTERMODEL BATCH {batch.get('batch_number')} | "
                f"status={batch.get('status')} | "
                f"seconds={float(batch.get('duration_seconds') or 0.0):.3f}"
            )
            for task in batch.get("task_reports", []):
                lines.append(
                    f"    {task.get('task_id')} | status={task.get('status')} | "
                    f"seconds={float(task.get('duration_seconds') or 0.0):.3f} | "
                    f"attempts={len(task.get('attempts', []))}"
                )
                for attempt in task.get("attempts", []):
                    append_slm_attempts(attempt, "      ")
            for repair in batch.get("format_repairs", []):
                lines.append(
                    f"    FORMAT REPAIR attempt={repair.get('attempt_number')} "
                    f"status={repair.get('status')}"
                )
                append_slm_attempts(repair, "      ")

    lines.extend(
        [
            "",
            "The JSON report contains all recorded request metadata, token counts,",
            "attempt timings, restart reasons, command arguments, return codes, stdout,",
            "stderr, file sizes, hashes, and error tracebacks.",
        ]
    )
    write_text_atomic(path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------


def process_full_scenario(
    ctx: RunContext,
    scenario_directory: Path,
    model_definitions: Sequence[ModelDefinition],
) -> dict[str, Any]:
    started_perf = time.perf_counter()
    scenario = create_scenario_run(ctx, scenario_directory)
    scenario.run_directory.mkdir(parents=True, exist_ok=True)

    scenario_report: dict[str, Any] = {
        "scenario_name": scenario.scenario_name,
        "scenario_directory": str(scenario.scenario_directory),
        "run_directory": str(scenario.run_directory),
        "started_at": iso_now(),
    }

    artifacts: list[ModelArtifact] = []
    task_reports: list[dict[str, Any]] = []
    discovered_folders = {
        item.name for item in scenario_directory.iterdir() if item.is_dir()
    }
    configured_folders = {
        model.scenario_folder for model in model_definitions if model.enabled
    }
    unknown_folders = sorted(discovered_folders - configured_folders)

    for model in model_definitions:
        if not model.enabled:
            continue
        for source_file in scenario_files(ctx, scenario, model):
            artifact, task_report = run_model_task(
                ctx, scenario, model, source_file
            )
            artifacts.append(artifact)
            task_reports.append(task_report)

    model_summary = summarize_attempts(task_reports)
    failed_model_tasks = sum(
        count
        for status, count in model_summary["statuses"].items()
        if status.startswith("FAILED")
    )
    scenario_report["model_generation"] = {
        "status": (
            "SKIPPED_NO_MODELS"
            if not task_reports
            else "COMPLETED_WITH_ERRORS"
            if failed_model_tasks
            else "OK"
        ),
        "unknown_scenario_model_folders": unknown_folders,
        "tasks": task_reports,
        "summary": model_summary,
    }

    write_artifact_manifest(scenario, artifacts)
    scenario_report["artifact_manifest"] = file_info(scenario.artifact_manifest)

    merge_report = run_adl_merger(ctx, scenario, artifacts)
    scenario_report["adl_merge"] = merge_report

    mode = dotted_get(ctx.config, "execution.mode", "full")
    if mode == "full":
        intermodel_report = run_intermodel_stage(
            ctx,
            scenario,
            artifacts,
            model_definitions,
            scenario.merged_adl,
        )
    else:
        intermodel_report = {
            "status": "SKIPPED_MODELS_ONLY",
            "duration_seconds": 0.0,
        }
    scenario_report["intermodel"] = intermodel_report

    stage_statuses = [
        scenario_report["model_generation"]["status"],
        merge_report.get("status"),
        intermodel_report.get("status"),
    ]
    scenario_report["status"] = (
        "COMPLETED_WITH_ERRORS"
        if any(
            status in {"FAILED", "FAILED_PYTHON", "COMPLETED_WITH_ERRORS"}
            or str(status).startswith("FAILED")
            for status in stage_statuses
        )
        else "OK"
    )
    scenario_report["finished_at"] = iso_now()
    scenario_report["duration_seconds"] = time.perf_counter() - started_perf
    return scenario_report


def process_intermodel_only_job(
    ctx: RunContext,
    job: Mapping[str, Any],
    model_definitions: Sequence[ModelDefinition],
) -> dict[str, Any]:
    started_perf = time.perf_counter()
    scenario, artifacts, input_adl = create_intermodel_only_scenario(ctx, job)
    scenario.run_directory.mkdir(parents=True, exist_ok=True)

    intermodel_report = run_intermodel_stage(
        ctx,
        scenario,
        artifacts,
        model_definitions,
        input_adl,
    )
    return {
        "scenario_name": scenario.scenario_name,
        "scenario_directory": str(scenario.scenario_directory),
        "run_directory": str(scenario.run_directory),
        "status": (
            "OK"
            if intermodel_report.get("status") in {"OK", "SKIPPED_EXISTING", "DISABLED"}
            else "COMPLETED_WITH_ERRORS"
        ),
        "started_at": iso_now(),
        "finished_at": iso_now(),
        "duration_seconds": time.perf_counter() - started_perf,
        "model_generation": {
            "status": "SKIPPED_INTERMODEL_ONLY",
            "tasks": [],
            "summary": summarize_attempts([]),
        },
        "adl_merge": {
            "status": "SKIPPED_INTERMODEL_ONLY",
            "duration_seconds": 0.0,
        },
        "artifact_manifest": file_info(scenario.artifact_manifest),
        "intermodel": intermodel_report,
    }


# ---------------------------------------------------------------------------
# Program entry point
# ---------------------------------------------------------------------------


def prepare_context(parameter_path: Path, config: dict[str, Any]) -> RunContext:
    project_root = parameter_path.parent.resolve()
    timestamp_format = str(dotted_get(config, "paths.timestamp_format", "%Y-%m-%dT%H-%M-%S"))
    timestamp = datetime.now().strftime(timestamp_format)

    run_id = str(uuid.uuid4())

    ctx = RunContext(
        parameter_path=parameter_path,
        project_root=project_root,
        config=config,
        run_id=run_id,
        timestamp=timestamp,
        started_at=iso_now(),
        started_perf=time.perf_counter(),
        report_json=Path(),
        report_text=Path(),
        event_log=Path(),
    )
    ctx.report = {
        "run_id": run_id,
        "status": "RUNNING",
        "mode": dotted_get(config, "execution.mode", "full"),
        "started_at": ctx.started_at,
        "parameter_file": file_info(parameter_path),
        "configuration": redact_secrets(config),
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "working_directory": str(Path.cwd()),
            "cpu_count": os.cpu_count(),
            "process_id": os.getpid(),
        },
        "scenarios": [],
        "unhandled_errors": [],
    }
    return ctx


def configure_runtime_report_paths(ctx: RunContext, report_directory: Path) -> None:
    """Store all runtime report files directly in the selected report directory."""
    report_values = {"timestamp": ctx.timestamp, "run_id": ctx.run_id}

    def report_path(config_key: str) -> Path:
        rendered_name = render_string(
            str(require(ctx.config, config_key)),
            report_values,
        )
        filename = Path(rendered_name)
        if filename.is_absolute() or len(filename.parts) != 1:
            raise PipelineError(
                f"{config_key} must be a filename without directory components, "
                f"got {rendered_name!r}."
            )
        return report_directory / filename

    report_directory.mkdir(parents=True, exist_ok=True)
    ctx.report_json = report_path("paths.report_json_filename")
    ctx.report_text = report_path("paths.report_text_filename")
    ctx.event_log = report_path("paths.event_log_filename")
    if ctx.event_log.exists():
        ctx.event_log.unlink()


def resolve_intermodel_report_directory(
    ctx: RunContext,
    job: Mapping[str, Any],
) -> Path:
    """
    Return the output_intermodel directory used by the selected job.

    The intermodel SLM directory normally ends in output_intermodel/slm.
    Runtime reports are stored in its parent so archiving output_intermodel
    automatically includes the JSON, TXT, and JSONL report files.
    """
    output_run_directory = resolve_path(
        ctx.project_root,
        require(job, "output_run_directory"),
    )
    raw_slm_directory = job.get("intermodel_slm_directory")
    if raw_slm_directory is not None:
        intermodel_slm_directory = resolve_path(
            ctx.project_root,
            raw_slm_directory,
        )
    else:
        intermodel_slm_directory = output_run_directory / str(
            require(ctx.config, "paths.intermodel_slm_directory")
        )
    return intermodel_slm_directory.parent


def finalize_context(ctx: RunContext, status: str) -> None:
    ctx.report["status"] = status
    ctx.report["finished_at"] = iso_now()
    ctx.report["duration_seconds"] = time.perf_counter() - ctx.started_perf
    ctx.report["summary"] = complete_run_summary(ctx.report)
    report_values = {"timestamp": ctx.timestamp, "run_id": ctx.run_id}
    visualization_enabled = enabled_setting(
        dotted_get(ctx.config, "visualization.enabled", "off"),
        "visualization.enabled",
    )
    visualization_path: Path | None = None
    if visualization_enabled:
        rendered = render_string(
            str(require(ctx.config, "visualization.filename")), report_values
        )
        filename = Path(rendered)
        if filename.is_absolute() or len(filename.parts) != 1:
            raise PipelineError(
                "visualization.filename must be a filename without directory components."
            )
        visualization_path = ctx.report_json.parent / filename

    ctx.report["report_files"] = {
        "json": str(ctx.report_json),
        "text": str(ctx.report_text),
        "events": str(ctx.event_log),
        "visualization": str(visualization_path) if visualization_path else None,
    }
    write_json_atomic(ctx.report_json, ctx.report)
    write_text_report(ctx.report_text, ctx.report)

    if visualization_path is not None:
        script_path = resolve_path(
            ctx.project_root, require(ctx.config, "visualization.script_path")
        )
        python_executable = str(
            dotted_get(ctx.config, "tools.python_executable", "") or sys.executable
        )
        visualization_result = run_external_command(
            ["{python}", "{script}", "{report_json}", "--output", "{output}"],
            {
                "python": python_executable,
                "script": script_path,
                "report_json": ctx.report_json,
                "output": visualization_path,
            },
            ctx.project_root,
            float(dotted_get(ctx.config, "visualization.timeout_seconds", 120)),
        )
        ctx.report["visualization"] = {
            "enabled": True,
            "status": (
                "OK"
                if visualization_result.ok and visualization_path.is_file()
                else "FAILED"
            ),
            "output": file_info(visualization_path),
            "command": command_report(ctx, visualization_result),
        }
        write_json_atomic(ctx.report_json, ctx.report)
        write_text_report(ctx.report_text, ctx.report)
    else:
        ctx.report["visualization"] = {"enabled": False, "status": "DISABLED"}
        write_json_atomic(ctx.report_json, ctx.report)
    ctx.event(
        "pipeline_finished",
        status=status,
        duration_seconds=ctx.report["duration_seconds"],
        summary=ctx.report["summary"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete configuration-driven 4EM pipeline."
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path(__file__).with_name("parameter.json"),
        help="parameter.json path. Default: next to pipeline.py.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default=None,
        help="Optional one-run override for execution.mode.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help=(
            "Optional one-run override for execution.selected_scenario. "
            "The value is relative to paths.scenarios_root."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parameter_path = args.parameters.expanduser().resolve()
    if not parameter_path.is_file():
        print(f"ERROR: Parameter file does not exist: {parameter_path}", file=sys.stderr)
        return 2

    scenario_directory: Path | None = None
    intermodel_only_job: Mapping[str, Any] | None = None

    try:
        config = load_parameter_file(parameter_path)
        if args.mode is not None:
            config.setdefault("execution", {})["mode"] = args.mode
        if args.scenario is not None:
            if not args.scenario.strip():
                raise PipelineError("--scenario must not be empty.")
            config.setdefault("execution", {})[
                "selected_scenario"
            ] = args.scenario.strip()
        ctx = prepare_context(parameter_path, config)
        mode = dotted_get(config, "execution.mode", "full")
        if mode in {"full", "models_only"}:
            scenario_directory = resolve_selected_scenario_directory(ctx)
            report_directory = create_scenario_run(
                ctx,
                scenario_directory,
            ).run_directory
        else:
            intermodel_only_job = resolve_selected_intermodel_only_job(ctx)
            report_directory = resolve_intermodel_report_directory(
                ctx,
                intermodel_only_job,
            )
        configure_runtime_report_paths(ctx, report_directory)
        ctx.event(
            "pipeline_started",
            mode=ctx.report["mode"],
            parameter_file=parameter_path,
        )
        model_definitions = parse_model_definitions(ctx)
        validate_static_paths(ctx, model_definitions)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    final_status = "OK"
    continue_after_failure = bool(
        dotted_get(config, "execution.continue_after_failure", True)
    )

    try:
        mode = dotted_get(config, "execution.mode", "full")
        if mode in {"full", "models_only"}:
            if scenario_directory is None:
                raise PipelineError("No scenario directory was resolved for this run.")
            try:
                scenario_report = process_full_scenario(
                    ctx, scenario_directory, model_definitions
                )
                ctx.report["scenarios"].append(scenario_report)
                if scenario_report.get("status") != "OK":
                    final_status = "COMPLETED_WITH_ERRORS"
            except Exception as exc:
                final_status = "COMPLETED_WITH_ERRORS"
                error_report = {
                    "scenario_name": scenario_directory.name,
                    "scenario_directory": str(scenario_directory),
                    "status": "FAILED_PIPELINE",
                    "started_at": iso_now(),
                    "finished_at": iso_now(),
                    "duration_seconds": 0.0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "model_generation": {
                        "tasks": [],
                        "summary": summarize_attempts([]),
                    },
                    "adl_merge": {},
                    "intermodel": {},
                }
                ctx.report["scenarios"].append(error_report)
                ctx.event(
                    "scenario_failed",
                    scenario=scenario_directory.name,
                    error=str(exc),
                )
                if not continue_after_failure:
                    raise
        else:
            if intermodel_only_job is None:
                raise PipelineError("No intermodel-only job was resolved for this run.")
            try:
                scenario_report = process_intermodel_only_job(
                    ctx, intermodel_only_job, model_definitions
                )
                ctx.report["scenarios"].append(scenario_report)
                if scenario_report.get("status") != "OK":
                    final_status = "COMPLETED_WITH_ERRORS"
            except Exception as exc:
                final_status = "COMPLETED_WITH_ERRORS"
                ctx.report["scenarios"].append(
                    {
                        "scenario_name": intermodel_only_job.get(
                            "scenario_name",
                            "unknown",
                        ),
                        "status": "FAILED_PIPELINE",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                        "model_generation": {
                            "tasks": [],
                            "summary": summarize_attempts([]),
                        },
                        "adl_merge": {},
                        "intermodel": {},
                    }
                )
                if not continue_after_failure:
                    raise
    except Exception as exc:
        final_status = "FAILED"
        ctx.report["unhandled_errors"].append(
            {
                "timestamp": iso_now(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        try:
            finalize_context(ctx, final_status)
        except Exception as exc:
            print(f"ERROR while writing reports: {exc}", file=sys.stderr)
            return 1

    print(f"Pipeline status: {final_status}")
    print(f"JSON report: {ctx.report_json}")
    print(f"Text report: {ctx.report_text}")
    print(f"Event log: {ctx.event_log}")
    visualization_file = dotted_get(ctx.report, "report_files.visualization")
    if visualization_file:
        print(f"Visualization: {visualization_file}")
    return 0 if final_status == "OK" else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
