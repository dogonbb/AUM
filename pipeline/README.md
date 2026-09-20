# 4EM Pipeline README

## 1. Overview

This project provides a configuration-driven pipeline for:

1. generating 4EM model outputs with an SLM,
2. converting generated TXT files into ADL,
3. merging model ADL files,
4. generating intermodel relationships,
5. integrating those relationships into a final ADL file.

The pipeline supports three execution modes:

- `full`: model generation, TXT-to-ADL conversion, ADL merging, intermodel generation, and final ADL integration.
- `models_only`: model generation, TXT-to-ADL conversion, and ADL merging only.
- `intermodel_only`: uses existing model outputs, an existing artifact manifest, and an existing merged ADL.

## 2. Folder Structure

```text
pipeline/
├── pipeline.py
├── parameter.json
├── scenarios/
│   ├── controlled_phase1/
│   ├── controlled_phase2/
│   ├── controlled_phase3/
│   ├── controlled_phase4/
│   ├── organized_phase1/
│   ├── organized_phase2/
│   ├── organized_phase3/
│   └── organized_phase4/
├── prompts/
│   ├── model_generation/
│   └── inter_model_generation/
│       ├── inter_model_connections_descriptions/
│       └── model_description/
├── scripts/
│   ├── restart_ollama_model.py
│   ├── adl_model_merger.py
│   ├── add_intermodel_relations_slm.py
│   └── slm_to_adl/
└── output/
    ├── runtime_reports/
    └── <scenario_name>_<timestamp>/
```

A normal run folder contains:

```text
output/<run_name>/
├── output_model_gen/
│   ├── slm/
│   │   └── generated model TXT files
│   ├── adl/
│   │   └── generated model ADL files
│   ├── artifacts.json
│   └── <scenario_name>_models.adl
└── output_intermodel/
    ├── slm/
    │   └── individual intermodel TXT files
    ├── intermodel_relations.txt
    └── <scenario_name>_with_intermodel_relations.adl
```

## 3. Important Configuration Settings

The main configuration file is `parameter.json`. It must be located next to `pipeline.py`.

### 3.1 Execution

```json
"execution": {
  "mode": "intermodel_only",
  "overwrite": true,
  "continue_after_failure": true,
  "selected_scenario": "controlled_phase1"
}
```

- `mode`: `full`, `models_only`, or `intermodel_only`
- `overwrite`: regenerate existing outputs when `true`
- `continue_after_failure`: continue with the next task after a failure
- `selected_scenario`: default scenario when no command-line scenario is supplied

### 3.2 SLM

```json
"slm": {
  "provider": "ollama",
  "base_url": "http://localhost:11434",
  "model": "gemma4:e4b",
  "thinking": true,
  "timeout_seconds": 1800,
  "log_thinking": false,
  "thinking_repetition_enabled": true,
  "thinking_repetition_limit": 10,
  "thinking_repetition_min_block_chars": 40,
  "show_stream_output": false,
  "keep_alive": "5m",
  "options": {
    "temperature": 0.2,
    "num_ctx": 131072,
    "top_p": 0.9,
    "seed": 42,
    "num_predict": 131072
  }
}
```

Important fields:

- `base_url`: local Ollama URL
- `model`: Ollama model name
- `thinking`: enables or disables thinking output
- `timeout_seconds`: maximum duration of one SLM request
- `log_thinking`: continuously save the thinking stream of every SLM attempt under the run's `thinking_logs` directory
- `thinking_repetition_enabled`: enable or disable automatic restarts caused by repeated thinking blocks
- `thinking_repetition_limit`: restart when the exact same complete thinking block has appeared this many times (only when repetition detection is enabled)
- `thinking_repetition_min_block_chars`: ignore blocks with fewer characters after surrounding whitespace is removed; `0` disables the length filter
- `show_stream_output`: print streamed thinking and answer text live in the terminal
- `temperature`: randomness
- `num_ctx`: context size
- `seed`: an integer improves reproducibility; `false` sends no seed to Ollama

### 3.3 Total SLM Timeout and Model Restart

```json
"retry": {
  "max_restart": 3,
  "seed_mode": "incremental",
  "seed_increment": 1
}
```

With `retry.seed_mode` set to `random`, every SLM attempt, including the first,
uses a new random 31-bit seed. Seeds are unique within one call and are stored
in every attempt report, so a result can be reproduced later. Set the mode to
`incremental` to use `slm.options.seed` for the first attempt and increase it
by `retry.seed_increment` for each retry (for example `42`, `43`, `44`, `45`).
Set `slm.options.seed` to `false` to omit the seed from every Ollama request;
this takes precedence over `retry.seed_mode`.

`slm.timeout_seconds` is one total wall-clock limit for a complete SLM call. It
includes connection setup, thinking, and answer generation. When
`slm.thinking_repetition_enabled` is `true`, empty or whitespace-only lines
separate complete thinking blocks. Blocks whose first non-whitespace text is
`ELEMENTS` or `CONNECTIONS` are ignored; all other blocks are counted. The
pipeline compares eligible blocks using exact, case-sensitive text equality. Whitespace, punctuation,
capitalization, and internal line endings must all match. When a complete block reaches
`slm.thinking_repetition_limit`, the current request is aborted and restarted.
Consecutive empty blocks are ignored.
Set `slm.thinking_repetition_enabled` to `false` for diagnostic runs that must
continue even when the model repeats its thinking.
Repeated blocks are also detected when other blocks occur between them.

When `slm.log_thinking` is enabled, every initial request and retry receives a
separate file under `thinking_logs/<task>/`. The file is flushed as chunks
arrive, so partial thinking remains available after a repetition abort, total
timeout, or Ctrl+C. Its filename contains the attempt number and seed, and the
same path is recorded in the JSON/JSONL runtime logs.

Behavior:

```text
SLM total timeout
→ abort request
→ unload the configured Ollama model
→ retry the identical request
→ repeat at most retry.max_restart times
```

The same restart sequence is used when a thinking repetition is detected.

```text
Python converter/integrator failure
→ no model restart
→ optional format-first validation repair
→ run the same Python script again
```

The restart command must be enabled:

```json
"tools": {
  "slm_restart": {
    "enabled": true,
    "cwd": ".",
    "timeout_seconds": 120,
    "command": [
        "{python}",
        "scripts/restart_ollama_model.py",
        "--model",
        "{model}",
        "--base-url",
        "{base_url}"
    ]
  }
}
```

### 3.4 Format-first Validation Repair

```json
"format_repair": {
  "enabled": true,
  "max_attempts": 2,
  "general_prompt_path": "prompts/format_repair/general_repair_prompt.txt"
}
```

Regular model repairs use `general_repair_prompt.txt`. Individual intermodel
repairs use the separately configured
`intermodel.repair_prompt_path` (`intermodel_repair_prompt.txt`) so that no
model-only `ELEMENTS`/`CONNECTIONS` instructions are included.

After every model or intermodel generation, the pipeline immediately validates
that individual output with the corresponding Python converter or integrator.
If validation fails, it sends only that malformed output together with the
Python error, relevant scenario source content, model explanation, generation
rules, and exact formatting specification to the SLM. For an intermodel task,
only the source/target pair is repaired; the aggregated intermodel file is not
sent to the repair model. Scenario and source-file names are omitted. The SLM
must first check the formatting.
If formatting is wrong, it may repair formatting only. If formatting is
already correct, it may make the smallest content change needed to resolve the
reported validation error, without inventing scenario information. Original
and repaired files are preserved below a `format_repair` directory and every
attempt is included in the runtime JSON.

The complete repair prompt and all model-specific explanations are in English.
It reuses each model's configured `description_path` and, for regular model
repairs, the rules from its original generation prompt (without the original
task suffix). Intermodel repairs reuse the task, connector, direction, output,
and output-rule sections of the rendered pair-generation prompt. The generated
repair-prompt sections are `General explanation`, `Scenario
texts`, `Model explanation`, `Original model generation rules`, `Required
output format`, `Python error`, and `Output to repair`. The scenario directory
name itself is intentionally not included.

### 3.5 Output Paths

```json
"paths": {
  "scenarios_root": "scenarios",
  "output_root": "output",
  "scenario_run_directory_template": "{scenario_name}_{timestamp}",
  "model_slm_directory": "output_model_gen/slm",
  "model_adl_directory": "output_model_gen/adl",
  "artifact_manifest_file": "output_model_gen/artifacts.json",
  "merged_adl_file": "output_model_gen/{scenario_name}_models.adl",
  "intermodel_slm_directory": "output_intermodel/slm",
  "intermodel_aggregate_slm_file": "output_intermodel/intermodel_relations.txt",
  "final_adl_file": "output_intermodel/{scenario_name}_with_intermodel_relations.adl"
}
```

Using:

```json
"scenario_run_directory_template": "{scenario_name}_{timestamp}"
```

creates a separate folder for every run.

### 3.6 Enable or Disable Models

Each model contains:

```json
"enabled": true
```

To test only one model type, enable that model and disable all others.

Available model IDs:

```text
GoalModel
BusinessProcessModel
ActorsResourcesModel
ConceptsModel
TechnicalComponentsRequirementsModel
ProductServiceModel
```

### 3.7 Intermodel-Only Jobs

The paths in an intermodel-only job must refer to an existing model-generation
run. Fixed directory names such as `controlled_phase1_run` below are examples;
adapt them to the actual `output_run_directory` configured for the run.

```json
"intermodel_only": {
  "jobs": [
    {
      "enabled": true,
      "scenario_name": "controlled_phase1",
      "scenario_directory": "scenarios/controlled_phase1",
      "artifact_manifest": "output/controlled_phase1_run/output_model_gen/artifacts.json",
      "input_adl": "output/controlled_phase1_run/output_model_gen/controlled_phase1_models.adl",
      "output_run_directory": "output/controlled_phase1_run"
    }
  ]
}
```

This maps the selected scenario to its scenario descriptions, artifacts, input ADL, and output folder.

### 3.8 Runtime Visualization

```json
"visualization": {
  "enabled": "on",
  "script_path": "scripts/visualize_runtime_report.py",
  "filename": "pipeline_{timestamp}_visualization.html",
  "timeout_seconds": 120
}
```

Set `enabled` to `"on"` or `"off"` to switch automatic visualization on or
off. Boolean `true` and `false` are also accepted. When enabled, every completed pipeline run creates a standalone HTML
dashboard next to its JSON and text reports. It shows total runtime, prompt and
output tokens, total tokens, SLM calls, timeout restarts, format repairs, and a
row for every individual SLM call and retry.

An existing JSON report can also be visualized manually:

```powershell
python scripts/visualize_runtime_report.py `
    output/path/to/pipeline_<timestamp>.json `
    --output output/path/to/pipeline_visualization.html
```

Open the generated HTML file directly in a browser. It has no external
JavaScript or library dependency.

## 4. Important Commands

Run all commands from the main `pipeline` folder.

### Full pipeline

```powershell
python pipeline.py --mode full --scenario controlled_phase1
```

### Model generation only

```powershell
python pipeline.py --mode models_only --scenario controlled_phase1
```

### Intermodel generation only

```powershell
python pipeline.py --mode intermodel_only --scenario controlled_phase1
```

### Run the default scenario from `parameter.json`

```powershell
python pipeline.py
```

## 5. Run All Eight Scenarios Once

```powershell
$scenarios = @(
    "controlled_phase1",
    "controlled_phase2",
    "controlled_phase3",
    "controlled_phase4",
    "organized_phase1",
    "organized_phase2",
    "organized_phase3",
    "organized_phase4"
)

foreach ($scenario in $scenarios) {
    python pipeline.py `
        --mode models_only `
        --scenario $scenario

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$scenario finished with an error."
    }
}
```

## 6. Run All Eight Scenarios Three Times

```powershell
$scenarios = @(
    "controlled_phase1",
    "controlled_phase2",
    "controlled_phase3",
    "controlled_phase4",
    "organized_phase1",
    "organized_phase2",
    "organized_phase3",
    "organized_phase4"
)

1..3 | ForEach-Object {
    $runNumber = $_

    foreach ($scenario in $scenarios) {
        Write-Host "Starting $scenario - run $runNumber of 3"

        python pipeline.py `
            --mode models_only `
            --scenario $scenario

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "$scenario - run $runNumber finished with an error."
        }
    }
}
```

This executes 24 runs in total.

## 7. Run All Eight Intermodel Scenarios Three Times and Save Every Run

The following example assumes fixed run directories named
`output/<scenario_name>_run`. If different directories are configured under
`intermodel_only.jobs`, adjust `$sourceFolder` accordingly.

```powershell
$scenarios = @(
    "controlled_phase1",
    "controlled_phase2",
    "controlled_phase3",
    "controlled_phase4",
    "organized_phase1",
    "organized_phase2",
    "organized_phase3",
    "organized_phase4"
)

$repetitions = 3

foreach ($runNumber in 1..$repetitions) {
    foreach ($scenario in $scenarios) {
        $sourceFolder = "output\${scenario}_run\output_intermodel"

        if (Test-Path $sourceFolder) {
            Remove-Item $sourceFolder -Recurse -Force
        }

        python pipeline.py `
            --mode intermodel_only `
            --scenario $scenario

        $exitCode = $LASTEXITCODE
        $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss-fff"
        $status = if ($exitCode -eq 0) { "SUCCESS" } else { "FAILED" }

        $saveFolder = "output\run_history\$scenario\run_${runNumber}_${timestamp}_${status}"
        $saveOutputFolder = Join-Path $saveFolder "output_intermodel"

        New-Item -ItemType Directory -Force -Path $saveOutputFolder | Out-Null

        if (Test-Path $sourceFolder) {
            Copy-Item `
                -Path "$sourceFolder\*" `
                -Destination $saveOutputFolder `
                -Recurse `
                -Force
        }
        else {
            "Pipeline exit code: $exitCode" |
                Set-Content `
                    -Path (Join-Path $saveFolder "NO_OUTPUT_CREATED.txt") `
                    -Encoding UTF8
        }
    }
}
```

Each saved run contains:

```text
output_intermodel/
├── slm/
├── intermodel_relations.txt
└── controlled_phase1_with_intermodel_relations.adl
```

The folder name includes the run number, timestamp, and status, so runs are not overwritten.

## 8. Directly Integrate an Existing TXT File into ADL

The paths in this example likewise assume the fixed example directory
`output/controlled_phase1_run` and must be adapted when timestamped or
model-specific run directories are used.

```powershell
python scripts/add_intermodel_relations_slm.py `
    output/controlled_phase1_run/output_model_gen/controlled_phase1_models.adl `
    output/controlled_phase1_run/output_intermodel/intermodel_relations.txt `
    --output output/controlled_phase1_run/output_intermodel/controlled_phase1_with_intermodel_relations.adl `
    --overwrite
```


The model solutions are the same for the matching phase. The source descriptions differ between Controlled and Organized scenarios.

## 9. Important Output Files

```text
output_model_gen/slm/
```

Generated model TXT files.

```text
output_model_gen/adl/
```

Converted model ADL files.

```text
output_model_gen/artifacts.json
```

Mapping between scenario files, model TXT files, ADL files, model IDs, and ADL model names.

```text
output_model_gen/<scenario_name>_models.adl
```

Merged ADL without intermodel relationships.

```text
output_intermodel/slm/
```

Individual intermodel TXT files.

```text
output_intermodel/intermodel_relations.txt
```

Aggregated intermodel relationships.

```text
output_intermodel/<scenario_name>_with_intermodel_relations.adl
```

Final ADL with integrated intermodel relationships.

## 10. Runtime Reports

Reports are written to:

```text
output/runtime_reports/
```

Typical files:

```text
pipeline_<timestamp>.json
pipeline_<timestamp>.txt
pipeline_<timestamp>.jsonl
```

The reports contain configuration data, task status, token usage, timeouts, restart counts, converter errors, and generated file paths.

## 11. Troubleshooting

### No final ADL was created

Check whether this file exists:

```text
output_intermodel/intermodel_relations.txt
```

Then run the integrator directly.

### The SLM hangs

Reduce:

```json
"timeout_seconds": 600
```

Enable:

```json
"max_restart": 1
```

and verify:

```json
"tools": {
  "slm_restart": {
    "enabled": true
  }
}
```

### Python conversion fails

Set:

```json
"format_repair": { "enabled": false, "max_attempts": 0 }
```

This prevents an additional SLM repair request after a converter failure.
Converter failures do not restart the configured Ollama model.

### Existing files are skipped

Set:

```json
"overwrite": true
```

### Old runs are overwritten

Use:

```json
"scenario_run_directory_template": "{scenario_name}_{timestamp}"
```
