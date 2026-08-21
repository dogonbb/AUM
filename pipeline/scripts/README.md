# Convert Saved Model SLM Outputs from Names to Descriptions

This tool does not call Ollama or another SLM.

It reads every scenario's:

```text
output/<scenario>_run/output_model_gen/complete_adl_file_here.adl
```

and creates description-based copies of all files below:

```text
output/<scenario>_run/output_model_gen/slm/
```

The copies are written to:

```text
output/<scenario>_run/output_model_gen/slm_description/
```

Original files are preserved.

## Install

Copy `create_description_slm_outputs.py` to the `scripts` directory.

## All scenarios

Run from the directory containing `pipeline.py`:

```powershell
python scripts/create_description_slm_outputs.py `
    --output-root output `
    --overwrite
```

## Only organized phase 4

```powershell
python scripts/create_description_slm_outputs.py `
    --output-root output `
    --scenario organized_phase4 `
    --overwrite
```

## Reports

Per scenario:

```text
output/<scenario>_run/output_model_gen/description_conversion_report/
├── report.json
└── report.txt
```

Master report:

```text
output/_description_conversion_report/
├── master_report.json
└── master_report.csv
```
