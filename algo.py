#!/usr/bin/env python3
"""
algo.py — AgroPortal annotator runner for Pontus-X / AgrospAI compute-to-data.

Inputs (mounted by platform):
  /data/inputs   (read-only)
Outputs:
  /data/outputs  (write results here)
Logs:
  /data/logs     (optional)

Environment variables:
  AGROPORTAL_API_KEY   (required)
  ONTOLOGIES           (optional)  - comma-separated list, defaults to "AGROVOC"
  INPUT_FILE           (optional)  - specific file inside /data/inputs
  SHEET                (optional)  - Excel sheet name
  COLUMN_MAP_JSON      (optional)  - JSON mapping
  SLEEP_BETWEEN_CALLS  (optional)
  MAX_CHARS_PER_CALL   (optional)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import agro_annotator_service as aas


INPUTS = Path("/data/inputs")
OUTPUTS = Path("/data/outputs")
LOGS = Path("/data/logs")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if not v else float(v)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if not v else int(v)


def _pick_input_file() -> Path:
    specified = os.getenv("INPUT_FILE")
    if specified:
        p = Path(specified)
        if not p.is_absolute():
            p = INPUTS / p
        if not p.exists():
            raise FileNotFoundError(f"INPUT_FILE does not exist: {p}")
        return p

    for ext in ("*.csv", "*.tsv", "*.xlsx", "*.xls"):
        files = sorted(INPUTS.rglob(ext))
        if files:
            return files[0]

    raise FileNotFoundError("No CSV/XLS/XLSX file found in /data/inputs")


def _normalize_col(c: str) -> str:
    return str(c).strip().lower()


def _auto_map_standard_columns(df: pd.DataFrame) -> Dict[str, str]:
    required = ["title", "keywords", "description"]
    norm = {_normalize_col(c): c for c in df.columns}

    missing = [c for c in required if c not in norm]
    if missing:
        raise ValueError(
            "Dataset must contain columns named "
            "'title', 'keywords', 'description' "
            f"(missing: {missing}). "
            "Use COLUMN_MAP_JSON to override."
        )

    return {k: norm[k] for k in required}


def _get_column_map(df: pd.DataFrame) -> Dict[str, str]:
    override = os.getenv("COLUMN_MAP_JSON")
    if override:
        column_map = json.loads(override)
        for k in ("title", "keywords", "description"):
            if k not in column_map:
                raise ValueError(f"COLUMN_MAP_JSON missing key '{k}'")
            if column_map[k] not in df.columns:
                raise ValueError(f"Column '{column_map[k]}' not found in dataset")
        return column_map

    return _auto_map_standard_columns(df)


def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    # Put cache folders in writable place
    os.chdir(str(OUTPUTS))

    if not os.getenv("AGROPORTAL_API_KEY"):
        print("ERROR: AGROPORTAL_API_KEY not set", file=sys.stderr)
        return 2

    input_file = _pick_input_file()
    sheet = os.getenv("SHEET")

    ontologies_env = os.getenv("ONTOLOGIES")
    if ontologies_env:
        ontologies = [o.strip() for o in ontologies_env.split(",") if o.strip()]
    else:
        ontologies = ["AGROVOC"]
        print("INFO: Defaulting ONTOLOGIES to AGROVOC", file=sys.stderr)

    sleep_between_calls = _env_float("SLEEP_BETWEEN_CALLS", 0.0)
    max_chars = _env_int("MAX_CHARS_PER_CALL", 4000)

    df = aas.load_table(str(input_file), sheet=sheet)
    column_map = _get_column_map(df)

    out_simple = OUTPUTS / "annotations_simple.csv"
    simple_df = aas.make_simple_annotations_csv_dedup(
        df=df,
        column_map=column_map,
        ontologies=ontologies,
        out_path=str(out_simple),
        max_chars_per_call=max_chars,
        sleep_between_calls=sleep_between_calls,
        use_proxy_score=True,
    )

    out_enriched = OUTPUTS / "annotations_enriched.csv"

    try:
        enriched_df = aas.enrich_labels_inplace(simple_df, lang="en")
        enriched_df.to_csv(out_enriched, index=False)
    except Exception as e:
        print(f"WARNING: label enrichment failed; continuing without enrichment: {e}", file=sys.stderr)
        # still write something so the job succeeds
        simple_df.to_csv(out_enriched, index=False)

    meta = {
        "input_file": str(input_file),
        "ontologies": ontologies,
        "column_map": column_map,
        "rows_input": len(df),
        "rows_output": len(enriched_df),
    }
    (OUTPUTS / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print("Algorithm finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())