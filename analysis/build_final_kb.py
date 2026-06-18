from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any
import ast
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "analysis" / "results" / "clean_rows.csv"
OUTPUT_DIR = BASE_DIR / "analysis" / "results"
OUTPUT_JSON = OUTPUT_DIR / "final_kb.json"
OUTPUT_JSONL = OUTPUT_DIR / "final_kb.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "final_kb.csv"


LIST_CATEGORICAL = {"colour", "shape", "material"}
SINGLE_CATEGORICAL = {"rigidity", "fragility"}

SIZE_TO_METERS = {
    "meters": 1.0,
    "meter": 1.0,
    "centimeters": 0.01,
    "centimeter": 0.01,
    "millimeters": 0.001,
    "millimeter": 0.001,
}

WEIGHT_TO_KG = {
    "kilograms": 1.0,
    "kilogram": 1.0,
    "grams": 0.001,
    "gram": 0.001,
}


def normalize_value_list(v: Any) -> list[str]:

    # handle None
    if v is None:
        return []

    # handle list values directly
    if isinstance(v, list):
        items = v
    else:
        # check NaN only for scalar types
        if isinstance(v, float) and pd.isna(v):
            return []

        # try parsing string lists
        if isinstance(v, str):
            s = v.strip()

            if s.startswith("[") or s.startswith("{"):
                try:
                    v = json.loads(s)
                except Exception:
                    try:
                        v = ast.literal_eval(s)
                    except Exception:
                        v = s

        if isinstance(v, list):
            items = v
        else:
            items = [v]

    out: list[str] = []

    for item in items:
        if item is None:
            continue

        s = str(item).strip().lower()
        if s:
            out.append(s)
            
    return out


def majority_list(values_per_run: list[list[str]], min_fraction: float = 0.5) -> list[str]:
    counts = Counter()
    n_runs = len(values_per_run)

    for run_vals in values_per_run:
        for val in set(run_vals):
            counts[val] += 1

    threshold = max(1, int(n_runs * min_fraction + 0.999999))
    kept = [label for label, c in counts.items() if c >= threshold]
    return sorted(kept)


def majority_label(labels: list[str]) -> str | None:
    clean = [str(x).strip().lower() for x in labels if str(x).strip()]
    if not clean:
        return None
    return Counter(clean).most_common(1)[0][0]


def robust_median(vals: list[float]) -> float | None:
    clean = [float(v) for v in vals if pd.notna(v)]
    if not clean:
        return None
    if len(clean) < 4:
        return round(median(clean), 4)

    med = median(clean)
    abs_dev = [abs(x - med) for x in clean]
    mad = median(abs_dev)

    if mad == 0:
        return round(med, 4)

    filtered = [x for x in clean if abs(x - med) <= 3 * mad]
    if not filtered:
        filtered = clean

    return round(median(filtered), 4)


def to_standard_numeric(domain: str, measurement: str, value: float) -> tuple[str, float] | None:
    m = str(measurement).strip().lower()

    if domain == "size":
        factor = SIZE_TO_METERS.get(m)
        if factor is None:
            return None
        return "m", value * factor

    if domain == "weight":
        factor = WEIGHT_TO_KG.get(m)
        if factor is None:
            return None
        return "kg", value * factor

    return None


def build_final_kb(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    kb: dict[str, dict[str, Any]] = {}

    for concept, concept_df in df.groupby("concept", sort=True):
        entry: dict[str, Any] = {"concept": concept, "properties": {}}

        # categorical list properties
        for domain in LIST_CATEGORICAL:
            domain_df = concept_df[
                (concept_df["domain"] == domain)
                & (concept_df["measurement"].fillna("") == "")
            ]
            if domain_df.empty:
                continue

            runs = [normalize_value_list(v) for v in domain_df["values"].tolist()]
            agg = majority_list(runs, min_fraction=0.5)
            entry["properties"][domain] = agg

        # categorical single-label properties
        for domain in SINGLE_CATEGORICAL:
            domain_df = concept_df[
                (concept_df["domain"] == domain)
                & (concept_df["measurement"].fillna("") == "")
            ]
            if domain_df.empty:
                continue

            label = majority_label(domain_df["values"].tolist())
            entry["properties"][domain] = label

        # numeric: size
        size_props: dict[str, float] = {}
        size_df = concept_df[concept_df["domain"] == "size"]
        if not size_df.empty:
            grouped = defaultdict(list)
            for _, row in size_df.iterrows():
                dim = str(row["dimension"]).strip().lower()
                parsed = to_standard_numeric("size", row["measurement"], float(row["values"]))
                if parsed is None:
                    continue
                _, standard_val = parsed
                grouped[dim].append(standard_val)

            for dim, vals in grouped.items():
                agg = robust_median(vals)
                if agg is not None:
                    size_props[f"{dim}_m"] = agg

        if size_props:
            entry["properties"]["size"] = size_props

        # numeric: weight
        weight_df = concept_df[concept_df["domain"] == "weight"]
        if not weight_df.empty:
            vals_kg = []
            for _, row in weight_df.iterrows():
                parsed = to_standard_numeric("weight", row["measurement"], float(row["values"]))
                if parsed is None:
                    continue
                _, standard_val = parsed
                vals_kg.append(standard_val)

            agg = robust_median(vals_kg)
            if agg is not None:
                entry["properties"]["weight_kg"] = agg

        kb[concept] = entry

    return kb


def kb_to_flat_rows(kb: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for concept, entry in kb.items():
        props = entry["properties"]

        row = {"concept": concept}

        for domain in LIST_CATEGORICAL:
            row[domain] = json.dumps(props.get(domain, []), ensure_ascii=False)

        for domain in SINGLE_CATEGORICAL:
            row[domain] = props.get(domain)

        size = props.get("size", {})
        row["height_m"] = size.get("height_m")
        row["length_m"] = size.get("length_m")
        row["width_m"] = size.get("width_m")
        row["weight_kg"] = props.get("weight_kg")

        rows.append(row)

    return rows


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # parse JSON-like values column where needed
    def parse_values(x: Any) -> Any:
        if pd.isna(x):
            return x

        if isinstance(x, (list, dict, int, float)):
            return x

        s = str(x).strip()
        if not s:
            return s

        if s.startswith("[") or s.startswith("{"):
            # Try JSON first
            try:
                return json.loads(s)
            except Exception:
                pass

            # Then try Python literal form like "['red', 'white']"
            try:
                return ast.literal_eval(s)
            except Exception:
                return s

        return s

    df["values"] = df["values"].apply(parse_values)

    kb = build_final_kb(df)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for concept in sorted(kb.keys()):
            f.write(json.dumps(kb[concept], ensure_ascii=False) + "\n")

    flat_rows = kb_to_flat_rows(kb)
    pd.DataFrame(flat_rows).to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_JSONL}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Objects in KB: {len(kb)}")


if __name__ == "__main__":
    main()