from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "analysis" / "results" / "final_kb.json"
OUT_DIR = BASE_DIR / "analysis" / "results"
OUT_CSV = OUT_DIR / "weird_values_report.csv"

LIST_PROPS = ["colour", "shape", "material"]
SINGLE_PROPS = ["fragility", "rigidity"]


def load_kb() -> dict[str, Any]:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def is_shape_suspicious(value: str) -> bool:
    suspicious = {
        "circle", "oval", "rectangle", "square", "triangle",
        "line", "curved", "rounded"
    }
    return value in suspicious


def is_colour_suspicious(value: str) -> bool:
    suspicious = {
        "blue", "green", "red", "white", "black"
    }
    return value in suspicious


def find_weird_values(kb: dict[str, Any]) -> pd.DataFrame:
    rows = []

    # collect global frequencies
    freq = {prop: Counter() for prop in LIST_PROPS + SINGLE_PROPS}

    for concept, entry in kb.items():
        props = entry.get("properties", {})
        for prop in LIST_PROPS:
            for v in props.get(prop, []):
                freq[prop][v] += 1
        for prop in SINGLE_PROPS:
            v = props.get(prop)
            if v:
                freq[prop][v] += 1

    for concept, entry in kb.items():
        props = entry.get("properties", {})

        # list properties
        for prop in LIST_PROPS:
            values = props.get(prop, [])
            for v in values:
                reasons = []

                if freq[prop][v] == 1:
                    reasons.append("rare_global_value")

                if prop == "shape" and concept == "person" and is_shape_suspicious(v):
                    reasons.append("possible_body_part_shape_or_generic_geometry")

                if prop == "colour" and concept == "person" and is_colour_suspicious(v):
                    reasons.append("possible_clothing_or_hair_colour_instead_of_object_colour")

                if prop == "material" and concept in {"person", "cat", "dog", "horse", "bird"}:
                    reasons.append("biological_material_interpretation")

                if reasons:
                    rows.append({
                        "concept": concept,
                        "property": prop,
                        "value": v,
                        "global_frequency": freq[prop][v],
                        "reason": "; ".join(reasons),
                    })

        # single-label properties
        for prop in SINGLE_PROPS:
            v = props.get(prop)
            if not v:
                continue

            reasons = []

            if freq[prop][v] == 1:
                reasons.append("rare_global_value")

            if prop == "rigidity" and concept in {"apple", "banana", "orange"} and v == "rigid":
                reasons.append("possibly_over-rigid_for_fruit")

            if prop == "fragility" and concept in {"bench", "chair", "car"} and v == "fragile":
                reasons.append("possibly_unexpected_fragility_label")

            if reasons:
                rows.append({
                    "concept": concept,
                    "property": prop,
                    "value": v,
                    "global_frequency": freq[prop][v],
                    "reason": "; ".join(reasons),
                })

    df = pd.DataFrame(rows).sort_values(
        by=["property", "concept", "global_frequency"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return df


def main() -> None:
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Missing KB file: {KB_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kb = load_kb()
    df = find_weird_values(kb)
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print(f"Flagged rows: {len(df)}")

    if not df.empty:
        print("\nSample flagged values:")
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()