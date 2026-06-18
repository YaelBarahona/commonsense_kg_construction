from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "analysis" / "results" / "final_kb.json"
GRAPH_DIR = BASE_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = GRAPH_DIR / "property_coverage.csv"
OUT_PNG = GRAPH_DIR / "property_coverage_heatmap.png"


PROPERTIES = [
    "colour",
    "shape",
    "material",
    "fragility",
    "rigidity",
    "size",
    "weight_kg",
]


def is_present(value):
    if value is None:
        return 0
    if isinstance(value, list):
        return 1 if len(value) > 0 else 0
    if isinstance(value, dict):
        return 1 if len(value) > 0 else 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    return 1


def main():
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Missing KB file: {KB_PATH}")

    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    rows = []
    for concept, entry in kb.items():
        props = entry.get("properties", {})
        row = {"concept": concept}
        for prop in PROPERTIES:
            row[prop] = is_present(props.get(prop))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("concept").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    heatmap_df = df.set_index("concept")

    plt.figure(figsize=(10, max(12, len(heatmap_df) * 0.3)))
    plt.imshow(heatmap_df.values, aspect="auto")
    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
    plt.colorbar(label="Coverage")
    plt.title("Property Coverage per Object")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    overall = heatmap_df.mean().sort_values(ascending=False)
    print("Saved:", OUT_CSV)
    print("Saved:", OUT_PNG)
    print("\nCoverage per property:")
    print((overall * 100).round(1).astype(str) + "%")

    # --- Property coverage bar chart ---

    sns.set_style("whitegrid")

    coverage_percent = (overall * 100).sort_values()

    plt.figure(figsize=(8,4))

    bars = plt.bar(
        coverage_percent.index,
        coverage_percent.values,
        color="#4C72B0",
        width=0.6
    )

    plt.ylabel("Coverage (%)", fontsize=11)
    plt.title("Property Coverage Across Objects", fontsize=13, pad=10)

    plt.ylim(0,105)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f"{height:.0f}%",
            ha='center',
            fontsize=10
        )

    plt.xticks(rotation=25)

    sns.despine()

    plt.tight_layout()

    BAR_PATH = GRAPH_DIR / "property_coverage_bar.png"
    plt.savefig(BAR_PATH, dpi=300)
    plt.close()

    print("Saved:", BAR_PATH)

if __name__ == "__main__":
    main()