from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "analysis" / "results" / "final_kb.csv"
GRAPH_DIR = BASE_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

def list_present(value):
    if pd.isna(value):
        return 0
    value = str(value).strip()
    if value == "" or value == "[]":
        return 0
    try:
        parsed = json.loads(value)
        return 1 if len(parsed) > 0 else 0
    except Exception:
        return 1

def value_present(value):
    if pd.isna(value):
        return 0
    return 1 if str(value).strip() else 0

rows = []

for _, row in df.iterrows():
    present = {
        "shape": list_present(row["shape"]),
        "colour": list_present(row["colour"]),
        "material": list_present(row["material"]),
        "fragility": value_present(row["fragility"]),
        "rigidity": value_present(row["rigidity"]),
        "size": int(
            pd.notna(row["height_m"])
            and pd.notna(row["length_m"])
            and pd.notna(row["width_m"])
        ),
        "weight_kg": value_present(row["weight_kg"]),
    }

    rows.append({
        "concept": row["concept"],
        "num_properties": sum(present.values())
    })

coverage_df = pd.DataFrame(rows)
summary = coverage_df["num_properties"].value_counts().sort_index()

summary.to_csv(GRAPH_DIR / "property_completeness_per_object.csv")

plt.figure(figsize=(6, 4))
plt.bar(summary.index.astype(str), summary.values)
plt.xlabel("Number of properties present")
plt.ylabel("Number of objects")
plt.title("Property Completeness per Object")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "property_completeness_per_object.png", dpi=300)
plt.close()

print(summary)
print("Saved:", GRAPH_DIR / "property_completeness_per_object.png")