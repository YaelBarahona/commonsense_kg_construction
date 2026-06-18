from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "inputs" / "mscoco concepts" / "properties.csv"
GRAPH_DIR = BASE_DIR / "graphs"
RESULTS_DIR = BASE_DIR / "analysis" / "results" / "existing_sources"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

sources = ["wordnet", "wikidata", "dbpedia", "yago", "ConceptNet"]

def has_value(x):
    if pd.isna(x):
        return False
    return str(x).strip() not in ["", "nan", "None"]

source_rows = []

for source in sources:
    count = df[source].apply(has_value).sum()
    percentage = count / len(df) * 100
    source_rows.append({
        "source": source,
        "values_found": count,
        "total_pairs": len(df),
        "coverage_percent": round(percentage, 2)
    })

source_df = pd.DataFrame(source_rows)
source_df.to_csv(RESULTS_DIR / "source_coverage.csv", index=False)
source_df.to_latex(RESULTS_DIR / "source_coverage.tex", index=False)

plt.figure(figsize=(7, 4))
plt.bar(source_df["source"], source_df["values_found"])
plt.ylabel("Object-property pairs with values")
plt.title("Coverage per Existing Knowledge Source")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "existing_source_coverage.png", dpi=300)
plt.close()

property_rows = []

for prop in sorted(df["property"].unique()):
    subset = df[df["property"] == prop]
    total = len(subset)

    available = subset[sources].notna().any(axis=1).sum()
    percentage = available / total * 100

    property_rows.append({
        "property": prop,
        "values_found": available,
        "total_pairs": total,
        "coverage_percent": round(percentage, 2)
    })

property_df = pd.DataFrame(property_rows)
property_df.to_csv(RESULTS_DIR / "property_coverage_existing_sources.csv", index=False)
property_df.to_latex(RESULTS_DIR / "property_coverage_existing_sources.tex", index=False)

property_df = property_df.sort_values("coverage_percent", ascending=False)

plt.figure(figsize=(9, 4))
plt.bar(property_df["property"], property_df["coverage_percent"])
plt.ylabel("Coverage (%)")
plt.title("Property Coverage in Existing Knowledge Sources")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "existing_property_coverage.png", dpi=300)
plt.close()

matrix_rows = []

for prop in sorted(df["property"].unique()):
    subset = df[df["property"] == prop]
    row = {"property": prop}

    for source in sources:
        row[source] = subset[source].apply(has_value).sum()

    matrix_rows.append(row)

matrix_df = pd.DataFrame(matrix_rows)
matrix_df.to_csv(RESULTS_DIR / "source_property_matrix.csv", index=False)
matrix_df.to_latex(RESULTS_DIR / "source_property_matrix.tex", index=False)

print("Saved tables to:", RESULTS_DIR)
print("Saved graphs to:", GRAPH_DIR)
print("\nCoverage per source:")
print(source_df)
print("\nCoverage per property:")
print(property_df)