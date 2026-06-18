# Load the filesavg

import os
from pathlib import Path
import pandas as pd
from kg_constructors.json_extractor import extract_json_from_string
from collections import defaultdict
import json
from dataclasses import dataclass, asdict
import numbers
from typing import Any
from utils.logger import setup_logger
from difflib import get_close_matches
import inflect
import re
from collections import Counter
import yaml

p = inflect.engine()
# Global unmatched tracker
unmatched_tokens_all = Counter()

@dataclass
class ErrorRecord:
    experiment_type: str
    model: str
    concept: str
    file: str
    file_path: str
    row_idx: int          # original DataFrame index
    error_category: str   # "syntax" | "semantic" | "factual"
    error_subtype: str    # e.g. "invalid_json", "many_keys", …
    message: str
    response_excerpt: str # keep it short; 1-2 kB tops

@dataclass
class DataOutput:
    experiment_type: str
    model: str
    concept: str
    file_path: str
    row_idx: int          
    domain: str   
    dimension: str    
    measurement: str
    values : Any 


# bucket for every clean row we accept
_agg_rows: list[dict] = []

# organised view, convenient if you want one record per concept later
#   (model, concept, domain, dimension, measurement) → {'avg': … , 'ranges': … , 'context': …}
_agg_by_key: defaultdict[tuple, dict] = defaultdict(dict)

#mode = ["context", "avg", "ranges"]
mode = ["avg"]
# book-keeping
files_checked = 0          # total *.json files opened successfully
rows_checked  = 0          # total rows iterated over

# Set up logging

logger = setup_logger()

errors: list[ErrorRecord] = []

RUNS = int(os.getenv("RUNS", 20))
condition_runs = {"context": RUNS, "avg": RUNS, "ranges": RUNS}
condition_files = {"context": 1, "avg": 35, "ranges": 11}
# Define the paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PARENT_DIR = BASE_DIR / "output"
OUTPUT  = BASE_DIR / "analysis" / "results"
OUTPUT_ANALYSIS_DIR = BASE_DIR / "analysis" / "results"

OUTPUT.mkdir(parents=True, exist_ok=True)
OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def completeness_analysis():
    """
    Check if all runs are complete for each model and concept.
    Saves the result into completeness_analysis_result.csv.
    """
    results = []

    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
            logger.info(f"Checking completeness for model: {model_dir.name}")
            for concept in model_dir.iterdir():
                if concept.is_dir():
                    num_files = len(list(concept.glob("*.json")))
                    if num_files != condition_files[experiment_type]:
                        warning_msg = (f"Incomplete files for {concept.name}: "
                                       f"expected {condition_files[experiment_type]}, found {num_files}")
                        logger.warning(f"{experiment_type}>{model_dir.name}: {warning_msg}")
                        results.append({
                            "model": model_dir.name,
                            "concept": concept.name,
                            "condition": experiment_type,
                            "warning": warning_msg
                        })

                    for file in concept.iterdir():
                        if file.suffix == ".json":
                            try:
                                data = pd.read_json(file)
                                runs = len(data)
                                if runs != condition_runs[experiment_type]:
                                    warning_msg = (f"Incomplete runs in {file.name}: "
                                                   f"expected {condition_runs[experiment_type]}, found {runs}")
                                    logger.warning(f"{experiment_type}>{model_dir.name}: {warning_msg}")
                                    results.append({
                                        "model": model_dir.name,
                                        "concept": concept.name,
                                        "condition": experiment_type,
                                        "warning": warning_msg
                                    })
                            except ValueError as e:
                                warning_msg = f"JSON parse error in {file.name}: {str(e)}"
                                logger.warning(f"{experiment_type}>{model_dir.name}: {warning_msg}")
                                results.append({
                                    "model": model_dir.name,
                                    "concept": concept.name,
                                    "condition": experiment_type,
                                    "warning": warning_msg
                                })
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_ANALYSIS_DIR / "completeness_analysis_result.csv", index=False)
        logger.info("Saved completeness analysis results to completeness_analysis_result.csv")
    else:
        logger.info("No completeness issues found. No CSV generated.")

def clean_up():
    """
    Remove folders that do not have any files in them.
    """
    folders_to_remove = []
    for experiment_type in mode:
        output_dir = OUTPUT_PARENT_DIR / experiment_type
        if not output_dir.exists():
            logger.error(f"Output directory {output_dir} does not exist.")
            continue

        for model_dir in output_dir.iterdir():
            logger.info(f"Checking completeness for model: {model_dir.name}")
            for concept in model_dir.iterdir():
                if concept.is_dir():
                    if not any(concept.iterdir()):
                        folders_to_remove.append(concept)
    
    for folder in folders_to_remove:
        logger.warning(f"Removing empty directory:{folder}")
        folder.rmdir()


def summarize_experiment_data(experiment_type):
    """
    Load the data for a specific experiment type.
    """
    output_dir = OUTPUT_PARENT_DIR / experiment_type
    if not output_dir.exists():
        logger.error(f"Output directory {output_dir} does not exist.")
        return pd.DataFrame()

    # Loop through all models and load their data
    for model_dir in output_dir.iterdir():
        logger.info(f"Loading data for model: {model_dir.name}")
        model_data = pd.DataFrame(columns=["model", "concept", "domain", "dimension", "measurement"] + ["run_" + str(i) for i in range(1, 21)])
        for concept in model_dir.iterdir():
            if concept.is_dir():
                logger.info(f"Loading data for concept: {concept.name}")
                for file in concept.glob("*.json"):
                    try:
                        data = pd.read_json(file)
                        if len(data) != condition_runs[experiment_type]:
                            logger.warning(f"Expected {condition_runs[experiment_type]} runs, but found {len(data)} in {file}.")

                        # model_data = pd.concat([model_data, data], ignore_index=True)
                    except ValueError as e:
                        logger.error(f"Error loading {file}: {e}")

def unwrap_value(v):
    """
    Flatten a value that might be nested JSON.

    Strategy (stop at first match):
        1.  if it's a JSON string, parse it
        2.  if dict → look for 'value' / 'avg' / 'mean'
        3.  if list  → recurse on the 1st element
        4.  else return as-is
    """
    # step 1 – string that *looks* like JSON
    if isinstance(v, str) and v.strip().startswith(('{', '[')):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return v          # leave as raw string

    # step 2 – dict
    if isinstance(v, dict):
        for k in ('value', 'avg', 'mean'):
            if k in v:
                return unwrap_value(v[k])
        # fallback: first numeric leaf in dict
        for _k, _v in v.items():
            if isinstance(_v, numbers.Number):
                return _v

    # step 3 – list/tuple
    if isinstance(v, (list, tuple)) and v:
        return unwrap_value(v[0])

    return v

def data_aggregation(dout: DataOutput):
    """
    Collect a DataOutput coming from semantic_check.

    • Keeps a flat list (`_agg_rows`) so you can dump it straight to CSV/JSONL.
    • Builds a keyed dict (`_agg_by_key`) so you can look up a concept quickly
      and see whether you already have the matching avg / range / context.
    """
    rec = asdict(dout)
    _agg_rows.append(rec)

    key = (
        dout.model,
        dout.concept,
        dout.domain,
        dout.dimension,
        dout.measurement,
    )
    _agg_by_key[key][dout.experiment_type] = dout.values

def dump_aggregated(out_dir: Path = OUTPUT):
    """
    Persist the aggregated clean data.

        clean_rows.csv            – every accepted row
        clean_rows.jsonl          – same in JSON Lines
        aggregated_by_key.json    – one object per (model, concept, …) key
    """
    if not _agg_rows:
        logger.info("No clean rows were gathered – nothing to dump.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. flat list -----------------------------------------------------------
    df = pd.DataFrame(_agg_rows)
    df.to_csv(out_dir / "clean_rows.csv", index=False)
    df.to_json(out_dir / "clean_rows.jsonl", orient="records", lines=True)

    # 2. keyed dict view  ----------------------------------------------------
    #    Convert tuple-keyed dict → list of objects that JSON can handle.
    serialisable = []
    for (model, concept, domain, dimension, measurement), vals in _agg_by_key.items():
        rec = dict(
            model=model,
            concept=concept,
            domain=domain,
            dimension=dimension,
            measurement=measurement,
            values=vals          # {'avg': …, 'ranges': …, 'context': …}
        )
        serialisable.append(rec)

    with open(out_dir / "aggregated_by_key.json", "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Aggregated %d clean rows across %d unique keys.",
        len(_agg_rows),
        len(_agg_by_key),
    )

def assemble_dictionary(concepts: list):
    custom_dict = ['average', 'avg', 'mean', 'value']
    out = []

    for w in concepts + custom_dict:
        if not isinstance(w, str) or not w.strip():
            continue
        w = w.strip().lower()
        out += [w, w.replace('_', ' ')]           # keep both
        if not w.endswith('s'):
            out += [w + 's', w.replace('_', ' ') + 's']

    # dedupe while preserving order
    seen, dedup = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            dedup.append(w)
    return dedup

def syntactic_check(response):
    return extract_json_from_string(response)


def load_yaml_dict(property_name):
    plural = property_name + "s"
    path = Path(__file__).resolve().parent.parent / "orka-properties" / f"{plural}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    return {k.lower(): v.lower() for k, v in mapping.items()}


def clean_with_yaml_row(response, domain):

    global unmatched_tokens_all
    p = inflect.engine()
    base = domain.strip().lower()
    mapping = load_yaml_dict(base)
    unmatched_counter = Counter()

    # 🧠 Extract and normalize the actual text
    try:
        # The thing is a dict like {'texture': ['smooth plastic', 'rubbery']}
        values = response.get(domain, [])
        if not isinstance(values, list):
            values = [values]
        text = ", ".join(values)
    except Exception as e:
        logger.error(f"[{base}] Invalid input format: {response}")
        return []

    # Actual cleaning logic
    text = text.lower()
    text = re.sub(r"[.;_/]", ",", text)
    text = re.sub(r"\s+", " ", text)

    raw_tokens = re.split(r"[,\n]", text)
    raw_tokens = [t.strip() for t in raw_tokens if t.strip()]

    cleaned = set()
    for phrase in raw_tokens:
        if phrase in mapping:
            cleaned.add(mapping[phrase])
            continue

        singular_phrase = " ".join([p.singular_noun(w) if p.singular_noun(w) else w for w in phrase.split()])
        if singular_phrase in mapping:
            cleaned.add(mapping[singular_phrase])
            continue

        words = phrase.split()
        matched = False
        for word in words:
            singular = p.singular_noun(word) if p.singular_noun(word) else word
            if singular in mapping:
                cleaned.add(mapping[singular])
                matched = True
        if not matched:
            for word in words:
                unmatched_counter[word] += 1
                unmatched_tokens_all[word + domain] += 1

    # Logging stats
    total_unmatched = sum(unmatched_counter.values())
    unique_unmatched = len(unmatched_counter)
    # if total_unmatched:
    #     logger.info(f"[{base}] Total unmatched entries: {total_unmatched}")
    #     logger.info(f"[{base}] Unique unmatched entries: {unique_unmatched}")
    #     top_unmatched = unmatched_counter.most_common(10)
    #     logger.info(f"[{base}] Top unmatched tokens: {top_unmatched}")

    return sorted(cleaned)



def semantic_check(response, row, experiment_type, file_path):
    """
    Validate `response`.  On success return (DataOutput, None);
    on failure return (None, <error_subtype>).
    """
    # ------------------------------------------------------------------
    # common meta
    meta = dict(
        experiment_type = experiment_type,
        model           = row.get("client"),
        concept         = row.get("concept"),
        file_path       = file_path,
        row_idx         = row.name,
        domain          = row.get("domain") if experiment_type != 'context' else '',
        dimension       = row.get("dimension"),
        measurement     = row.get("measurement"),
    )

    # ==============================================================
    # 1 · AVG
    # --------------------------------------------------------------
    if experiment_type == "avg":
        domain = row.get("domain")
        if isinstance(response, list):
            if not row.get("measurement"):
                if domain in {'colour', 'shape', 'material', 'location', 'disposition', 'pattern', 'texture'}:
                    clean_response = [str(x).strip().lower() for x in response if str(x).strip()]
                    data_out = DataOutput(**meta, values=clean_response)
                    return data_out, None
                else:
                    return None, "Lists of JSONs"
            else:
                return None, "Lists of JSONs"

        try:
            keys = list(response.keys())
            values = list(response.values())
        except AttributeError:
            return None, "Lists of JSONs"

        # ---- pick the key ---------------------------------------------------
        if not values or all(v is None for v in values):
            return None, "Response is None"
        
        # ---- Processing categorical values ----------------------------
        if not row.get("measurement"):
            if domain in {'function', 'scenario', 'context'}:
                return None, 'Skipped domain'

            elif domain in {'colour', 'shape', 'material', 'location', 'disposition', 'pattern', 'texture'}:
                clean_response = clean_with_yaml_row(response, domain)
                data_out = DataOutput(**meta, values=clean_response)
                return data_out, None

            elif domain == "rigidity":
                val = response.get("rigidity")
                if isinstance(val, list):
                    if len(val) == 0:
                        return None, "Empty rigidity value"
                    val = str(val[0]).strip().lower()
                else:
                    val = str(val).strip().lower()

                allowed = {"rigid", "flexible", "soft"}
                if val not in allowed:
                    return None, "Incorrect rigidity value"

                data_out = DataOutput(**meta, values=val)
                return data_out, None

            elif domain == "fragility":
                val = response.get("fragility")
                if isinstance(val, list):
                    if len(val) == 0:
                        return None, "Empty fragility value"
                    val = str(val[0]).strip().lower()
                else:
                    val = str(val).strip().lower()

                allowed = {"fragile", "sturdy"}
                if val not in allowed:
                    return None, "Incorrect fragility value"

                data_out = DataOutput(**meta, values=val)
                return data_out, None

            else:
                return None, "Unrecognized domain"
        # ---- Processing measurements ----------------------------
        if len(keys) != 1:
            domains_variants = assemble_dictionary(
                [row.get("domain"), row.get("dimension"),
                row.get("measurement"), row.get("concept")]
            )
            key_candidates = [k for k in keys
                            if k.strip().lower().replace("_", " ") in domains_variants]

            if len(key_candidates) == 1:
                target_key = key_candidates[0]
            elif "answer" in keys:
                target_key = "answer"
            else:
                return None, "Too many keys" if key_candidates else "Incorrect key name"
        else:
            target_key = keys[0]

        # ---- validate value -------------------------------------------------
        val = unwrap_value(response[target_key])
        try:
            float(val)
            
        except (ValueError, TypeError):
            return None, "Incorrect data type"

        data_out = DataOutput(**meta, values=float(val))
        return data_out, None

    # ------------------------------------------------------------------
    else:
        return None, "Unknown experiment_type"

def factual_check(response):
    pass

def analyse(exp):
    out_dir = OUTPUT_PARENT_DIR / exp
    if not out_dir.exists():
        logger.error("Missing directory %s", out_dir); return

    for model_dir in out_dir.iterdir():
        for concept_dir in model_dir.iterdir():
            for f in concept_dir.glob("*.json"):
                try:
                    df = pd.read_json(f)
                except ValueError as e:
                    errors.append(ErrorRecord(
                        exp, model_dir.name, concept_dir.name, f.name, str(f), -1,
                        "syntax", "file_not_json", str(e), ""
                    ))
                    continue

                global files_checked, rows_checked
                files_checked += 1
                rows_checked  += len(df)

                # completeness of runs
                # if len(df) != condition_runs[exp]:
                    # errors.append(ErrorRecord(
                        # exp, model_dir.name, concept_dir.name, f.name, str(f), -1,
                        # "other", "incomplete_runs",
                        # f"found {len(df)} rows, expected {condition_runs[exp]}",
                        # ""
                    #))

                # row-level checks
                for idx, row in df.iterrows():
                    raw = row["response"]
                    
                    parsed = syntactic_check(raw)
                    if parsed is None:
                        errors.append(ErrorRecord(
                            exp, model_dir.name, concept_dir.name, f.name, str(f), idx,
                            "syntax", "invalid_json", "could not parse", raw[:800]
                        ))
                        continue

                    response, err = semantic_check(parsed, row, exp, str(f))
                    if response is None:          # semantic_check returns None on failure
                        errors.append(ErrorRecord(
                            exp, model_dir.name, concept_dir.name, f.name, str(f), idx,
                            "semantic", err, f"key/value mismatch: {err}", str(parsed)[:800]
                        ))
                        continue

                    data_aggregation(response)

def summarize_error_stats(error_file: str | Path,
                          condition: str,
                          total_files: int | None = None,
                          total_rows:  int | None = None):
    """
    Summarise the frequency of error categories and sub-types
    from the consolidated error log.

    Parameters
    ----------
    error_file : str | Path
        Path to error_summary.csv or error_summary.jsonl

    Returns
    -------
    tuple(dict, pandas.DataFrame)
        (category_counts, sub_type_counts_df)
    """
    error_file = Path(error_file)

    # --- Load -----------------------------------------------------------------
    if error_file.suffix == ".csv":
        df = pd.read_csv(error_file)
    elif error_file.suffix in {".jsonl", ".json"}:
        df = pd.read_json(error_file, lines=True)
    else:
        raise ValueError(f"Unsupported file type: {error_file.suffix}")

    if df.empty:
        logger.info("No rows found in %s – nothing to summarise.", error_file)
        return {}, pd.DataFrame()
    
    # --- Aggregate ------------------------------------------------------------
    # how many files had ≥1 error
    err_files = df["file"].nunique()
    err_rows  = len(df)

    category_counts = df["error_category"].value_counts().to_dict()

    sub_type_counts = (
        df.groupby(["error_category", "error_subtype"])
          .size()
          .reset_index(name="count")
          .sort_values(["error_category", "count"], ascending=[True, False])
    )
    # if grand totals were given, compute ratios
    file_ratio = f"{err_files}/{total_files}  ({err_files/total_files:.1%})" \
                 if total_files else f"{err_files} (total ?)"
    row_ratio  = f"{err_rows}/{total_rows}   ({err_rows/total_rows:.1%})" \
                 if total_rows  else f"{err_rows} (total ?)"
    
    # --- Print ----------------------------------------------------------------
    print(f"\n=== Error summary: {condition} ===")
    print(f"Files with errors : {file_ratio}")
    print(f"Rows  with errors : {row_ratio}")

    print(f"\n--- Error category counts: {condition}---")
    for cat, n in category_counts.items():
        print(f"{cat:<10} : {n}")

    print(f"\n--- Error sub-type counts {condition}---")
    for _, row in sub_type_counts.iterrows():
        print(f"{row.error_category:<10} > {row.error_subtype:<25} : {row['count']}")

    return category_counts, sub_type_counts

def dump_errors(err_list: list[ErrorRecord], exp):
    if not err_list:
        logger.info("✅ No errors found.")
        return

    df = pd.DataFrame([asdict(e) for e in err_list])
    df.to_csv(OUTPUT / f"error_summary_{exp}.csv", index=False)
    df.to_json(OUTPUT / f"error_summary_{exp}.json", orient="records", lines=True)
    logger.info("Wrote %d error rows to error_summary.*", len(df))

if __name__ == "__main__":
    conditions = ['avg']
    for exp in conditions:
        errors.clear()
        files_checked = 0
        rows_checked  = 0

        analyse(exp)

        dump_errors(errors, exp)
        error_file = OUTPUT / f"error_summary_{exp}.csv"
if error_file.exists():
    summarize_error_stats(
            error_file,
            condition=exp,
            total_files=files_checked,
            total_rows=rows_checked
        )
else:
    logger.info(f"No error file for {exp}, skipping error summary.")
    
logger.info(f"Collected clean rows: {len(_agg_rows)}")
logger.info(f"Collected unique keys: {len(_agg_by_key)}")
dump_aggregated(OUTPUT)
logger.info(f"🔎 GLOBAL unmatched stats across all domains:")
logger.info(f"    Total unmatched: {sum(unmatched_tokens_all.values())}")
logger.info(f"    Unique unmatched: {len(unmatched_tokens_all)}")
logger.info(f"    Top offenders: {unmatched_tokens_all.most_common(20)}")
