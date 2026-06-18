import pandas as pd
import os
import glob
from pathlib import Path
import json
from collections import defaultdict, Counter
from utils.logger import setup_logger
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, friedmanchisquare, wilcoxon, chi2_contingency, hypergeom, mannwhitneyu, ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from itertools import combinations
import scikit_posthocs as sp
import csv
import logging
import ast



logger = setup_logger(level=logging.INFO)

COLUMNS = ['model_name', 'concept', 'domain', 'measurement', 'dimension', 'response']
INPUTS = Path(__file__).parent.parent / "inputs"
EXTRACTED_FOLDER = Path(__file__).parent.parent / "data" / "extracted_knowledge"
RESULTS_FOLDER = Path(__file__).parent.parent / "data" / "results"
GRAPH_FOLDER = Path(__file__).parent.parent / "graphs"
PARSED_FOLDER = Path(__file__).parent.parent / "data" / "parsed"
SUMMARY_FOLDER = Path(__file__).parent.parent / "logs" / "summaries" 
SUMMARY_FOLDER.mkdir(parents=True, exist_ok=True) 

# ============================
# Utility Functions
# ----------------------------

def plot_mae_barcharts(mae_df):
    output_dir = GRAPH_FOLDER / "mae"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving unit-based MAE charts to: {output_dir}")

    # ==== Overall Performance per Unit (All concepts combined) ====
    for metric in ["mean_absolute_error", "mean_normalized_error"]:
        if metric not in mae_df.columns:
            continue
        metric_name = "MAE" if "absolute" in metric else "Normalized Error"
        metric_label = metric.replace("_", " ").capitalize()

        overall = mae_df.groupby(["measurement", "model_name"])[metric].mean().reset_index()

        plt.figure(figsize=(12, 6))
        sns.barplot(data=overall, x="measurement", y=metric, hue="model_name", palette="Set2")
        plt.title(f"{metric_name} per Unit (All Concepts)")
        plt.ylabel(metric_label)
        plt.xlabel("Unit")
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        path = output_dir / f"{metric}_per_unit.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved chart: {path.name}")

    # ==== Per-Concept, Per-Unit ====
    for concept in mae_df["concept"].unique():
        concept_df = mae_df[mae_df["concept"] == concept]
        for metric in ["mean_absolute_error", "mean_normalized_error"]:
            if metric not in concept_df.columns:
                continue
            metric_name = "MAE" if "absolute" in metric else "Normalized Error"
            metric_label = metric.replace("_", " ").capitalize()

            grouped = concept_df.groupby(["measurement", "model_name"])[metric].mean().reset_index()

            plt.figure(figsize=(12, 6))
            sns.barplot(data=grouped, x="measurement", y=metric, hue="model_name", palette="Dark2")
            plt.title(f"{metric_name} for Concept '{concept}'")
            plt.ylabel(metric_label)
            plt.xlabel("Unit")
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            fname = f"{metric}_{concept.replace(' ', '_')}_per_unit.png"
            path = output_dir / fname
            plt.savefig(path)
            plt.close()
            logger.info(f"Saved concept chart: {path.name}")

def combine_extracted_knowledge():
    df = pd.DataFrame(columns=COLUMNS)
    for file in EXTRACTED_FOLDER.glob('*.csv'):
        file_pd = pd.read_csv(file)
        df = pd.concat([df, file_pd], ignore_index=True)
    return df

def mean_average_error(df):
    # Filter ground truth
    units = {'length' : ["feet", "centimeters", "millimeters", "inches", "meters"],
            'width' : ["feet", "centimeters", "millimeters", "inches", "meters"],
            'height' : ["feet", "centimeters", "millimeters", "inches", "meters"],
            'weight' : ['kilograms', 'grams', 'pounds', 'ounces'],
            'temperature' : ['kelvin', 'fahrenheit', 'celsius']}
    df_gt = df[df["model_name"] == "mturk"]
    concepts = set(df['concept'])
    mae_columns = df.columns.tolist() + [
        f'mean_{dimension}_rel_error_{unit}'
        for dimension in units.keys()
        for unit in units[dimension]
    ]

    mae_df = pd.DataFrame(columns=mae_columns)
    for dimension in units.keys():
        df_results = pd.DataFrame(columns=['concept', 'model_name', 'dimension'] + [f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]])
        for concept in concepts:
            concept_df = df[df['concept'] == concept]
            gt_rows = df_gt[df_gt['concept'] == concept]
            for model_name in concept_df['model_name'].unique():
                if model_name == 'mturk':
                    continue  # Skip GT
                model_df = concept_df[concept_df['model_name'] == model_name]
                error_dict = {'concept': concept, 'model_name': model_name, 'dimension': dimension}

                for unit in units[dimension]:
                    # pred_values = model_df[f'{dimension}_{unit}'].dropna().values
                    pred_values = model_df[(model_df['dimension'] == dimension) & (model_df['measurement'] == unit)]['response'].dropna().values
                    gt_value = gt_rows[(gt_rows['dimension'] == dimension) & (gt_rows['measurement'] == unit)]['response'].dropna().values
                    gt_value = pd.to_numeric(gt_value, errors="coerce")
                    gt_value = gt_value[~np.isnan(gt_value)]  # remove NaNs if any remain
                    pred_values = pd.to_numeric(pred_values, errors="coerce")
                    pred_values = pred_values[~np.isnan(pred_values)]  # remove NaNs if any remain                    
                    logger.debug(f"Examining {dimension} for {model_name} and {concept} across {unit}")
                    if len(gt_value) == 0 or len(pred_values) == 0:
                        error_mre = np.nan
                        error_mae = np.nan
                        
                    else:
                        error_mre = np.mean(np.abs(pred_values - np.mean(gt_value)) / np.mean(gt_value))
                        gt_ref = np.mean(gt_value)
                        error_mae = np.mean(np.abs(pred_values - gt_ref))
                        mae_std = np.std(np.abs(pred_values - gt_ref))
                    error_dict[f'mean_{dimension}_rel_error_{unit}'] = error_mre
                    error_dict[f'mean_{dimension}_abs_error_{unit}'] = error_mae
                    error_dict[f'mean_{dimension}_abs_error_std_{unit}'] = mae_std
                df_results = pd.concat([df_results, pd.DataFrame([error_dict])], ignore_index=True)     
        df_results.to_csv(str(RESULTS_FOLDER / f'df_results_{dimension}.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
        unit_cols = [f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]]
        check_normality(df_results, unit_cols)
        friedman_test_units(df_results, f"mean_{dimension}_rel_error", units[dimension])
        posthoc_nemenyi_units(df_results, f"mean_{dimension}_rel_error", units[dimension])
        median_errors = df_results[[f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]]].median().round(3)
        mean_errors = df_results[[f'mean_{dimension}_rel_error_{unit}' for unit in units[dimension]]].mean().round(3)
        print(median_errors.sort_values())
        print(mean_errors.sort_values())
        mae_df = pd.concat([mae_df,df_results], ignore_index=True )
    return mae_df


def check_normality(df, unit_cols):
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        
        for unit_col in unit_cols:
            values = model_df[unit_col].dropna().values
            if len(values) < 3:
                continue  # Shapiro needs at least 3 data points

            stat, p = shapiro(values)
            if p > 0.005:
                print(f'Model: {model}, Unit: {unit_col}, Shapiro p-value: {p:.4f}')

def friedman_test_units(df, metric_prefix, units):
    unit_cols = [f'{metric_prefix}_{unit}' for unit in units]
    
    # Drop rows with any NaNs across the unit columns
    clean_df = df.dropna(subset=unit_cols)
    
    # Collect error arrays per unit
    data = [clean_df[col].values for col in unit_cols]
    k = len(unit_cols)
    stat, p = friedmanchisquare(*data)
    print(f"Friedman test for units in {metric_prefix}: stat={stat:.4f}, p-value={p:.4e}")
    
    if p < 0.05:
        print(f"⇒ Significant differences between units: χ²(df={k - 1}) = {stat:.2f}, p < {p:.4f}")
    else:
        print("⇒ No significant difference between units.") 

def posthoc_nemenyi_units(df, metric_prefix, units):
    unit_cols = [f'{metric_prefix}_{unit}' for unit in units]
    clean = df.dropna(subset=unit_cols)
    # Data matrix: rows=concepts, cols=units
    matrix = clean[unit_cols].values  # shape: (n_concepts x n_units)
    
    # Run Nemenyi post-hoc for Friedman
    p_matrix = sp.posthoc_nemenyi_friedman(matrix)
    p_matrix.index = units
    p_matrix.columns = units
    
    print("Pairwise p-values (Nemenyi post-hoc):")
    print(p_matrix.round(3))
    
    # Optional: significance plot
    sp.sign_plot(p_matrix, annot=True)


def is_evaluated_list(x):
    try:
        return isinstance(ast.literal_eval(x), list)
    except (ValueError, SyntaxError, TypeError):
        return False

def calculate_majority_votes(df, majority=0.6):
    df_models_clean = pd.DataFrame(columns=['model_name', 'concept', 'domain', 'response'])

    # Filter for categorical domains
    categorical = set(df[df['measurement'].isna()]['domain'])


    for model in df['model_name'].unique():
        df_model = df[df['model_name'] == model]
        
        for concept in df_model['concept'].unique():
            df_concept = df_model[df_model['concept'] == concept]
            
            for domain in categorical:  
                df_domain = df_concept[df_concept['domain'] == domain]

                # Row-level unique counting
                response_counts = Counter()

                for response in df_domain['response']:
                    try:
                        response_evaled = ast.literal_eval(response)
                    except (ValueError, SyntaxError, TypeError):
                        pass
                    if isinstance(response_evaled, list):
                        for label in set(response_evaled):
                            response_counts[label] += 1
                majority = len(df_domain) / 2
                ultimate_list = [label for label, c in response_counts.items() if c >= majority]

                if ultimate_list:
                    df_models_clean = pd.concat([df_models_clean, pd.DataFrame([{
                        'model_name': model,
                        'concept': concept,
                        'domain': domain,
                        'response': ultimate_list
                    }])], ignore_index=True)

    df_models_clean.to_csv(str(RESULTS_FOLDER / 'voted_categorical_values.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    return df_models_clean


def save_catscores_to_csv(score_dict):
    """
    Flatten a nested dictionary of the form:
    {model: {concept: {dimension: {unit: [values]}}}} and save to CSV.

    Parameters:
        score_dict (dict): Nested dictionary with scores.
        output_path (str): File path to save the CSV to.
    """
    rows = []
    for model, concept_dict in score_dict.items():
        for concept, dim_dict in concept_dict.items():
            for dim, values in dim_dict.items():
                for val in values:
                    rows.append({
                        "model": model,
                        "concept": concept,
                        "dimension": dim,
                        "unit": '',
                        "value": val
                    })
    df = pd.DataFrame(rows)
    df.to_csv(str(RESULTS_FOLDER / 'CAT_RAW_SUMMARY.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    return df  # optionally return for inspection

def save_numscores_to_csv(score_dict):
    """
    Flatten a nested dictionary of the form:
    {model: {concept: {dimension: {unit: [values]}}}} and save to CSV.

    Parameters:
        score_dict (dict): Nested dictionary with scores.
        output_path (str): File path to save the CSV to.
    """
    rows = []
    for model, concept_dict in score_dict.items():
        for concept, dim_dict in concept_dict.items():
            for dim, unit_dict in dim_dict.items():
                for unit, values in unit_dict.items():
                    for val in values:
                        rows.append({
                            "model": model,
                            "concept": concept,
                            "dimension": dim,
                            "unit": unit,
                            "value": val
                        })
    df = pd.DataFrame(rows)
    df.to_csv(str(RESULTS_FOLDER / 'NUM_RAW_SUMMARY.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    return df  # optionally return for inspection

# ============================
# Analysis Functions
# ----------------------------

def analyze_measurements(df_complete):
    logger.debug(f"Staring the analysis of measurement units...")
    return mean_average_error(df_complete)

def analyze_performance(df, mae_df):
    units = {'length' : ["feet", "centimeters", "millimeters", "inches", "meters"],
            'width' : ["feet", "centimeters", "millimeters", "inches", "meters"],
            'height' : ["feet", "centimeters", "millimeters", "inches", "meters"],
            'weight' : ['kilograms', 'grams', 'pounds', 'ounces'],
            'temperature' : ['kelvin', 'fahrenheit', 'celsius']}
    numerical_units = ["feet", "centimeters", "millimeters", "inches", "meters", 'kilograms', 'grams', 'pounds', 'ounces', 'kelvin', 'fahrenheit', 'celsius']
    df_majority = calculate_majority_votes(df)
    df_gt_clean = df_majority[df_majority['model_name'] == 'mturk']
    models = [model for model in df['model_name'].unique() if model != 'mturk']
    concepts = [concept for concept in df['concept'].unique()]
    categories = set(df[df['measurement'].isna()]['domain'])
    numerical = set(df[df['measurement'].notna()]['dimension'])
    model_num_score = {model : {concept: 0 for concept in concepts} for model in models}
    model_cat_score = {model : {concept: 0 for concept in concepts} for model in models}
    model_raw_cat_scores = {model : {concept: {dim : [] for dim in categories} for concept in concepts} for model in models}
    model_raw_num_scores = {model : {concept: {dim : {unit : [] for unit in numerical_units} for dim in numerical} for concept in concepts} for model in models}
    jaccard = lambda pred, gold: len(set(pred) & set(gold)) / len(set(pred) | set(gold)) if pred or gold else 1.0
    models_final_num_scores = {model: 0 for model in models}
    models_final_cat_scores = {model: 0 for model in models}
    for model in models:
        df_model = df_majority[df_majority['model_name'] == model]
        for concept in concepts:

            logger.debug("Calculating categorical scores")

            cat_scores = []
            for cat in categories:
                G_cd = df_gt_clean.loc[(df_gt_clean['concept'] == concept) & (df_gt_clean['domain'] == cat), 'response']
                P_mcd = df_model.loc[(df_model['concept'] == concept) & (df_model['domain'] == cat), 'response']
                G_cd_flattened = [item for resp in G_cd if isinstance(resp, list) for item in resp]
                P_mcd_flattened = [item for resp in P_mcd if isinstance(resp, list) for item in resp]

                if G_cd_flattened or P_mcd_flattened:  # skip totally empty comparisons
                    score = jaccard(P_mcd_flattened, G_cd_flattened)
                    cat_scores.append(score)
                    model_raw_cat_scores[model][concept][cat].append(score)
                    logger.debug(f"Model: {model}, Concept: {concept}, Domain: {cat}")
                    logger.debug(f"Gold: {set(G_cd_flattened)}")
                    logger.debug(f"Pred: {set(P_mcd_flattened)}")
                    logger.debug(f"Jaccard: {score:.3f}")   
            if cat_scores:
                model_cat_score[model][concept] = sum(cat_scores) / len(cat_scores)
            else:
                model_cat_score[model][concept] = None  # or 0 or float('nan') or whatever existential void you prefer
            logger.debug("Calculating numerical scores")
            num_scores = []
            for num_domain in numerical:
                domain_df = mae_df[mae_df['dimension'] == num_domain]
                for col in domain_df.columns:
                    if num_domain in col:
                        logger.debug(f"Model: {model} Concept {concept} Domain: {num_domain} Column: {col}")
                        
                        mean_dif = domain_df.loc[(domain_df['concept'] == concept) & (domain_df['model_name'] == model), col].tolist()[0]
                        e_t = 1/(1 + abs(mean_dif))
                        num_scores.append(e_t)
                        unit = col.split('_')[-1]
                        model_raw_num_scores[model][concept][num_domain][unit].append(e_t)
                        logger.debug(f"\te: {e_t}")
            if num_scores:
                model_num_score[model][concept] = sum(num_scores) / len(num_scores)
                logger.info(f"\tFinal num score for {model} and {concept}: { model_num_score[model][concept]}")
            else:
                model_num_score[model][concept] = None
            logger.info(f"\tFinal num scores for {model} : { model_num_score[model]}")

        models_final_num_scores[model] = np.nanmean(list(model_num_score[model].values()))
        models_final_cat_scores[model] = sum(model_cat_score[model].values()) / len(model_cat_score[model])
        logger.info(models_final_num_scores)
    save_numscores_to_csv(model_raw_num_scores)
    save_catscores_to_csv(model_raw_cat_scores)
    df_cat_model_score = pd.DataFrame(list(model_cat_score.items()), columns=['concept', 'num_score'])
    df_cat_model_score.to_csv(str(RESULTS_FOLDER /f'models_cat_scores.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    df_num_model_score = pd.DataFrame(list(model_num_score.items()), columns=['concept', 'num_score'])
    df_num_model_score.to_csv(str(RESULTS_FOLDER /f'models_num_scores.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    df_cat_scores = pd.DataFrame(models_final_cat_scores.items(), columns=['model_name', 's_cat'])
    df_cat_scores.to_csv(str(RESULTS_FOLDER /'cat_scores.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    df_num_scores = pd.DataFrame(models_final_num_scores.items(), columns=['model_name', 's_num'])
    df_num_scores.to_csv(str(RESULTS_FOLDER /'num_scores.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    return df_majority


def calculate_exact_matches():
    cat_scores = pd.read_csv(str(RESULTS_FOLDER / 'CAT_RAW_SUMMARY.csv'))
    df_ones = cat_scores[cat_scores['value'] == 1.0]
    models = set(df_ones['model'])
    for model in models:
        df_model = df_ones[df_ones['model'] == model]
        logger.info(f"{model} has {df_model['value'].sum()/400} exact matches")
    perfect_by_model = df_ones['model'].value_counts().reset_index()
    perfect_by_model.columns = ['model', 'num_perfect_predictions']

    perfect_by_concept = df_ones['concept'].value_counts().reset_index()
    perfect_by_concept.columns = ['concept', 'num_perfect_predictions']

    perfect_by_dimension = df_ones['dimension'].value_counts().reset_index()
    perfect_by_dimension.columns = ['dimension', 'num_perfect_predictions']

    print(df_ones[['model', 'concept', 'dimension']])
    df_ones.to_csv(str(RESULTS_FOLDER /'exact_matches.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)

def proportions_z_test(successes, total):
        stat, pval = proportions_ztest(successes, total)
        logger.info(f"\tZ-test statistic: {stat:.4f}, p-value: {pval:.6f}")
        if pval < 0.05:
            logger.info("\t\t➤ Significant difference in success rates.")
        else:
            logger.info("\t\t➤ No significant difference in success rates.")

def wilcoxon_test(model1_scores, model2_scores):
    paired = [(x, y) for x, y in zip(model1_scores, model2_scores) if not (np.isnan(x) or np.isnan(y))]

    if len(paired) < 2:
        logger.warning("Too few valid data points after removing NaNs — skipping Wilcoxon test.")
    else:
        m1_clean, m2_clean = zip(*paired)
        
        if np.allclose(m1_clean, m2_clean, equal_nan=True):
            logger.info("All cleaned numerical scores are equal — skipping Wilcoxon test.")
        else:
            try:
                stat, p = wilcoxon(m1_clean, m2_clean)
                logger.info(f"\tWilcoxon statistic: {stat:.2e}, p-value: {p:.6f}")
                if p < 0.05:
                    logger.info("\t\t➤ Significant difference in numerical performance.")
                else:
                    logger.info("\t\t➤ No significant difference in numerical performance.")
            except ValueError as e:
                logger.warning(f"Wilcoxon test failed: {e}")

def friedman_test(*model_score_lists, model_names=None):
    # Filter out rows (i.e., test cases) with any NaNs
    zipped = list(zip(*model_score_lists))
    cleaned = [row for row in zipped if not any(pd.isna(x) for x in row)]

    if len(cleaned) < 2:
        logger.warning("Too few valid data points after removing NaNs — skipping Friedman test.")
        return

    try:
        # Transpose back into separate model lists
        cleaned_lists = list(zip(*cleaned))
        stat, p = friedmanchisquare(*cleaned_lists)
        logger.info(f"\tFriedman statistic: {stat:.2e}, p-value: {p:.6f}")

        if p < 0.05:
            logger.info("\t\t➤ Significant difference among models.")
        else:
            logger.info("\t\t➤ No significant difference among models.")

        if model_names:
            logger.info(f"\tModels compared: {', '.join(model_names)}")

    except ValueError as e:
        logger.warning(f"Friedman test failed: {e}")

def proportions_chi2_test(successes, totals, model_names=None):
    """
    successes: list of success counts for each model
    totals: list of total counts for each model
    model_names: optional list of model names
    """
    failures = [t - s for s, t in zip(successes, totals)]
    contingency_table = [successes, failures]

    stat, pval, dof, expected = chi2_contingency(contingency_table)

    logger.info(f"\tChi-squared statistic: {stat:.4f}, p-value: {pval:.6f}, df: {dof}")
    if pval < 0.05:
        logger.info("\t\t➤ Significant difference in success rates across models.")
    else:
        logger.info("\t\t➤ No significant difference in success rates across models.")

    if model_names:
        logger.info(f"\tModels compared: {', '.join(model_names)}")

def compare_models(models: list):
    logger.info(f"Starting model comparison for: {models}")

    # Load CSVs
    num_scores = pd.read_csv(str(RESULTS_FOLDER / 'NUM_RAW_SUMMARY.csv'))
    cat_scores = pd.read_csv(str(RESULTS_FOLDER / 'CAT_RAW_SUMMARY.csv'))
    syn_errors = pd.read_csv(str(SUMMARY_FOLDER / 'syntax_summary.csv'))
    sem_errors = pd.read_csv(str(SUMMARY_FOLDER / 'semantic_summary.csv'))
    logger.info("Loaded all result CSVs.")

    model_stats = {}

    for model in models:
        stats = {}

        # Numerical scores
        num_vals = num_scores[num_scores['model'] == model]['value'].dropna().tolist()
        stats['numerical'] = num_vals
        logger.info(f"[{model}] Numerical scores: {num_vals}")

        # Categorical scores
        cat_vals = cat_scores[cat_scores['model'] == model]['value'].dropna().tolist()
        stats['categorical'] = cat_vals
        logger.info(f"[{model}] Categorical scores: {cat_vals}")

        # Syntax valid parses
        syn_val = syn_errors[syn_errors['model'] == model]['valid'].tolist()
        if syn_val:
            stats['syntax_valid'] = syn_val[0]
            logger.info(f"[{model}] Valid syntax parses: {syn_val[0]}")
        else:
            stats['syntax_valid'] = None
            logger.warning(f"[{model}] No syntax parse info found.")

        # Semantic valid parses
        sem_val = sem_errors[sem_errors['model'] == model]['valid'].tolist()
        if sem_val:
            stats['semantic_valid'] = sem_val[0]
            logger.info(f"[{model}] Valid semantic parses: {sem_val[0]}")
        else:
            stats['semantic_valid'] = None
            logger.warning(f"[{model}] No semantic parse info found.")

        model_stats[model] = stats

    if len(models) == 2:
        logger.info("Performing pairwise comparison.")

        m1, m2 = models[0], models[1]

        # Syntax
        logger.info(f"[{m1}] Valid parses: {model_stats[m1]['syntax_valid']}")
        logger.info(f"[{m2}] Valid parses: {model_stats[m2]['syntax_valid']}")
        logger.info("Running proportions z-test on syntax parsing rates:")
        proportions_z_test(
            [model_stats[m1]['syntax_valid'], model_stats[m2]['syntax_valid']],
            [43200, 43200]
        )

        # Semantic
        logger.info("Running proportions z-test on semantic parsing rates:")
        proportions_z_test(
            [model_stats[m1]['semantic_valid'], model_stats[m2]['semantic_valid']],
            [43200, 43200]
        )

        # Numerical scores
        logger.info("Running Wilcoxon signed-rank test on numerical scores:")
        wilcoxon_test(
            model_stats[m1]['numerical'],
            model_stats[m2]['numerical']
        )

        # Categorical scores
        logger.info("Running Wilcoxon signed-rank test on categorical scores:")
        wilcoxon_test(
            model_stats[m1]['categorical'],
            model_stats[m2]['categorical']
        )
    else:
        logger.info("Performing multi-model comparison.")

        # Extract score lists from model_stats
        num_scores_all = [model_stats[m]['numerical'] for m in models]
        cat_scores_all = [model_stats[m]['categorical'] for m in models]
        syntax_valid_all = [model_stats[m]['syntax_valid'] for m in models]
        semantic_valid_all = [model_stats[m]['semantic_valid'] for m in models]

        # Friedman test (Numerical)
        logger.info("Running Friedman test on numerical scores:")
        friedman_test(*num_scores_all, model_names=models)

        # Friedman test (Categorical)
        logger.info("Running Friedman test on categorical scores:")
        friedman_test(*cat_scores_all, model_names=models)

        # Chi-squared proportions test (Syntax)
        logger.info("Running Chi-squared test on syntax parsing rates:")
        proportions_chi2_test(syntax_valid_all, [43200] * len(models))
        
        # Chi-squared proportions test (Semantic)
        logger.info("Running Chi-squared test on semantic parsing rates:")
        proportions_chi2_test(semantic_valid_all, [43200] * len(models))

        logger.info("Running Pair-wise combinations on semantic errors:")
        for m1, m2 in combinations(models, 2):
            count = [model_stats[m1]['semantic_valid'], model_stats[m2]['semantic_valid']]
            nobs = [43200, 43200]
            stat, pval = proportions_ztest(count, nobs)
            print(f"{m1} vs {m2}: z={stat:.2f}, p={pval:.4f}")

        logger.info("Running Pair-wise combinations on synxat errors:")
        for m1, m2 in combinations(models, 2):
            count = [model_stats[m1]['syntax_valid'], model_stats[m2]['syntax_valid']]
            nobs = [43200, 43200]
            stat, pval = proportions_ztest(count, nobs)
            print(f"{m1} vs {m2}: z={stat:.2f}, p={pval:.4f}")

        long_data = []
        for model, scores in zip(models, num_scores_all):
            for task_id, score in enumerate(scores):
                long_data.append({
                    'model': model,
                    'score': score,
                    'task': task_id  # or concept/dimension name if available
                })

        # Step 2: Convert to DataFrame
        df_scores = pd.DataFrame(long_data)
        num_before = len(df_scores)
        df_scores['score'] = pd.to_numeric(df_scores['score'], errors='coerce')
        df_scores = df_scores.dropna(subset=['score'])
        num_after = len(df_scores)

        df_scores['model'] = df_scores['model'].astype(str)
        df_scores['task'] = df_scores['task'].astype(str)
        logger.info(f"Dropped {num_before - num_after} rows with non-numeric scores.")
        # Step 3: Run Nemenyi post-hoc test
        # p_values_matrix = sp.posthoc_nemenyi_friedman(df_scores, y_col='score', group_col='model', block_col='task')



def analyze_context():

    def categorical_context():
        votes = pd.read_csv(RESULTS_FOLDER / 'voted_categorical_values.csv')
        df = pd.read_csv(RESULTS_FOLDER / "data_summed.csv")
        concepts = set(df['concept'])
        cat_domains = set(df[df['measurement'].isna()]['domain'])
        models = set(df['model_name'])
        all_combinations = len([f"{d} of a {c}" for d in cat_domains for c in concepts])
        missing_dimensions = {model : [f"{d} of a {c}" for d in cat_domains for c in concepts] for model in models}
        for model in models:
            model_df = votes[votes['model_name'] == model]
            for index, row in model_df.iterrows():
                found_concept_domain = f"{row['domain']} of a {row['concept']}"
                logger.info(f"[{model}] Found {found_concept_domain}")
                missing_dimensions[model].remove(found_concept_domain)
        # print(cat_domains)
        # Missing values per domain
        # Check overlap
        for model in models:
            # Parameters
            M = all_combinations   # total number of possible items (e.g., all dimensions)
            K = len(missing_dimensions['mturk'])    # number of ground truth relevant dimensions
            n = len(missing_dimensions[model])   # number of predicted dimensions
            x = len(set(missing_dimensions['mturk']) & set(missing_dimensions[model]))   # observed overlap

            # P(X >= x): probability of getting x or more overlaps by chance
            p_value = hypergeom.sf(x - 1, M, K, n)  # sf = 1 - cdf(x - 1)
            logger.info(f"Checking {model}")
            logger.info(f"\tObserved overlap: {x}")
            logger.info(f"\tp-value: {p_value:.4f}")
            pass

    def numerical_context_avg_temp():
        df = pd.read_csv(RESULTS_FOLDER / "NUM_RAW_SUMMARY.csv")

        results = []
        stats = {}
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            stats[model] = 0
            for concept in model_df['concept'].unique():
                subset = model_df[model_df['concept'] == concept]
                
                temp_errors = subset[subset['dimension'] == 'temperature']['value'].dropna()
                other_errors = subset[subset['dimension'] != 'temperature']['value'].dropna()
                
                if len(temp_errors) >= 2 and len(other_errors) >= 2:
                    stat, p = mannwhitneyu(temp_errors, other_errors, alternative='two-sided')
                    relevant = p < 0.05
                    if relevant:
                        stats[model] += 1
                    result = {
                        'model': model,
                        'concept': concept,
                        'temp_mean': temp_errors.mean(),
                        'other_mean': other_errors.mean(),
                        'statistic': stat,
                        'p_value': p,
                        'n_temp': len(temp_errors),
                        'n_other': len(other_errors),
                        'relevant': relevant
                    }
                    results.append(result)

                    logger.info(
                        f"[{model} | {concept}] Temperature (n={len(temp_errors)}, μ={temp_errors.mean():.3f}) "
                        f"vs. Other dimension (n={len(other_errors)}, μ={other_errors.mean():.3f}) → "
                        f"U={stat:.2f}, p={p:.6f} {'✓' if p < 0.05 else '✗'}"
                    )
                else:
                    logger.warning(
                        f"[{model} | {concept}] Not enough data points to compare "
                        f"(temperature={len(temp_errors)}, others={len(other_errors)})"
                    )

        # Save to CSV
        results_df = pd.DataFrame(results)
        output_path = RESULTS_FOLDER / "temperature_vs_other_domains.csv"
        results_df.to_csv(str(RESULTS_FOLDER /output_path), index=False, quoting=csv.QUOTE_NONNUMERIC)
        logger.info(f"Saved comparison results to {output_path}")
        print(stats)


    def compare_temperature_to_others(input_path="data_summed.csv", output_path="temperature_vs_others.csv"):
        df = pd.read_csv(RESULTS_FOLDER / input_path)
        df = df[(df['model_name'] != 'mturk') & (df['measurement'].notna())]
        results = []

        grouped = df.groupby(['model_name', 'concept'])

        for (model, concept), group in grouped:
            temperature_values = group[group['dimension'] == 'temperature']['response'].dropna().astype(float)
            other_values = group[group['dimension'] != 'temperature']['response'].dropna().astype(float)

            if len(temperature_values) < 2 or len(other_values) < 2:
                logger.warning(f"Skipping {model}, {concept} due to insufficient data for comparison.")
                continue

            t_stat, p_value = ttest_ind(temperature_values, other_values, equal_var=False)

            temp_mean = temperature_values.mean()
            temp_median = temperature_values.median()
            temp_std = temperature_values.std()
            other_mean = other_values.mean()
            other_std = other_values.std()

            logger.info(
                f"[{model} | {concept}] {temp_median}: Temperature MRE: {temp_mean:.3f}±{temp_std:.3f} "
                f"vs Others: {other_mean:.3f}±{other_std:.3f} (p={p_value:.4f})"
            )

            results.append({
                'model': model,
                'concept': concept,
                'temperature_mre': temp_mean,
                'temperature_std': temp_std,
                'other_mre': other_mean,
                'other_std': other_std,
                't_stat': t_stat,
                'p_value': p_value
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(
            str(RESULTS_FOLDER / output_path),
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )
        # # Missing values per domain
        # df = pd.read_csv(RESULTS_FOLDER / "data_summed.csv")
        # df_num = pd.read_csv(RESULTS_FOLDER / "NUM_RAW_SUMMARY.csv")
        # df_gt = df[df['model_name'] == 'mturk']
        # models_df = df_num[df_num['model'] != 'mturk']
        # models = set(models_df['model'])
        # irrelevant = []
        # for concept in df_gt['concept'].unique():
        #     concept_df =  df_gt[df_gt['concept'] == concept]
        #     for domain in concept_df['domain'].unique():
        #         domain_df = concept_df[concept_df['domain'] == domain]
        #         na_count = domain_df['response'].isna().sum()
        #         total = len(domain_df['response'])
        #         if na_count/total >= 0.5:
        #             irrelevant.append([concept, domain])
        #             logger.info(f"{concept} and {domain} NA: {domain_df['response'].isna().sum()} over {len(domain_df['response'])}")
        # For each model, check mean relative error and std and see if it's statistically significant for all the other properties within the concept
        
        # for i in irrelevant:
        #     concept = i[0]
        #     domain = i[1]
        #     for model in models:
        #         irrelevant_concept_df = models_df[(models_df['concept'] == concept) & (models_df['model'] == model)]
        #         irrelevant_concept_df[temp]
        #         irrelevant_error
        # pass

    # categorical_context()
    compare_temperature_to_others()

def analyze_ground_truth():
    pass

def get_mre_std():

    # Load and clean data
    num_scores = pd.read_csv(RESULTS_FOLDER / 'mae.csv')
    num_scores.replace([np.inf, -np.inf], np.nan, inplace=True)

    rel_error_cols = [col for col in num_scores.columns if "_rel_" in col]
    models = set(num_scores['model_name'])

    # Collect values: (row index, column name, concept, value)
    mean_error_dict = {model: [] for model in models}
    for index, row in num_scores.iterrows():
        for col in rel_error_cols:
            mean_error_dict[row['model_name']].append((index, col, row['concept'], row[col]))

    results = []
    outliers = []

    for model in sorted(models):
        entries = mean_error_dict[model]
        values = np.array([x[3] for x in entries], dtype=np.float64)

        # Remove NaNs
        clean_mask = ~np.isnan(values)
        entries = [e for e, keep in zip(entries, clean_mask) if keep]
        values = values[clean_mask]

        # Identify outliers
        outlier_mask = np.abs(values) > 1e3
        outlier_entries = [e for e, is_out in zip(entries, outlier_mask) if is_out]

        # Print top 5
        if outlier_entries:
            logger.warning(f"[{model}] {len(outlier_entries)} outliers (>|1000|), top 5:")
            for o in sorted(outlier_entries, key=lambda x: abs(x[3]), reverse=True)[:5]:
                logger.warning(f"\tRow: {o[0]}, Column: {o[1]}, Concept: {o[2]}, Value: {o[3]}")
            outliers.extend([
                {
                    "model": model,
                    "row": o[0],
                    "column": o[1],
                    "concept": o[2],
                    "value": o[3]
                } for o in outlier_entries
            ])

        # Compute filtered stats
        filtered_vals = [v for v in values if abs(v) <= 1e3]
        mean = np.nanmean(filtered_vals)
        std = np.nanstd(filtered_vals)

        results.append({
            "model": model,
            "mean": mean,
            "std": std,
            "count": len(filtered_vals),
            "num_outliers": len(outlier_entries)
        })

    # Save results
    pd.DataFrame(results).to_csv(RESULTS_FOLDER / "mean_std_cleaned.csv", index=False)
    pd.DataFrame(outliers).to_csv(str(RESULTS_FOLDER /'outliers.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)

    logger.info("Saved summary stats to mean_std_cleaned.csv")
    logger.info("Saved outlier details (with concepts) to outliers.csv")
    # Plot distributions
    # Prepare tidy DataFrame for seaborn
    plot_data = []

    for model in sorted(models):
        entries = mean_error_dict[model]
        values = np.array([x[3] for x in entries], dtype=np.float64)
        values = values[~np.isnan(values)]
        values = values[np.abs(values) <= 1e3]

        for v in values:
            plot_data.append({"model": model, "relative_error": v})

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=plot_df, x="model", y="relative_error")
    plt.title("Relative Error Distribution per Model (Boxplot)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_FOLDER / "rel_error_boxplot.png")
    plt.close()

if __name__ == "__main__":
    # # Combine responses
    # df = combine_extracted_knowledge()
    # mae_df = analyze_measurements(df[df['measurement'].notna() & (df['measurement'] != '')])
    # mae_df.to_csv(str(RESULTS_FOLDER /'mae.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    # rq1_df = analyze_performance(df, mae_df)
    # df.to_csv(str(RESULTS_FOLDER /'data_summed.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    # compare_models(['phi3mini_4k_instruct_fp16', 'phi3mini_4k_instruct_q4'])
    # compare_models(['qwen25_1b_standard', 'qwen25_7b_standard'])
    # compare_models(['llama31_8b_instruct', 'llama31_8b_instant', 'llama31_8b_standard'])
    # calculate_exact_matches()
    # get_mre_std()
    analyze_context()