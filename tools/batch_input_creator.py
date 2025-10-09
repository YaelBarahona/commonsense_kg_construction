from prompts.template_manager import load_template, preprocess_template
import json
import pandas as pd
from pathlib import Path
import os
import yaml
import csv



# --- Input variables ---
project = 'emotions'
definition = 'Cambridge'

# MODELS = {'gpt41': 'gpt-4.1'}
MODELS = {'claude4' : 'claude-sonnet-4-5-20250929'}
RUNS = 20
TEMPLATE_NAMES = [project]
INPUT_DIR = Path(__file__).parent.parent / 'inputs' / project
OUTPUT_DIR = INPUT_DIR / 'batches'
INPUT_CONCEPT_PATH = os.path.join(INPUT_DIR, f"{project}.json")
INPUT_PROPERTY_PATH = os.path.join(INPUT_DIR, "properties_emotions_condensed.yaml")


def create_batch_file_openai_groq(model_name, model_path, output_file=None, runs=20):
    # Input variables
    output_directory = OUTPUT_DIR / model_name
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file  = output_directory / f"batch_input_{model_name}_{runs}runs_{definition}.jsonl"

    # Loading the files
    with open(INPUT_PROPERTY_PATH, "r") as f:
        properties = yaml.safe_load(f)
    with open(INPUT_CONCEPT_PATH, "r") as f:
        concepts = json.load(f)
    
    id=0
    concepts_to_check = {}

    for key in concepts.keys():
        concepts_to_check[concepts[key]["name"]] = concepts[key]["definitions"][definition]    

    batch_info_df = pd.DataFrame(columns=['model_name', 'concept', 'domain', 'custom_id'])

    with output_file.open("w") as f:
        for concept, description in concepts_to_check.items():
            for template in TEMPLATE_NAMES:
                template_data = load_template(template)
                system_prompt=template_data.get("system_prompt", "You are a commonsense knowledge engineer. Return **ONLY** valid JSON.")

                domains = properties.get('dimensional', {})

                if isinstance(domains, list):
                    domains = {domain: {'quality_dimensions': [''], 'units': ['']} for domain in domains}
                for domain, details in domains.items():
                    for run in range(runs):
                        custom_id = f"emotions-{id}-run{run}"
                        user_prompt = preprocess_template(
                            template_data["template"],
                            concept=concept,
                            description=description,
                            domain="",
                            dimension=domain,
                            return_range="yes",
                            measurement="",
                            dimension_description = details['definition'],
                            dimension_range = details['range']
                        )

                        payload = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model_path,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                "max_tokens": 500
                            }
                        }
                        f.write(json.dumps(payload) + "\n")
                        data = [model_name, concept, domain, custom_id]
                        temp_df = pd.DataFrame([data], columns=batch_info_df.columns)
                        batch_info_df = pd.concat([temp_df, batch_info_df], ignore_index=True)
                    id+=1
    batch_info_df['concept'] = batch_info_df['concept'].astype(str).str.strip()

    batch_info_df.to_csv(
        str(output_directory / f"{model_name}_batch_info.csv"),
        index=False,
        quoting=csv.QUOTE_NONNUMERIC
    )

def create_batch_file_anthropic(model_name, model_path, output_file=None, runs=20):
    # --- Input setup ---
    output_directory = OUTPUT_DIR / model_name
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = output_directory / f"batch_input_{model_name}_{runs}runs_{definition}.jsonl"

    # --- Load concept + property files ---
    with open(INPUT_PROPERTY_PATH, "r") as f:
        properties = yaml.safe_load(f)
    with open(INPUT_CONCEPT_PATH, "r") as f:
        concepts = json.load(f)

    id = 0
    concepts_to_check = {v["name"]: v["definitions"][definition] for v in concepts.values()}
    batch_info_df = pd.DataFrame(columns=["model_name", "concept", "domain", "custom_id"])

    # --- Write Anthropic-style payloads ---
    with output_file.open("w") as f:
        for concept, description in concepts_to_check.items():
            for template in TEMPLATE_NAMES:
                template_data = load_template(template)
                system_prompt = template_data.get(
                    "system_prompt",
                    "You are a commonsense knowledge engineer. Return ONLY valid JSON."
                )
                domains = properties.get("dimensional", {})
                if isinstance(domains, list):
                    domains = {d: {"definition": "", "range": ""} for d in domains}

                for domain, details in domains.items():
                    for run in range(runs):
                        custom_id = f"emotions-{id}-run{run}"
                        user_prompt = preprocess_template(
                            template_data["template"],
                            concept=concept,
                            description=description,
                            domain="",
                            dimension=domain,
                            return_range="yes",
                            measurement="",
                            dimension_description=details.get("definition", ""),
                            dimension_range=details.get("range", "")
                        )

                        # Anthropic message schema
                        payload = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/messages",
                            "body": {
                                "model": model_path,
                                "system": system_prompt,
                                "max_output_tokens": 500,
                                "messages": [
                                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
                                ]
                            }
                        }

                        f.write(json.dumps(payload) + "\n")
                        batch_info_df.loc[len(batch_info_df)] = [model_name, concept, domain, custom_id]
                    id += 1

    batch_info_path = output_directory / f"{model_name}_batch_info.csv"
    batch_info_df.to_csv(batch_info_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Saved Anthropic batch: {output_file}")

if __name__ == "__main__":

    for model_name, model_path in MODELS.items():
        # create_batch_file_openai_groq(model_name, model_path, runs=RUNS)
        create_batch_file_anthropic(model_name, model_path, runs=RUNS)