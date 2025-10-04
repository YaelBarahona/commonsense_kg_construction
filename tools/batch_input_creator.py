from prompts.template_manager import load_template, preprocess_template
import json
import pandas as pd
from pathlib import Path
import os
import yaml
import csv

def create_batch_file(model_name, model_path, output_path=None, runs=20):
    # Input variables
    OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'batches' / model_name
    INPUT_DIR = 'inputs'
    id=0
    input_path_concept = os.path.join(INPUT_DIR, "concepts_emotions.json")
    input_path_property = os.path.join(INPUT_DIR, "properties_emotions_condensed.yaml")

    templates = ['emotions']

    output_path  = OUTPUT_DIR / f"batch_input_{model_name}_{runs}runs.jsonl"

    # Loading the files
    with open(input_path_property, "r") as f:
        properties = yaml.safe_load(f)
    with open(input_path_concept, "r") as f:
        concepts = json.load(f)
    

    concepts_to_check = {}
    for key in concepts.keys():
        concepts_to_check[concepts[key]["name"]] = concepts[key]["definition"]    
    print(concepts_to_check)
    batch_info_df = pd.DataFrame(columns=['model_name', 'concept', 'domain', 'custom_id'])

    with output_path.open("w") as f:
        for concept, description in concepts_to_check.items():
            for template in templates:
                template_data = load_template(template)
                system_prompt=template_data.get("system_prompt", "You are a commonsense knowledge engineer. Return **ONLY** valid JSON.")

                domains = properties.get('dimensional', {})

                print(domains)

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
        str(OUTPUT_DIR / f"{model_name}_batch_info.csv"),
        index=False,
        quoting=csv.QUOTE_NONNUMERIC
    )

if __name__ == "__main__":
    # groq_models = {'deepseekr1_distill_llama_70b':'deepseek-r1-distill-llama-70b',
    #                 'llama4scout_17b16e_instruct': 'meta-llama/llama-4-scout-17b-16e-instruct'}
    models = {'gpt41': 'gpt-4.1'}#,'deepseekr1_distill_llama_70b':'deepseek-r1-distill-llama-70b', 
              #'llama4scout_17b16e_instruct': 'meta-llama/llama-4-scout-17b-16e-instruct',
              #'llama31_8b_instant' : 'llama-3.1-8b-instant'}
    runs = 20
    # models = {'llama3_8b_instant' : 'llama-3.1-8b-instant'}
    for model_name, model_path in models.items():
        create_batch_file(model_name, model_path, runs=runs)