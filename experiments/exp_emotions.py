import os
import json
import yaml
from runners.runner import Runner
from kg_constructors.json_constructor import JsonConstructor
from llm_clients.local_client import LocalClient
from llm_clients.groq_client import GroqClient
from llm_clients.nebula_client import NebulaClient
from config.config_loader import load_model_config
from dotenv import load_dotenv
from utils.logger import setup_logger
from copy import deepcopy
from pathlib import Path

logger = setup_logger()

INPUT_DIR = "inputs"
concept_file= "concepts_emotions.json"
property_file = "properties_emotions.yaml"
RUNS = int(os.getenv("RUNS", 20))
OUTPUT_PARENT_DIR = "output"
condition = 'emotions'

def get_checkpoint(model_name, concepts):
    """Get the checkpoint for a given model name."""
    concepts_remaining = deepcopy(concepts)
    logger.info(f"Checking model: {model_name}")
    concepts_to_check = {}
    for key in concepts.keys():
        concepts_to_check[concepts[key]["name"]] = key
    output_path = Path("output") / Path(condition) / Path(model_name)
    if not output_path.exists():
        logger.info(f"Output path {output_path} does not exist. Creating it.")
        output_path.mkdir(parents=True, exist_ok=True)
        return concepts_remaining
    for folder in output_path.iterdir():
        if str(folder.name) in list(concepts_to_check.keys()):
            logger.info(f"Found existing output for concept: {folder.name}")
            del concepts_remaining[concepts_to_check[str(folder.name)]]
    if not concepts_remaining:
        logger.info(f"All concepts processed for model {model_name}.")
        return None
    else:
        logger.info(f"Remaining concepts for model {model_name}: {len(list(concepts_remaining.keys()))}")
        return concepts_remaining

def run_experiment(current_client, condition):
    runner = Runner(clients=[current_client], serializer=JsonConstructor())
    model_name = current_client.model_name
    input_path_concept = os.path.join(INPUT_DIR, concept_file)
    with open(input_path_concept, "r") as f:
        concepts = json.load(f)
    concepts = get_checkpoint(model_name, concepts)
    if concepts == None:
        return None
    input_path_property = os.path.join(INPUT_DIR, property_file)
    with open(input_path_property, "r") as f:
        properties = yaml.safe_load(f)

    for _, obj_info in concepts.items():
        name = obj_info.get("name", "")
        definition = obj_info.get("definition", "")
        
        logger.info(f"Processing concept: {name} ({definition})")
        output_dir = os.path.join(OUTPUT_PARENT_DIR, condition ,current_client.model_name, name)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Processing dimensional emotions")
        measurable = properties.get("dimensional", {})
        for dimensional_model, details in measurable.items():
            template_name = "emotions"
            for qd, dq_details in details.get("quality_dimensions", []).items():
                dimension_description = dq_details.get("definition", [])
                dimension_range =  dq_details.get("range", [])
                
                output_path = os.path.join(output_dir, f"{RUNS}_{dimensional_model}_{name}_{qd}.json")
                logger.info(f"Running {model_name} and saving output to:\n\t\t{output_path}")

                runner.run(
                    concept=name.lower(),
                    description=definition,
                    domain="",
                    dimension=qd,
                    template_name=template_name,
                    runs=int(RUNS),
                    return_range="yes",
                    measurement="",
                    output_path=output_path,
                    dimension_range = dimension_range,
                    dimension_description = dimension_description

                )
def run_batch():
    load_dotenv()
    logger.info("Starting batch run...")

    model_config = load_model_config()

        
    # for entry in model_config.get("groq", []):
    #     logger.info(f"Loading Groq model: {entry['model_path']}")
    #     model_name = entry["model_path"]
    #     current_client = GroqClient(api_key=os.getenv("GROQ_API_KEY"), model_name=model_name)
    #     logger.info(f"Loaded client: {model_name}")
    #     run_experiment(current_client, condition)

    # for entry in model_config.get("local", []):
    #     logger.info(f"Loading local model: {entry['model_path']}")
    #     model_name = entry['name']
    #     current_client = LocalClient(model_path=entry["model_path"], model_name=model_name)
    #     logger.info(f"Loaded client: {model_name}")
    #     run_experiment(current_client, condition)


    for entry in model_config.get("nebula", []):
        model_path = entry["model_path"]
        model_name = entry["name"]
        logger.info(f"Loading Nebula model: {model_name}")
        current_client = NebulaClient(api_key=os.getenv("NEBULA_API_KEY"), model_name=model_name, model_path=model_path)
        logger.info(f"Loaded client: {model_name}")
        run_experiment(current_client, condition)
    


    logger.info("Batch run completed.")

if __name__ == "__main__":
    run_batch()
    logger.info("Batch runner script executed directly.")