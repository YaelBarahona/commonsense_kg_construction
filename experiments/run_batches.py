import os
# from groq import Groq
from openai import OpenAI
from prompts.template_manager import load_template, preprocess_template
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
from utils.logger import setup_logger
import time
import pandas as pd
import csv

logger = setup_logger()

load_dotenv()

def upload_batch(client, file_path = "batch_file.jsonl"):
    logger.info(f"\tUploading batch...")
    batch_input_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch")
    return batch_input_file

def run_batch(batch_input_file, client):
    logger.info(f"\tRunning batch...")
    batch_input_file_id = batch_input_file.id
    response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h")
    return response

def check_batch_status(batch_response, client):
    logger.info(f"\tChecking batch status...")
    response = client.batches.retrieve(batch_response.id)
    return response

def return_batch(output_file_id, client, result_name="batch_results.jsonl"):
    file_response = client.files.content(output_file_id)
    file_response.write_to_file(result_name)

def start_batch_pipeline(models, client_name, runs):
    if client_name == 'openai':
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    elif client_name == 'groq':
        client = Groq(api_key=os.getenv('GROQ_BATCH_API_KEY'))
    else:
        return 'Invalid client name. Must be Groq or OpenAI.'
    
    batch_run_info = {model_name : {} for model_name in models.keys()}

    for model_name, model_path in models.items():
        logger.info(f"Starting batch process for model: {model_name}")
        
        batch_file = Path(__file__).parent.parent / 'inputs' / f'batch_input_{model_name}_{runs}runs.jsonl'
        
        batch_input_file = upload_batch(client, batch_file)
        logger.info(f"Batch file uploaded for {model_name}: {batch_input_file}")
        batch_run_info[model_name]['batch_input_file'] = batch_input_file
        
        response = run_batch(batch_input_file, client)
        logger.info(f"Batch run started for {model_name}, response: {response}")
        batch_run_info[model_name]['run_response'] = response
        
        status = check_batch_status(response, client)
        logger.info(f"Initial status check for {model_name}: {status}")
        batch_run_info[model_name]['status_response'] = status
    
    logger.info("All batches initialized")
    logger.debug(batch_run_info)

    batch_completed = {model_name : False for model_name in models.keys()}
    logger.info("Successfully started all batches, beginning monitoring loop.")
    time.sleep(60)

    while not all(batch_completed.values()):
        for model_name, model_path in models.items():
            response = batch_run_info[model_name]['run_response']
            batch_run_info[model_name]['status_response'] = check_batch_status(response, client)
            logger.info(f"\t...Checked status for {model_name}: {batch_run_info[model_name]['status_response'].status}")
            if batch_run_info[model_name]['status_response'].status == 'failed':
                logger.info(f"\t Batch for model {model_name} failed!")
                return False
            if batch_run_info[model_name]['status_response'].status == 'completed':
                logger.info(f"\t Batch for model {model_name} finished!")
                batch_completed[model_name] = True
                output_file_id = batch_run_info[model_name]['status_response'].output_file_id
                return_batch(output_file_id, client, result_name=f"results_{model_name}.jsonl")
            time.sleep(60)
    return True

if __name__ == "__main__":
    # groq_models = {'deepseekr1_distill_llama_70b':'deepseek-r1-distill-llama-70b',
    #                 'llama4scout_17b16e_instruct': 'meta-llama/llama-4-scout-17b-16e-instruct'}
    models = {'gpt41': 'gpt-4.1'}
    # models = {'llama3_8b_instant' : 'llama-3.1-8b-instant'}
    start_batch_pipeline(models, 'groq', 20)


    
 