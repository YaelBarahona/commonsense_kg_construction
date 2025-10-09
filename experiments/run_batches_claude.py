import os
from groq import Groq
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
import json
from pathlib import Path
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


logger = setup_logger()

load_dotenv()



# --- Config ---
BATCH_FILE = Path(__file__).parent.parent / "inputs/emotions/batches/claude4/batch_input_claude4_20runs_Cambridge.jsonl"
MODEL = "claude-sonnet-4-5-20250929"   # model name should match your batch file



def create_message_batch():
    # --- Load requests from JSONL ---
    requests = []
    with open(BATCH_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            custom_id = entry["custom_id"]
            body = entry["body"]

            # Build Anthropic request
            req = Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=body.get("model", MODEL),
                    system=body.get("system", None),
                    max_tokens=body.get("max_output_tokens", 500),
                    messages=body["messages"]
                )
            )
            requests.append(req)

    print(f"Loaded {len(requests)} requests from {BATCH_FILE}")

    # --- Submit batch ---
    message_batch = client.messages.batches.create(requests=requests)
    print("Batch created successfully:")
    print(message_batch)

def start_batch_pipeline(models, client_name, batch_path):
    create_message_batch()

def monitor_batch(batch_id):
    message_batch = client.messages.batches.retrieve(
        batch_id,
    )
    print(f"Batch {message_batch.id} processing status is {message_batch.processing_status}")

def download_batch_results_stream(batch_id, output_path="anthropic_results.jsonl"):
    """Download batch results using the modern streaming interface."""
    print(f"Streaming results for batch {batch_id}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch_id):
            # Convert Anthropic's Result object into plain dict
            result_dict = result.model_dump()
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

            match result.result.type:
                case "succeeded":
                    print(f"✅ Success: {result.custom_id}")
                case "errored":
                    err = result.result.error
                    if err and err.type == "invalid_request":
                        print(f"⚠️ Validation error in {result.custom_id}")
                    else:
                        print(f"❌ Server error in {result.custom_id}")
                case "expired":
                    print(f"⌛ Request expired: {result.custom_id}")

    print(f"✅ All results streamed and saved to {output_path}")


if __name__ == "__main__":
    # groq_models = {'deepseekr1_distill_llama_70b':'deepseek-r1-distill-llama-70b',
    #                 'llama4scout_17b16e_instruct': 'meta-llama/llama-4-scout-17b-16e-instruct'}
    models = {'gpt41': 'gpt-4.1'}
    # meta-llama/llama-4-scout-17b-16e-instruct
    # meta-llama/llama-4-maverick-17b-128e-instruct
    project = 'emotions'
    client = anthropic.Anthropic()          # uses ANTHROPIC_API_KEY from env
    batch_file = Path(__file__).parent.parent / 'inputs' / project / 'batches' / 'gpt41' / 'batch_input_gpt41_20runs_Cambridge.jsonl'
    # start_batch_pipeline(models, 'openai', batch_file)
    batch_id = 'msgbatch_01TamNrAx3ERSHWg32vcoTbH'
    monitor_batch(batch_id)
    download_batch_results_stream(batch_id)

    
 