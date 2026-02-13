import argparse
import os
import logging
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import filter_repo_objects

# Configure logging
logger = logging.getLogger(__name__)

def list_models(search_query, limit=10):
    logger.debug(f"Starting list_models with query='{search_query}', limit={limit}")
    api = HfApi()
    print(f"Searching for models with query: '{search_query}' (GGUF)...")
    
    # Filter for models that likely have GGUF files
    logger.debug("Calling HfApi.list_models...")
    models = api.list_models(
        search=search_query,
        limit=limit * 2, # Get more to filter
        sort="downloads",
        direction=-1,
        tags=["gguf"]
    )
    logger.debug(f"Retrieved {len(list(models))} models (before display limit)")
    
    # Reset iterator because we consumed it for logging (oops, list_models returns a generator, so we should be careful. 
    # Actually list_models returns an iterator. Let's re-call or just iterate carefully.
    # To be safe and simple debug, let's just log "Called HfApi.list_models" and iterate.
    
    # Re-calling API to be safe with generator consumption if I were to inspect it, but let's just restart the call logic cleanly 
    # or better yet, just iterate.
    
    models = api.list_models(
        search=search_query,
        limit=limit * 2, 
        sort="downloads",
        direction=-1,
        tags=["gguf"]
    )

    count = 0
    print(f"{'Model ID':<50} | {'Downloads':<10} | {'Likes':<10}")
    print("-" * 76)
    
    for model in models:
        logger.debug(f"Processing model: {model.modelId}")
        if count >= limit:
            logger.debug("Limit reached, stopping iteration.")
            break
        print(f"{model.modelId:<50} | {model.downloads:<10} | {model.likes:<10}")
        count += 1

def download_model(model_id, filename=None, output_dir="models"):
    logger.debug(f"Starting download_model: id={model_id}, filename={filename}, dir={output_dir}")
    print(f"Downloading {model_id} to {output_dir}...")
    
    # Ensure output directory exists, creating the root 'models' if needed
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Ensured output directory exists: {output_dir}")
    
    if not filename:
        logger.debug("No filename provided. Initiating auto-discovery...")
        print("No filename specified. Searching for Q4_K_M or Q4_0 GGUF files...")
        api = HfApi()
        try:
            logger.debug(f"Listing repo files for {model_id}...")
            files = api.list_repo_files(repo_id=model_id)
            logger.debug(f"Found {len(files)} files in repo.")
        except Exception as e:
            logger.error(f"Failed to list repo files: {e}")
            print(f"Error accessing repository: {e}")
            return

        gguf_files = [f for f in files if f.endswith(".gguf")]
        logger.debug(f"Found {len(gguf_files)} GGUF files: {gguf_files}")
        
        if not gguf_files:
            print(f"No GGUF files found in {model_id}.")
            return
            
        # Heuristic to find a good default quantum
        preferred_quants = [
            "Q4_K_M.gguf", "Q4_0.gguf", "Q5_K_M.gguf", "Q8_0.gguf",
            "q4_k_m.gguf", "q4_0.gguf", "q5_k_m.gguf", "q8_0.gguf",
            "q4.gguf", "Q4.gguf", "-Q4_K_M.gguf", "-Q4_0.gguf"
        ]
        selected_file = None
        
        for quant in preferred_quants:
            logger.debug(f"Checking for preference: {quant}")
            for f in gguf_files:
                if f.endswith(quant):
                    selected_file = f
                    logger.debug(f"Match found: {selected_file}")
                    break
            if selected_file:
                break
        
        if not selected_file:
            logger.debug("No preferred quantum matched. Asking user to specify.")
            print("Could not automatically select a quant. Available files:")
            for f in gguf_files:
                print(f" - {f}")
            print("Please run again with --filename <filename>")
            return
            
        print(f"Selected default file: {selected_file}")
        filename = selected_file
        
    try:
        logger.debug(f"Calling hf_hub_download for {filename}...")
        file_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        logger.debug(f"Download complete. Path: {file_path}")
        print(f"Successfully downloaded to: {file_path}")
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        print(f"Error downloading model: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download GGUF models from Hugging Face")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available GGUF models")
    list_parser.add_argument("search", nargs="?", default="", help="Search query (e.g. 'llama3')")
    list_parser.add_argument("--limit", type=int, default=10, help="Number of results to show")
    
    # Download command
    dl_parser = subparsers.add_parser("download", help="Download a model file")
    dl_parser.add_argument("--model-id", required=True, help="Hugging Face Model ID (e.g. 'microsoft/Phi-3-mini-4k-instruct-gguf')")
    dl_parser.add_argument("--filename", help="Specific filename to download (e.g. 'Phi-3-mini-4k-instruct-q4.gguf')")
    dl_parser.add_argument("--output-dir", default="models", help="Directory to save the model")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.command == "list":
        list_models(args.search, args.limit)
    elif args.command == "download":
        download_model(args.model_id, args.filename, args.output_dir)

if __name__ == "__main__":
    main()
