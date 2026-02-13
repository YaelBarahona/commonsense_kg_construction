
import argparse
import os
import sys
from kg_constructors.pipeline import ConstructionPipeline
from config.config_loader import load_model_config

def main():
    parser = argparse.ArgumentParser(description="Commonsense Knowledge Graph Construction Pipeline")
    
    # Required arguments
    parser.add_argument("--concept", type=str, required=True, help="The concept to process (e.g., 'apple', 'love')")
    parser.add_argument("--model", type=str, required=True, help="The name of the model to use (must be defined in config/models.yaml)")
    
    # Optional arguments
    parser.add_argument("--property", type=str, default=None, help="The property to query (e.g., 'size', 'color')")
    parser.add_argument("--template", type=str, default="emotions", help="The prompt template name to use (default: 'emotions')")
    parser.add_argument("--ontology", type=str, default="ontology.owl", help="Path to the output ontology file")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the model (can also be set via env vars like GROQ_API_KEY)")
    
    args = parser.parse_args()

    # Environment variable handling for specific models if api_key not provided
    if not args.api_key:
        # Try to find a relevant API key in environment variables
        # This is a heuristic; specific clients might check their own env vars.
        if "groq" in args.model.lower():
            args.api_key = os.getenv("GROQ_API_KEY")
        elif "openai" in args.model.lower() or "gpt" in args.model.lower():
             args.api_key = os.getenv("OPENAI_API_KEY")
        elif "nebula" in args.model.lower():
             args.api_key = os.getenv("NEBULA_API_KEY")

    
    print(f"Starting pipeline for concept: {args.concept}")
    
    try:
        pipeline = ConstructionPipeline(
            concept=args.concept,
            prompt_template_name=args.template,
            property=args.property,
            model_name=args.model,
            ontology_path=args.ontology,
            api_key=args.api_key 
        )
        
        pipeline.run()
        print("Knowledge graph construction completed successfully.")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
