# Commonsense Knowledge Graph Construction with Large Language Models

This repository provides a framework for constructing commonsense knowledge graphs using large language models (LLMs). The system extracts structured knowledge about real-world concepts by querying LLMs with carefully designed prompts, then aggregates and analyzes the results.
The raw data used for the K-CAP submission is available [here](https://zenodo.org/records/16743477)
![High-level overview of the system](docs/cskg-overview.png "Overview")

## Repository Structure

- **prompts/**: Contains all prompt templates used to query LLMs. Modify these files to experiment with different prompt designs or extraction strategies.
- **inputs/**: Stores the lists of concepts and property definitions used as input to the pipeline (e.g., `concepts_mscoco.json`, `exp_properties.yaml`).
- **data/**: Central location for all data files.
  - `data/raw_data/`: Raw outputs from LLM queries and initial data dumps.
  - `data/preprocessed/`: Cleaned and preprocessed data ready for analysis.
  - `data/parsed/`: Results of the syntaxtic parsing.
  - `data/extracted_knowledge/`: Results of the semantic parsing.
  - `data/results/`: Output statistics
- **kg_constructors/**: Main pipeline code for knowledge extraction.
- **experiments/**: Scripts for running batch experiments (e.g., for MS COCO or ImageNet).
- **analysis/**: Scripts and notebooks for analyzing and visualizing results.
- **output/**, **logs/**: Output files and logs generated during runs.

## Overview

The pipeline operates on two main inputs:
- **Concepts**: The entities to be described (e.g., "apple", "hammer").
- **Quality Dimensions**: The domains or properties relevant to each concept (e.g., colour, shape, weight).

A key step is determining which quality dimensions are relevant for each concept. This is achieved using a "context vector", which acts as a binary filter to indicate the applicability of each dimension.

The framework supports two main use cases:
1. **Irrelevant Quality Dimensions**: When a dimension does not apply to a concept (e.g., the speed of an apple).
2. **Non-defining Features**: When a value can take any form and is not a defining feature (e.g., the colour of a mug).

The initial experiments focus on 80 concepts from the MS COCO dataset, with extensions to 1000 concepts from ImageNet-1k.

## Reproducing the Experiment

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/commonsense_kg_construction.git
cd commonsense_kg_construction
```

### 2. Install Dependencies

It is recommended to use a virtual environment:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare Input Data

- Place your concepts and property definitions in the `inputs/` directory.
- In the paper the files used: `concepts_mscoco.json`, `exp_properties.yaml`,`properties.csv` .

### 4. Run the Knowledge Generation Pipeline

Knowledge generation for the 80 MS COCO object concepts can be performed using:

```sh
python experiments/exp_mscoco.py
```

This script queries the configured Large Language Model for each concept-property pair and stores the generated outputs as JSON files.

### 5. Analyse Results

The generated outputs can be validated and processed using:

```sh
python analysis/main_analysis.py
python analysis/build_final_kb.py
python analysis/error_count.py
python analysis/find_weird_values.py
```

### 6. Visualize Results

Visualization scripts are available in `analysis/visualisations/`.

Visualisation scripts are available in the `analysis/` directory.

```sh
python analysis/property_coverage.py
python scripts/existing_knowledge.py
```

Generated figures are stored in the `graphs/` directory.

**Note:**  
- You may need API keys for LLM providers (e.g., OpenAI, Groq, Nebula). Set these in a `.env` file in the project root.
- Output and intermediate files are stored in the `output/`, `data/`, and `logs/` directories.

## Citation

If you use this codebase, please cite or acknowledge the repository.
