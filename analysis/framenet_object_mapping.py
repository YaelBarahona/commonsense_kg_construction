from __future__ import annotations

# Standard Python libraries used throughout the pipeline
import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

# Import official NLTK FrameNet and WordNet corpora
from nltk.corpus import framenet as fn
from nltk.corpus import wordnet as wn

# Local helper function for sending prompts to the Nebula LLM API
from llm_client import call_llm


# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
# This allows progress updates to be shown in the terminal whilst the
# script is running.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# File paths
# ------------------------------------------------------------------
# BASE_DIR points to the current analysis folder.
BASE_DIR = Path(__file__).resolve().parent

# PROJECT_ROOT points to the main thesis project directory.
PROJECT_ROOT = BASE_DIR.parent

# JSON file containing the custom object knowledge base created earlier.
KB_FILE = BASE_DIR / "results" / "final_kb.json"

# CSV file containing MSCOCO object grounding information such as:
# WordNet synsets, Wikidata, DBpedia, YAGO and CSKG identifiers.
GROUNDING_FILE = (
    PROJECT_ROOT
    / "inputs"
    / "mscoco concepts"
    / "mscoco-groundtruth.csv"
)

# Final output file containing FrameNet assignments.
OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "framenet_object_mapping.json"
)


# ------------------------------------------------------------------
# Selected FrameNet frames
# ------------------------------------------------------------------
# These are official FrameNet frames loaded from NLTK.
# The project focuses only on this subset of frames.
FRAME_NAMES = [
    "Animals",
    "Plants",
    "Food",
    "Vehicle",
    "Containing",
    "Function",
    "Cooking_creation",
    "Use_vehicle",
    "Transportation_status",
    "Locative_relation",
    "Getting_scenario",
    "Arranging",
    "Attempt_obtain_food_scenario",
]


# ------------------------------------------------------------------
# Load JSON helper
# ------------------------------------------------------------------
def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file and return it as a Python dictionary.
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Load grounding CSV helper
# ------------------------------------------------------------------
def load_grounding_csv(path: Path) -> dict[str, dict[str, str]]:
    """
    Load the MSCOCO grounding CSV.

    Each object contains:
    - WordNet synset
    - Wikidata URI
    - DBpedia URI
    - YAGO URI
    - CSKG identifier
    """

    grounding = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            object_name = row["name"].strip()
            grounding[object_name] = row

    return grounding


# ------------------------------------------------------------------
# WordNet enrichment
# ------------------------------------------------------------------
def get_wordnet_info(wn_synset: str | None) -> dict[str, Any]:
    """
    Use WordNet to retrieve semantic information for the object.

    This provides:
    - definition
    - lemmas
    - hypernyms

    This semantic grounding helps the LLM make more informed
    FrameNet assignments.
    """

    if not wn_synset or wn_synset == "None":
        return {}

    try:
        syn = wn.synset(wn_synset)

        return {
            "synset": wn_synset,
            "definition": syn.definition(),
            "lemmas": [lemma.name() for lemma in syn.lemmas()],
            "hypernyms": [h.name() for h in syn.hypernyms()],
        }

    except Exception:
        return {
            "synset": wn_synset,
            "error": "Could not load WordNet synset"
        }


# ------------------------------------------------------------------
# Load official FrameNet frames
# ------------------------------------------------------------------
def load_framenet_frames() -> dict[str, Any]:
    """
    Load the selected official FrameNet frames from NLTK.

    Each frame contains:
    - frame name
    - frame ID
    - frame elements (roles)
    - frame definition
    """

    frames = {}

    for frame_name in FRAME_NAMES:

        # Search for the frame by name in FrameNet
        matches = fn.frames(rf"(?i)^{re.escape(frame_name)}$")

        if not matches:
            logger.warning(
                f"Frame not found in NLTK FrameNet: {frame_name}"
            )
            continue

        # Load the full frame object
        frame_obj = fn.frame(matches[0].ID)

        frames[frame_name] = frame_obj

    return frames


# ------------------------------------------------------------------
# Create readable FrameNet description text
# ------------------------------------------------------------------
def build_allowed_frames_text(frames: dict[str, Any]) -> str:
    """
    Convert FrameNet frames into readable text for the LLM prompt.

    The LLM receives:
    - frame name
    - frame ID
    - definition
    - official frame elements
    """

    lines = []

    for frame_name, frame_obj in frames.items():

        roles = sorted(frame_obj.FE.keys())

        definition = getattr(frame_obj, "definition", "")

        lines.append(
            f"- {frame_name} (ID: {frame_obj.ID})\n"
            f"  Definition: {definition}\n"
            f"  Roles: {', '.join(roles)}"
        )

    return "\n".join(lines)


# ------------------------------------------------------------------
# Build role validation structure
# ------------------------------------------------------------------
def build_allowed_roles(
    frames: dict[str, Any]
) -> dict[str, set[str]]:
    """
    Create a dictionary mapping each frame to its official roles.

    This is later used to validate the LLM output.
    """

    return {
        frame_name: set(frame_obj.FE.keys())
        for frame_name, frame_obj in frames.items()
    }


# ------------------------------------------------------------------
# Convert KB properties into readable text
# ------------------------------------------------------------------
def summarize_properties(properties: dict[str, Any]) -> str:
    """
    Convert the custom object KB properties into readable text
    for the LLM prompt.
    """

    parts = []

    for key in [
        "shape",
        "colour",
        "material",
        "fragility",
        "rigidity",
        "size",
        "weight_kg"
    ]:

        value = properties.get(key)

        if value is None:
            continue

        if isinstance(value, list):

            parts.append(
                f"{key}: "
                f"{', '.join(map(str, value)) if value else 'unknown'}"
            )

        elif isinstance(value, dict):

            inner = ", ".join(f"{k}={v}" for k, v in value.items())

            parts.append(f"{key}: {inner}")

        else:
            parts.append(f"{key}: {value}")

    return "\n".join(parts)


# ------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------
def build_prompt(
    object_name: str,
    kb_entry: dict[str, Any],
    grounding: dict[str, str],
    wordnet_info: dict[str, Any],
    allowed_frames_text: str,
) -> str:
    """
    Construct the full prompt sent to the LLM.

    The prompt contains:
    - object name
    - KB properties
    - external semantic grounding
    - WordNet information
    - official FrameNet frames and roles

    The LLM must choose valid FrameNet mappings.
    """

    properties = kb_entry.get("properties", {})

    property_summary = summarize_properties(properties)

    grounding_text = "\n".join(
        f"{k}: {v}"
        for k, v in grounding.items()
        if v not in [None, "", "None"]
    )

    wordnet_text = json.dumps(
        wordnet_info,
        indent=2,
        ensure_ascii=False
    )

    return f"""
You are assigning objects to official FrameNet frames
and frame elements.

Use ONLY the allowed FrameNet frames and roles listed below.

Do NOT invent frames.
Do NOT invent roles.
Do NOT invent objects.

Return ONLY valid JSON.

Object:
{object_name}

Object properties from my KB:
{property_summary}

External grounding:
{grounding_text}

WordNet information:
{wordnet_text}

Allowed FrameNet frames and roles:
{allowed_frames_text}

Task:
Assign this object only to the most central and useful
FrameNet frame-role pairs.

Important rules:
- Prefer precision over coverage.
- Return at most 3 assignments.
- Only use frames that describe the object's main category, function, or typical role.
- Do NOT include weak or trivial relations.
- Do NOT assign Containing just because an object has internal parts.
- Do NOT assign Transportation_status unless the object is actually a vehicle.
- Do NOT assign Use_vehicle unless the object is actually used as a vehicle.
- Do NOT assign Animals roles like Descriptor or Characteristic unless they are essential.
- If the object is food, prefer Food/Food.
- If the object is a container, prefer Containing/Container.
- If the object is a vehicle, prefer Vehicle/Vehicle.
- If the object is an animal, prefer Animals/Animal.
- If the object is a device or tool, prefer Function/Entity.

Output schema:
{{
  "object": "{object_name}",
  "assignments": [
    {{
      "frame": "...",
      "role": "...",
      "confidence": "high|medium|low",
      "reason": "short reason"
    }}
  ]
}}
""".strip()


# ------------------------------------------------------------------
# Extract JSON from LLM output
# ------------------------------------------------------------------
def extract_json(text: str) -> dict[str, Any]:
    """
    Attempt to extract valid JSON from the LLM response.

    Some LLMs occasionally add extra text around the JSON,
    so this function attempts to recover the first JSON object.
    """

    text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:

        try:
            return json.loads(match.group())

        except json.JSONDecodeError:
            pass

    return {
        "error": "Could not parse JSON",
        "raw_output": text,
        "assignments": []
    }


# ------------------------------------------------------------------
# Validate FrameNet assignments
# ------------------------------------------------------------------
def validate_output(
    parsed: Any,
    object_name: str,
    allowed_roles: dict[str, set[str]],
) -> dict[str, Any]:
    """
    Validate that:
    - the frame exists
    - the role belongs to that frame

    Invalid assignments are removed automatically.
    """

    if not isinstance(parsed, dict):

        return {
            "object": object_name,
            "assignments": [],
            "error": "Top-level output was not a JSON object",
            "raw_output": parsed,
        }

    assignments = parsed.get("assignments", [])

    if not isinstance(assignments, list):
        assignments = []

    valid_assignments = []
    invalid_assignments = []

    for assignment in assignments:

        if not isinstance(assignment, dict):
            invalid_assignments.append(assignment)
            continue

        frame = assignment.get("frame")
        role = assignment.get("role")

        # Check whether the frame exists
        if frame not in allowed_roles:
            invalid_assignments.append(assignment)
            continue

        # Check whether the role belongs to that frame
        if role not in allowed_roles[frame]:
            invalid_assignments.append(assignment)
            continue

        valid_assignments.append({
            "frame": frame,
            "role": role,
            "confidence": assignment.get(
                "confidence",
                "unknown"
            ),
            "reason": assignment.get("reason", "")
        })

    return {
        "object": object_name,
        "assignments": valid_assignments,
        "invalid_assignments_removed": invalid_assignments,
    }


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def main() -> None:
    """
    Main execution pipeline.

    The pipeline:
    1. loads the KB
    2. loads the grounding CSV
    3. loads FrameNet frames
    4. enriches objects with WordNet
    5. sends prompts to the LLM
    6. validates outputs
    7. saves final FrameNet assignments
    """

    if not KB_FILE.exists():
        raise FileNotFoundError(
            f"Missing KB file: {KB_FILE}"
        )

    if not GROUNDING_FILE.exists():
        raise FileNotFoundError(
            f"Missing grounding CSV: {GROUNDING_FILE}"
        )

    logger.info("Loading KB...")
    kb = load_json(KB_FILE)

    logger.info("Loading grounding CSV...")
    grounding_rows = load_grounding_csv(GROUNDING_FILE)

    logger.info("Loading FrameNet frames...")
    frames = load_framenet_frames()

    allowed_frames_text = build_allowed_frames_text(frames)

    allowed_roles = build_allowed_roles(frames)

    results = {}

    # Process all objects individually
    for object_name, kb_entry in sorted(kb.items()):

        logger.info(f"Processing object: {object_name}")

        grounding = grounding_rows.get(object_name, {})

        wn_synset = grounding.get("wn_synset")

        wordnet_info = get_wordnet_info(wn_synset)

        prompt = build_prompt(
            object_name=object_name,
            kb_entry=kb_entry,
            grounding=grounding,
            wordnet_info=wordnet_info,
            allowed_frames_text=allowed_frames_text,
        )

        # Send prompt to the LLM
        raw_output = call_llm(prompt)

        # Retry once if response is empty
        if not raw_output.strip():

            logger.warning(
                f"Empty response for {object_name}, retrying..."
            )

            raw_output = call_llm(prompt)

        # Parse and validate output
        parsed = extract_json(raw_output)

        validated = validate_output(
            parsed,
            object_name,
            allowed_roles
        )

        # Store additional metadata
        validated["kb_properties"] = kb_entry.get(
            "properties",
            {}
        )

        validated["grounding"] = grounding

        validated["wordnet"] = wordnet_info

        results[object_name] = validated

        # Small delay to reduce API instability
        time.sleep(1)

    # Create results folder if necessary
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save final JSON output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    logger.info(f"Saved output to: {OUTPUT_FILE}")

    logger.info(
        f"Processed objects: {len(results)}"
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()