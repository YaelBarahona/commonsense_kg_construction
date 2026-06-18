from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import yaml

from llm_client import call_llm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

FRAME_MAPPING_FILE = BASE_DIR / "results" / "framenet_object_mapping.json"
PROMPT_FILE = PROJECT_ROOT / "prompts" / "templates" / "scenario_generation.yaml"
OUTPUT_FILE = BASE_DIR / "results" / "semantic_scenarios.json"

SLEEP_SECONDS = 1


GROUP_FRAME_PRIORITY = {
    "food_activity": [
        "Food",
        "Cooking_creation",
        "Attempt_obtain_food_scenario",
        "Containing",
        "Arranging",
    ],

    "transport_activity": [
        "Vehicle",
        "Use_vehicle",
        "Transportation_status",
        "Locative_relation",
    ],

    "animal_interaction": [
        "Animals",
        "Locative_relation",
        "Function",
    ],

    "plant_activity": [
        "Plants",
        "Food",
        "Locative_relation",
    ],

    "object_containment": [
        "Containing",
        "Arranging",
        "Function",
    ],

    "tool_and_device_use": [
        "Function",
        "Arranging",
        "Locative_relation",
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_prompt_yaml(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_json(text: str) -> Any:
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
        "scenarios": [],
    }


def get_object_frames(mapping_entry: dict[str, Any]) -> list[str]:
    frames = []

    for assignment in mapping_entry.get("assignments", []):
        frame = assignment.get("frame")

        if frame and frame not in frames:
            frames.append(frame)

    return frames


def build_semantic_groups(
    frame_mapping: dict[str, Any],
) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = {}

    for group_name, priority_frames in GROUP_FRAME_PRIORITY.items():
        groups[group_name] = {
            "objects": [],
            "frames": priority_frames,
        }

        for object_name, entry in frame_mapping.items():
            object_frames = get_object_frames(entry)

            if any(frame in object_frames for frame in priority_frames):
                groups[group_name]["objects"].append(object_name)

    groups = {
        group_name: data
        for group_name, data in groups.items()
        if data["objects"]
    }

    return groups


def build_prompt(
    prompt_yaml: dict[str, str],
    group_name: str,
    objects: list[str],
    frames: list[str],
) -> str:
    system_prompt = prompt_yaml["system"]
    user_template = prompt_yaml["user_template"]

    object_text = "\n".join(f"- {obj}" for obj in objects)
    frame_text = "\n".join(f"- {frame}" for frame in frames)

    user_prompt = user_template.format(
        group_name=group_name,
        objects=object_text,
        frames=frame_text,
    )

    return f"{system_prompt}\n\n{user_prompt}"


def clean_list(
    values: Any,
    allowed_values: set[str],
) -> list[str]:
    if not isinstance(values, list):
        return []

    cleaned = []

    for value in values:
        if value in allowed_values and value not in cleaned:
            cleaned.append(value)

    return cleaned


def validate_group_output(
    parsed: Any,
    group_name: str,
    allowed_objects: list[str],
    allowed_frames: list[str],
) -> dict[str, Any]:
    allowed_object_set = set(allowed_objects)
    allowed_frame_set = set(allowed_frames)

    if not isinstance(parsed, dict):
        return {
            "semantic_group": group_name,
            "scenarios": [],
            "error": "Top-level output was not a JSON object",
            "raw_output": parsed,
        }

    scenarios = parsed.get("scenarios", [])

    if not isinstance(scenarios, list):
        scenarios = []

    valid_scenarios = []

    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue

        scenario_name = scenario.get("name", "")
        scenario_objects = scenario.get("relevant_objects", [])
        scenario_frames = clean_list(
            scenario.get("collaborating_frames", []),
            allowed_frame_set,
        )

        subframes = scenario.get("subframes", [])

        if not isinstance(subframes, list):
            subframes = []

        valid_subframes = []

        for subframe in subframes:
            if not isinstance(subframe, dict):
                continue

            subframe_name = subframe.get("name", "")
            subframe_objects = clean_list(
                subframe.get("relevant_objects", []),
                allowed_object_set,
            )
            subframe_frames = clean_list(
                subframe.get("collaborating_frames", []),
                allowed_frame_set,
            )

            if not subframe_name:
                continue

            valid_subframes.append({
                "step": subframe.get("step"),
                "name": subframe_name,
                "relevant_objects": subframe_objects,
                "collaborating_frames": subframe_frames,
                "description": subframe.get("description", ""),
            })

        if not scenario_name:
            continue

        valid_scenarios.append({
            "name": scenario_name,
            "relevant_objects": scenario_objects,
            "collaborating_frames": scenario_frames,
            "subframes": valid_subframes,
        })

    print(json.dumps(parsed, indent=2))

    return {
        "semantic_group": group_name,
        "available_objects": allowed_objects,
        "available_frames": allowed_frames,
        "scenarios": valid_scenarios,
    }


def collect_used_objects(results: dict[str, Any]) -> set[str]:
    used_objects = set()

    for group in results.get("semantic_groups", []):
        for scenario in group.get("scenarios", []):
            used_objects.update(scenario.get("relevant_objects", []))

            for subframe in scenario.get("subframes", []):
                used_objects.update(subframe.get("relevant_objects", []))

    return used_objects


def main() -> None:
    if not FRAME_MAPPING_FILE.exists():
        raise FileNotFoundError(f"Missing FrameNet mapping file: {FRAME_MAPPING_FILE}")

    logger.info("Loading FrameNet object mapping...")
    frame_mapping = load_json(FRAME_MAPPING_FILE)

    logger.info("Loading prompt file...")
    prompt_yaml = load_prompt_yaml(PROMPT_FILE)

    logger.info("Building semantic groups from FrameNet mappings...")
    semantic_groups = build_semantic_groups(frame_mapping)

    results = {
        "semantic_groups": [],
        "coverage": {},
    }

    for group_name, group_data in semantic_groups.items():
        logger.info(f"Generating scenarios for group: {group_name}")

        objects = sorted(group_data["objects"])
        frames = group_data["frames"]

        prompt = build_prompt(
            prompt_yaml=prompt_yaml,
            group_name=group_name,
            objects=objects,
            frames=frames,
        )

        raw_output = call_llm(prompt)
        parsed_output = extract_json(raw_output)

        validated_output = validate_group_output(
            parsed=parsed_output,
            group_name=group_name,
            allowed_objects=objects,
            allowed_frames=frames,
        )

        results["semantic_groups"].append(validated_output)

        used_objects = collect_used_objects(results)
        all_objects = set(frame_mapping.keys())

        results["coverage"] = {
            "total_objects": len(all_objects),
            "used_objects": len(used_objects),
            "unused_objects": sorted(all_objects - used_objects),
        }

        save_json(OUTPUT_FILE, results)

        time.sleep(SLEEP_SECONDS)

    save_json(OUTPUT_FILE, results)

    logger.info(f"Saved semantic scenarios to: {OUTPUT_FILE}")
    logger.info(
        f"Object coverage: {results['coverage']['used_objects']} / "
        f"{results['coverage']['total_objects']}"
    )


if __name__ == "__main__":
    main()