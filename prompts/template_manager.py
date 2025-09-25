import yaml
import os

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

def load_template(name: str) -> dict:
    path = os.path.join(TEMPLATE_DIR, f"{name}.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def fill_template(template_str: str, **kwargs) -> str:
    return template_str.format(**{k: v.replace('_', ' ') if isinstance(v, str) else v for k, v in kwargs.items()})

def preprocess_template(template: str, **kwargs) -> str:
    description = kwargs.get("description", "")
    dimension = kwargs.get("dimension", "")
    domain = kwargs.get("domain", "")
    dim_range = kwargs.get("dimension_range", "")
    dimension_description = kwargs.get("dimension_description", "")
    kwargs["measurement"] = kwargs.get("measurement", "")
    kwargs["description_clause"] = f"(which is {description})" if description else ""
    kwargs["dimension_description_clause"] = f"(as in {dimension_description})" if dimension_description else ""
    kwargs["dimension_clause"] = f" as in {dimension}" if dimension else ""
    kwargs["range_clause"] = f"{dim_range}" if dim_range else ""
    kwargs["properties_clause"] = "\n".join(f"- {d}" for d in domain) if domain else ""
    result = template.format(**{k: v.replace('_', ' ') if isinstance(v, str) else v for k, v in kwargs.items()})
    return result