from typing import List, Dict
from prompts.template_manager import load_template, preprocess_template
from kg_constructors.json_constructor import JsonConstructor
from llm_clients.base_client import LLMClient
from groq import InternalServerError, RateLimitError
import time
import requests

class Runner:
    def __init__(self, clients: List[LLMClient], serializer: JsonConstructor):
        self.clients = clients
        self.serializer = serializer
        self.max_retries = 3
        self.initial_delay = 1  # seconds

    def run(
        self,
        concept: str,
        description: str,
        domain: List[str],
        dimension: str,
        template_name: str,
        runs: int = 1,
        return_range: str = "",
        measurement: str = None,
        output_path: str = "output.json",
        **kwargs
    ) -> List[Dict]:
        template_data = load_template(template_name)
        results = []

        dimension_range = ""
        dimension_description = ""

        if "dimension_description" in kwargs:
            dimension_description = kwargs["dimension_description"]
   
        if "dimension_range" in kwargs:
            dimension_range = kwargs["dimension_range"]

        for client in self.clients:
            for i in range(runs):
                user_prompt = preprocess_template(
                    template_data["template"],
                    concept=concept,
                    description=description,
                    domain=domain,
                    dimension=dimension,
                    return_range=return_range,
                    measurement=measurement,
                    dimension_description = dimension_description,
                    dimension_range = dimension_range

                )

                response = None
                for attempt in range(1, self.max_retries + 1):
                    try:
                        response = client.generate(
                            system_prompt=template_data.get("system_prompt", "You are a commonsense knowledge engineer. Return **ONLY** valid JSON."),
                            user_prompt=user_prompt
                        )
                        break  # Success
                    except requests.exceptions.ConnectTimeout as e:
                            print(f"[Attempt {attempt}] ConnectTimeout for client {client.model_name}: {e}")
                            if attempt == self.max_retries:
                                print("Skipping due to timeout.")
                            else:
                                sleep_time = self.initial_delay * (2 ** (attempt - 1))
                                time.sleep(sleep_time)
                    except requests.exceptions.RequestException as e:
                        print(f"[Attempt {attempt}] General request error for {client.model_name}: {e}")
                        if attempt == self.max_retries:
                            print("Skipping due to persistent request failure.")
                        else:
                            sleep_time = self.initial_delay * (2 ** (attempt - 1))
                            time.sleep(sleep_time)
                    except InternalServerError as e:
                        print(f"[Attempt {attempt}] Error with client {client.model_name}: {e}")
                        if attempt == self.max_retries:
                            print(f"Max retries reached for client {client.model_name}. Skipping this run.")
                        else:
                            sleep_time = self.initial_delay * (2 ** (attempt - 1))
                            time.sleep(sleep_time)
                    except RateLimitError as e:
                        print(f"Rate limit error at run {i} for client {client.model_name}: {e}")
                        return results

                if response is not None:
                    results.append({
                        "client": client.model_name,
                        "concept": concept,
                        "domain": domain,
                        "measurement": measurement,
                        "dimension": dimension,
                        "format": "range" if return_range else "avg",
                        "response": response
                    })

        self.serializer.serialize(results, output_path)
        return results
