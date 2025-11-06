
from datasets import load_dataset
import json
ds = load_dataset("google-research-datasets/mbpp", "sanitized")

data = []
for example in ds["test"]:
    data.append({
        "prompt": [
            {
                "from": "user",
                "value": example["prompt"]
            }
        ],
        "final_answer": {
            "task_id": example["task_id"],
            "code": example["code"],
            "test_list": example["test_list"],
            "test_import": example["test_imports"]
        }
    })
with open("mbpp.json", "w") as f:
    json.dump(data, f,indent=4)
