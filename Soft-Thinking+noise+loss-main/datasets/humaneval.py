from datasets import load_dataset
import json
ds = load_dataset("openai/openai_humaneval")

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
            "canonical_solution": example["canonical_solution"],
            "test": example["test"],
            "entry_point": example["entry_point"]
        }
    })
with open("humaneval.json", "w") as f:
    json.dump(data, f,indent=4)
