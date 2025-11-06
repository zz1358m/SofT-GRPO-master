from datasets import load_dataset
import json
from datetime import datetime

ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5")

data = []

filter_date = datetime.strptime("2024-08-01T00:00:00", "%Y-%m-%dT%H:%M:%S")

for example in ds["test"]:
    contest_date_str = example["contest_date"]
    contest_date = datetime.strptime(contest_date_str, "%Y-%m-%dT%H:%M:%S")

    if contest_date >= filter_date:
        data.append({
            "prompt": [
                {
                    "from": "user",
                    "value": example.pop("question_content", "")
                }
            ],
            "final_answer": {
                "question_title": example["question_title"],
                "question_id": example["question_id"],
                "contest_id": example["contest_id"],
                "contest_date": example["contest_date"],
                "starter_code": example["starter_code"],
                "difficulty": example["difficulty"]
            }
        })
with open("livecodebench.json", "w") as f:
    json.dump(data, f,indent=4)

