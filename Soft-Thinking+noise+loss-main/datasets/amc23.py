from datasets import load_dataset
import json
ds = load_dataset("zwhe99/amc23", "default")

data = []
for example in ds["test"]:
    print(example["question"])
    print(example["answer"])
    # extract the answer after ####
    answer = example["answer"]
    data.append({
        "prompt": [
            {
                "from": "user",
                "value": example["question"]
            }
        ],
        "final_answer": str(int(answer))
        })
with open("amc23.json", "w") as f:
    json.dump(data, f,indent=4)
