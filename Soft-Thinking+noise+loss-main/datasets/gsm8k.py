from datasets import load_dataset
import json
ds = load_dataset("openai/gsm8k", "main")

data = []
for example in ds["test"]:
    print(example["question"])
    print(example["answer"])
    # extract the answer after ####
    answer = example["answer"].split("####")[1].strip()
    data.append({
        "prompt": [
            {
                "from": "user",
                "value": example["question"]
            }
        ],
        "final_answer": answer
        })
with open("gsm8k.json", "w") as f:
    json.dump(data, f,indent=4)
