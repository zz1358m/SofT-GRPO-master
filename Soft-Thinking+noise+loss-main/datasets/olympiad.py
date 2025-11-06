from datasets import load_dataset
import json
import json

from pyarrow.dataset import dataset

data = []
with open('olympiad_bench_raw.json', 'r') as f:
    dataset = json.load(f)
for example in dataset:
    print(example["problem"])
    print(example["answer"])
    # extract the answer after ####
    answer = example["answer"]
    data.append({
        "prompt": [
            {
                "from": "user",
                "value": example["problem"]
            }
        ],
        "final_answer": str(answer[0])
        })
with open("olympiad_bench.json", "w") as f:
    json.dump(data, f,indent=4)
