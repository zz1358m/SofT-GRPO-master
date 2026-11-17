"""[ST-OPT] tiny per-step profiler, enabled with SOFT_PROFILE=1."""
import json
import os
import time

import torch

ENABLED = os.environ.get("SOFT_PROFILE", "0") == "1"
OUT = os.environ.get("SOFT_PROFILE_OUT", "/tmp/st_prof.jsonl")
acc = {}


def sync():
    if ENABLED:
        torch.cuda.synchronize()


def add(key, dt):
    acc[key] = acc.get(key, 0.0) + dt


def step_done(every=256):
    acc["steps"] = acc.get("steps", 0) + 1
    if acc["steps"] % every == 0:
        n = acc["steps"]
        rec = {"steps": n, **{k: round(1000 * v / n, 4) for k, v in acc.items() if k != "steps"}}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
