#!/usr/bin/env python3
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "google/gemma-3-270m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Simple GSM8K style test
problems = [
    "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer:",
    "Question: A robe takes 2 bolts of blue fiber and half that much white fiber. It takes 3 times as much yellow fiber as white fiber. How many bolts of yellow fiber does it take?\nAnswer:",
    "Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?\nAnswer:",
]

results = []
for i, p in enumerate(problems):
    inputs = tokenizer(p, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append({"problem": i, "prompt": p, "output": text})

out = {
    "model": model_name,
    "timestamp": datetime.now().isoformat(),
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "results": results
}

with open(os.path.join(os.path.dirname(__file__), "..", "results", f"quick_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
