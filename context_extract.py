import json
import random

with open('context/context.json') as f:
    context = json.load(f)

data = []

for key, value in context.items():
    if value:
        random_value = random.choice(value)
        data.append({
            "topic": key,
            "worksheet": random_value
        })
        print(f"{key}: {random_value}")
    else:
        print(f"{key}: No values available")

with open('random_extracts.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"\nExtracted {len(data)} random values from {len(context)} categories")