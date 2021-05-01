# -*- coding: utf-8 -*-
import json

input = "output.json"
json_obj = None
with open(input, mode='r', encoding='utf-8') as f:
    json_obj = json.loads(f.read())

output = "topic_labels.json"
with open(output, mode='w', encoding='utf-8') as f:
    f.write(json.dumps(json_obj, indent=4, separators=(',', ':')))

print("Finish and exit.")