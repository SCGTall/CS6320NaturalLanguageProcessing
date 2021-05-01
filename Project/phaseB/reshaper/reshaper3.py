# -*- coding: utf-8 -*-
import json
from random import sample

input = "final_train_labels.json"
json_obj = []
with open(input, mode='r', encoding='utf-8') as f:
    json_obj = json.loads(f.read())

m_dict = {}
for (i, tweet) in enumerate(json_obj):
    label = tweet["m_label"]
    if label not in m_dict.keys():
        m_dict[label] = [i]
    else:
        m_dict[label].append(i)
minimum = len(json_obj)
for v_list in m_dict.values():
    if len(v_list) < minimum:
        minimum = len(v_list)
print(f"Smallest set size: {minimum}")
selected = []
for v_list in m_dict.values():
    s = sample(v_list, minimum)
    for i in s:
        selected.append(json_obj[i])

output = "fixed_train_tweets.json"
with open(output, mode='w', encoding='utf-8') as f:
    f.write(json.dumps(selected, indent=4, separators=(',', ':')))

print("Finish and exit.")