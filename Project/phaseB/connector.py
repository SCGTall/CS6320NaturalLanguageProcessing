# -*- coding: utf-8 -*-
import json
import os

folder = "connector/"
files= os.listdir(folder)
connected = []
for name in files:
    if "json" not in name:
        continue
    with open(folder + name, mode='r', encoding='utf-8') as f:
        line = f.readline()
        lineNum = 0
        while line:
            lineNum += 1
            tweet = json.loads(line)
            connected.append(tweet)
            #print(f"{lineNum}: {line}", end="\r")
            line = f.readline()
    print(f"{lineNum} records in {name}.")

print(f"{len(connected)} records in all.")

output = "final_train_labels.json"
with open(output, mode='w', encoding='utf-8') as f:
    f.write(json.dumps(connected, indent=4, separators=(',', ':')))
print(f"Save as {output}.")
