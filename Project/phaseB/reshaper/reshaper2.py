# -*- coding: utf-8 -*-
import json

input = "test_tweets.json"
json_obj = []
with open(input, mode='r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        tweet = json.loads(line)
        json_obj.append(tweet)
        line = f.readline()

output = "final_test_tweets.json"
with open(output, mode='w', encoding='utf-8') as f:
    f.write(json.dumps(json_obj, indent=4, separators=(',', ':')))

print("Finish and exit.")