# -*- coding: utf-8 -*-
import json

input1 = "phaseA/train_tweets.json"
id_dict = {}
with open(input1, mode='r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        tweet = json.loads(line)
        id_dict[tweet['id']] = tweet
        line = f.readline()

count = 0
input2 = "phaseB/test_tweets.json"
with open(input2, mode='r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        tweet = json.loads(line)
        if tweet['id'] not in id_dict.keys():
            print(tweet)
            count += 1
        line = f.readline()
print(count)