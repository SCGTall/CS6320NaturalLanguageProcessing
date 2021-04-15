# -*- coding: utf-8 -*-
import json

# init
FOLDER_PREFIX = "data/"
MISINFORMATION_TARGETS = "misinformation_targets.json"
TRAIN_TWEETS = "train_tweets.json"
TRAIN_LABELS = "test.json"
#TRAIN_LABELS = "example_train_labels.json"

ms_dict = {}  # misinformation targets
with open(FOLDER_PREFIX + MISINFORMATION_TARGETS, mode='r', encoding='utf-8') as f:
    ms_dict = json.loads(f.read())

with open(FOLDER_PREFIX + TRAIN_LABELS, mode='w', encoding='utf-8') as f:
    json.dumps(ms_dict, f)

print("Finish and exit.")

