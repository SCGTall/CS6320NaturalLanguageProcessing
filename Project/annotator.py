# -*- coding: utf-8 -*-
import json

# init
FOLDER_PREFIX = "data/"
MISINFORMATION_TARGETS = "misinformation_targets.json"
ms_dict = {}  # misinformation targets
TRAIN_TWEETS = "train_tweets.json"
tt_dict = {}  # train tweets
TRAIN_LABELS = "test.json"
tl_dict = {}  # train labels
#TRAIN_LABELS = "example_train_labels.json"

with open(FOLDER_PREFIX + MISINFORMATION_TARGETS, mode='r', encoding='utf-8') as f:
    ms_dict = json.loads(f.read())

with open(FOLDER_PREFIX + TRAIN_LABELS, mode='w', encoding='utf-8') as f:
    json.dumps(ms_dict, f)

print("Finish and exit.")

