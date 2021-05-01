"""
main1 and data_utils_test are used to get a file of tweet results
"""

import logging
import random
import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nltk
nltk.download('stopwords')

# !pip install pytorch_transformers
from pytorch_transformers import AdamW  # Adam's optimization w/ fixed weight decay

from models.finetuned_models import FineTunedBert
from utils.data_utils_test import IMDBDataset
from utils.model_utils import train, test

# Disable unwanted warning messages from pytorch_transformers
# NOTE: Run once without the line below to check if anything is wrong, here we target to eliminate
# the message "Token indices sequence length is longer than the specified maximum sequence length"
#21 since we already take care of it within the tokenize() function through fixing sequence length
logging.getLogger('pytorch_transformers').setLevel(logging.CRITICAL)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE FOUND: %s" % DEVICE)


NUM_EPOCHS = 1
BATCH_SIZE = 1

PRETRAINED_MODEL_NAME = 'bert-base-cased'
NUM_PRETRAINED_BERT_LAYERS = 4
MAX_TOKENIZATION_LENGTH = 512
NUM_CLASSES = 2
TOP_DOWN = True
NUM_RECURRENT_LAYERS = 0
HIDDEN_SIZE = 128
REINITIALIZE_POOLER_PARAMETERS = False
USE_BIDIRECTIONAL = False
DROPOUT_RATE = 0.20
AGGREGATE_ON_CLS_TOKEN = True
CONCATENATE_HIDDEN_STATES = False

APPLY_CLEANING = False
TRUNCATION_METHOD = 'head-only'
NUM_WORKERS = 0

BERT_LEARNING_RATE = 3e-5
CUSTOM_LEARNING_RATE = 1e-3
BETAS = (0.9, 0.999)
BERT_WEIGHT_DECAY = 0.01
EPS = 1e-8

#Initialize to-be-finetuned Bert model
model = FineTunedBert(pretrained_model_name=PRETRAINED_MODEL_NAME,
                      num_pretrained_bert_layers=NUM_PRETRAINED_BERT_LAYERS,
                      max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                      num_classes=NUM_CLASSES,
                      top_down=TOP_DOWN,
                      num_recurrent_layers=NUM_RECURRENT_LAYERS,
                      use_bidirectional=USE_BIDIRECTIONAL,
                      hidden_size=HIDDEN_SIZE,
                      reinitialize_pooler_parameters=REINITIALIZE_POOLER_PARAMETERS,
                      dropout_rate=DROPOUT_RATE,
                      aggregate_on_cls_token=AGGREGATE_ON_CLS_TOKEN,
                      concatenate_hidden_states=CONCATENATE_HIDDEN_STATES,
                      use_gpu=True if torch.cuda.is_available() else False)

model.load_state_dict(torch.load('finetuned-bert-model-1LBD.pt', map_location=DEVICE))




test_dataset = IMDBDataset(input_directory='test_tweets.jsonl',
                           tokenizer=model.get_tokenizer(),
                           apply_cleaning=APPLY_CLEANING,
                           max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                           truncation_method=TRUNCATION_METHOD,
                           device=DEVICE)
tweet_id = test_dataset.tweets_id
                       

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)
print(test_loader)

# Define loss function
criterion = nn.CrossEntropyLoss()



# Define optimizer
#optimizer = AdamW(grouped_model_parameters)

#Place model & loss function on GPU
model, criterion = model.to(DEVICE), criterion.to(DEVICE)

#Start actual training, check test loss after each epoch
best_test_loss = float('inf')
result = []
tweet_sentiment = []


result = test(model=model,
                                iterator=test_loader,
                                criterion=criterion,
                                device=DEVICE,
                                result = result,
                                include_bert_masks=True)

len_result = len(result)

for i in range(len_result):
    json_dict = {}
    if int(result[i]) == 0:
        json_dict = {"id":tweet_id[i], "sentiment":"negative","sentiment_id":int(result[i])}
    else:
        json_dict = {"id":tweet_id[i], "sentiment":"positive","sentiment_id":int(result[i])}

    tweet_sentiment.append(json_dict)
    

json_string = json.dumps(tweet_sentiment, indent=4, separators=(',', ':'))
jsonFile = open("output1.jsonl", "w")
jsonFile.write(json_string)
jsonFile.close()
