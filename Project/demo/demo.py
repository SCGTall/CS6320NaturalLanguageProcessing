# -*- coding: utf-8 -*-
import torch

from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from transformers import BertForSequenceClassification
import pandas as pd
import numpy as np

import random
import json
import sys


def get_dictionary(_lst):
    return {v: i for (i, v) in enumerate(_lst)}


def get_inverse_dictionary(_dic):
    return {v: k for (k, v) in _dic.items()}


valid_mode = ["-s", "-e", "-t"]
sentiments = ["negative", "positive"]
emotions = ["happiness", "sadness"]
topics = ["immunity", "side effect", "infertility", "microchip control", "people die",
          "dna", "aborted fetus tissue", "bell's palsy", "vaccine"]
model_dirs = {"": "model/finetuned_BERT_epoch_4.model",
              "-s": "model/finetuned_BERT_epoch_x2_5.model",
              "-e": "model/finetuned_BERT_epoch_x3_9.model",
              "-t": "model/finetuned_BERT_epoch_x4_9.model"}

mode = ""
model = ""
arguments = []
other_dic = {}
inverse_other_dic = {}
df = pd.DataFrame(columns = ["tweet_id", "m_id", "other_id", "text"]) # collect all data in one table
# prepare data
if len(sys.argv) > 1 and sys.argv[1] in valid_mode:
    mode = sys.argv[1]
    arguments = sys.argv[2:]
else:
    arguments = sys.argv[1:]

# test tweet texts (tweet_id, text)
df_tts = pd.read_json("json/final_test_tweets.json", orient='records', dtype=False)
df_tts.rename(columns={'id':'tweet_id'}, inplace = True)
# test candidates (tweet_id, m_id)
df_tcs = None
# other inputs (tweet_id, o_id)
df_ots = None
if mode == "-s":
    df_ots = pd.read_json("json/sentiment_test_tweet.json", orient='records', dtype=False)
    df_ots.rename(columns={'id': 'tweet_id'}, inplace=True)
elif mode == "-e":
    df_ots = pd.read_json("json/emotion_test_tweet.json", orient='records', dtype=False)
    df_ots.rename(columns={'id': 'tweet_id'}, inplace=True)
    df_ots.rename(columns={'#emotion': 'other_id'}, inplace=True)
elif mode == "-t":
    df_ots = pd.read_json("json/topic_test_tweet.json", orient='records', dtype=False)
    df_ots.rename(columns={'id': 'tweet_id'}, inplace=True)
    df_ots['tweet_id'] = df_ots['tweet_id'].astype('int64')
    df_ots['topic_id'] = df_ots.topic.replace(get_dictionary(topics))

if len(arguments) == 0:  # use default json file or input arguments
    df_tcs = pd.read_json("json/final_test_candidates.json", orient='records', dtype=False)
    df = pd.merge(df_tcs, df_tts, how='left', on=['tweet_id'])
    if mode != "":
        df = pd.merge(df, df_ots[['tweet_id', 'other_id']], how='left', on=['tweet_id'])
else:
    if mode == "" and len(arguments) != 2:
        print("Unvalid input. Try like: <mode>(optional), <text/tweet_id>, <m_id>, <other_id>(if mode is assigned)")
        sys.exit(1)
    if mode != "" and len(arguments) != 3:
        print("Unvalid input. Try like: <mode>(optional), <text/tweet_id>, <m_id>, <other_id>(if mode is assigned)")
        sys.exit(1)
    tweet_id = arguments[0]
    text = ""
    if isinstance(tweet_id, int):
        if tweet_id in df_tts['tweet_id'].values:
            text = df_tts.loc[df_tts['tweet_id']==tweet_id, 'text']
        else:
            print(f"Cannot find tweet from given tweet_id: {tweet_id}")
            sys.exit(1)
    else:
        text = tweet_id
        if text in df_tts['text'].values:
            tweet_id = df_tts.loc[df_tts['text']==text, 'tweet_id']
        else:
            tweet_id = -1
    m_id = arguments[1]
    other_id = arguments[2] if mode != "" else -1
    if mode != "":
        other_lst = []
        if mode == "-s":
            other_lst = sentiments
        elif mode == "-e":
            other_lst = emotions
        elif mode == "-t":
            other_lst = topics

        if isinstance(other_id, int):
            if other_id not in range(len(other_lst)):
                print(f"Cannot find other label from given other_id: {other_id}")
                sys.exit(1)
        else:
            label = other_id
            other_dic = get_dictionary(other_lst)
            inverse_other_dic = get_inverse_dictionary(other_lst)
            if label in other_dic.keys():
                other_id = other_dic[label]
            else:
                print(f"Cannot find other label from given other_label: {label}")
                sys.exit(1)

    df = pd.DataFrame({'tweet_id': tweet_id, 'm_id': m_id, 'other_id': other_id, 'text': text,}, index=[1])

if mode != "":
    df['input'] = df['m_id'].map(lambda x: str(x) + " ") + df['other_id'].map(lambda x: str(x) + " ") + df['text']
else:
    df['input'] = df['m_id'].map(lambda x: str(x) + " ") + df['text']
df['label'] = 0

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)

label_dict = {'agree': 0, 'disagree': 1, 'no_stance': 2, 'not_relevant': 3}
label_dict_inverse = {v: k for k, v in label_dict.items()}

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

model_dir = model_dirs[mode]
model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

encoded_data_pred = tokenizer.batch_encode_plus(
    df.input.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors='pt'
)

input_ids_pred = encoded_data_pred['input_ids']
attention_masks_pred = encoded_data_pred['attention_mask']
labels_pred = torch.tensor(df.label.values)  # pretend to be all 0s. Proved that labels does not change results.

dataset_pred = TensorDataset(input_ids_pred, attention_masks_pred, labels_pred)

dataloader_prediction = DataLoader(dataset_pred,
                              sampler=SequentialSampler(dataset_pred),
                              batch_size=1)


def predict(dataloader_val):
    model.eval()

    predictions = []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    predictions = np.concatenate(predictions, axis=0)

    return np.argmax(predictions, axis=1).flatten()

test_predictions = predict(dataloader_prediction)
df['label'] = pd.Series(test_predictions).values
df['m_label'] = df.label.replace(label_dict_inverse)

print_time = 10  # print first 10 lines
with open("result.json", mode='w', encoding='utf-8') as f:
    for elem in json.loads(df[['tweet_id', 'm_id', 'm_label']].to_json(orient='records')):
        line = json.dumps(elem) + "\n"
        if print_time > 0:
            print(line)
            print_time -= 1
        f.write(line)