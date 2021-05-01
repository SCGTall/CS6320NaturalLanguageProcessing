# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_json("phaseB/fixed_train_tweets.json", orient='records')

a = df.tweet_id.values
print(type(a))
print(a)

b = list(zip(df.tweet_id.values, df.m_id.values))
print(type(b))
print(b)

c = np.column_stack((df.tweet_id.values, df.m_id.values))
print(type(c))
print(c)
