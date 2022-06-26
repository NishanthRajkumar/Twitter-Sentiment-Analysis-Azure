#!/usr/bin/env python
# coding: utf-8

# ## AzureTextAnalytics_Sentiments
# 
# 
# 

# In[1]:


# Import Required libraries
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


# In[3]:
conf = SparkConf().setAppName("Sentiment")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

df = spark.read.load('abfss://twitter-asa@cfpdlstorage.dfs.core.windows.net/output/tweets.csv', format='csv', header=True)


# In[4]:


tweet_list = df.select('Tweet').rdd.flatMap(lambda x: x).collect()


# In[6]:


# Connect Text Analytics Client to Azure Language Service
endpoint = sys.argv[1]
key = sys.argv[2]
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# In[10]:


no_of_docs = len(tweet_list)
no_of_batches = int(no_of_docs/10)
excess = no_of_docs%10

tweet_batches = []
for i in range(0,no_of_batches):
    start = i*10
    stop = start+10
    tweet_batches.append(tweet_list[start:stop])
if excess != 0:
    excess_start = 10*no_of_batches
    tweet_batches.append(tweet_list[excess_start:])


# In[12]:


sentiments = []
for tweet_batch in tweet_batches:
    result = text_analytics_client.analyze_sentiment(tweet_batch)
    docs = [doc for doc in result if not doc.is_error]
    for idx, doc in enumerate(docs):
        sentiments.append(doc.sentiment)


# In[13]:


predicted_sentiment_dict = {'Tweet': tweet_list, 'Sentiment': sentiments}
pred_df = pd.DataFrame(predicted_sentiment_dict)
pred_df.to_csv('abfss://twitter-asa@cfpdlstorage.dfs.core.windows.net/sentiments/tweet_sentiments.csv', index=False)

