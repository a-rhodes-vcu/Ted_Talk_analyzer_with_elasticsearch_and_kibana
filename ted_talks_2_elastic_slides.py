import pickle
import pandas as pd
from elasticsearch import helpers, Elasticsearch
import json
import requests



with open('ted_w_topic.pkl', 'rb') as picklefile:
    ted_w_topic = pickle.load(picklefile)


ted_w_topic['film_date'] = pd.to_datetime(ted_w_topic['film_date'],unit='s')
ted_w_topic['published_date'] = pd.to_datetime(ted_w_topic['published_date'],unit='s')

#print (ted_w_topic.tags.head())

ted_w_topic['tags'] = ted_w_topic['tags'].replace([r"\[","\]","'"],"",regex=True).str.lower()

#print (ted_w_topic.speaker_occupation.head(8))

ted_w_topic['speaker_occupation'] = ted_w_topic['speaker_occupation'].replace(['\/','\;'],', ',regex=True).str.lower()


#print (ted_w_topic.isna().sum())

ted_w_topic.speaker_occupation.fillna('None',inplace=True)

ted_w_topic['film_date'] = pd.to_datetime(ted_w_topic['film_date'],unit='s')
ted_w_topic['published_date'] = pd.to_datetime(ted_w_topic['published_date'],unit='s')

ted_w_topic.speaker_occupation.fillna('None',inplace=True)


ted_w_topic['speaker_occupation'] = ted_w_topic['speaker_occupation'].replace(['\/','\;'],', ',regex=True).str.lower()

ted_w_topic['tags'] = ted_w_topic['tags'].replace([r"\[","\]","'"],"",regex=True).str.lower()

# drop cols I don't want
ted_ingest = ted_w_topic.drop(columns=['ratings','related_talks'])
print (ted_ingest.head())

ted_ingest_d = ted_ingest.to_dict(orient='records')


es = Elasticsearch("localhost:9200")
es_client = Elasticsearch(http_compress=True)
Elasticsearch.info(es)

ted_ingest.to_csv('ted_w_topic.csv',index=False, encoding='utf8')


helpers.bulk(es_client, ted_ingest_d, index='ted_talks_1')



#curl -X POST "http://localhost:9200/index/ted_curl" -H "content-type: application/json" --data-binary "@ted_w_topic.json"
