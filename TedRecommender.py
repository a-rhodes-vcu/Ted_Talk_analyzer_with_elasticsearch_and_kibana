import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

with open('ted_all.pkl', 'rb') as picklefile:
    ted_all = pickle.load(picklefile)

with open('cleaned_talks.pkl', 'rb') as picklefile:
    cleaned_talks = pickle.load(picklefile)

with open('topic_names.pkl', 'rb') as picklefile:
    topic_names = pickle.load(picklefile)

with open('lda_mod.pkl', 'rb') as picklefile:
    lda_mod = pickle.load(picklefile)

with open('vect_mod.pkl', 'rb') as picklefile:
    vect_mod = pickle.load(picklefile)

with open('lda_data.pkl', 'rb') as picklefile:
    lda_data = pickle.load(picklefile)



def get_recommendations(first_article, num_of_recs, topics, ted_data, model, vectorizer, training_vectors):
    new_vec = model.transform(
        vectorizer.transform([first_article]))

    nn = NearestNeighbors(n_neighbors=num_of_recs, metric='cosine', algorithm='brute')
    nn.fit(training_vectors)

    results = nn.kneighbors(new_vec)

    recommend_list = results[1][0]
    scores = results[0]

    ss = np.array(scores).flat
    for i, resp in enumerate(recommend_list):
        print('\n--- ID ---\n', + resp)
        print('--- distance ---\n', + ss[i])
        print('--- topic ---')
        print(topics.iloc[resp, 0])
        print(ted_data.iloc[resp, 1])
        print('--- teds tags ---')
        print(ted_data.iloc[resp, -3])

    return recommend_list, ss


rec_list, scores = get_recommendations(cleaned_talks[804],10, topic_names, ted_all,
                                       lda_mod, vect_mod, lda_data)


print ("########")
print (rec_list)
