import nltk, re, pickle, os
import pandas as pd
import numpy as np


from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize, MWETokenizer
from nltk.stem import porter, WordNetLemmatizer

from nltk.corpus import stopwords
from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation,  TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors

import pyLDAvis, pyLDAvis.sklearn
from IPython.display import display


from sklearn.preprocessing  import  StandardScaler

import seaborn as sns
#%matplotlib inline
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

with open('ted_all.pkl', 'rb') as picklefile:
    ted_all = pickle.load(picklefile)

with open('cleaned_talks.pkl', 'rb') as picklefile:
    cleaned_talks = pickle.load(picklefile)


def topic_mod_lda(data, topics=5, iters=10, ngram_min=1, ngram_max=3, max_df=0.6, max_feats=5000):
    """ vectorizer - turn words into numbers for each document(rows)
    then use Latent Dirichlet Allocation to get topics"""

    vectorizer = CountVectorizer(ngram_range=(ngram_min, ngram_max),
                                 stop_words='english',
                                 max_df=max_df,
                                 max_features=max_feats)

    #  `fit (train), then transform` to convert text to a bag of words

    vect_data = vectorizer.fit_transform(data)

    lda = LatentDirichletAllocation(n_components=topics,
                                    max_iter=iters,
                                    random_state=42,
                                    learning_method='online',
                                    n_jobs=-1)

    lda_dat = lda.fit_transform(vect_data)

    # to display a list of topic words and their scores

    def display_topics(model, feature_names, no_top_words):
        for ix, topic in enumerate(model.components_):
            print("Topic ", ix)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))

    display_topics(lda, vectorizer.get_feature_names(), 20)

    return vectorizer, vect_data, lda, lda_dat



vect_mod, vect_data, lda_mod, lda_data = topic_mod_lda(cleaned_talks,
                                                       topics=20, iters=100,
                                     ngram_min=1,
                                     ngram_max=2,
                                     max_df=0.5,
                                     max_feats=2000)
topic_ind = np.argmax(lda_data, axis=1)
topic_ind.shape
y=topic_ind

# create text labels for plotting
tsne_labels = pd.DataFrame(y)

# save to csv
tsne_labels.to_csv('tsne_labels.csv')

topic_names = tsne_labels

topic_names.to_csv('topic_names.csv')

with open('topic_names.pkl', 'wb') as picklefile:
    pickle.dump(topic_names, picklefile)

with open('lda_mod.pkl', 'wb') as picklefile:
    pickle.dump(lda_mod, picklefile)

with open('lda_data.pkl', 'wb') as picklefile:
    pickle.dump(lda_data, picklefile)

with open('vect_data.pkl', 'wb') as picklefile:
    pickle.dump(vect_data, picklefile)

with open('vect_mod.pkl', 'wb') as picklefile:
    pickle.dump(vect_mod, picklefile)


# Setup to run in Jupyter notebook
#pyLDAvis.enable_notebook()

# Create the visualization
#vis = pyLDAvis.sklearn.prepare(lda_mod, vect_data, vect_mod)

# Export as a standalone HTML web page
#pyLDAvis.save_html(vis, 'lda.html')

# # Let's view it!
#display(vis)

def plot_tsne(X, y, v1=0, v2=0):
    """ pass in the X from pca.transform ,and the corresponding y values and a string to add to the title
    plots the first three PCA directions/eigenvectors with target values as the color
    ___________________________________________________________________"""

    fig = plt.figure(1, figsize=(13, 10))
    ax = Axes3D(fig, elev=-150, azim=110)

    # plot transformed values (the three features that we have decomposed to) , colors correspond to target values
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,
               cmap=plt.cm.hot, edgecolor='k', s=50)
    ax.set_title("tSNE Ted Topics ", fontsize=16)
    ax.set_xlabel("1st ", fontsize=16)
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd", fontsize=16)
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd", fontsize=16)
    ax.w_zaxis.set_ticklabels([])
    ax.view_init(v1, v2)


from sklearn.manifold import TSNE

# a t-SNE model
# angle value close to 1 means sacrificing accuracy for speed
# pca initializtion usually leads to better results
tsne_model = TSNE(n_components=3, verbose=1, random_state=44, angle=.50,
                  perplexity=18,early_exaggeration=1,learning_rate=50.0)#, init='pca'

# 20-D -> 3-D
tsne_lda = tsne_model.fit_transform(lda_data)

tsne_data = pd.DataFrame(tsne_lda)
tsne_data.to_csv('tsne_lda.csv')

plot_tsne(tsne_lda,y, 0, 60)


