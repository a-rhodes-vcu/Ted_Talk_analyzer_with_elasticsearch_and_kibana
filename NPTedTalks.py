from __future__ import print_function
import nltk, re, pickle, os
import pandas as pd
import numpy as np

from collections import Counter
from operator import itemgetter


from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize, MWETokenizer
from nltk.stem import porter, WordNetLemmatizer

from nltk.corpus import stopwords
from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation,  TruncatedSVD

nltk.download('punkt')
nltk.download('brown')
nltk.download('stopwords')
nltk.download('wordnet')

ted_main = pd.read_csv('ted_main.csv',encoding='utf-8')
ted_trans = pd.read_csv('transcripts.csv',encoding='utf-8')

ted_all = pd.merge(ted_trans,right=ted_main,on='url')
#print (ted_all.head(5))

with open('ted_all.pkl', 'wb') as picklefile:
    pickle.dump(ted_all, picklefile)

ted_all['id'] = ted_all.index

talks = ted_all['transcript']



def clean_text(text):
    """
    Takes in a corpus of documents and cleans. ONly works with multiple docs for now

    1. remove parentheticals
    2. tokenize into words using wordpunct
    3. lowercase and remove stop words
    4. lemmatize
    5. lowercase and remove stop words


    OUT: cleaned text = a list (documents) of lists (cleaned word in each doc)
    """

    lemmizer = WordNetLemmatizer()
    stemmer = porter.PorterStemmer()

    stop = stopwords.words('english')
    stop += ['.', ',', ':', '...', '!"', '?"', "'", '"', ' - ', ' — ', ',"', '."', '!', ';', '♫♫', '♫', \
             '.\'"', '[', ']', '—', ".\'", 'ok', 'okay', 'yeah', 'ya', 'stuff', ' 000 ', ' em ', \
             ' oh ', 'thank', 'thanks', 'la', 'was', 'wa', '?', 'like', 'go', ' le ', ' ca ', ' I ', " ? ", "s", " t ",
             "ve", "re"]
    # stop = set(stop)

    cleaned_text = []

    for post in text:
        cleaned_words = []

        # remove parentheticals
        clean_parens = re.sub(r'\([^)]*\)', ' ', post)

        #clean_parens = [line.decode('utf-8').strip() for line in clean_parens]

        # tokenize into words
        for word in wordpunct_tokenize(clean_parens):


            # lowercase and throw out any words in stop words
            if word.lower() not in stop:

                # lemmatize  to roots
                low_word = lemmizer.lemmatize(word)

                # stem and lowercase ( an alternative to lemmatize)
                # low_word = stemmer.stem(root.lower())

                # keep if not in stopwords (yes, again)
                if low_word.lower() not in stop:
                    # put into a list of words for each document
                    cleaned_words.append(low_word.lower())

        # keep corpus of cleaned words for each document
        cleaned_text.append(' '.join(cleaned_words))


    return cleaned_text


cleaned_talks = clean_text(talks)



with open('cleaned_talks.pkl', 'wb') as picklefile:
    pickle.dump(cleaned_talks, picklefile)

#print (cleaned_talks[0][0:300])

counter = Counter()

n = 2
for doc in cleaned_talks:
    words = TextBlob(doc).words
    bigrams = ngrams(words, n)
    counter += Counter(bigrams)

#for phrase, count in counter.most_common(30):
#    print('%20s %i' % (" ".join(phrase), count))


# CountVectorizer is a class; so `vectorizer` below represents an instance of that object.
c_vectorizer = CountVectorizer(ngram_range=(1,3),
                             stop_words='english',
                             max_df = 0.6,
                             max_features=10000)

t_vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                   stop_words='english',
                                   token_pattern="\\b[a-z][a-z]+\\b",
                                   lowercase=True,
                                   max_df = 0.6)


# call `fit` to build the vocabulary
c_vectorizer.fit(cleaned_talks)
# finally, call `transform` to convert text to a bag of words
c_x = c_vectorizer.transform(cleaned_talks)


# call `fit` to build the vocabulary
t_vectorizer.fit(cleaned_talks)
# finally, call `transform` to convert text to a bag of words
t_x = t_vectorizer.transform(cleaned_talks)

