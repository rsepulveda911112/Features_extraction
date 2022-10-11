from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
from gensim.test.utils import datapath
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
import pickle


class TFIDF:

    def __init__(self, file, text1, text2):
        if os.path.isfile(os.getcwd() + file):
            # Load model to disk
            temp_file = datapath(os.getcwd()+file)
            self.tfidf = pickle.load(open(temp_file, 'rb'))
        else:
            self.tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', max_features=5000, use_idf=True,
                                    norm='l2', strip_accents='unicode')
            self.tfidf.fit(text1 + text2)
            # Save model to disk.
            if not os.path.isdir(os.getcwd() +'/resource'):
                os.mkdir(os.getcwd() +'/resource')
            pickle.dump(self.tfidf, open(os.getcwd() + file, "wb"))

    def get_cosine_similarity(self, text1, text2):
        X_head = self.tfidf.transform([text1])
        X_body = self.tfidf.transform([text2])
        return cosine_similarity(X_head, X_body)[0][0]
