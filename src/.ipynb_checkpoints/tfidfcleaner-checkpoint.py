import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn import feature_extraction, linear_model, model_selection, metrics



class dataCleaner:
    
    
    def __init__(self):
        
        self.featureExtractor = feature_extraction.text.TfidfVectorizer(
                                                strip_accents = 'ascii',
                                                lowercase = True,
                                                tokenizer = token.tokenize,
                                                stop_words = stop_words,
                                                ngram_range = (1, 3),
                                                max_features = 20000
        )

    

    
    def fit(self, X):
        self.featureExtractor.fit(X)
    
    def transform(self, X):
        return self.featureExtractor.transform(X)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)