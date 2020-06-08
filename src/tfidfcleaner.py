import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn import feature_extraction, linear_model, model_selection, metrics



class dataCleaner:
    
    
    def __init__(self, ngrams = 3, max_features = 20000):
        """
        Input:
            ngrams: how deep you want your ngrams to be
            max_features: how many features to create using tfidf
        Output:
            None
            
        This just sets up our tfidf pipeline by initializing the feature extractor
        """
        
        
        self.token = nltk.tokenize.casual.TweetTokenizer()
        self.stop_words = set(stopwords.words('english')) 
        
        self.featureExtractor = feature_extraction.text.TfidfVectorizer(
                                                strip_accents = 'ascii',
                                                lowercase = True,
                                                tokenizer = self.token.tokenize,
                                                stop_words = self.stop_words,
                                                ngram_range = (1, ngrams),
                                                max_features = max_features
        )

    

    
    def fit(self, X):
        """
        Input:
            X: a list of words to fit our tfidf vectorizer to
        Output:
            None
            
        This fits X to our tfidf vectorizer
        """
        lstripped = map(lambda x: x.lstrip('!'), list(X))
        self.featureExtractor.fit(lstripped)
    
    def transform(self, X):
        """
        Input:
            X: a list of words to use our tfidf vectorizer on
        Output:
            A transformed version of X
            
        This just runs our tfidf vectorizer on X
        """
        lstripped = map(lambda x: x.lstrip('!'), list(X))
        return self.featureExtractor.transform(X)
    
    def fit_transform(self, X):
        """
        Input:
            X: a list of words to fit and then use our tfidf vectorizer on
        Output:
            A transformed version of X
            
        This combines the fit and transform steps
        """
        lstripped = map(lambda x: x.lstrip('!'), list(X))
        
        return self.featureExtractor.fit_transform(lstripped)