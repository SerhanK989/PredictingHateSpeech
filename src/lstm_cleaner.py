import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn import feature_extraction, linear_model, model_selection, metrics
from sklearn import ensemble
from scipy import sparse
import sys
sys.path.append("..")
from src.make_model_lstm import hate_speech_model



class lstm_cleaner():
    
    
    def __init__(self, token = nltk.tokenize.casual.TweetTokenizer(), 
                 sw = set(stopwords.words('english')),
                lem = WordNetLemmatizer()):
        """
        Input: 
            token: the tokenizer to use
            sw: the set of stop words to use
            lem: the lemmatizer to use
        Output:
            None
        
        This function just saves the parameters of our lstm cleaner in our model
        """
        
        self.stop_words = sw
        self.token = token
        self.lem = WordNetLemmatizer()
        self.wordDict = None
        self.maxWords = 0
        
    
    def fit(self, X):
        """
        Input:
            X: the list of tweets we fit our model to
        Output:
            None
            
        This function fits X to our cleaner
        Mainly it learns the word count and the maximum number of words in a tweet
        """
        
        lstripped = map(lambda x: x.lstrip('!'), list(X))
        
        
        self.wordDict = {}
        i = 1
        for tweet in lstripped:
            tokenized = self.token.tokenize(tweet)
            newSent = []
            for word in tokenized:
                if word not in self.stop_words:
                    newWord = self.lem.lemmatize(word)
                    if newWord not in self.wordDict:
                        self.wordDict[newWord] = i
                        i += 1
                    newSent.append(newWord)
            
            self.maxWords = max(self.maxWords, len(newSent))
            
    def transform(self, X):
        """
        Input:
            X: the list of tweets we transform
        Output:
            A list of tweets with each one transformed into an integer sequence
            
        This takes in X as our input and maps it to an array of integers to run on our LSTM.
        """
        
        lstripped = map(lambda x: x.lstrip('!'), list(X))
        
        cleaned = []
        for tweet in lstripped:
            tokenized = self.token.tokenize(tweet)
            newSent = []
            for word in tokenized:
                if word not in self.stop_words:
                    newWord = self.lem.lemmatize(word)
                    if newWord not in self.wordDict:
                        newSent.append('')
                    else:
                        newSent.append(newWord)
            
            cleaned.append(newSent)
                    
        def create_seq(sent, vocab, maxWords):
            n = len(sent)
            numZeros = maxWords - n
            result = [0]*numZeros

            for word in sent:
                if word in vocab:
                    result.append(vocab[word] + 2)
                else:
                    result.append(1)

            return result
            
        sequences = np.array(list(map(lambda x: create_seq(x, self.wordDict, self.maxWords), cleaned)))
        return sequences
    
    def fit_transform(self, X):
        """
        Input:
            X: the list of tweets we fit and then transform
        Output:
            A list of tweets with each one transformed into an integer sequence
            
        This runs the fit and transform functions together
        """
        lstripped = map(lambda x: x.lstrip('!'), list(X))
        
        cleaned = []
        self.wordDict = {}
        i = 1
        for tweet in lstripped:
            tokenized = self.token.tokenize(tweet)
            newSent = []
            for word in tokenized:
                if word not in self.stop_words:
                    newWord = self.lem.lemmatize(word)
                    if newWord not in self.wordDict:
                        self.wordDict[newWord] = i
                        i += 1
                    newSent.append(newWord)
            
            cleaned.append(newSent)
            
        self.maxWords = max(map(len, cleaned))
        
        def create_seq(sent, vocab, maxWords):
            n = len(sent)
            numZeros = maxWords - n
            result = [0]*numZeros

            for word in sent:
                if word in vocab:
                    result.append(vocab[word] + 2)
                else:
                    result.append(1)

            return result
            
        sequences = np.array(list(map(lambda x: create_seq(x, self.wordDict, self.maxWords), cleaned)))
        return sequences
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    