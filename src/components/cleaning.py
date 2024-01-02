import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm import tnrange, tqdm_notebook, tqdm
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

#from dataclasses import dataclass

#from src.components.data_ingestion import DataIngestion
#from src.components.data_ingestion import DataIngestionConfig



   

class cleaning_process:
    def __init__(self):
        pass

    @staticmethod
    def data_cleaning(self,dd):
        logging.info("data cleaning started")
        for i,s in enumerate(tqdm(dd['text'],position=0, leave=True)):
            text = str(dd.loc[i, 'text'])
            text = text.replace("#", "")
            text = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text, flags=re.MULTILINE)
            text = re.sub('@\\w+ *', '', text, flags=re.MULTILINE)
            dd.loc[i, 'text'] = text
        return dd
    
    @staticmethod
    def cleaning(data):
        # nltk.download('wordnet')
        # nltk.download('stopwords')
        # nltk.download('punkt')
        stop_words = nltk.corpus.stopwords.words(['english'])
        lem = WordNetLemmatizer()
        #remove urls
        tweet_without_url = re.sub(r'http\S+',' ', data)

        #remove hashtags
        tweet_without_hashtag = re.sub(r'#\w+', ' ', tweet_without_url)

        #3. Remove mentions and characters that not in the English alphabets
        tweet_without_mentions = re.sub(r'@\w+',' ', tweet_without_hashtag)
        precleaned_tweet = re.sub('[^A-Za-z]+', ' ', tweet_without_mentions)

        #2. Tokenize
        tweet_tokens = TweetTokenizer().tokenize(precleaned_tweet)

        #3. Remove Puncs
        tokens_without_punc = [w for w in tweet_tokens if w.isalpha()]

        #4. Removing Stopwords
        tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]

        #5. lemma
        text_cleaned = [lem.lemmatize(t) for t in tokens_without_sw]

        #6. Joining
        return " ".join(text_cleaned)
    
    @staticmethod
    def getSubjectivity(tweet):
        return TextBlob(tweet).sentiment.subjectivity
    
    @staticmethod
    def getPolarity(tweet):
        return TextBlob(tweet).sentiment.polarity
    
    @staticmethod
    def crypto_price_cate(score):
        if score < 1:
            return 'negative'
        elif score == 1:
            return 'neutral'
        else:
            return 'positive'
    
    @staticmethod
    def observe_period(period,crypto_usd_grouped):
        res = crypto_usd_grouped.shift(period)/crypto_usd_grouped
        res = res.apply(cleaning_process.crypto_price_cate)
        return res 
    
    @staticmethod
    def getSentiment(score):
        if score < 0:
            return 'negative'
        elif score == 0:
            return 'neutral'
        else:
            return 'positive'
            









