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


from dataclasses import dataclass

# from src.components.data_ingestion import DataIngestion
# from src.components.data_ingestion import DataIngestionConfig

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer


from src.components.cleaning import *

@dataclass
class DataPreprocessingConfig:
    #all the output from this file will be stored in artifact folder
    clean_tweets_path: str=os.path.join('artifacts',"clean_data.csv")
   

class DataPreprocessing:
    def __init__(self):
        self.processing_config=DataPreprocessingConfig()

    

    def initiate_data_processing(self,btc_price,raw_tweets):
        logging.info("Entered the data preprocessing method or component")
        cleaning_obj=cleaning_process()
        try:
            btc_price_df=pd.read_csv(btc_price)
            raw_tweets_df=pd.read_csv(raw_tweets)

            logging.info("Read btc price data and tweets data completed")
            raw_tweets_df = raw_tweets_df.sort_values(by = 'date')
            dd = raw_tweets_df.sample(frac=0.01, replace=False, random_state=1) #taking 10 % data for further analysis
            dd.reset_index(inplace=True)

            dd=cleaning_obj.data_cleaning(self,dd)
            #adding column compound
            df_clean = dd.copy()
            analyzer = SentimentIntensityAnalyzer()
            compound = []
            for i,s in enumerate(tqdm(df_clean['text'],position=0, leave=True)):
                # print(i,s)
                vs = analyzer.polarity_scores(str(s))
                compound.append(vs["compound"])
            df_clean["compound"] = compound
            logging.info("compund column added")

            #adding column socores
            scores = []
            for i, s in tqdm(df_clean.iterrows(), total=df_clean.shape[0],position=0, leave=True):
                try:
                    scores.append(s["compound"] * ((int(s["user_followers"]))) * ((int(s["user_favourites"])+1)/int(s['user_followers']+1)) *((int(s["is_retweet"])+1)))
                except:
                    scores.append(np.nan)
            df_clean["score"] = scores
            logging.info("score column added")

            #started with btc price dataset 
            btc_price_df.Date = pd.to_datetime(btc_price_df.Date)


            # sentiment analysis 
            df_clean = df_clean.drop_duplicates()
            tweets = df_clean.copy()
            tweets['date'] = pd.to_datetime(tweets['date'],utc=True)
            tweets.date = tweets.date.dt.tz_localize(None)
            tweets.index = tweets['date']

            tweets_grouped = tweets.resample('1h')['score'].sum()

            crypto_usd = btc_price_df.copy()
            crypto_usd['Date'] = pd.to_datetime(crypto_usd['Date'], unit='s')
            crypto_usd.index = crypto_usd['Date']
           
            crypto_usd_grouped = crypto_usd.resample('D')['Close'].mean()

            beggining = max(tweets_grouped.index.min().replace(tzinfo=None), crypto_usd_grouped.index.min())
            end = min(tweets_grouped.index.max().replace(tzinfo=None), crypto_usd_grouped.index.max())
            tweets_grouped = tweets_grouped[beggining:end]
            crypto_usd_grouped = crypto_usd_grouped[beggining:end]

            logging.info("grouped data sentiment analysis done")

            df = df_clean.copy()
            df.dropna(subset=['hashtags'], inplace=True)
            df = df[['text']] 
            df.columns = ['tweets']

            df['cleaned_tweets'] = df['tweets'].apply(cleaning_obj.cleaning)
            df['date'] = df_clean['date']
            df['date_clean'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df.drop(columns='date',inplace=True)

            time_sentiment = cleaning_obj.observe_period(7,crypto_usd_grouped) # compare price ratio in 7 days. price_7_days_later/ price_now 
            df['crypto_sentiment'] = df.date_clean.apply(lambda x: time_sentiment[x] if x in time_sentiment else np.nan)

            df['subjectivity'] = df['cleaned_tweets'].apply(cleaning_obj.getSubjectivity)
            df['polarity'] = df['cleaned_tweets'].apply(cleaning_obj.getPolarity)

            df['sentiment'] = df['polarity'].apply(cleaning_obj.getSentiment)
            df['target'] = df['sentiment'] == df['crypto_sentiment']
            df.to_csv(self.processing_config.clean_tweets_path,index=False,header=True)
            logging.info("file clean data created in artifact folder")


            return(
                self.processing_config.clean_tweets_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        








