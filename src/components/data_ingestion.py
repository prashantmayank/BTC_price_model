import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np


from dataclasses import dataclass

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from src.components.data_preprocessing import DataPreprocessing
from src.components.data_preprocessing import DataPreprocessingConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    #all the output from this file will be stored in artifact folder
    # train_data_path: str=os.path.join('artifacts',"train.csv")
    # test_data_path: str=os.path.join('artifacts',"test.csv")
    btc_price_data_path: str=os.path.join('artifacts',"btc_price.csv")
    raw_tweets_data_path: str=os.path.join('artifacts',"raw_tweets.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df_price = pd.read_csv('/Users/prashantkumarmayank/Documents/BTC_price_model/notebook/data/BTC-USD.csv')
            df_raw = pd.read_csv('/Users/prashantkumarmayank/Documents/BTC_price_model/notebook/data/Bitcoin_tweets.csv')
            logging.info('Read the dataset as dataframe')

            #os.makedirs(os.path.dirname(self.ingestion_config.raw_tweets_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.btc_price_data_path),exist_ok=True) ##it should be false
            #this is like initiallising the directory

            df_price.to_csv(self.ingestion_config.btc_price_data_path,index=False,header=True)
            df_raw.to_csv(self.ingestion_config.raw_tweets_data_path,index=False,header=True)



            #logging.info("Train test split initiated")
            # train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            # train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            # test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.btc_price_data_path,
                self.ingestion_config.raw_tweets_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    btc_price,raw_tweets=obj.initiate_data_ingestion()

    data_preprocessing=DataPreprocessing()
    clean_data=data_preprocessing.initiate_data_processing(btc_price,raw_tweets)

    modeltrainer=ModelTrainer()
    y_test,y_pred_test=modeltrainer.initiate_model_trainer(clean_data)
    print('Accuracy:\t{:0.1f}%'.format(accuracy_score(np.argmax(y_test,axis=1),y_pred_test)*100))
    print(classification_report(np.argmax(y_test,axis=1), y_pred_test))







