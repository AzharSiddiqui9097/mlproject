import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exception import CustomException
from src.logger import logging
from data_transformation import DataTransformation,DataTransformationConfig
from model_trainer import ModelTrainer,ModelTrainerConfig

import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestionConfig:
    train_data_path = os.path.join('Artifacts','Train.csv')
    test_data_path = os.path.join('Artifacts','Test.csv')
    raw_data_path = os.path.join('Artifacts','Raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the data as Dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train and Test split Initiated")
            train_set,test_set = train_test_split(df,test_size=0.25,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)
            logging.info("Data ingestion is completed")
            return self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e,sys)
            # pass
        
if __name__=='__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    modelTrainer = ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_arr,test_arr))