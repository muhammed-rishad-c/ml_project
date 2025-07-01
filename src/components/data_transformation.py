import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    
class DataTranformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_tranformation_object(self):
        logging.info("getting starting for data tranformation")
        try:
            cat_feature=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
        'test_preparation_course']
            
            num_feature=['reading_score', 'writing_score']
            
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one hot encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor=ColumnTransformer(
                [
                    ("standard scaler",num_pipeline,num_feature),
                    ("onehot encoder",cat_pipeline,cat_feature)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_pt,test_pt):
        try:
            train_df=pd.read_csv(train_pt)
            test_df=pd.read_csv(test_pt)
            logging.info("read train and test data")
            logging.info("obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_tranformation_object()
            
            target_column_name="math_score"
            numerical_columns=['reading_score', 'writing_score']
            
            input_feature_train_df=train_df.drop(target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info("applying preprocessing on training and test data")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
                
        except Exception as e:
            raise CustomException(e,sys)
            