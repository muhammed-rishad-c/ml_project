import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("splitting train and test data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                "catboost":CatBoostRegressor(verbose=0),
                "adaboost":AdaBoostRegressor(),
                "gradientboost":GradientBoostingRegressor(),
                "random forest":RandomForestRegressor(),
                "linear regression":LinearRegression(),
                "kneighbor":KNeighborsRegressor(),
                "decision tree":DecisionTreeRegressor(),
                "xgboost":XGBRegressor()
            }
            model_report:dict=evaluation(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("no best model found")
            
            logging.info("found the best model for training and test data")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            y_pred=best_model.predict(x_test)
            
            r2score=r2_score(y_test,y_pred)
            return r2score
        except Exception as e:
            raise CustomException(e,sys)
            