import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):
        try:
            logging.info('Splitting training and testing input')




            ## X_train,y_train,X_test,y_test=(
               # train_array[:,:-1],
               # train_array[:,-1],
               # test_array[:,:-1],
                #test_array[:,-1]
               # )
            
            model=LogisticRegression()
            model.fit(X_train,y_train)
            score=model.score(X_train,y_train)
            #print(f"model score is {score}")
            y_pred=model.predict(X_test)
            accuracy=accuracy_score(y_test,y_pred)
            #print(f'accuracy score is {accuracy}')
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            return accuracy


        except Exception as e:
            raise CustomException(e,sys)