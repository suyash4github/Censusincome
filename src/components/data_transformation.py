import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_features =['workclass', 'education', 'married_status', 'occupation','relationship', 'race', 'sex', 'native_country']
            integer_features=['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss','hrs_pr_week']
            num_pipeline=Pipeline(
                steps=[
                ('scaler',StandardScaler())]
            )
            cat_pipeline=Pipeline(
                steps=[               
                ('onehotencoder',OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            logging.info('Numerical columns standard scaling compeleted')
            logging.info('categorical columns onehot encoding completed')

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,integer_features),
                ('cat_pipeline',cat_pipeline,categorical_features)
            ])

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name='income'
            

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            ##train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            
            ##test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessing_obj
            )

            return(
                input_feature_train_arr,
                target_feature_train_df,
                input_feature_test_arr,
                target_feature_test_df,
                self.data_transformation_config.preprocessor_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

