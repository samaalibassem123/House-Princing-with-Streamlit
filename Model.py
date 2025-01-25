from sklearn.preprocessing import OneHotEncoder, StandardScaler,QuantileTransformer, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer,make_column_transformer, make_column_selector
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
import numpy as np
import pandas as pd

class Model():
    def __init__(self, df):
        self.df = df
        self.num_features = make_column_selector(dtype_include='number')
        self.cat_features = make_column_selector(dtype_exclude='number')
        self.x = self.df.drop('SalePrice',axis=1)
        self.y = self.df.SalePrice
        self.model = None
    def train(self):
        #Initilize the functions for preprocessing data
        one_H = OneHotEncoder(handle_unknown='ignore') # to transform categoral varibels to numerical ones
        Quant = QuantileTransformer(n_quantiles=974) # scaling data
        Normal  = MinMaxScaler() # scaling data
        std  = StandardScaler(with_mean=False) # scaling data
        impute = SimpleImputer() # to fill the null values
       
        # Creating piplines
        numerical_pipline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='mean')), # first we need to fill the null values
                ('std', std), # scale the data                    
                ('qt', Quant),  # another scaler  
             # another scaler  
            ],
        )
        categoral_pipline = Pipeline(
            [
                ('oneHot', one_H),
                ('imputer', SimpleImputer(strategy='constant')),
                ('qt', Quant),  
                ('std', std ),
                # another scaler  
            ]
        )

        # create the transformer for data preprocessing 
        TransformC = ColumnTransformer(
            [
                ('transform_num', numerical_pipline, self.num_features),# transform the numerical columns
                ('transform_cat', categoral_pipline, self.cat_features),# transform the categorall columns
            ],
            remainder='drop'
        )

        # create the piplines that activate the models
        model_pipe = Pipeline([ ('Preprocess', TransformC), ('model', LinearRegression()) ])
   

        #create the grid params for each model
        r1_param = {
            'model__fit_intercept': [True, False],
            'model':[LinearRegression()]
        }
        r2_param = {
            "model__loss": ["squared_error", "huber", "epsilon_insensitive"],
            "model__penalty": ["l2", "l1", "elasticnet"],
            'model':[SGDRegressor()]
        }
        r3_param = {
            "model__n_estimators": [50, 100, 200],
            'model':[RandomForestRegressor()]
        }
        r4_param = {
            'model__kernel': ["linear", "poly", "rbf", "sigmoid"],
            "model__C": [0.1, 1, 10, 100],
            'model':[SVR()]
        }

        prams = [r1_param, r2_param, r3_param, r4_param]

        
        #find the best model 
        self.model = RandomizedSearchCV(estimator=model_pipe, param_distributions=prams, n_iter=1000, random_state=46, cv=3, n_jobs=-1)
        self.model.fit(self.x,self.y)
        print(self.model.best_estimator_)
        print(self.model.best_params_)
        print(self.model.best_score_)

    def Predict(self, x):
        res = self.model.predict(x)
        return res
        