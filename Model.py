from sklearn.preprocessing import OneHotEncoder, StandardScaler,QuantileTransformer, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer,make_column_transformer, make_column_selector
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

class Model():
    def __init__(self, df):
        self.df = df
        self.num_features = make_column_selector(dtype_include='number')
        self.cat_features = make_column_selector(dtype_exclude='number')
        self.x = self.df.drop(['Id', 'Alley', 'EnclosedPorch', '3SsnPorch', 'PoolQC', 'Fence','MiscFeature','SalePrice'],axis=1)
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
             
            ],
        )
        categoral_pipline = Pipeline(
            [
                ('oneHot', one_H),
                ('imputer', SimpleImputer(strategy='constant')),
        
               
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

            'model':[LinearRegression()]
        }
        r2_param = {

            'model':[SGDRegressor()]
        }
      
        r4_param = {
         
            'model':[SVR()]
        }

        prams = [r1_param, r2_param, r4_param]

        
        #find the best model 
        self.model = GridSearchCV(estimator=model_pipe, param_grid=prams, cv=30, n_jobs=-1)
        #split the data
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=45)
        self.model.fit(x_train,y_train)
        print(self.model.best_estimator_)
        print(self.model.best_params_)
        print(self.model.best_score_)
        pred = self.model.predict(x_test)
        print(self.model.score(x_test, y_test))
        print(r2_score(y_test, pred))

    def Predict(self, x):
        x = x.drop(['Id', 'Alley', 'EnclosedPorch', '3SsnPorch', 'PoolQC', 'Fence','MiscFeature'], axis=1)
        return self.model.predict(x)
    
    def Accuracy(self, x , y): 
        print(r2_score(x,y))
        return r2_score(x,y)

        