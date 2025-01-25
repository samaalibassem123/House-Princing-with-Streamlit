from sklearn.preprocessing import OneHotEncoder, StandardScaler,QuantileTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
import numpy as np
import pandas as pd
import time
import sys

class Model():
    def __init__(self, df):
        self.df = df
        self.num_features = ['MSSubClass','LotArea']
        self.cat_features = ['MSZoning']
        self.x = self.df[['MSSubClass','MSZoning','LotArea']]
        self.y = self.df.SalePrice
        self.mdl = None
    def preprocess(self, X):
        #define the methods for piplines
        imputer = SimpleImputer()
        oneH = OneHotEncoder(handle_unknown='ignore')
        Stdr = StandardScaler(with_mean=False)
        qt = QuantileTransformer()
        #create piplines
        p_numF = make_pipeline(imputer,qt, Stdr)
        p_catF = make_pipeline(oneH, imputer, qt, Stdr)
        #transform or preprocess the columns
        Preprocess = make_column_transformer(
            (p_numF, self.num_features),
            (p_catF, self.cat_features),
            remainder='drop'
        )
        #training the transformer 
        Preprocess.fit_transform(self.df)
        #return the result
        return Preprocess.transform(X)

    def train(self):
        #preprocessing the data 
        x_process = self.preprocess(self.x)
        # linear regression 
        lin_reg = LinearRegression()
        reg_params = {
            "fit_intercept": [True, False],  
        }
        # SGDRegressor
        sgdr = SGDRegressor()
        sgdr_params = {
             "loss": ["squared_error", "huber", "epsilon_insensitive"]
        }
        # randomforest
        rdf = RandomForestRegressor()
        rdf_params = {
             "n_estimators": [50, 100, 200],      
        }
        #svr
        svr = SVR()
        svr_params ={
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        } 
        
        #GridSearchccv
        self.mdl = GridSearchCV(estimator=rdf, param_grid=rdf_params ,cv=30)
        self.mdl.fit(x_process, self.y)
        print('sayer')
        print(self.mdl.best_score_)
    
    def Predict(self, x):
        x_prepro = self.preprocess(x)
        res = self.mdl.predict(x_prepro)
        return res
        