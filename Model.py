from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import time
import sys

class Model():
    def __init__(self, df):
        self.num_features = ['MSSubClass','LotArea']
        self.cat_features = ['MSZoning']
        self.x = df[['MSSubClass','LotArea','MSZoning']]
        self.y = df.SalePrice
    def preprocess(self, df):
        #define piplines
        p_numF = Pipeline(steps=[
            ('Imp', SimpleImputer()),
            ('std', StandardScaler(with_mean=True, with_std=True))
        ])
        p_catF = Pipeline(steps=[
            ('mput',  SimpleImputer(strategy='most_frequent')),
            ('ondH', OneHotEncoder())
        ])
        #transform or preprocess the columns
        Preprocess = ColumnTransformer(
            transformers=
            [
                ('numCol', p_numF, self.num_features),
                ('catCol', p_catF, self.cat_features)
            ],
             remainder='drop',
        )
        
        return Preprocess
    def loading_animation(self, duration):
        symbols = ['-', '\\', '|', '/']
        for i in range(int(duration * 10)):
            for symbol in symbols:
                sys.stdout.write(f'\rLoading {symbol}')
                sys.stdout.flush()
                time.sleep(0.1)
        print("\rLoading complete!")

    def train(self):
        #preprocessing the data 
        Preprocess = self.preprocess(self.x)

        # linear regression 
        lin_reg = LinearRegression()
        reg_params = {
            "fit_intercept": [True, False],  
            "copy_X": [True, False],        
            "positive": [True, False],      # Force coefficients to be positive
            "n_jobs": [None, -1]  
        }
        #GridSearchccv
        Gcv = GridSearchCV(estimator=lin_reg, param_grid=reg_params, cv=3)
        
        model = make_pipeline(Preprocess, Gcv)
        model.fit(self.x, self.y)
        print(model.score(self.x, self.y))