import streamlit as st
import pandas as pd
from Model import Model
def preprocessing():
    pass


#training data
df = pd.read_csv('Data/train.csv')

#testing data
test = pd.read_csv('Data/test.csv')

#validation data 

st.write('# House Pricing prediction')

st.write(df.head())

features = ['MSSubClass','MSZoning','LotArea']
mSZoning = ['RL', 'RM', 'C', 'FV', 'RH']

ms_class = st.slider('MSSubClass',min_value=df.MSSubClass.min(), max_value=df.MSSubClass.max())
ms_zoning = st.selectbox("Select the MsZoning",options=mSZoning)
lot_ar = st.number_input('LotArea')


print(ms_class, ms_zoning, lot_ar)


model = Model(df)
model.train()
