import streamlit as st
import pandas as pd
from Model import Model



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
#transforming data to a dataFrame
x = [
    {
        'MSSubClass':ms_class,
        'MSZoning':ms_zoning,
        'LotArea':lot_ar
    }
]
predict_df = pd.DataFrame(x)
st.write(predict_df.head())

#train the model

model = Model(df)
model.train()


# predict a a given inputs
def on_click():
    res = model.Predict(test)
    st.write(res)
st.button('click me ',on_click=on_click)



