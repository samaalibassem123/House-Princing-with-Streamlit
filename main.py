import streamlit as st
import pandas as pd
from Model import Model




# Initilize the model
df = pd.read_csv('Data/train.csv')

@st.cache_data
def initializeModel(df):
    model = Model(df)
    model.train()
    return model

model = initializeModel(df)



#validation data 
valid = pd.read_csv('Data/sample_submission.csv')

st.write('# House Pricing prediction')

st.write(df.head())


test_df = st.file_uploader('drop the test file')

if test_df:
    test = pd.read_csv(test_df)
    st.write(test)
    st.write(model.x)
 

# predict a a given inputs
def on_click():
    global res,ok
    try:
        res = model.Predict(test)
    except:
        st.warning('Choose a file')
        return
    st.session_state['res'] = res

st.button('click me ',on_click=on_click)

# print the result
if 'res' in st.session_state:
    res = st.session_state['res']
    st.write(res)
    st.write(model.Accuracy(valid.SalePrice,res))
    st.write('## sale prediction with the real ones')
    chart_df = pd.DataFrame({
        'Real SalePrice':valid.SalePrice,
        'Prediction SalePrice':res,
    })

    st.scatter_chart(chart_df, color=["#FF0000", "#0000FF"])
    
