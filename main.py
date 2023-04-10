import streamlit as st
import pandas as pd
import pickle
import time
# from gtts import gTTS

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

data = pd.read_csv('Cleaned_data.csv', index_col=[0])
pipe = pickle.load(open('LinearModel.pkl', 'rb'))

with header:
    st.title('House Price Predictor')
    st.text('In this project, I try to predict the price of the house based on locality, size, and other features of the house')

with dataset:
    st.header('Bengaluru House price data')
    st.text(
        'I used the the Bengaluru House price dataset from Kaggle. It is 5 years old.')

    st.write(data.head())


with features:
    st.header('Features used in training the model')
    st.text('Used 4 features to predict the price of the house based on locality, size, and other features of the house')

with modelTraining:
    st.header('Predict the price of the house')
    st.text('Please input the values in the required fields below. It will help train the model and predict the price of the house you want to buy in the respective locality in Bengaluru.')

    sel_col, dis_col = st.columns(2)
    location = sel_col.selectbox(
        'Locality of the house', options=data['location'].unique(), index=0)
    bhk = sel_col.slider('Number of bedrooms needed(in BHK)',
                         min_value=1, max_value=30)
    total_area = sel_col.number_input(
        'Total area of the house(in sqft)', min_value=300)
    bath = sel_col.slider('Number of bathrooms needed',
                          min_value=1, max_value=30, )

    input = pd.DataFrame([[location, float(total_area), float(bath), float(bhk)]], columns=[
        'location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0]

    def y_on_y_inflation(price):
        ans = price*1.06*1.06*1.06*1.06*1.06
        return ans
    dis_col.subheader('Predicted price of the house')
    dis_col.write(location)
    dis_col.write(total_area)
    dis_col.write(bhk)
    dis_col.write(bath+2)
    dis_col.write(input)
    price = prediction*1e5
    dis_col.write("Prediction: Rs. "+str(price))
    dis_col.write("Inflation adjusted price according to 2022: Rs. " +
                  str(y_on_y_inflation(price)))
    dis_col.write("Price per sqft: Rs. " +
                  str(y_on_y_inflation(price)/total_area))
