#import pickle
#import numpy as np
import streamlit as st
import model

#model = pickle.load(open('/home/swanand/Desktop/Python Projects/Car Price Prediction/model.pkl', 'rb'))

st.title("Used Car Price Predictor")
km_driven = st.text_input('KMs driven')
no_of_owners = st.text_input('No. of owners')
mileage = st.text_input('Mileage (kmpl)')
engine = st.text_input('Engine (cc)')
seats = st.text_input('No. of seats')
age = st.text_input('Age in years')
fuel_type = st.selectbox('Choose fuel type', ('CNG', 'Diesel', 'LPG', 'Petrol'))
seller_type = st.selectbox('Choose seller type', ('Individual', 'Dealer', 'Trustmark Dealer'))
transmission = st.selectbox('Choose transmission type', ('Manual', 'Automatic'))

if st.button('Predict'):
    res = model.predict_price(int(km_driven), int(no_of_owners), float(mileage), int(engine), int(seats), int(age), fuel_type, seller_type, transmission)
    st.subheader('Predicted price : %d Rs.' % (res))
