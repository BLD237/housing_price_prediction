import streamlit as st
import numpy as np
import joblib
import time 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
st.header('House price estimator')
with st.form('form'):
    st.subheader('House Description') 
    area =  st.slider("Squared feet Area", 1, 20000)
    bedrooms =  st.slider("Number of bedrooms", 1, 5) 
    bathrooms =  st.slider("Number of bathrooms", 1, 3)  
    stories =  st.slider("Number of Stories", 1, 4)   
    parking = st.slider("parking space", 1, 4)
    mainroad =  st.radio("Around the Main road", ("Yes", "No"))
    guestroom =  st.radio("Need A guest room?", ("Yes", "No"))
    basement =  st.radio("Need A Basement?", ("Yes", "No"))
    hotwaterheating = st.radio("Need hot water heating?", ("Yes", "No"))
    airconditioning = st.radio("Need Air conditioning?", ("Yes", "No"))   
    prefarea = st.radio("Need prefarea?", ("Yes", "No"))
    furnishingstatus = st.radio("Funishing status", ("semi-furnished", "furnished", 'unfurnished'))    
    summited = st.form_submit_button('Predict')
if summited:    
    col1, col2 = st.columns([3, 1])
    progress_bar = col1.progress(0)
    col2.write("Predicting...")
    time.sleep(2)  
    progress_bar.progress(10) 
    model = joblib.load('models/housing.joblib')
    time.sleep(2) 
    progress_bar.progress(30) 
    data = {
        "area":area,
        'bedrooms':bedrooms,
        'bathrooms':bathrooms,
        'stories':stories,
        'mainroad':mainroad,
        'guestroom':guestroom,
        'basement':basement,
        'hotwaterheating':hotwaterheating,
        'airconditioning':airconditioning,
        'parking':parking,
        'prefarea':prefarea,
        'furnishingstatus':furnishingstatus,        
    }
    le = LabelEncoder()
    df = pd.DataFrame([data])   
    df.mainroad = le.fit_transform(df.mainroad)
    df.guestroom = le.fit_transform(df.guestroom)
    df.basement =  le.fit_transform(df.basement)
    df.hotwaterheating = le.fit_transform(df.hotwaterheating)
    df.airconditioning =  le.fit_transform(df.airconditioning)
    df.prefarea =  le.fit_transform(df.prefarea)
    df.furnishingstatus = le.fit_transform(df.furnishingstatus)  
     
    predection = model.predict(df)
    st.write(df)
    st.write(f"Predicted price: {predection[0]}")
    
    
    


    