import streamlit as st 
import pandas as pd 
import tensorflow
import pickle
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import  load_model

data=pd.read_csv("final_scout_not_dummy.csv")
data=data[data.make_model!="Audi A2"]
data=data[data.make_model!="Renault Duster"]


car_counts=st.selectbox("Plots",["Car Ages","Price","Consumption"])

if car_counts=="Car Ages":
    st.subheader("Average Ages")
    st.bar_chart(data.groupby("make_model").mean()["age"])
        
if car_counts=="Price":
    st.subheader("Average Prices")
    st.bar_chart(data.groupby("make_model").mean()["price"])
if car_counts=="Consumption":    
    st.subheader("Average gas consumption")
    st.bar_chart(data.groupby("make_model").mean()["cons_comb"])

model=load_model("deeplr_model.h5")
transformer=pickle.load(open("transformer","rb"))
scaler=pickle.load(open("scaler","rb"))
age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
km=st.sidebar.slider("What is the km of your car", 0,350000, step=1000)
consumption=st.sidebar.slider("What is the mpg of your car", 1.0,10.0, step=0.1)
car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))

xgb_model=pickle.load(open("xgb_model.pkl","rb"))
xgb_col_transformer=pickle.load(open("xgb_col_transformer","rb"))
my_dict = {
    "age": age,
    "hp_kW": hp,
    "km": km,
    'cons_comb':consumption,
    "make_model": car_model
    
}



df = pd.DataFrame.from_dict([my_dict])

st.table(df)
new_df=transformer.transform(pd.DataFrame([my_dict])).ravel()
new_df_sc=scaler.transform(new_df.reshape(-1,1)).ravel()

if st.button("Predict with Neural Network"):
    a=model.predict(new_df_sc.reshape(1,-1))
    b=scaler.inverse_transform(a.reshape(-1,1))
    st.success(f"The estimated price is : $ {int(b)}")

if st.button("Predict with XGBoost") :
    df_transformed=xgb_col_transformer.transform(df)
    xgb_prediction=xgb_model.predict(df_transformed)
    st.success(f"The estimated price is :${int(xgb_prediction)}")
    
