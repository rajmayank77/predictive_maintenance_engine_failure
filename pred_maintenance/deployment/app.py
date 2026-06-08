import streamlit as st 
import pandas as pd 
import joblib from huggingface_hub 
import hf_hub_download 

model_path = hf_hub_download(
    repo_id="rajmayank092018/predictive-maintenance-model", 
    filename="xgboost_predictive_maintenance.pkl" 
    ) 
model = joblib.load(model_path) 
st.title("Predictive Maintenance System") 
rpm = st.number_input("Engine RPM") 
oil_pressure = st.number_input("Lub Oil Pressure") 
fuel_pressure = st.number_input("Fuel Pressure") 
coolant_pressure = st.number_input("Coolant Pressure") 
oil_temp = st.number_input("Lub Oil Temperature") 
coolant_temp = st.number_input("Coolant Temperature") 
if st.button("Predict"): 
  data = pd.DataFrame({ 
          'Engine_RPM':[rpm], 
          'Lub_Oil_Pressure':[oil_pressure], 
          'Fuel_Pressure':[fuel_pressure], 
          'Coolant_Pressure':[coolant_pressure], 
          'Lub_Oil_Temperature':[oil_temp], 
          'Coolant_Temperature':[coolant_temp] 
          }) 
  pred = model.predict(data) 
  if pred[0] == 1:
    st.error("Maintenance Required") 
  else: 
    st.success("Engine Healthy")
