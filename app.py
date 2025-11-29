import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('geo_encoder.pkl','rb') as file:
    label_geo = pickle.load(file)

with open('gender_encoder.pkl','rb') as file:
    label_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')

#User input
geography = st.selectbox('Geography',label_geo.categories_[0])
gender = st.selectbox('Gender',label_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_Score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',1,10)
num_of_products = st.slider('Number of products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Activte Member',[0,1])

input_data = {
    'Geography': geography,   # <---- FIX ADDED
    'CreditScore': credit_Score,
    'Gender': label_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

# Geography OHE
geo_encoded = label_geo.transform([[input_data['Geography']]])
geo_df = pd.DataFrame(
    geo_encoded,
    columns=label_geo.get_feature_names_out(['Geography'])
)

# Convert input_data to DF
df_num = pd.DataFrame([input_data])

df_num = df_num.drop(columns=['Geography'])

# Final DF
final_df = pd.concat([df_num, geo_df], axis=1)

# Scaling
input_scaled = scaler.transform(final_df)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to Churn')
else:
    st.write('The customer is not likely to Churn')
