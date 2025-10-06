import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# --- 1. LOAD THE TRAINED MODEL AND PREPROCESSORS ---

# Load the trained Keras model
try:
    Model = tf.keras.models.load_model("model.h5")
except AttributeError:
    Model = tf.keras.load_model("model.h5")


# Load the encoders and scaler
with open("label_encoder-gender.pkl", "rb") as file:
    label_encoder_genders = pickle.load(file)

with open("One_hot_encoder1.pkl", "rb") as file:
    One_hot_encoder11 = pickle.load(file)

with open("scaler1.pkl", "rb") as file:
    scaler11 = pickle.load(file)


# --- 2. STREAMLIT APP INTERFACE ---

st.title("Customer Churn Prediction 📊")
st.write("Enter the customer details to predict the likelihood of churn.")

# User input
Geography = st.selectbox("Geography", One_hot_encoder11.categories_[0])
Gender = st.selectbox("Gender", label_encoder_genders.classes_)
age = st.slider("Age", 18, 100, 35)

# CHANGED: Replaced number_input with slider as requested
Balance = st.slider("Balance", 50, 100000, 50000)

# CHANGED: Replaced number_input with slider as requested
Estimated_salary = st.slider("Estimated Salary", 50, 100000, 50000)

tensure = st.slider("Tenure (years with the bank)", 0, 10, 5)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card? (1=Yes, 0=No)", [0, 1])
is_active_member = st.selectbox("Is Active Member? (1=Yes, 0=No)", [0, 1])


# --- 3. PREDICTION LOGIC ---

if st.button("Predict Churn"):

    # Create initial DataFrame (your method)
    input_data = pd.DataFrame({
        'Gender': [label_encoder_genders.transform([Gender])[0]],
        'Age': [age],
        'Tenure': [tensure],
        'Balance': [Balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [Estimated_salary]
    })

    # One-hot encode "Geography"
    geo_encoded_array = One_hot_encoder11.transform([[Geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded_array, columns=One_hot_encoder11.get_feature_names_out(["Geography"]))

    # Combine DataFrames
    final_input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Ensure correct column order
    model_feature_order = [
        'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_France', 'Geography_Germany', 'Geography_Spain'
    ]
    final_input_df = final_input_df[model_feature_order]

    # Scale the input
    input_data_scaled = scaler11.transform(final_input_df)

    # Predict churn
    prediction = Model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Display the result
    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {prediction_proba:.2f}")

    if prediction_proba > 0.5:
        st.error("The customer is LIKELY to churn.")
    else:
        st.success("The customer is NOT likely to churn.")