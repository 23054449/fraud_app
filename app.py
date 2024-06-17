import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from io import BytesIO
import requests
import gdown

def download_and_load_model(url, model_path):
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model1_url = 'https://drive.google.com/uc?export=download&id=1-H7qGZvWYJPl_Zd2DFLd3T8EHxGIlRem'
model2_url = 'https://drive.google.com/uc?export=download&id=1-7R1OEDAa3Le9t8vnTZ76V4naCXxLra6'
model3_url = 'https://drive.google.com/uc?export=download&id=1-S-MlnEvsfDRBXBAIyXLSYwaJ9-Zqbq3'
model4_url = 'https://drive.google.com/uc?export=download&id=1-DC2SGX_KAN7Q8GScvMAyoOm5rzhwXmy'
model5_url = 'https://drive.google.com/uc?export=download&id=1-6uaVpwkZjJWmyzBwH9ZtbgD1gdtaPVi'

# Paths to save the downloaded models
model1_path = 'LR_over.pkl'
model2_path = 'RF_norm.pkl'
model3_path = 'KNN_under.pkl'
model4_path = 'NB_over.pkl'
model5_path = 'DT_norm.pkl'

# Download and load the trained models
model1 = download_and_load_model(model1_url, model1_path)
model2 = download_and_load_model(model2_url, model2_path)
model3 = download_and_load_model(model3_url, model3_path)
model4 = download_and_load_model(model4_url, model4_path)
model5 = download_and_load_model(model5_url, model5_path)

# model1 = pickle.load(open('LR_over.pkl', 'rb'))
# model2 = pickle.load(open('RF_norm.pkl', 'rb'))
# model3 = pickle.load(open('KNN_under.pkl', 'rb'))
# model4 = pickle.load(open('NB_over.pkl', 'rb'))
# model5 = pickle.load(open('DT_norm.pkl', 'rb'))

# Title header
st.title("Fraud Detection App")

# File uploader (accepts both .xlsx and .csv files)
uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])

if "data" not in st.session_state:
    st.session_state.data = None

if "new_data" not in st.session_state:
    st.session_state.new_data = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Select a model"

# Submit button
if st.button("Submit"):
    if uploaded_file is not None:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.xlsx'):
            st.session_state.data = pd.read_excel(uploaded_file, index_col=[0])
        elif uploaded_file.name.endswith('.csv'):
            st.session_state.data = pd.read_csv(uploaded_file, index_col=[0])
        else:
            st.write("Unsupported file type. Please upload a .xlsx or .csv file.")
            st.stop()
    else:
        st.write("Please upload an Excel or CSV file first.")

# Create a container for displaying data
data_container = st.container()

with data_container:
    if st.session_state.data is not None:
        st.write("Here is the data from the file you uploaded:")
        st.write(st.session_state.data)
        # Display the total number of rows
        num_rows = st.session_state.data.shape[0]
        st.write(f"Total number of rows: {num_rows}")

# Title for the dropdown list using HTML for precise spacing
st.markdown("<h3 style='margin-bottom:0'>Select Prediction Model</h3>", unsafe_allow_html=True)

# Dropdown list
option = st.selectbox(
    "",
    ["Select a model", "Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Naive Bayes", "Decision Tree", "All"]
)

# Check if the selected model has changed
if option != st.session_state.selected_model and option != "Select a model":
    st.session_state.new_data = None
    st.session_state.selected_model = option

# Display the selected model and load it
if option and option != "Select a model":
    st.write("You have selected ", option)

# Submit button
if st.button("Start Fraud Detection!"):
    if option and option != "Select a model":
        if option == "Logistic Regression":
            y = model1.predict(st.session_state.data)
        elif option == "Random Forest":
            y = model2.predict(st.session_state.data)
        elif option == "K-Nearest Neighbors":
            y = model3.predict(st.session_state.data)
        elif option == "Naive Bayes":
            y = model4.predict(st.session_state.data)
        elif option == "Decision Tree":
            y = model5.predict(st.session_state.data)
        else:
            y = np.mean([model1.predict(st.session_state.data), model2.predict(st.session_state.data), model3.predict(st.session_state.data), model4.predict(st.session_state.data), model5.predict(st.session_state.data)], axis=0)
        st.session_state.new_data = y

    else:
        st.write("Please select a model to use first.")

# Create a container for displaying data
data_container1 = st.container()
with data_container1:
    if st.session_state.new_data is not None:
        st.write("Here are the predicted results")
        results_df = st.session_state.data.assign(Predicted=st.session_state.new_data)
        st.write(results_df)
        # Calculate the number of fraud rows
        fraud_rows = np.sum(st.session_state.new_data != 0)
        st.write(f"Number of fraud transactions: {fraud_rows}")

        # Convert the DataFrame to CSV and create a download button
        csv = results_df.to_csv(index=False).encode()
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )