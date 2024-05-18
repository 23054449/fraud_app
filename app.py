import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from pathlib import Path

model1 = pickle.load(open('LR_norm.pkl', 'rb'))
model2 = pickle.load(open('RF_norm.pkl', 'rb'))
model3 = pickle.load(open('KNN_norm.pkl', 'rb'))
model4 = pickle.load(open('NB_norm.pkl', 'rb'))
model5 = pickle.load(open('DT_norm.pkl', 'rb'))


# Get the user's home directory
home_directory = str(Path.home())

# Specify the Downloads folder path
downloads_directory = os.path.join(home_directory, "Downloads")

# Define the file path where you want to save the CSV
file_path = "/mount/src/fraud_app/output/predictions.csv"

# Ensure the directory exists
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

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
        st.write(st.session_state.data.assign(Predicted=st.session_state.new_data))
        # Display the total number of fraud rows
        # Calculate the number of fraud rows
        fraud_rows = np.sum(st.session_state.new_data != 0)
        st.write(f"Number of fraud transactions: {fraud_rows}")

        # Button to save the new_data to Excel
        if st.button("Save Predictions to Excel"):
            # Construct the full file path for the Downloads folder
            #file_path = os.path.join(downloads_directory, "predicted_data.csv")
            st.session_state.data.assign(Predicted=st.session_state.new_data).to_csv(file_path, index=False)
            st.write("Predicted data saved to 'predicted_data.csv'")