import streamlit as st  
import pandas as pd  
from src.data_preprocessing import load_and_clean_data, split_data, create_preprocessor
from src.model_training import train_and_evaluate
from config import DATA_PATH, TARGET, SCALER_TYPE

st.title("Cement Strength Prediction")

# Optionally upload a file or use the default dataset from DATA_PATH
uploaded_file = st.file_uploader("Upload your CSV file (optional)", type=["csv"])

if uploaded_file:
    # If a user uploads a file, load it
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(df.head())
else:
    # If no file is uploaded, use the default dataset from DATA_PATH
    st.write(f"Using the default data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        st.write("Loaded Data from default path:")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error loading data from {DATA_PATH}: {e}")
        st.stop()

# Debugging: Display the type of 'df' to ensure it's a DataFrame
st.write(f"Data type after loading: {type(df)}")

# Cleaning the data: Removing duplicates
st.write("Cleaning the Data...")
try:
    # Change this line to load the data from DATA_PATH, not df
    df_cleaned = load_and_clean_data(DATA_PATH)  # Corrected to pass DATA_PATH
    st.write("Data cleaned successfully!")
    st.write(df_cleaned.head())
except Exception as e:
    st.error(f"Error in cleaning the data: {e}")
    st.stop()

# Splitting the data into training and testing sets
st.write("Splitting the data into training and testing sets...")
xtrain, xtest, ytrain, ytest = split_data(df_cleaned, TARGET)

# Create the preprocessing pipeline
preprocessor = create_preprocessor(SCALER_TYPE)

# Train and evaluate the models
st.write("Training and Evaluating Models...")
try:
    result = train_and_evaluate(xtrain, ytrain, xtest, ytest, preprocessor)
    st.write(result)  # Display the result in the Streamlit app
except Exception as e:
    st.error(f"Error during training and evaluation: {e}")
