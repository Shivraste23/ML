import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Load the trained model
model_file = "model1.pkl"

# Check if the model file exists
if not os.path.exists(model_file):
    st.error(f"Model file not found at: {model_file}")
else:
    with open(model_file, 'rb') as f:
        trained_model = pickle.load(f)

# Function to preprocess data and make predictions
def preprocessDataAndPredict(Summary, Humidity, WindSpeed, WindBearing, Visibility, Pressure):
    test_data = [[Summary, Humidity, WindSpeed, WindBearing, Visibility, Pressure]]
    test_data = np.array(test_data)
    test_data = pd.DataFrame(test_data)
    prediction = trained_model.predict(test_data)
    return prediction

# Streamlit app
def main():
    st.title("Weather Prediction App")
    st.write("Enter the weather details to get the prediction.")

    # Input fields
    Summary = st.text_input("Summary")
    Humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=50.0)
    WindSpeed = st.number_input("Wind Speed", min_value=0.0, value=10.0)
    WindBearing = st.number_input("Wind Bearing", min_value=0.0, max_value=360.0, value=180.0)
    Visibility = st.number_input("Visibility", min_value=0.0, value=10.0)
    Pressure = st.number_input("Pressure", min_value=0.0, value=1013.0)

    # Prediction button
    if st.button("Predict"):
        try:
            prediction = preprocessDataAndPredict(Summary, Humidity, WindSpeed, WindBearing, Visibility, Pressure)
            st.success(f"Prediction: {round(prediction[0], 2)}")
        except ValueError:
            st.error("Error: Please enter valid numerical values for all fields.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    main()