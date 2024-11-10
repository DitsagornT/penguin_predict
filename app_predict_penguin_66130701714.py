import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained model and encoders
with open('model_penguin_66130701714.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Function to make predictions
def predict_penguin(sex, island, bill_length, bill_depth, flipper_length, body_mass):
    try:
        # Check if the input labels are valid
        sex_transformed = sex_encoder.transform([sex])[0]
        island_transformed = island_encoder.transform([island])[0]
        
        # Debugging outputs
        st.write(f"Sex Transformed: {sex_transformed}")
        st.write(f"Island Transformed: {island_transformed}")
        
        # Prepare the input data
        input_data = np.array([[sex_transformed, island_transformed, bill_length, bill_depth, flipper_length, body_mass]])
        
        # Debugging input data
        st.write(f"Input Data: {input_data}")
        
        # Predict the species
        species_pred = model.predict(input_data)
        species = species_encoder.inverse_transform(species_pred)[0]
        
        return species
    
    except ValueError as e:
        # If there is an unseen label, handle it
        return f"Error: {str(e)}"

# Streamlit UI
st.title("Penguin Species Prediction")

# Get user inputs
sex = st.selectbox("Sex of the penguin", ['Male', 'Female'])
island = st.selectbox("Island of the penguin", ['Torgersen', 'Biscoe', 'Dream'])
bill_length = st.slider("Bill Length (mm)", 30.0, 70.0, 45.0)
bill_depth = st.slider("Bill Depth (mm)", 10.0, 25.0, 15.0)
flipper_length = st.slider("Flipper Length (mm)", 170.0, 230.0, 200.0)
body_mass = st.slider("Body Mass (g)", 2500.0, 6500.0, 4500.0)

# Button to make the prediction
if st.button('Predict'):
    species = predict_penguin(sex, island, bill_length, bill_depth, flipper_length, body_mass)
    st.write(f"The predicted species is: {species}")
