import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained model and encoders
with open('model_penguin_66130701714.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Function to make predictions
def predict_penguin(island, culmen_length, culmen_depth, flipper_length, body_mass, sex):
    # Create a DataFrame for the input data
    x_new = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length],
        'culmen_depth_mm': [culmen_depth],
        'flipper_length_mm': [flipper_length],
        'body_mass_g': [body_mass],
        'sex': [sex]
    })

    # Transform categorical features
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])

    # Predict the species
    y_pred_new = model.predict(x_new)
    result = species_encoder.inverse_transform(y_pred_new)
    
    return result[0]

# Streamlit UI
st.title("Penguin Species Prediction")

# Get user inputs
island = st.selectbox("Island of the penguin", island_encoder.classes_)
culmen_length = st.slider("Culmen Length (mm)", 30.0, 70.0, 46.5)
culmen_depth = st.slider("Culmen Depth (mm)", 10.0, 25.0, 17.9)
flipper_length = st.slider("Flipper Length (mm)", 170.0, 230.0, 192.0)
body_mass = st.slider("Body Mass (g)", 2500.0, 6500.0, 3500.0)
sex = st.selectbox("Sex of the penguin", sex_encoder.classes_)

# Button to make the prediction
if st.button('Predict'):
    species = predict_penguin(island, culmen_length, culmen_depth, flipper_length, body_mass, sex)
    st.write(f"The predicted species is: {species}")
