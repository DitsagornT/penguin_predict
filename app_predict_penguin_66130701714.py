import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained model and encoders
with open('model_penguin_66130701714.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Function to make predictions
def predict_penguin(x_new):
    # Prepare the input data for prediction
    input_data = np.array([[
        x_new['sex'], 
        x_new['island'], 
        x_new['culmen_length_mm'], 
        x_new['culmen_depth_mm'], 
        x_new['flipper_length_mm'], 
        x_new['body_mass_g']
    ]])
    
    # Predict the species
    species_pred = model.predict(input_data)
    species = species_encoder.inverse_transform(species_pred)[0]
    
    return species

# Streamlit UI
st.title("Penguin Species Prediction")

# Get user inputs
sex = st.selectbox("Sex of the penguin", sex_encoder.classes_)
island = st.selectbox("Island of the penguin", island_encoder.classes_)
culmen_length = st.slider("Culmen Length (mm)", 30.0, 70.0, 46.5)
culmen_depth = st.slider("Culmen Depth (mm)", 10.0, 25.0, 17.9)
flipper_length = st.slider("Flipper Length (mm)", 170.0, 230.0, 192.0)
body_mass = st.slider("Body Mass (g)", 2500.0, 6500.0, 3500.0)

# Create DataFrame for x_new
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

# Button to make the prediction
if st.button('Predict'):
    species = predict_penguin(x_new.iloc[0])  # Pass the first row of the DataFrame
    st.write(f"The predicted species is: {species}")
