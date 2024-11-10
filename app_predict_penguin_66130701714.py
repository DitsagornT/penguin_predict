import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model and encoders
with open('model_penguin_66130701714.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Function to safely transform input values
def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        st.error(f"Unseen label '{value}' encountered!")
        return encoder.transform([encoder.classes_[0]])[0]  # Default to the first class

# Function to make predictions
def predict_penguin(sex, island, bill_length, bill_depth, flipper_length, body_mass):
    sex_value = safe_transform(sex_encoder, sex)
    island_value = safe_transform(island_encoder, island)

    input_data = np.array([[sex_value, island_value, bill_length, bill_depth, flipper_length, body_mass]])
    species_pred = model.predict(input_data)
    species = species_encoder.inverse_transform(species_pred)[0]
    return species

# Streamlit UI
st.title("Penguin Species Prediction")

# Get user inputs with the current value displayed
sex = st.selectbox("Sex of the penguin", ['Male', 'Female'])
island = st.selectbox("Island of the penguin", ['Torgersen', 'Biscoe', 'Dream'])

bill_length = st.slider("Bill Length (mm)", 30.0, 70.0, 45.0, step=0.1)
bill_depth = st.slider("Bill Depth (mm)", 10.0, 25.0, 15.0, step=0.1)

# Debugging: Check values for flipper_length
flipper_length = st.slider("Flipper Length (mm)", 170, 230, 200, step=1)
st.write(f"Flipper Length: {flipper_length} mm")

body_mass = st.slider("Body Mass (g)", 2500.0, 6500.0, 4500.0, step=10)

# Display the current values
st.write(f"Bill Length: {bill_length} mm")
st.write(f"Bill Depth: {bill_depth} mm")
st.write(f"Body Mass: {body_mass} g")

# Button to make the prediction
if st.button('Predict'):
    species = predict_penguin(sex, island, bill_length, bill_depth, flipper_length, body_mass)
    st.write(f"The predicted species is: {species}")
