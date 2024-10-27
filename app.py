# app.py
import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title and description
st.title("Iris Species Prediction App")
st.markdown("<h2>Enter the details below to predict the species of an Iris flower:</h2>", unsafe_allow_html=True)

# Render HTML form
st.markdown("""
    <style>
        .form-control { margin-bottom: 1em; }
    </style>
    <div>
        <form action="" method="post">
            <label>Sepal Length (cm): </label><br>
            <input class="form-control" name="sepal_length" type="number" step="0.1" min="0" required><br>
            
            <label>Sepal Width (cm): </label><br>
            <input class="form-control" name="sepal_width" type="number" step="0.1" min="0" required><br>
            
            <label>Petal Length (cm): </label><br>
            <input class="form-control" name="petal_length" type="number" step="0.1" min="0" required><br>
            
            <label>Petal Width (cm): </label><br>
            <input class="form-control" name="petal_width" type="number" step="0.1" min="0" required><br>
            
            <button class="btn btn-primary w-100">Predict</button>
        </form>
    </div>
""", unsafe_allow_html=True)

# Form submission
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Prediction button
if st.button("Predict"):
    # Get prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    iris_species = ['Setosa', 'Versicolor', 'Virginica']
    result = iris_species[prediction[0]]
    
    st.success(f"The predicted species is: {result}")
