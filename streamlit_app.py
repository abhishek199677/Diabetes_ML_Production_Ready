import numpy as np
import pickle
import streamlit as st
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_model():
    """Loads the trained model and scaler from a pickle file."""
    with open("trained_model.pkl", 'rb') as file:
        classifier, scaler = pickle.load(file)
    return classifier, scaler

def diabetes_prediction(input_data, classifier, scaler):
    """Makes a diabetes prediction based on input data."""
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Scale the input data before prediction
    scaled_input = scaler.transform(input_data_reshaped)
    
    prediction = classifier.predict(scaled_input)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Custom navbar HTML
navbar_html = """
    <style>
    .navbar {
        padding: 10px;
        position: fixed;
        top: 0;
        width: 100%;
        z-index: 1000;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .navbar a {
        color: black;
        margin: 0 20px;
        text-decoration: none;
    }
    .navbar a img {
        vertical-align: middle;
        width: 24px;
        height: 24px;
    }
    .navbar a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="navbar">
        <a href="https://github.com/abhishek199677" target="_blank">
            <img src="https://img.icons8.com/material-outlined/24/000000/github.png" alt="GitHub"/> 
        </a>
    </div>
"""

def main():
    st.title('Diabetes Scan Prediction 💉')
    
    # Display custom navbar
    components.html(navbar_html, height=50, scrolling=False)

    try:
        # Input fields for user data
        Pregnancies = float(st.text_input('Number of Pregnancies', ''))
        Glucose = float(st.text_input('Glucose Level', ''))
        BloodPressure = float(st.text_input('Blood Pressure value', ''))
        SkinThickness = float(st.text_input('Skin Thickness value', ''))
        Insulin = float(st.text_input('Insulin Level', ''))
        BMI = float(st.text_input('BMI value', ''))
        DiabetesPedigreeFunction = float(st.text_input('Diabetes Pedigree Function value', ''))
        Age = float(st.text_input('Age of the Person', ''))
        
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        
        # Load model and scaler
        classifier, scaler = load_model()
        
        if st.button('Diabetes Test Result'):
            diagnosis = diabetes_prediction(input_data, classifier, scaler)
            st.success(diagnosis)
    except ValueError:
        st.error("Please enter valid numerical values for all inputs.")

    # Custom footer HTML
    footer_html = """
    <style>
    .footer {
        text-align: center;
        padding: 10px;
        position: fixed;
        bottom: 0;
        width: 100%;
        border-top: 1px solid #ddd;
    }

    </style>
    <div class="footer">
        <p>&copy; 2023 Abhishek....! All rights reserved.</p>
    </div>
    """
    
    # Display custom logo
    components.html(
        '<div style="text-align: center;"><img src="https://img.freepik.com/free-vector/medical-blood-glucose-measurement_1308-17807.jpg?t=st=1740464756~exp=1740468356~hmac=818733790b9067e9ceb4ab22bacda7735e7d356659fa4d16a520c7c2d2f9023f&w=1480" alt="Diabetes Scan Prediction" style="width: 200px;"></div>',
        height=300, scrolling=False
    )

    # Display disclaimer
    st.markdown(
        "This is a simple web application that uses machine learning to predict if a person has diabetes based on their medical data. The model used in this application is a logistic regression model trained on the Pima Indians Diabetes dataset. The accuracy of the model is approximately 77.69%."
    )

    # Display source code link
    st.markdown(
        "[Source Code](https://github.com/abhishek199677/Diabetes-_ML_Production_Ready)"  
    )
    # Display live demo link
    st.markdown(
        "[Live Demo](https://diabetes-scan-prediction.herokuapp.com/)"
    )

    # Display custom footer
    components.html(footer_html, height=50, scrolling=False)

if __name__ == "__main__":
    main()
