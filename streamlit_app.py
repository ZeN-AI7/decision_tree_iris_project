# streamlit_app.py

import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# --- Inject custom CSS for the color palette ---
st.markdown(
    """
    <style>
    /* Main app background: Electric Tangerine */
    [data-testid="stAppViewContainer"] {
        background-color: #FF6D1F !important;
    }

    /* Text color: Black Hole */
    body, .main {
        color: #222222 !important;
    }
    
    /* Sidebar background: Sustainable Linen */
    [data-testid="stSidebar"] {
        background-color: #FAF3E1 !important;
    }
    
    /* Slider styling: Recycled Cotton */
    /* This sets the primary color for the BaseWeb slider component used by Streamlit */
    [data-baseweb="slider"] {
        --primary-color: #F5E7C6 !important;
    }
    
    /* In case the slider thumb needs additional styling */
    [data-baseweb="slider"] .slider__thumb {
        background-color: #F5E7C6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar: Place sliders on the left ---
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width  = st.sidebar.slider("Petal Width (cm)",  min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# --- Main content: Centered interface with a fun header and instructions ---
st.title("Iris Species Prediction")
st.markdown("<div class='main'><h2 class='main-header'>Welcome to the Fun Iris Classifier!</h2></div>", unsafe_allow_html=True)
st.markdown("<div class='main'>Use the sliders in the sidebar to adjust the iris features and click the button below to predict the species.</div>", unsafe_allow_html=True)

# --- Load pre-trained model and iris dataset ---
model = joblib.load('iris_model.pkl')
iris = load_iris()
target_names = iris.target_names

# --- Prediction button ---
if st.button("Predict"):
    # Prepare the feature array for prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    predicted_species = target_names[prediction[0]]
    
    # Display the prediction with an animated result
    st.markdown(f"<div class='main'><div class='result'>Predicted Iris Species: {predicted_species}</div></div>", unsafe_allow_html=True)

# --- Optional: Add a fun animated image or GIF for extra engagement ---
# st.image("https://media.giphy.com/media/26xBwdIuRJiAIqHwA/giphy.gif", width=300)
