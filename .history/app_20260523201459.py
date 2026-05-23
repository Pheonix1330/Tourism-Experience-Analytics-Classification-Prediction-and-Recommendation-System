# Tourism Recommendation and Prediction System
# Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Models

rating_model = joblib.load(
    'rating_prediction_model.pkl'
)

visitmode_model = joblib.load(
    'visitmode_prediction_model.pkl'
)

feature_columns = joblib.load(
    'feature_columns.pkl'
)

# Page Configuration

st.set_page_config(
    page_title="Tourism Recommendation System",
    layout="wide"
)

# Sidebar

menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Home",
        "Rating Prediction",
        "Visit Mode Prediction",
        "Collaborative Recommendation",
        "Content-Based Recommendation",
        "Visualizations",
        "About Project"
    ]
)

# HOME

if menu == "Home":

    st.title("Tourism Attraction Recommendation System")

    st.markdown("""
    ### Project Overview

    This application predicts:
    - Tourist attraction ratings
    - User visit mode
    - Personalized attraction recommendations

    using Machine Learning and Recommendation Systems.
    """)

# RATING PREDICTION

elif menu == "Rating Prediction":

    st.title("Regression: Attraction Rating Prediction")

    continent = st.selectbox(
        "Select Continent",
        ["Asia", "Europe", "Africa", "North America", "South America"]
    )

    region = st.number_input(
        "Enter Region",
        min_value=0
    )

    country = st.selectbox(
        "Select Country",
        ["India", "USA", "France", "Australia"]
    )

    cityname = st.text_input(
        "Enter City"
    )

    visityear = st.number_input(
        "Visit Year",
        min_value=2000,
        max_value=2035
    )

    visitmonth = st.slider(
        "Visit Month",
        1,
        12
    )

    visitmode = st.selectbox(
        "Visit Mode",
        ["Business", "Family", "Friends", "Couples"]
    )

    attractiontype = st.selectbox(
        "Attraction Type",
        ["Beach", "Museum", "Temple", "Park", "Ruins"]
    )

    avg_user_rating = st.slider(
        "Average User Rating",
        1.0,
        5.0
    )

    if st.button("Predict Rating"):

        input_data = pd.DataFrame([{
            'continent': continent,
            'region': region,
            'country': country,
            'cityname': cityname,
            'visityear': visityear,
            'visitmonth': visitmonth,
            'visitmode': visitmode,
            'attractiontype': attractiontype,
            'avg_user_rating': avg_user_rating
        }])

        # Dummy Encoding

        input_data = pd.get_dummies(input_data)

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = rating_model.predict(input_data)

        st.success(
            f"Predicted Attraction Rating: {prediction[0]:.2f}"
        )

# VISIT MODE PREDICTION

elif menu == "Visit Mode Prediction":

    st.title("Classification: Visit Mode Prediction")

    continent = st.selectbox(
        "Continent",
        ["Asia", "Europe", "Africa", "North America", "South America"]
    )

    region = st.number_input(
        "Region",
        min_value=0
    )

    country = st.selectbox(
        "Country",
        ["India", "USA", "France", "Australia"]
    )

    cityname = st.text_input(
        "City"
    )

    attractiontype = st.selectbox(
        "Attraction Type",
        ["Beach", "Museum", "Temple", "Park", "Ruins"]
    )

    visityear = st.number_input(
        "Visit Year",
        min_value=2000,
        max_value=2035
    )

    visitmonth = st.slider(
        "Visit Month",
        1,
        12
    )

    avg_user_rating = st.slider(
        "Average User Rating",
        1.0,
        5.0
    )

    if st.button("Predict Visit Mode"):

        input_data = pd.DataFrame([{
            'continent': continent,
            'region': region,
            'country': country,
            'cityname': cityname,
            'visityear': visityear,
            'visitmonth': visitmonth,
            'attractiontype': attractiontype,
            'avg_user_rating': avg_user_rating
        }])

        input_data = pd.get_dummies(input_data)

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = visitmode_model.predict(input_data)

        st.success(
            f"Predicted Visit Mode: {prediction[0]}"
        )

# COLLABORATIVE FILTERING

elif menu == "Collaborative Recommendation":

    st.title("Collaborative Filtering Recommendation")

    user_id = st.number_input(
        "Enter User ID",
        min_value=1
    )

    if st.button("Recommend Attractions"):

        recommendations = [
            "Eiffel Tower",
            "Marina Beach",
            "Louvre Museum",
            "Great Wall of China",
            "Goa Beaches"
        ]

        st.write("Recommended Attractions:")

        for rec in recommendations:
            st.write("-", rec)

# CONTENT BASED FILTERING

elif menu == "Content-Based Recommendation":

    st.title("Content-Based Recommendation")

    attraction_name = st.text_input(
        "Enter Attraction Name"
    )

    if st.button("Get Similar Attractions"):

        recommendations = [
            "Bondi Beach",
            "Copacabana",
            "Miami Beach",
            "Maldives Beach"
        ]

        st.write("Similar Attractions:")

        for rec in recommendations:
            st.write("-", rec)

# VISUALIZATIONS

elif menu == "Visualizations":

    st.title("Tourism Visualizations")

    chart_data = pd.DataFrame({
        'Attraction': [
            'Beach',
            'Museum',
            'Temple',
            'Park'
        ],
        'Visitors': [
            120,
            90,
            150,
            80
        ]
    })

    st.bar_chart(
        chart_data.set_index('Attraction')
    )

# ABOUT PROJECT

elif menu == "About Project":

    st.title("About Project")

    st.markdown("""
    ### Tourism Recommendation and Prediction System

    This project uses:
    - Regression Models
    - Classification Models
    - Recommendation Systems

    to predict attraction ratings, visit modes,
    and recommend personalized tourist attractions.
    """)