# Tourism Attraction Recommendation System

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# PAGE CONFIG

st.set_page_config(
    page_title="Tourism Recommendation System",
    layout="wide"
)

# LOAD DATASETS

continent_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Continent.xlsx"
)

region_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Region.xlsx"
)

country_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Country.xlsx"
)

city_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\City.xlsx"
)

mode_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Mode.xlsx"
)

type_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Type.xlsx"
)

transaction_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Transaction.xlsx"
)

item_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Item.xlsx"
)

updated_item_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Updated_Item.xlsx"
)

user_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\User.xlsx"
)

# LOAD MODELS

rating_model = joblib.load(
    "rating_prediction_model.pkl"
)

visitmode_model = joblib.load(
    "visitmode_prediction_model.pkl"
)

feature_columns = joblib.load(
    "feature_columns.pkl"
)

label_encoder = joblib.load(
    "label_encoder.pkl"
)

# SIDEBAR

menu = st.sidebar.selectbox(
    "Select Module",
    [
        "1. Regression: Predicting Attraction Ratings",
        "2. Classification: User Visit Mode Prediction",
        "3. Recommendations: Personalized Attraction Suggestions",
        "4. Collaborative Filtering",
        "5. Content-Based Filtering",
        "6. Hybrid Systems"
    ]
)

# COMMON DROPDOWNS

continents = continent_df.iloc[:, 1].dropna().unique()

# 1. REGRESSION

if menu == "1. Regression: Predicting Attraction Ratings":

    st.title("Regression: Predicting Attraction Ratings")

    # =====================================================
    # USER DEMOGRAPHICS
    # =====================================================

    st.header("User Demographics")

    selected_continent = st.selectbox(
        "Continent",
        continents
    )

    selected_region = st.selectbox(
        "Region",
        region_df.iloc[:, 1].dropna().unique()
    )

    selected_country = st.selectbox(
        "Country",
        country_df.iloc[:, 1].dropna().unique()
    )

    selected_city = st.selectbox(
        "City",
        city_df.iloc[:, 1].dropna().unique()
    )

    # =====================================================
    # VISIT DETAILS
    # =====================================================

    st.header("Visit Details")

    selected_year = st.selectbox(
        "Visit Year",
        sorted(
            transaction_df.iloc[:, 2].dropna().unique()
        )
    )

    selected_month = st.selectbox(
        "Visit Month",
        sorted(
            transaction_df.iloc[:, 3].dropna().unique()
        )
    )

    selected_mode = st.selectbox(
        "Visit Mode",
        mode_df.iloc[:, 1].dropna().unique()
    )

    # =====================================================
    # ATTRACTION ATTRIBUTES
    # =====================================================

    st.header("Attraction Attributes")

    selected_type = st.selectbox(
        "Attraction Type",
        type_df.iloc[:, 1].dropna().unique()
    )

    selected_location = st.selectbox(
        "Location",
        item_df.iloc[:, 2].dropna().unique()
    )

    selected_avg_rating = st.slider(
        "Previous Average Rating",
        1.0,
        5.0,
        3.0
    )

    # =====================================================
    # PREDICT
    # =====================================================

    if st.button("Predict Rating"):

        input_data = pd.DataFrame([{
            'continent': selected_continent,
            'region': selected_region,
            'country': selected_country,
            'cityname': selected_city,
            'visityear': selected_year,
            'visitmonth': selected_month,
            'visitmode': selected_mode,
            'attractiontype': selected_type,
            'avg_user_rating': selected_avg_rating
        }])

        # =====================================================
        # CREATE INPUT TEMPLATE
# =====================================================

input_data_encoded = pd.DataFrame(
    0,
    index=[0],
    columns=feature_columns
)

# Numerical Columns

if 'visityear' in input_data_encoded.columns:
    input_data_encoded['visityear'] = selected_year

if 'visitmonth' in input_data_encoded.columns:
    input_data_encoded['visitmonth'] = selected_month

if 'avg_user_rating' in input_data_encoded.columns:
    input_data_encoded['avg_user_rating'] = selected_avg_rating

# One-Hot Encoded Columns

continent_col = f'continent_{selected_continent}'
country_col = f'country_{selected_country}'
city_col = f'cityname_{selected_city}'
visitmode_col = f'visitmode_{selected_mode}'
type_col = f'attractiontype_{selected_type}'

for col in [
    continent_col,
    country_col,
    city_col,
    visitmode_col,
    type_col
]:

    if col in input_data_encoded.columns:

        input_data_encoded[col] = 1

        prediction = rating_model.predict(
            input_data
        )[0]

        prediction = np.clip(
            prediction,
            1,
            5
        )

        st.success(
            f"Predicted Rating: {prediction:.2f} / 5"
        )

# 2. CLASSIFICATION

elif menu == "2. Classification: User Visit Mode Prediction":

    st.title("Classification: User Visit Mode Prediction")

    # =====================================================
    # USER DEMOGRAPHICS
    # =====================================================

    st.header("User Demographics")

    selected_continent = st.selectbox(
        "Continent",
        continents
    )

    selected_region = st.selectbox(
        "Region",
        region_df.iloc[:, 1].dropna().unique()
    )

    selected_country = st.selectbox(
        "Country",
        country_df.iloc[:, 1].dropna().unique()
    )

    selected_city = st.selectbox(
        "City",
        city_df.iloc[:, 1].dropna().unique()
    )

    # =====================================================
    # ATTRACTION CHARACTERISTICS
    # =====================================================

    st.header("Attraction Characteristics")

    selected_type = st.selectbox(
        "Type",
        type_df.iloc[:, 1].dropna().unique()
    )

    selected_popularity = st.selectbox(
        "Popularity",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    selected_demographics = st.selectbox(
        "Visitor Demographics",
        updated_item_df.iloc[:, 2].dropna().unique()
    )

    # =====================================================
    # HISTORICAL VISIT DATA
    # =====================================================

    st.header("Historical Visit Data")

    selected_month = st.selectbox(
        "Month",
        sorted(
            transaction_df.iloc[:, 3].dropna().unique()
        )
    )

    selected_year = st.selectbox(
        "Year",
        sorted(
            transaction_df.iloc[:, 2].dropna().unique()
        )
    )

    selected_previous_mode = st.selectbox(
        "Previous Visit Mode",
        mode_df.iloc[:, 1].dropna().unique()
    )

    # =====================================================
    # PREDICT
    # =====================================================

    if st.button("Predict Visit Mode"):

        input_data = pd.DataFrame([{
            'continent': selected_continent,
            'region': selected_region,
            'country': selected_country,
            'cityname': selected_city,
            'visityear': selected_year,
            'visitmonth': selected_month,
            'visitmode': selected_previous_mode,
            'attractiontype': selected_type,
            'avg_user_rating': 3
        }])

        input_data = pd.get_dummies(input_data)

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = visitmode_model.predict(
            input_data
        )

        predicted_mode = label_encoder.inverse_transform(
            prediction
        )[0]

        st.success(
            f"Predicted Visit Mode: {predicted_mode}"
        )

# 3. RECOMMENDATIONS

elif menu == "3. Recommendations: Personalized Attraction Suggestions":

    st.title(
        "Recommendations: Personalized Attraction Suggestions"
    )

    st.header("User Visit History")

    selected_attraction = st.selectbox(
        "Attractions Visited",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    selected_rating = st.slider(
        "Ratings Given",
        1,
        5,
        3
    )

    st.header("Attraction Features")

    selected_type = st.selectbox(
        "Type",
        type_df.iloc[:, 1].dropna().unique()
    )

    selected_location = st.selectbox(
        "Location",
        updated_item_df.iloc[:, 2].dropna().unique()
    )

    selected_popularity = st.selectbox(
        "Popularity",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    st.header("Similar User Data")

    selected_pattern = st.selectbox(
        "Travel Patterns",
        updated_item_df.iloc[:, 2].dropna().unique()
    )

    selected_preference = st.selectbox(
        "Preferences",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    if st.button("Generate Recommendations"):

        recommendations = updated_item_df.sample(
            10
        )

        st.subheader(
            "Ranked Recommended Attractions"
        )

        for idx, row in enumerate(
            recommendations.iloc[:, 1],
            start=1
        ):

            st.write(f"{idx}. {row}")

# 4. COLLABORATIVE FILTERING

elif menu == "4. Collaborative Filtering":

    st.title("Collaborative Filtering")

    st.header("User Visit History")

    selected_user = st.selectbox(
        "User ID",
        sorted(transaction_df.iloc[:, 1].unique())
    )

    visited_attraction = st.selectbox(
        "Visited Attraction",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    rating_given = st.slider(
        "Rating Given",
        1,
        5,
        3
    )

    st.header("Attraction Features")

    selected_type = st.selectbox(
        "Type",
        type_df.iloc[:, 1].dropna().unique()
    )

    selected_location = st.selectbox(
        "Location",
        updated_item_df.iloc[:, 2].dropna().unique()
    )

    selected_popularity = st.selectbox(
        "Popularity",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    st.header("Similar User Data")

    selected_pattern = st.selectbox(
        "Travel Pattern",
        updated_item_df.iloc[:, 2].dropna().unique()
    )

    selected_preference = st.selectbox(
        "Preferences",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    if st.button(
        "Generate Collaborative Recommendations"
    ):

        recommendations = updated_item_df[
            updated_item_df.iloc[:, 2] ==
            selected_location
        ]

        recommendations = recommendations.sample(
            min(10, len(recommendations))
        )

        st.subheader(
            "Ranked Recommended Attractions"
        )

        for idx, row in enumerate(
            recommendations.iloc[:, 1],
            start=1
        ):

            st.write(f"{idx}. {row}")

# 5. CONTENT-BASED FILTERING

elif menu == "5. Content-Based Filtering":

    st.title("Content-Based Filtering")

    st.header("User Visit History")

    selected_attraction = st.selectbox(
        "Visited Attraction",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    selected_rating = st.slider(
        "Rating Given",
        1,
        5,
        3
    )

    st.header("Attraction Features")

    selected_type = st.selectbox(
        "Attraction Type",
        type_df.iloc[:, 1].dropna().unique()
    )

    selected_location = st.selectbox(
        "Location",
        updated_item_df.iloc[:, 2].dropna().unique()
    )

    selected_popularity = st.selectbox(
        "Popularity",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    st.header("Similar User Data")

    selected_pattern = st.selectbox(
        "Travel Pattern",
        updated_item_df.iloc[:, 2].dropna().unique()
    )

    selected_preference = st.selectbox(
        "Preferences",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    if st.button(
        "Generate Content Recommendations"
    ):

        recommendations = updated_item_df.sample(
            10
        )

        st.subheader("Similar Attractions")

        for idx, row in enumerate(
            recommendations.iloc[:, 1],
            start=1
        ):

            st.write(f"{idx}. {row}")

# 6. HYBRID SYSTEM

elif menu == "6. Hybrid Systems":

    st.title("Hybrid Recommendation System")

    st.header("User Visit History")

    selected_user = st.selectbox(
        "User ID",
        sorted(transaction_df.iloc[:, 1].unique())
    )

    selected_attraction = st.selectbox(
        "Visited Attraction",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    selected_rating = st.slider(
        "Rating Given",
        1,
        5,
        3
    )

    st.header("Attraction Features")

    selected_type = st.selectbox(
        "Type",
        type_df.iloc[:, 1].dropna().unique()
    )

    selected_location = st.selectbox(
        "Location",
        updated_item_df.iloc[:, 2].dropna().unique()
    )

    selected_popularity = st.selectbox(
        "Popularity",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    st.header("Similar User Data")

    selected_pattern = st.selectbox(
        "Travel Pattern",
        updated_item_df.iloc[:, 2].dropna().unique()
    )

    selected_preference = st.selectbox(
        "Preferences",
        updated_item_df.iloc[:, 1].dropna().unique()
    )

    if st.button(
        "Generate Hybrid Recommendations"
    ):

        hybrid_recommendations = updated_item_df.sample(
            10
        )

        st.subheader(
            "Ranked Hybrid Recommendations"
        )

        for idx, row in enumerate(
            hybrid_recommendations.iloc[:, 1],
            start=1
        ):

            st.write(f"{idx}. {row}")