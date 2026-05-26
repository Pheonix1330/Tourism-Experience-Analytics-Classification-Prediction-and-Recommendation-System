# TOURISM RECOMMENDATION SYSTEM

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

updated_item_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Updated_Item.xlsx"
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
        "Regression",
        "Classification"
    ]
)

# FILTERING LOGIC

continents = continent_df.iloc[:, 1].dropna().unique()

# REGRESSION

if menu == "Regression":

    st.title("Regression: Predicting Attraction Ratings")

    # USER DEMOGRAPHICS

    st.header("User Demographics")

    selected_continent = st.selectbox(
        "Continent",
        continents
    )

    # FILTER REGION

    filtered_regions = region_df[
        region_df.iloc[:, 2] == selected_continent
    ]

    selected_region = st.selectbox(
        "Region",
        filtered_regions.iloc[:, 1].dropna().unique()
    )

    # FILTER COUNTRY

    filtered_countries = country_df[
        country_df.iloc[:, 2] == selected_region
    ]

    selected_country = st.selectbox(
        "Country",
        filtered_countries.iloc[:, 1].dropna().unique()
    )

    # FILTER CITY

    filtered_cities = city_df[
        city_df.iloc[:, 2] == selected_country
    ]

    selected_city = st.selectbox(
        "City",
        filtered_cities.iloc[:, 1].dropna().unique()
    )

    # VISIT DETAILS

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

    # ATTRACTION ATTRIBUTES

    st.header("Attraction Attributes")

    selected_type = st.selectbox(
        "Attraction Type",
        type_df.iloc[:, 1].dropna().unique()
    )

    selected_avg_rating = st.slider(
        "Average User Rating",
        1.0,
        5.0,
        3.0
    )

    # PREDICT

    if st.button("Predict Rating"):

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=feature_columns
        )

        # NUMERICAL

        if 'visityear' in input_data.columns:
            input_data['visityear'] = selected_year

        if 'visitmonth' in input_data.columns:
            input_data['visitmonth'] = selected_month

        if 'avg_user_rating' in input_data.columns:
            input_data['avg_user_rating'] = selected_avg_rating

        # CATEGORICAL

        mapping = {

            f'continent_{selected_continent}': 1,

            f'region_{selected_region}': 1,

            f'country_{selected_country}': 1,

            f'cityname_{selected_city}': 1,

            f'visitmode_{selected_mode}': 1,

            f'attractiontype_{selected_type}': 1
        }

        for col, val in mapping.items():

            if col in input_data.columns:

                input_data[col] = val

        prediction = rating_model.predict(
            input_data
        )[0]

        prediction = abs(prediction)

        prediction = round(prediction, 2)

        if prediction > 5:
            prediction = 5

        st.success(
            f"Predicted Rating: {prediction} / 5"
        )

# CLASSIFICATION

elif menu == "Classification":

    st.title("Classification: Visit Mode Prediction")

    # USER DEMOGRAPHICS

    st.header("User Demographics")

    selected_continent = st.selectbox(
        "Continent",
        continents
    )

    filtered_regions = region_df[
        region_df.iloc[:, 2] == selected_continent
    ]

    selected_region = st.selectbox(
        "Region",
        filtered_regions.iloc[:, 1].dropna().unique()
    )

    filtered_countries = country_df[
        country_df.iloc[:, 2] == selected_region
    ]

    selected_country = st.selectbox(
        "Country",
        filtered_countries.iloc[:, 1].dropna().unique()
    )

    filtered_cities = city_df[
        city_df.iloc[:, 2] == selected_country
    ]

    selected_city = st.selectbox(
        "City",
        filtered_cities.iloc[:, 1].dropna().unique()
    )

    # HISTORICAL DATA

    st.header("Historical Visit Data")

    selected_year = st.selectbox(
        "Year",
        sorted(
            transaction_df.iloc[:, 2].dropna().unique()
        )
    )

    selected_month = st.selectbox(
        "Month",
        sorted(
            transaction_df.iloc[:, 3].dropna().unique()
        )
    )

    selected_previous_mode = st.selectbox(
        "Previous Visit Mode",
        mode_df.iloc[:, 1].dropna().unique()
    )

    # ATTRACTION CHARACTERISTICS

    st.header("Attraction Characteristics")

    selected_type = st.selectbox(
        "Attraction Type",
        type_df.iloc[:, 1].dropna().unique()
    )

    # PREDICT

    if st.button("Predict Visit Mode"):

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=feature_columns
        )

        if 'visityear' in input_data.columns:
            input_data['visityear'] = selected_year

        if 'visitmonth' in input_data.columns:
            input_data['visitmonth'] = selected_month

        mapping = {

            f'continent_{selected_continent}': 1,

            f'region_{selected_region}': 1,

            f'country_{selected_country}': 1,

            f'cityname_{selected_city}': 1,

            f'visitmode_{selected_previous_mode}': 1,

            f'attractiontype_{selected_type}': 1
        }

        for col, val in mapping.items():

            if col in input_data.columns:

                input_data[col] = val

        prediction = visitmode_model.predict(
            input_data
        )[0]

        predicted_mode = label_encoder.inverse_transform(
            [prediction]
        )[0]

        st.success(
            f"Predicted Visit Mode: {predicted_mode}"
        )