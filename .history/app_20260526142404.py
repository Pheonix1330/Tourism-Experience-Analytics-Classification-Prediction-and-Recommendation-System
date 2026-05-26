# =========================================================
# TOURISM EXPERIENCE ANALYTICS SYSTEM
# FINAL STREAMLIT APP
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Tourism Recommendation System",
    layout="wide"
)

# =========================================================
# LOAD DATASETS
# =========================================================

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

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

dfs = [
    continent_df,
    region_df,
    country_df,
    city_df,
    mode_df,
    type_df,
    transaction_df,
    updated_item_df
]

for df in dfs:
    df.columns = df.columns.str.strip()

# =========================================================
# CONVERT IDS
# =========================================================

transaction_df["VisitMode"] = (
    transaction_df["VisitMode"]
    .astype(str)
    .str.zfill(2)
)

mode_df["VisitModeId"] = (
    mode_df["VisitModeId"]
    .astype(str)
    .str.zfill(2)
)

transaction_df["AttractionId"] = (
    transaction_df["AttractionId"]
    .astype(str)
)

updated_item_df["AttractionId"] = (
    updated_item_df["AttractionId"]
    .astype(str)
)

# =========================================================
# MERGE REGION + COUNTRY + CONTINENT
# =========================================================

merged_country = country_df.merge(
    region_df,
    on="RegionId",
    how="left"
)

merged_country = merged_country.merge(
    continent_df,
    on="ContinentId",
    how="left"
)

# =========================================================
# MERGE CITY
# =========================================================

merged_city = city_df.merge(
    merged_country,
    on="CountryId",
    how="left"
)

# =========================================================
# MERGE VISIT MODE
# =========================================================

merged_visit_df = pd.merge(
    transaction_df,
    mode_df,
    left_on="VisitMode",
    right_on="VisitModeId",
    how="left"
)

# =========================================================
# LOAD MODELS
# =========================================================

rating_model = joblib.load(
    "rating_prediction_model.pkl"
)

# =========================================================
# SIDEBAR
# =========================================================

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

# =========================================================
# 1. REGRESSION
# =========================================================

if menu == "1. Regression: Predicting Attraction Ratings":

    st.title(
        "Regression: Predicting Attraction Ratings"
    )

    # =====================================================
    # USER DEMOGRAPHICS
    # =====================================================

    st.header("User Demographics")

    # CONTINENT

    continents = sorted(
        merged_country["Continent"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_continent = st.selectbox(
        "Continent",
        continents
    )

    # REGION

    regions = sorted(
        merged_country[
            merged_country["Continent"]
            == selected_continent
        ]["Region"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_region = st.selectbox(
        "Region",
        regions
    )

    # COUNTRY

    countries = sorted(
        merged_country[
            (
                merged_country["Continent"]
                == selected_continent
            )
            &
            (
                merged_country["Region"]
                == selected_region
            )
        ]["Country"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_country = st.selectbox(
        "Country",
        countries
    )

    # CITY

    cities = sorted(
        merged_city[
            (
                merged_city["Continent"]
                == selected_continent
            )
            &
            (
                merged_city["Region"]
                == selected_region
            )
            &
            (
                merged_city["Country"]
                == selected_country
            )
        ]["CityName"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_city = st.selectbox(
        "City",
        cities
    )

    # =====================================================
    # VISIT DETAILS
    # =====================================================

    st.header("Visit Details")

    # YEAR

    year_options = sorted(
        merged_visit_df["VisitYear"]
        .dropna()
        .unique()
    )

    selected_year = st.selectbox(
        "Visit Year",
        year_options
    )

    # MONTH

    month_options = list(range(1, 13))

    selected_month = st.selectbox(
        "Visit Month",
        month_options
    )

    # VISIT MODE

    visit_mode_options = sorted(
        merged_visit_df["VisitMode_y"]
        .dropna()
        .unique()
    )

    selected_mode = st.selectbox(
        "Visit Mode",
        visit_mode_options
    )

    # =====================================================
    # ATTRACTION ATTRIBUTES
    # =====================================================

    st.header("Attraction Attributes")

    # TYPE

    type_options = sorted(
        updated_item_df["AttractionType"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_type = st.selectbox(
        "Attraction Type",
        type_options
    )

    # FILTER LOCATIONS

    filtered_type_df = updated_item_df[
        updated_item_df["AttractionType"]
        == selected_type
    ]

    location_options = sorted(
        filtered_type_df["AttractionAddress"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_location = st.selectbox(
        "Location",
        location_options
    )

    # GET ATTRACTION ID

    selected_row = filtered_type_df[
        filtered_type_df["AttractionAddress"]
        == selected_location
    ]

    selected_attraction_id = str(
        selected_row.iloc[0]["AttractionId"]
    )

    # MATCH TRANSACTIONS

    matched_transaction_df = transaction_df[
        transaction_df["AttractionId"]
        == selected_attraction_id
    ]

    # AVG RATING

    if len(matched_transaction_df) > 0:

        avg_rating = round(
            matched_transaction_df["Rating"].mean(),
            2
        )

    else:

        avg_rating = 3.0

    # PREVIOUS AVG RATING

    selected_avg_rating = st.slider(
        "Previous Average Rating",
        1.0,
        5.0,
        float(min(max(avg_rating, 1), 5))
    )

    # =====================================================
    # PREDICT RATING
    # =====================================================

    st.header("Predicted Rating")

    if st.button("Predict Rating"):

        input_df = pd.DataFrame({

            "Continent": [selected_continent],

            "Country": [selected_country],

            "CityName": [selected_city],

            "VisitYear": [selected_year],

            "VisitMonth": [selected_month],

            "VisitMode": [selected_mode],

            "AttractionType": [selected_type]
        })

        prediction = rating_model.predict(
            input_df
        )[0]

        prediction = round(prediction, 2)

        prediction = max(1, min(5, prediction))

        st.success(
            f"Predicted Rating : {prediction} / 5"
        )

        # SUMMARY

        st.subheader("Prediction Summary")

        st.write("Continent :", selected_continent)

        st.write("Region :", selected_region)

        st.write("Country :", selected_country)

        st.write("City :", selected_city)

        st.write("Visit Year :", selected_year)

        st.write("Visit Month :", selected_month)

        st.write("Visit Mode :", selected_mode)

        st.write("Attraction Type :", selected_type)

        st.write("Location :", selected_location)

        st.write(
            "Previous Avg Rating :",
            selected_avg_rating
        )

# =========================================================
# 2. CLASSIFICATION
# =========================================================

elif menu == "2. Classification: User Visit Mode Prediction":

    st.title(
        "Classification: User Visit Mode Prediction"
    )

    st.header("User Demographics")

    # CONTINENT

    continents = sorted(
        merged_country["Continent"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_continent = st.selectbox(
        "Continent",
        continents
    )

    # REGION

    regions = sorted(
        merged_country[
            merged_country["Continent"]
            == selected_continent
        ]["Region"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_region = st.selectbox(
        "Region",
        regions
    )

    # COUNTRY

    countries = sorted(
        merged_country[
            (
                merged_country["Continent"]
                == selected_continent
            )
            &
            (
                merged_country["Region"]
                == selected_region
            )
        ]["Country"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_country = st.selectbox(
        "Country",
        countries
    )

    # CITY

    cities = sorted(
        merged_city[
            (
                merged_city["Continent"]
                == selected_continent
            )
            &
            (
                merged_city["Region"]
                == selected_region
            )
            &
            (
                merged_city["Country"]
                == selected_country
            )
        ]["CityName"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_city = st.selectbox(
        "City",
        cities
    )

    st.success("Region Added Successfully")