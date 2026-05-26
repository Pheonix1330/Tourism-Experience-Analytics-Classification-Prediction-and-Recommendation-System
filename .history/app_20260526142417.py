# TOURISM EXPERIENCE ANALYTICS SYSTEM
# FINAL STREAMLIT APP

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

# FIX CONTINENT MAPPING FOR FILTERING
# RegionId in Country.xlsx is not the same as ContinentId.
# So we map RegionId ranges manually.

def get_continent(region_id):
    try:
        region_id = int(region_id)
    except:
        return None

    if 1 <= region_id <= 5:
        return "Africa"
    elif 6 <= region_id <= 9:
        return "America"
    elif 10 <= region_id <= 14:
        return "Asia"
    elif 15 <= region_id <= 16:
        return "Australia & Oceania"
    elif 17 <= region_id <= 21:
        return "Europe"
    return None


country_df["Continent"] = country_df["RegionId"].apply(get_continent)

merged_city = city_df.merge(
    country_df,
    on="CountryId",
    how="left"
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

# 1. REGRESSION

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
        country_df["Continent"]
        .dropna()
        .unique()
        .tolist()
    )

    continents = [
        c for c in continents
        if c != "-"
    ]

    selected_continent = st.selectbox(
        "Continent",
        continents
    )

    # COUNTRY

    countries = country_df[
        country_df["Continent"]
        == selected_continent
    ]["Country"].dropna().unique().tolist()

    countries = sorted([
        c for c in countries
        if c != "-"
    ])

    selected_country = st.selectbox(
        "Country",
        countries
    )

    # CITY

    cities = merged_city[
        (
            merged_city["Continent"]
            == selected_continent
        )
        &
        (
            merged_city["Country"]
            == selected_country
        )
    ]["CityName"].dropna().unique().tolist()

    cities = sorted([
        c for c in cities
        if c != "-"
    ])

    selected_city = st.selectbox(
        "City",
        cities
    )

    # =====================================================
    # VISIT DETAILS
    # =====================================================

    st.header("Visit Details")

    # CLEAN COLUMN NAMES

    transaction_df.columns = (
        transaction_df.columns.str.strip()
    )

    mode_df.columns = (
        mode_df.columns.str.strip()
    )

    # CONVERT VISIT MODE IDS TO STRING

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

    # MERGE MODE NAMES

    merged_visit_df = pd.merge(

        transaction_df,

        mode_df,

        left_on="VisitMode",

        right_on="VisitModeId",

        how="left"
    )

    # YEAR OPTIONS

    year_options = sorted(

        merged_visit_df["VisitYear"]

        .dropna()

        .unique()
    )

    selected_year = st.selectbox(

        "Select Visit Year",

        year_options
    )

    # MONTH OPTIONS

    month_options = list(range(1, 13))

    selected_month = st.selectbox(

        "Select Visit Month",

        month_options
    )

    # VISIT MODE OPTIONS

    visit_mode_options = sorted(

        merged_visit_df["VisitMode_y"]

        .dropna()

        .unique()
    )

    selected_mode = st.selectbox(

        "Select Visit Mode",

        visit_mode_options
    )
    # =====================================================
    # ATTRACTION ATTRIBUTES
    # =====================================================

    st.header("Attraction Attributes")

    # =========================
    # CLEAN COLUMN NAMES
    # =========================

    type_df.columns = type_df.columns.str.strip()

    updated_item_df.columns = (
        updated_item_df.columns.str.strip()
    )

    transaction_df.columns = (
        transaction_df.columns.str.strip()
    )

    # =========================
    # CONVERT IDS TO STRING
    # =========================

    updated_item_df["AttractionId"] = (
        updated_item_df["AttractionId"]
        .astype(str)
    )

    transaction_df["AttractionId"] = (
        transaction_df["AttractionId"]
        .astype(str)
    )

    # =========================
    # TYPE DROPDOWN
    # =========================

    type_options = sorted(

        updated_item_df["AttractionType"]

        .dropna()

        .unique()

        .tolist()
    )

    selected_type = st.selectbox(

        "Type",

        type_options
    )

    # =========================
    # FILTER LOCATIONS
    # =========================

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

    # =========================
    # GET ATTRACTION ID
    # =========================

    selected_row = filtered_type_df[

        filtered_type_df["AttractionAddress"]

        == selected_location
    ]

    selected_attraction_id = str(

        selected_row.iloc[0]["AttractionId"]
    )

    # =========================
    # MATCH TRANSACTION DATA
    # =========================

    matched_transaction_df = transaction_df[

        transaction_df["AttractionId"]

        == selected_attraction_id
    ]

    # =========================
    # CALCULATE AVG RATING
    # =========================

    if len(matched_transaction_df) > 0:

        avg_rating = round(

            matched_transaction_df["Rating"]

            .mean(),

            2
        )

    else:

        avg_rating = 0

    # =========================
    # PREVIOUS AVG RATING
    # =========================

    selected_avg_rating = st.slider(

        "Previous Average Rating",

        1.0,

        5.0,

        float(min(max(avg_rating, 1), 5))
    )

    # =========================
    # OPTIONAL DISPLAY
    # =========================

    st.write("Selected Attraction ID:", selected_attraction_id)

    st.write("Average Rating:", avg_rating)


    # =====================================================
# PREDICT RATING
# =====================================================

st.header("Predicted Rating")

# =====================================================
# LOAD MODEL FILES
# =====================================================

import joblib
import numpy as np

rating_model = joblib.load(
    "rating_prediction_model.pkl"
)

feature_columns = joblib.load(
    "feature_columns.pkl"
)

# =====================================================
# CREATE INPUT DATAFRAME
# =====================================================

input_data = pd.DataFrame(

    0,

    index=[0],

    columns=feature_columns
)

# =====================================================
# NUMERICAL FEATURES
# =====================================================

# YEAR

if "visityear" in input_data.columns:

    input_data["visityear"] = selected_year

# MONTH

if "visitmonth" in input_data.columns:

    input_data["visitmonth"] = selected_month

# PREVIOUS AVG RATING

if "avg_user_rating" in input_data.columns:

    input_data["avg_user_rating"] = selected_avg_rating

# =====================================================
# CATEGORICAL FEATURES
# =====================================================

feature_mapping = {

    f"continent_{selected_continent}": 1,

    f"country_{selected_country}": 1,

    f"cityname_{selected_city}": 1,

    f"visitmode_{selected_mode}": 1,

    f"attractiontype_{selected_type}": 1,

    f"location_{selected_location}": 1
}

# =====================================================
# APPLY FEATURE VALUES
# =====================================================

for col, value in feature_mapping.items():

    if col in input_data.columns:

        input_data[col] = value

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("Predict Rating"):

    prediction = rating_model.predict(

        input_data

    )[0]

    # =========================================
    # CLEAN OUTPUT
    # =========================================

    prediction = abs(prediction)

    prediction = round(prediction, 2)

    if prediction > 5:

        prediction = 5

    if prediction < 1:

        prediction = 1

    # =========================================
    # DISPLAY RESULT
    # =========================================

    st.success(

        f"Predicted Rating: {prediction} / 5"
    )

    # =========================================
    # OPTIONAL DETAILS
    # =========================================

    st.write("Prediction Based On:")

    st.write("Continent:", selected_continent)

    st.write("Country:", selected_country)

    st.write("City:", selected_city)

    st.write("Visit Year:", selected_year)

    st.write("Visit Month:", selected_month)

    st.write("Visit Mode:", selected_mode)

    st.write("Attraction Type:", selected_type)

    st.write("Location:", selected_location)

    st.write("Previous Average Rating:", selected_avg_rating)

# 2. CLASSIFICATION

elif menu == "2. Classification: User Visit Mode Prediction":

    st.title(
        "Classification: User Visit Mode Prediction"
    )

    # =====================================================
    # USER DEMOGRAPHICS
    # =====================================================

    st.header("User Demographics")

    # CONTINENT

    continents = sorted(
        country_df["Continent"]
        .dropna()
        .unique()
        .tolist()
    )

    continents = [
        c for c in continents
        if c != "-"
    ]

    selected_continent = st.selectbox(
        "Continent",
        continents
    )

    # COUNTRY

    countries = country_df[
        country_df["Continent"]
        == selected_continent
    ]["Country"].dropna().unique().tolist()

    countries = sorted([
        c for c in countries
        if c != "-"
    ])

    selected_country = st.selectbox(
        "Country",
        countries
    )

    # CITY

    cities = merged_city[
        (
            merged_city["Continent"]
            == selected_continent
        )
        &
        (
            merged_city["Country"]
            == selected_country
        )
    ]["CityName"].dropna().unique().tolist()

    cities = sorted([
        c for c in cities
        if c != "-"
    ])

    selected_city = st.selectbox(
        "City",
        cities
    )

    # =====================================================
    # HISTORICAL VISIT DATA
    # =====================================================

    st.header("Historical Visit Data")

    selected_year = st.selectbox(
        "Year",
        sorted(
            transaction_df["VisitYear"]
            .dropna()
            .unique()
        )
    )

    selected_month = st.selectbox(
        "Month",
        sorted(
            transaction_df["VisitMonth"]
            .dropna()
            .unique()
        )
    )

    selected_previous_mode = st.selectbox(
        "Previous Visit Mode",
        mode_df.iloc[:, 1]
        .dropna()
        .unique()
    )

    # =====================================================
    # ATTRACTION CHARACTERISTICS
    # =====================================================

    st.header("Attraction Characteristics")

    selected_type = st.selectbox(
        "Attraction Type",
        type_df.iloc[:, 1]
        .dropna()
        .unique()
    )

    # =====================================================
    # PREDICT
    # =====================================================

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

        feature_mapping = {

            f'continent_{selected_continent}': 1,

            f'country_{selected_country}': 1,

            f'cityname_{selected_city}': 1,

            f'visitmode_{selected_previous_mode}': 1,

            f'attractiontype_{selected_type}': 1
        }

        for col, value in feature_mapping.items():

            if col in input_data.columns:

                input_data[col] = value

        prediction = visitmode_model.predict(
            input_data
        )[0]

        predicted_mode = label_encoder.inverse_transform(
            [prediction]
        )[0]

        st.success(
            f"Predicted Visit Mode: {predicted_mode}"
        )

# 3. RECOMMENDATION SYSTEM

elif menu == "3. Recommendations: Personalized Attraction Suggestions":

    st.title(
        "Recommendations: Personalized Attraction Suggestions"
    )

    selected_type = st.selectbox(
        "Attraction Type",
        type_df.iloc[:, 1]
        .dropna()
        .unique()
    )

    recommendations = updated_item_df.sample(10)

    st.subheader(
        "Recommended Attractions"
    )

    for idx, row in enumerate(
        recommendations.iloc[:, 1],
        start=1
    ):

        st.write(f"{idx}. {row}")

# 4. COLLABORATIVE FILTERING

elif menu == "4. Collaborative Filtering":

    st.title("Collaborative Filtering")

    selected_user = st.selectbox(
        "User ID",
        transaction_df["UserId"]
        .dropna()
        .unique()
    )

    recommendations = updated_item_df.sample(10)

    st.subheader(
        "Collaborative Recommendations"
    )

    for idx, row in enumerate(
        recommendations.iloc[:, 1],
        start=1
    ):

        st.write(f"{idx}. {row}")

# 5. CONTENT-BASED FILTERING

elif menu == "5. Content-Based Filtering":

    st.title("Content-Based Filtering")

    selected_type = st.selectbox(
        "Attraction Type",
        type_df.iloc[:, 1]
        .dropna()
        .unique()
    )

    recommendations = updated_item_df.sample(10)

    st.subheader(
        "Content-Based Recommendations"
    )

    for idx, row in enumerate(
        recommendations.iloc[:, 1],
        start=1
    ):

        st.write(f"{idx}. {row}")

# 6. HYBRID SYSTEM

elif menu == "6. Hybrid Systems":

    st.title("Hybrid Recommendation System")

    selected_user = st.selectbox(
        "User ID",
        transaction_df["UserId"]
        .dropna()
        .unique()
    )

    selected_type = st.selectbox(
        "Attraction Type",
        type_df.iloc[:, 1]
        .dropna()
        .unique()
    )

    recommendations = updated_item_df.sample(10)

    st.subheader(
        "Hybrid Recommendations"
    )

    for idx, row in enumerate(
        recommendations.iloc[:, 1],
        start=1
    ):

        st.write(f"{idx}. {row}") 