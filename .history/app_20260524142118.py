# Tourism Attraction Recommendation System
# Dynamic Streamlit App

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
    r"D:\Tourism Experience Analytics\DATA SET\Continent(1).xlsx"
)

region_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Region(1).xlsx"
)

country_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Country(1).xlsx"
)

city_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\City(1).xlsx"
)

mode_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Mode(1).xlsx"
)

type_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Type(1).xlsx"
)

transaction_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Transaction(1).xlsx"
)

item_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Item(1).xlsx"
)

updated_item_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\Updated_Item(1).xlsx"
)

user_df = pd.read_excel(
    r"D:\Tourism Experience Analytics\DATA SET\User(1).xlsx"
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

# SIDEBAR MENU

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

# COMMON DROPDOWN LOGIC

continents = continent_df["Continent"].dropna().unique()

# 1. REGRESSION

if menu == "1. Regression: Predicting Attraction Ratings":

    st.title("Regression: Predicting Attraction Ratings")

    # USER DEMOGRAPHICS

    st.header("User Demographics")

    selected_continent = st.selectbox(
        "Continent",
        continents
    )

    continent_id = continent_df[
        continent_df["Continent"] == selected_continent
    ]["ContinentId"].values[0]

    filtered_regions = region_df[
        region_df["ContinentId"] == continent_id
    ]

    selected_region = st.selectbox(
        "Region",
        filtered_regions["Region"].unique()
    )

    region_id = filtered_regions[
        filtered_regions["Region"] == selected_region
    ]["RegionId"].values[0]

    filtered_countries = country_df[
        country_df["RegionId"] == region_id
    ]

    selected_country = st.selectbox(
        "Country",
        filtered_countries["Country"].unique()
    )

    country_id = filtered_countries[
        filtered_countries["Country"] == selected_country
    ]["CountryId"].values[0]

    filtered_cities = city_df[
        city_df["CountryId"] == country_id
    ]

    selected_city = st.selectbox(
        "City",
        filtered_cities["CityName"].unique()
    )

    # VISIT DETAILS

    st.header("Visit Details")

    selected_year = st.selectbox(
        "Visit Year",
        sorted(
            transaction_df["VisitYear"].unique()
        )
    )

    selected_month = st.selectbox(
        "Visit Month",
        sorted(
            transaction_df["VisitMonth"].unique()
        )
    )

    selected_mode = st.selectbox(
        "Mode of Visit",
        mode_df["VisitMode"].unique()
    )

    # ATTRACTION ATTRIBUTES

    st.header("Attraction Attributes")

    selected_type = st.selectbox(
        "Attraction Type",
        type_df["AttractionType"].unique()
    )

    selected_location = st.selectbox(
        "Location",
        item_df["AttractionAddress"].dropna().unique()
    )

    selected_avg_rating = st.slider(
        "Previous Average Rating",
        1.0,
        5.0,
        3.0
    )

    # PREDICTION

    if st.button("Predict Rating"):

        input_data = pd.DataFrame([{
            "continent": selected_continent,
            "region": selected_region,
            "country": selected_country,
            "cityname": selected_city,
            "visityear": selected_year,
            "visitmonth": selected_month,
            "visitmode": selected_mode,
            "attractiontype": selected_type,
            "avg_user_rating": selected_avg_rating
        }])

        input_data = pd.get_dummies(input_data)

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = rating_model.predict(input_data)[0]

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

    # USER DEMOGRAPHICS

    st.header("User Demographics")

    selected_continent = st.selectbox(
        "Continent",
        continents
    )

    continent_id = continent_df[
        continent_df["Continent"] == selected_continent
    ]["ContinentId"].values[0]

    filtered_regions = region_df[
        region_df["ContinentId"] == continent_id
    ]

    selected_region = st.selectbox(
        "Region",
        filtered_regions["Region"].unique()
    )

    region_id = filtered_regions[
        filtered_regions["Region"] == selected_region
    ]["RegionId"].values[0]

    filtered_countries = country_df[
        country_df["RegionId"] == region_id
    ]

    selected_country = st.selectbox(
        "Country",
        filtered_countries["Country"].unique()
    )

    country_id = filtered_countries[
        filtered_countries["Country"] == selected_country
    ]["CountryId"].values[0]

    filtered_cities = city_df[
        city_df["CountryId"] == country_id
    ]

    selected_city = st.selectbox(
        "City",
        filtered_cities["CityName"].unique()
    )

    # ATTRACTION CHARACTERISTICS

    st.header("Attraction Characteristics")

    selected_type = st.selectbox(
        "Type",
        type_df["AttractionType"].unique()
    )

    selected_popularity = st.selectbox(
        "Popularity",
        updated_item_df["Attraction"].unique()
    )

    selected_demographic = st.selectbox(
        "Visitor Demographics",
        updated_item_df["AttractionAddress"].unique()
    )

    # HISTORICAL VISIT DATA

    st.header("Historical Visit Data")

    selected_month = st.selectbox(
        "Month",
        sorted(transaction_df["VisitMonth"].unique())
    )

    selected_year = st.selectbox(
        "Year",
        sorted(transaction_df["VisitYear"].unique())
    )

    selected_previous_mode = st.selectbox(
        "Previous Visit Mode",
        mode_df["VisitMode"].unique()
    )

    # PREDICT VISIT MODE

    if st.button("Predict Visit Mode"):

        input_data = pd.DataFrame([{
            "continent": selected_continent,
            "region": selected_region,
            "country": selected_country,
            "cityname": selected_city,
            "visityear": selected_year,
            "visitmonth": selected_month,
            "visitmode": selected_previous_mode,
            "attractiontype": selected_type,
            "avg_user_rating": 3
        }])

        input_data = pd.get_dummies(input_data)

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = visitmode_model.predict(input_data)

        predicted_mode = label_encoder.inverse_transform(
            prediction
        )[0]

        st.success(
            f"Predicted Visit Mode: {predicted_mode}"
        )

# 3. RECOMMENDATION SYSTEM

elif menu == "3. Recommendations: Personalized Attraction Suggestions":

    st.title(
        "Recommendations: Personalized Attraction Suggestions"
    )

    st.header("User Visit History")

    selected_attraction = st.selectbox(
        "Attractions Visited",
        updated_item_df["Attraction"].unique()
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
        type_df["AttractionType"].unique()
    )

    selected_location = st.selectbox(
        "Location",
        updated_item_df["AttractionAddress"].unique()
    )

    selected_popularity = st.selectbox(
        "Popularity",
        updated_item_df["Attraction"].unique()
    )

    st.header("Similar User Data")

    selected_pattern = st.selectbox(
        "Travel Patterns",
        updated_item_df["AttractionAddress"].unique()
    )

    selected_preference = st.selectbox(
        "Preferences",
        updated_item_df["Attraction"].unique()
    )

    if st.button("Generate Recommendations"):

        recommendations = updated_item_df[
            updated_item_df["AttractionTypeId"] ==
            updated_item_df[
                updated_item_df["Attraction"] ==
                selected_popularity
            ]["AttractionTypeId"].values[0]
        ]["Attraction"].head(10)

        st.subheader(
            "Ranked Recommended Attractions"
        )

        rank = 1

        for rec in recommendations:

            st.write(f"{rank}. {rec}")

            rank += 1

# 4. COLLABORATIVE FILTERING

elif menu == "4. Collaborative Filtering":

    st.title("Collaborative Filtering")

    selected_user = st.selectbox(
        "User ID",
        transaction_df["UserId"].unique()
    )

    recommendations = transaction_df[
        transaction_df["UserId"] != selected_user
    ]["AttractionId"].head(10)

    st.subheader("Recommended Attractions")

    for i in recommendations:

        attraction_name = item_df[
            item_df["AttractionId"] == i
        ]["Attraction"].values

        if len(attraction_name) > 0:

            st.write("-", attraction_name[0])

# 5. CONTENT BASED FILTERING

elif menu == "5. Content-Based Filtering":

    st.title("Content-Based Filtering")

    selected_attraction = st.selectbox(
        "Select Attraction",
        updated_item_df["Attraction"].unique()
    )

    attraction_type_id = updated_item_df[
        updated_item_df["Attraction"] ==
        selected_attraction
    ]["AttractionTypeId"].values[0]

    similar_attractions = updated_item_df[
        updated_item_df["AttractionTypeId"] ==
        attraction_type_id
    ]["Attraction"].head(10)

    st.subheader("Similar Attractions")

    for attraction in similar_attractions:

        st.write("-", attraction)

# 6. HYBRID SYSTEM

elif menu == "6. Hybrid Systems":

    st.title("Hybrid Recommendation System")

    selected_attraction = st.selectbox(
        "Select Attraction",
        updated_item_df["Attraction"].unique()
    )

    hybrid_recommendations = updated_item_df.sample(
        10
    )["Attraction"]

    st.subheader(
        "Hybrid Recommended Attractions"
    )

    for attraction in hybrid_recommendations:

        st.write("-", attraction)