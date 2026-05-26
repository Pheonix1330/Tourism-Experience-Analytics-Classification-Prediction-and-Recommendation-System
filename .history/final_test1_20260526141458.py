# =========================================================
# TOURISM EXPERIENCE ANALYTICS
# RATING PREDICTION STREAMLIT APP
# =========================================================

import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="Tourism Rating Prediction",

    layout="wide"
)

st.title("Tourism Rating Prediction System")

# =========================================================
# LOAD DATASETS
# =========================================================

continent_df = pd.read_excel("DATA SET/Continent.xlsx")

country_df = pd.read_excel("DATA SET/Country.xlsx")

city_df = pd.read_excel("DATA SET/City.xlsx")

user_df = pd.read_excel("DATA SET/User.xlsx")

mode_df = pd.read_excel("DATA SET/Mode.xlsx")

transaction_df = pd.read_excel("DATA SET/Transaction.xlsx")

updated_item_df = pd.read_excel("Updated_Item.xlsx")

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(

    "rating_prediction_model.pkl"
)

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

dfs = [

    continent_df,

    country_df,

    city_df,

    user_df,

    mode_df,

    transaction_df,

    updated_item_df
]

for df in dfs:

    df.columns = df.columns.str.strip()

# =========================================================
# CONVERT IDS
# =========================================================

transaction_df["AttractionId"] = (
    transaction_df["AttractionId"]
    .astype(str)
)

updated_item_df["AttractionId"] = (
    updated_item_df["AttractionId"]
    .astype(str)
)

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

# =========================================================
# MERGE COUNTRY + CONTINENT
# =========================================================

merged_country = country_df.merge(

    continent_df,

    left_on="RegionId",

    right_on="ContinentId",

    how="left"
)

# =========================================================
# MERGE CITY + COUNTRY + CONTINENT
# =========================================================

merged_city = city_df.merge(

    merged_country,

    on="CountryId",

    how="left"
)

# =========================================================
# USER DEMOGRAPHICS
# =========================================================

st.header("User Demographics")

# CONTINENT

continents = sorted(

    merged_city["Continent"]

    .dropna()

    .unique()

    .tolist()
)

selected_continent = st.selectbox(

    "Continent",

    continents
)

# COUNTRY

countries = sorted(

    merged_city[

        merged_city["Continent"]

        == selected_continent

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

# =========================================================
# VISIT DETAILS
# =========================================================

st.header("Visit Details")

# MERGE VISIT MODE

visit_df = transaction_df.merge(

    mode_df,

    left_on="VisitMode",

    right_on="VisitModeId",

    how="left"
)

# YEAR

year_options = sorted(

    visit_df["VisitYear"]

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

    visit_df["VisitMode_y"]

    .dropna()

    .unique()
)

selected_mode = st.selectbox(

    "Visit Mode",

    visit_mode_options
)

# =========================================================
# ATTRACTION ATTRIBUTES
# =========================================================

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

filtered_location_df = updated_item_df[

    updated_item_df["AttractionType"]

    == selected_type
]

location_options = sorted(

    filtered_location_df["AttractionAddress"]

    .dropna()

    .unique()

    .tolist()
)

selected_location = st.selectbox(

    "Location",

    location_options
)

# =========================================================
# GET ATTRACTION ID
# =========================================================

selected_row = filtered_location_df[

    filtered_location_df["AttractionAddress"]

    == selected_location
]

selected_attraction_id = str(

    selected_row.iloc[0]["AttractionId"]
)

# =========================================================
# GET AVG RATING
# =========================================================

matched_transactions = transaction_df[

    transaction_df["AttractionId"]

    == selected_attraction_id
]

if len(matched_transactions) > 0:

    avg_rating = round(

        matched_transactions["Rating"]

        .mean(),

        2
    )

else:

    avg_rating = 3.0

# =========================================================
# PREVIOUS AVG RATING
# =========================================================

selected_avg_rating = st.slider(

    "Previous Average Rating",

    1.0,

    5.0,

    float(min(max(avg_rating, 1), 5))
)

# =========================================================
# PREDICT BUTTON
# =========================================================

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

    prediction = model.predict(

        input_df
    )[0]

    # CLEAN OUTPUT

    prediction = round(prediction, 2)

    prediction = max(1, min(5, prediction))

    # DISPLAY

    st.success(

        f"Predicted Rating : {prediction} / 5"
    )

    st.subheader("Prediction Summary")

    st.write("Continent :", selected_continent)

    st.write("Country :", selected_country)

    st.write("City :", selected_city)

    st.write("Visit Year :", selected_year)

    st.write("Visit Month :", selected_month)

    st.write("Visit Mode :", selected_mode)

    st.write("Attraction Type :", selected_type)

    st.write("Location :", selected_location)

    st.write("Previous Avg Rating :", selected_avg_rating)