# TOURISM EXPERIENCE ANALYTICS SYSTEM
# FINAL STREAMLIT APP

# TOURISM EXPERIENCE ANALYTICS SYSTEM
# FINAL STREAMLIT APP - FIXED RATING PREDICTION

import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Tourism Recommendation System",
    layout="wide"
)

# =====================================================
# DATA PATH
# =====================================================

DATA_PATH = r"D:\Tourism Experience Analytics\DATA SET"

# =====================================================
# LOAD DATASETS
# =====================================================

region_df = pd.read_excel(
    fr"{DATA_PATH}\Region.xlsx"
)

continent_df = pd.read_excel(
    fr"{DATA_PATH}\Continent.xlsx"
)

country_df = pd.read_excel(
    fr"{DATA_PATH}\Country.xlsx"
)

city_df = pd.read_excel(
    fr"{DATA_PATH}\City.xlsx"
)

mode_df = pd.read_excel(
    fr"{DATA_PATH}\Mode.xlsx"
)

type_df = pd.read_excel(
    fr"{DATA_PATH}\Type.xlsx"
)

transaction_df = pd.read_excel(
    fr"{DATA_PATH}\Transaction.xlsx"
)

updated_item_df = pd.read_excel(
    fr"{DATA_PATH}\Updated_Item.xlsx"
)

# =====================================================
# LOAD MODEL
# =====================================================

rating_model = joblib.load(
    "rating_prediction_model.pkl"
)

# =====================================================
# CLEAN COLUMN NAMES
# =====================================================

for df in [
    region_df,
    continent_df,
    country_df,
    city_df,
    mode_df,
    type_df,
    transaction_df,
    updated_item_df
]:
    df.columns = df.columns.str.strip()

# =====================================================
# MERGE COUNTRY → REGION → CONTINENT
# =====================================================

country_df = country_df.merge(
    region_df,
    on="RegionId",
    how="left"
)

country_df = country_df.merge(
    continent_df,
    on="ContinentId",
    how="left"
)

merged_city = city_df.merge(
    country_df,
    on="CountryId",
    how="left"
)

# =====================================================
# CLEAN IDS
# =====================================================

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

# =====================================================
# MERGE VISIT MODE
# =====================================================

merged_visit_df = transaction_df.merge(
    mode_df,
    left_on="VisitMode",
    right_on="VisitModeId",
    how="left"
)

# =====================================================
# SIDEBAR
# =====================================================

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

# =====================================================
# 1. REGRESSION
# =====================================================

if menu == "1. Regression: Predicting Attraction Ratings":

    st.title(
        "Regression: Predicting Attraction Ratings"
    )

    # =====================================================
    # USER DEMOGRAPHICS
    # =====================================================

    st.header("User Demographics")

    # ======================
    # CONTINENT DROPDOWN
    # ======================

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

    # ======================
    # REGION DROPDOWN
    # ======================

    regions = country_df[
        country_df["Continent"] == selected_continent
    ]["Region"].dropna().unique().tolist()

    regions = sorted([
        r for r in regions
        if r != "-"
    ])

    selected_region = st.selectbox(
        "Region",
        regions
    )

    # ======================
    # COUNTRY DROPDOWN
    # ======================

    countries = country_df[
        (
            country_df["Continent"] == selected_continent
        )
        &
        (
            country_df["Region"] == selected_region
        )
    ]["Country"].dropna().unique().tolist()

    countries = sorted([
        c for c in countries
        if c != "-"
    ])

    selected_country = st.selectbox(
        "Country",
        countries
    )

    # ======================
    # CITY DROPDOWN
    # ======================

    cities = merged_city[
        (
            merged_city["Continent"] == selected_continent
        )
        &
        (
            merged_city["Region"] == selected_region
        )
        &
        (
            merged_city["Country"] == selected_country
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

    selected_year = st.selectbox(
        "Select Visit Year",
        sorted(
            transaction_df["VisitYear"]
            .dropna()
            .unique()
        )
    )

    selected_month = st.selectbox(
        "Select Visit Month",
        list(range(1, 13))
    )

    selected_mode = st.selectbox(
        "Select Visit Mode",
        sorted(
            merged_visit_df["VisitMode_y"]
            .dropna()
            .unique()
        )
    )

    # =====================================================
    # ATTRACTION ATTRIBUTES
    # =====================================================

    st.header("Attraction Attributes")

    selected_type = st.selectbox(
        "Attraction Type",
        sorted(
            updated_item_df["AttractionType"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    # =====================================================
    # FILTER LOCATIONS BY TYPE + COUNTRY + CITY
    # =====================================================

    filtered_items = updated_item_df[
        (
            updated_item_df["AttractionType"] == selected_type
        )
        &
        (
            updated_item_df["AttractionAddress"]
            .astype(str)
            .str.contains(
                selected_country,
                case=False,
                na=False
            )
        )
        &
        (
            updated_item_df["AttractionAddress"]
            .astype(str)
            .str.contains(
                selected_city,
                case=False,
                na=False
            )
        )
    ]

    # Fallback 1: type + city
    if filtered_items.empty:
        filtered_items = updated_item_df[
            (
                updated_item_df["AttractionType"] == selected_type
            )
            &
            (
                updated_item_df["AttractionAddress"]
                .astype(str)
                .str.contains(
                    selected_city,
                    case=False,
                    na=False
                )
            )
        ]

    # Fallback 2: type + country
    if filtered_items.empty:
        filtered_items = updated_item_df[
            (
                updated_item_df["AttractionType"] == selected_type
            )
            &
            (
                updated_item_df["AttractionAddress"]
                .astype(str)
                .str.contains(
                    selected_country,
                    case=False,
                    na=False
                )
            )
        ]

    # Fallback 3: only type
    if filtered_items.empty:
        filtered_items = updated_item_df[
            updated_item_df["AttractionType"] == selected_type
        ]

    selected_location = st.selectbox(
        "Location",
        sorted(
            filtered_items["AttractionAddress"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    selected_row = filtered_items[
        filtered_items["AttractionAddress"] == selected_location
    ]

    selected_attraction_id = str(
        selected_row.iloc[0]["AttractionId"]
    )

    matched_transaction_df = transaction_df[
        transaction_df["AttractionId"] == selected_attraction_id
    ]

    if len(matched_transaction_df) > 0:
        avg_rating = round(
            matched_transaction_df["Rating"].mean(),
            2
        )
    else:
        avg_rating = 0

    st.write(
        "Selected Attraction ID:",
        selected_attraction_id
    )

    st.write(
        "Existing Average Rating:",
        avg_rating
    )

    # =====================================================
    # PREDICT RATING
    # =====================================================

    st.header("Predicted Rating")

    if st.button("Predict Rating"):

        input_data = pd.DataFrame([{
            "Continent": selected_continent,
            "Country": selected_country,
            "CityName": selected_city,
            "VisitYear": selected_year,
            "VisitMonth": selected_month,
            "VisitMode": selected_mode,
            "AttractionType": selected_type
        }])

        prediction = rating_model.predict(
            input_data
        )[0]

        prediction = round(
            float(prediction),
            2
        )

        if prediction > 5:
            prediction = 5

        if prediction < 1:
            prediction = 1

        st.success(
            f"Predicted Rating: {prediction} / 5"
        )

        st.write("Prediction Based On:")

        st.write("Continent:", selected_continent)
        st.write("Region:", selected_region)
        st.write("Country:", selected_country)
        st.write("City:", selected_city)
        st.write("Visit Year:", selected_year)
        st.write("Visit Month:", selected_month)
        st.write("Visit Mode:", selected_mode)
        st.write("Attraction Type:", selected_type)
        st.write("Location:", selected_location)

# =====================================================
# 2. CLASSIFICATION
# =====================================================

elif menu == "2. Classification: User Visit Mode Prediction":

    st.title(
        "Classification module not updated in this file"
    )

    # =====================================================
    # USER DEMOGRAPHICS
    # =====================================================

    st.header("User Demographics")

    # ======================
    # CONTINENT DROPDOWN
    # ======================

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

    # ======================
    # REGION DROPDOWN
    # ======================

    regions = country_df[
        country_df["Continent"] == selected_continent
    ]["Region"].dropna().unique().tolist()

    regions = sorted([
        r for r in regions
        if r != "-"
    ])

    selected_region = st.selectbox(
        "Region",
        regions
    )

    # ======================
    # COUNTRY DROPDOWN
    # ======================

    countries = country_df[
        (
            country_df["Continent"] == selected_continent
        )
        &
        (
            country_df["Region"] == selected_region
        )
    ]["Country"].dropna().unique().tolist()

    countries = sorted([
        c for c in countries
        if c != "-"
    ])

    selected_country = st.selectbox(
        "Country",
        countries
    )

    # ======================
    # CITY DROPDOWN
    # ======================

    cities = merged_city[
        (
            merged_city["Continent"] == selected_continent
        )
        &
        (
            merged_city["Region"] == selected_region
        )
        &
        (
            merged_city["Country"] == selected_country
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

    selected_year = st.selectbox(
        "Select Visit Year",
        sorted(
            transaction_df["VisitYear"]
            .dropna()
            .unique()
        )
    )

    selected_month = st.selectbox(
        "Select Visit Month",
        list(range(1, 13))
    )

    selected_mode = st.selectbox(
        "Select Visit Mode",
        sorted(
            merged_visit_df["VisitMode_y"]
            .dropna()
            .unique()
        )
    )

    # =====================================================
    # ATTRACTION ATTRIBUTES
    # =====================================================

    st.header("Attraction Attributes")

    selected_type = st.selectbox(
        "Attraction Type",
        sorted(
            updated_item_df["AttractionType"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    # =====================================================
    # FILTER LOCATIONS BY TYPE + COUNTRY + CITY
    # =====================================================

    filtered_items = updated_item_df[
        (
            updated_item_df["AttractionType"] == selected_type
        )
        &
        (
            updated_item_df["AttractionAddress"]
            .astype(str)
            .str.contains(
                selected_country,
                case=False,
                na=False
            )
        )
        &
        (
            updated_item_df["AttractionAddress"]
            .astype(str)
            .str.contains(
                selected_city,
                case=False,
                na=False
            )
        )
    ]

    # Fallback 1: type + city
    if filtered_items.empty:
        filtered_items = updated_item_df[
            (
                updated_item_df["AttractionType"] == selected_type
            )
            &
            (
                updated_item_df["AttractionAddress"]
                .astype(str)
                .str.contains(
                    selected_city,
                    case=False,
                    na=False
                )
            )
        ]

    # Fallback 2: type + country
    if filtered_items.empty:
        filtered_items = updated_item_df[
            (
                updated_item_df["AttractionType"] == selected_type
            )
            &
            (
                updated_item_df["AttractionAddress"]
                .astype(str)
                .str.contains(
                    selected_country,
                    case=False,
                    na=False
                )
            )
        ]

    # Fallback 3: only type
    if filtered_items.empty:
        filtered_items = updated_item_df[
            updated_item_df["AttractionType"] == selected_type
        ]

    selected_location = st.selectbox(
        "Location",
        sorted(
            filtered_items["AttractionAddress"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    selected_row = filtered_items[
        filtered_items["AttractionAddress"] == selected_location
    ]

    selected_attraction_id = str(
        selected_row.iloc[0]["AttractionId"]
    )

    matched_transaction_df = transaction_df[
        transaction_df["AttractionId"] == selected_attraction_id
    ]

    if len(matched_transaction_df) > 0:
        avg_rating = round(
            matched_transaction_df["Rating"].mean(),
            2
        )
    else:
        avg_rating = 0

    st.write(
        "Selected Attraction ID:",
        selected_attraction_id
    )

    st.write(
        "Existing Average Rating:",
        avg_rating
    )

# =====================================================
# 3. RECOMMENDATIONS
# =====================================================

elif menu == "3. Recommendations: Personalized Attraction Suggestions":

    st.title(
        "Recommendations: Personalized Attraction Suggestions"
    )

    selected_type = st.selectbox(
        "Attraction Type",
        sorted(
            updated_item_df["AttractionType"]
            .dropna()
            .unique()
        )
    )

    recommendations = updated_item_df[
        updated_item_df["AttractionType"] == selected_type
    ].sample(
        min(10, len(updated_item_df[updated_item_df["AttractionType"] == selected_type]))
    )

    st.subheader(
        "Recommended Attractions"
    )

    for idx, row in enumerate(
        recommendations["Attraction"],
        start=1
    ):
        st.write(f"{idx}. {row}")

# =====================================================
# 4. COLLABORATIVE FILTERING
# =====================================================

elif menu == "4. Collaborative Filtering":

    st.title("Collaborative Filtering")

    selected_user = st.selectbox(
        "User ID",
        sorted(
            transaction_df["UserId"]
            .dropna()
            .unique()
        )
    )

    recommendations = updated_item_df.sample(
        min(10, len(updated_item_df))
    )

    st.subheader(
        "Collaborative Recommendations"
    )

    for idx, row in enumerate(
        recommendations["Attraction"],
        start=1
    ):
        st.write(f"{idx}. {row}")

# =====================================================
# 5. CONTENT-BASED FILTERING
# =====================================================

elif menu == "5. Content-Based Filtering":

    st.title("Content-Based Filtering")

    selected_type = st.selectbox(
        "Attraction Type",
        sorted(
            updated_item_df["AttractionType"]
            .dropna()
            .unique()
        )
    )

    recommendations = updated_item_df[
        updated_item_df["AttractionType"] == selected_type
    ].sample(
        min(10, len(updated_item_df[updated_item_df["AttractionType"] == selected_type]))
    )

    st.subheader(
        "Content-Based Recommendations"
    )

    for idx, row in enumerate(
        recommendations["Attraction"],
        start=1
    ):
        st.write(f"{idx}. {row}")

# =====================================================
# 6. HYBRID SYSTEM
# =====================================================

elif menu == "6. Hybrid Systems":

    st.title("Hybrid Recommendation System")

    selected_user = st.selectbox(
        "User ID",
        sorted(
            transaction_df["UserId"]
            .dropna()
            .unique()
        )
    )

    selected_type = st.selectbox(
        "Attraction Type",
        sorted(
            updated_item_df["AttractionType"]
            .dropna()
            .unique()
        )
    )

    recommendations = updated_item_df[
        updated_item_df["AttractionType"] == selected_type
    ].sample(
        min(10, len(updated_item_df[updated_item_df["AttractionType"] == selected_type]))
    )

    st.subheader(
        "Hybrid Recommendations"
    )

    for idx, row in enumerate(
        recommendations["Attraction"],
        start=1
    ):
        st.write(f"{idx}. {row}")

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

    # =====================================================
# PREDICT VISIT MODE
# =====================================================

st.header("Predicted Visit Mode")

if st.button("Predict Visit Mode"):

    # =========================================
    # CREATE INPUT DATAFRAME
    # =========================================

    input_data = pd.DataFrame([{

        "Continent": selected_continent,

        "Region": selected_region,

        "Country": selected_country,

        "CityName": selected_city,

        "VisitYear": selected_year,

        "VisitMonth": selected_month,

        "VisitMode": selected_mode,

        "AttractionType": selected_type
    }])

    # =========================================
    # MODEL PREDICTION
    # =========================================

    prediction = visitmode_model.predict(
        input_data
    )[0]

    # =========================================
    # DISPLAY OUTPUT
    # =========================================

    st.success(
        f"Predicted Visit Mode : {prediction}"
    )

    # =========================================
    # SUMMARY
    # =========================================

    st.subheader("Prediction Summary")

    st.write("Continent :", selected_continent)

    st.write("Region :", selected_region)

    st.write("Country :", selected_country)

    st.write("City :", selected_city)

    st.write("Visit Year :", selected_year)

    st.write("Visit Month :", selected_month)

    st.write("Previous Visit Mode :", selected_mode)

    st.write("Attraction Type :", selected_type)
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