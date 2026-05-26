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
# LOAD CLASSIFICATION MODEL
# =====================================================

visitmode_model = joblib.load(
    "visitmode_prediction_model.pkl"
)

# =====================================================
# LOAD LABEL ENCODER
# =====================================================

label_encoder = joblib.load(
    "label_encoder.pkl"
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
        "3. Collaborative Filtering",
        "4. Content-Based Filtering",
        "5. Hybrid Systems"
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

    # =====================================================
    # FIX REGION COLUMN
    # =====================================================

    # Region.xlsx already contains:
    # RegionId | Region | Continent

    # So merge only country + region

    merged_country = country_df.merge(

        region_df,

        on="RegionId",

        how="left"
    )

    # =====================================================
    # MERGE CITY
    # =====================================================

    merged_city = city_df.merge(

        merged_country,

        on="CountryId",

        how="left"
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
    # TITLE
    # =====================================================

    st.title(
        "Classification: User Visit Mode Prediction"
    )

    # =====================================================
    # USER DEMOGRAPHICS
    # =====================================================

    st.header("User Demographics")

    # ======================
    # CONTINENT
    # ======================

    continents = sorted(
        merged_country["Continent"]
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
        continents,
        key="class_continent"
    )

    # ======================
    # REGION
    # ======================

    regions = sorted(
        merged_country[
            merged_country["Continent"]
            == selected_continent
        ]["Region_x"]
        .dropna()
        .unique()
        .tolist()
    )

    regions = [
        r for r in regions
        if r != "-"
    ]

    selected_region = st.selectbox(
        "Region",
        regions,
        key="class_region"
    )

    # ======================
    # COUNTRY
    # ======================

    countries = sorted(
        merged_country[
            (
                merged_country["Continent"]
                == selected_continent
            )
            &
            (
                merged_country["Region_x"]
                == selected_region
            )
        ]["Country"]
        .dropna()
        .unique()
        .tolist()
    )

    countries = [
        c for c in countries
        if c != "-"
    ]

    selected_country = st.selectbox(
        "Country",
        countries,
        key="class_country"
    )

    # ======================
    # CITY
    # ======================

    cities = sorted(
        merged_city[
            (
                merged_city["Continent"]
                == selected_continent
            )
            &
            (
                merged_city["Region_x"]
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

    cities = [
        c for c in cities
        if c != "-"
    ]

    selected_city = st.selectbox(
        "City",
        cities,
        key="class_city"
    )

    # =====================================================
    # VISIT DETAILS
    # =====================================================

    st.header("Visit Details")

    selected_year = st.selectbox(
        "Visit Year",
        sorted(
            transaction_df["VisitYear"]
            .dropna()
            .unique()
        ),
        key="class_year"
    )

    selected_month = st.selectbox(
        "Visit Month",
        list(range(1, 13)),
        key="class_month"
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
        ),
        key="class_type"
    )

    # =====================================================
    # FILTER LOCATIONS
    # =====================================================

    filtered_items = updated_item_df[
        updated_item_df["AttractionType"]
        == selected_type
    ]

    if filtered_items.empty:

        filtered_items = updated_item_df

    selected_location = st.selectbox(
        "Location",
        sorted(
            filtered_items["AttractionAddress"]
            .dropna()
            .unique()
            .tolist()
        ),
        key="class_location"
    )

    # =====================================================
    # GET ATTRACTION ID
    # =====================================================

    selected_row = filtered_items[
        filtered_items["AttractionAddress"]
        == selected_location
    ]

    selected_attraction_id = str(
        selected_row.iloc[0]["AttractionId"]
    )

    # =====================================================
    # MATCH TRANSACTION DATA
    # =====================================================

    matched_transaction_df = transaction_df[
        transaction_df["AttractionId"]
        == selected_attraction_id
    ]

    # =====================================================
    # AVERAGE RATING
    # =====================================================

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
    # PREDICT VISIT MODE
    # =====================================================

    st.header("Predicted Visit Mode")

    if st.button(
        "Predict Visit Mode",
        key="classification_predict_button"
    ):

        # =========================================
        # INPUT DATAFRAME
        # =========================================

        input_data = pd.DataFrame([{

            "Continent": selected_continent,

            "Region": selected_region,

            "Country": selected_country,

            "CityName": selected_city,

            "VisitYear": selected_year,

            "VisitMonth": selected_month,

            "AttractionType": selected_type
        }])

        # =========================================
        # MODEL PREDICTION
        # =========================================

        prediction_encoded = visitmode_model.predict(
            input_data
        )[0]

        # =========================================
        # CONVERT LABEL
        # =========================================

        prediction = label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        # =========================================
        # DISPLAY RESULT
        # =========================================

        st.success(
            f"Predicted Visit Mode : {prediction}"
        )

        # =========================================
        # SUMMARY
        # =========================================

        st.subheader("Prediction Summary")

        st.write(
            "Continent :",
            selected_continent
        )

        st.write(
            "Region :",
            selected_region
        )

        st.write(
            "Country :",
            selected_country
        )

        st.write(
            "City :",
            selected_city
        )

        st.write(
            "Visit Year :",
            selected_year
        )

        st.write(
            "Visit Month :",
            selected_month
        )

        st.write(
            "Attraction Type :",
            selected_type
        )

        st.write(
            "Location :",
            selected_location
        )

        st.write(
            "Average Rating :",
            avg_rating
        )

# =====================================================
# 3. COLLABORATIVE FILTERING
# =====================================================

elif menu == "3. Collaborative Filtering":

    st.title("Collaborative Filtering")

    st.header("User Visit History")

    selected_user = st.selectbox(

        "User ID",

        sorted(
            transaction_df["UserId"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    # =====================================================
    # USER HISTORY
    # =====================================================

    user_history = transaction_df[

        transaction_df["UserId"]
        .astype(str)
        == str(selected_user)
    ]

    visited_ids = user_history[
        "AttractionId"
    ].astype(str).unique()

    avg_user_rating = round(

        user_history["Rating"].mean(),

        2
    )

    st.write(
        "Average User Rating:",
        avg_user_rating
    )

    # =====================================================
    # FIND SIMILAR USERS
    # =====================================================

    preferred_types = updated_item_df[

        updated_item_df["AttractionId"]
        .astype(str)
        .isin(visited_ids)

    ]["AttractionType"].unique()

    similar_users = transaction_df[

        transaction_df["AttractionId"]
        .astype(str)
        .isin(visited_ids)
    ]

    similar_user_ids = similar_users[
        "UserId"
    ].astype(str).unique()

    collaborative_transactions = transaction_df[

        transaction_df["UserId"]
        .astype(str)
        .isin(similar_user_ids)
    ]

    top_attractions = collaborative_transactions.groupby(

        "AttractionId"

    )["Rating"].mean().reset_index()

    top_attractions = top_attractions.sort_values(

        by="Rating",

        ascending=False
    )

    recommended_df = updated_item_df.merge(

        top_attractions,

        on="AttractionId",

        how="inner"
    )

    recommended_df = recommended_df[

        ~recommended_df["AttractionId"]
        .astype(str)
        .isin(visited_ids)
    ]

    recommended_df = recommended_df.head(10)

    st.subheader(
        "Ranked List of Recommended Attractions"
    )

    for idx, row in enumerate(

        recommended_df.itertuples(),

        start=1
    ):

        st.write(

            f"{idx}. {row.Attraction}"

        )

        st.write(
            "Type:",
            row.AttractionType
        )

        st.write(
            "Location:",
            row.AttractionAddress
        )

        st.write(
            "Predicted Rating:",
            round(row.Rating, 2)
        )

        st.markdown("---")

# =====================================================
# 4. CONTENT-BASED FILTERING
# =====================================================

elif menu == "4. Content-Based Filtering":

    st.title("Content-Based Filtering")

    st.header("Attraction Features")

    selected_type = st.selectbox(

        "Attraction Type",

        sorted(
            updated_item_df["AttractionType"]
            .dropna()
            .unique()
        )
    )

    filtered_locations = updated_item_df[

        updated_item_df["AttractionType"]
        == selected_type

    ]["AttractionAddress"].dropna().unique()

    selected_location = st.selectbox(

        "Location",

        sorted(filtered_locations)
    )

    # =====================================================
    # FILTER CONTENT
    # =====================================================

    filtered_df = updated_item_df[

        (
            updated_item_df["AttractionType"]
            == selected_type
        )
        &
        (
            updated_item_df["AttractionAddress"]
            == selected_location
        )
    ]

    attraction_ids = filtered_df[
        "AttractionId"
    ].astype(str)

    ratings_df = transaction_df[

        transaction_df["AttractionId"]
        .astype(str)
        .isin(attraction_ids)
    ]

    popularity_df = ratings_df.groupby(

        "AttractionId"

    )["Rating"].mean().reset_index()

    popularity_df = popularity_df.sort_values(

        by="Rating",

        ascending=False
    )

    recommendations = updated_item_df.merge(

        popularity_df,

        on="AttractionId",

        how="inner"
    )

    recommendations = recommendations.sort_values(

        by="Rating",

        ascending=False
    ).head(10)

    st.subheader(
        "Ranked List of Recommended Attractions"
    )

    for idx, row in enumerate(

        recommendations.itertuples(),

        start=1
    ):

        st.write(
            f"{idx}. {row.Attraction}"
        )

        st.write(
            "Type:",
            row.AttractionType
        )

        st.write(
            "Location:",
            row.AttractionAddress
        )

        st.write(
            "Popularity Rating:",
            round(row.Rating, 2)
        )

        st.markdown("---")

# =====================================================
# 5. HYBRID SYSTEM
# =====================================================

elif menu == "5. Hybrid Systems":

    st.title("Hybrid Recommendation System")

    # =====================================================
    # USER HISTORY
    # =====================================================

    selected_user = st.selectbox(

        "User ID",

        sorted(
            transaction_df["UserId"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    # =====================================================
    # CONTENT FEATURES
    # =====================================================

    selected_type = st.selectbox(

        "Attraction Type",

        sorted(
            updated_item_df["AttractionType"]
            .dropna()
            .unique()
        )
    )

    filtered_locations = updated_item_df[

        updated_item_df["AttractionType"]
        == selected_type

    ]["AttractionAddress"].dropna().unique()

    selected_location = st.selectbox(

        "Location",

        sorted(filtered_locations)
    )

    # =====================================================
    # USER HISTORY
    # =====================================================

    user_history = transaction_df[

        transaction_df["UserId"]
        .astype(str)
        == str(selected_user)
    ]

    visited_ids = user_history[
        "AttractionId"
    ].astype(str).unique()

    # =====================================================
    # CONTENT FILTER
    # =====================================================

    content_df = updated_item_df[

        (
            updated_item_df["AttractionType"]
            == selected_type
        )
        &
        (
            updated_item_df["AttractionAddress"]
            == selected_location
        )
    ]

    content_ids = content_df[
        "AttractionId"
    ].astype(str)

    collaborative_df = transaction_df[

        transaction_df["AttractionId"]
        .astype(str)
        .isin(content_ids)
    ]

    hybrid_scores = collaborative_df.groupby(

        "AttractionId"

    )["Rating"].mean().reset_index()

    hybrid_scores = hybrid_scores.sort_values(

        by="Rating",

        ascending=False
    )

    recommendations = updated_item_df.merge(

        hybrid_scores,

        on="AttractionId",

        how="inner"
    )

    recommendations = recommendations[

        ~recommendations["AttractionId"]
        .astype(str)
        .isin(visited_ids)
    ]

    recommendations = recommendations.head(10)

    st.subheader(
        "Ranked List of Recommended Attractions"
    )

    for idx, row in enumerate(

        recommendations.itertuples(),

        start=1
    ):

        st.write(
            f"{idx}. {row.Attraction}"
        )

        st.write(
            "Type:",
            row.AttractionType
        )

        st.write(
            "Location:",
            row.AttractionAddress
        )

        st.write(
            "Hybrid Score:",
            round(row.Rating, 2)
        )

        st.markdown("---")