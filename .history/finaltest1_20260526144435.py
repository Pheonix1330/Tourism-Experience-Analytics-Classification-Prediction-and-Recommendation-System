import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Tourism Rating Prediction Test",
    layout="wide"
)

# =====================================================
# DATA PATH
# =====================================================

DATA_PATH = r"D:\Tourism Experience Analytics\DATA SET"

# =====================================================
# LOAD DATA
# =====================================================

continent_df = pd.read_excel(
    fr"{DATA_PATH}\Continent.xlsx"
)

region_df = pd.read_excel(
    fr"{DATA_PATH}\Region.xlsx"
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

    continent_df,

    region_df,

    country_df,

    city_df,

    mode_df,

    transaction_df,

    updated_item_df
]:

    df.columns = df.columns.str.strip()

# =====================================================
# MERGE REGION + CONTINENT
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
# TITLE
# =====================================================

st.title(
    "Tourism Rating Prediction Test"
)

# =====================================================
# CONTINENT
# =====================================================

selected_continent = st.selectbox(

    "Continent",

    sorted(

        country_df["Continent"]

        .dropna()

        .unique()
    )
)

# =====================================================
# REGION
# =====================================================

selected_region = st.selectbox(

    "Region",

    sorted(

        country_df[

            country_df["Continent"]

            == selected_continent

        ]["Region"]

        .dropna()

        .unique()
    )
)

# =====================================================
# COUNTRY
# =====================================================

selected_country = st.selectbox(

    "Country",

    sorted(

        country_df[

            (

                country_df["Continent"]

                == selected_continent

            )

            &

            (

                country_df["Region"]

                == selected_region

            )

        ]["Country"]

        .dropna()

        .unique()
    )
)

# =====================================================
# CITY
# =====================================================

selected_city = st.selectbox(

    "City",

    sorted(

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
    )
)

# =====================================================
# YEAR
# =====================================================

selected_year = st.selectbox(

    "Visit Year",

    sorted(

        transaction_df["VisitYear"]

        .dropna()

        .unique()
    )
)

# =====================================================
# MONTH
# =====================================================

selected_month = st.selectbox(

    "Visit Month",

    list(range(1, 13))
)

# =====================================================
# VISIT MODE
# =====================================================

selected_mode = st.selectbox(

    "Visit Mode",

    sorted(

        merged_visit_df["VisitMode_y"]

        .dropna()

        .unique()
    )
)

# =====================================================
# ATTRACTION TYPE
# =====================================================

selected_type = st.selectbox(

    "Attraction Type",

    sorted(

        updated_item_df["AttractionType"]

        .dropna()

        .unique()
    )
)

# =====================================================
# FILTER LOCATION
# =====================================================

filtered_items = updated_item_df[

    (

        updated_item_df["AttractionType"]

        == selected_type

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

# =====================================================
# FALLBACK CITY
# =====================================================

if filtered_items.empty:

    filtered_items = updated_item_df[

        (

            updated_item_df["AttractionType"]

            == selected_type

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

# =====================================================
# FALLBACK COUNTRY
# =====================================================

if filtered_items.empty:

    filtered_items = updated_item_df[

        (

            updated_item_df["AttractionType"]

            == selected_type

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

# =====================================================
# FINAL FALLBACK
# =====================================================

if filtered_items.empty:

    filtered_items = updated_item_df[

        updated_item_df["AttractionType"]

        == selected_type
    ]

# =====================================================
# LOCATION DROPDOWN
# =====================================================

selected_location = st.selectbox(

    "Location",

    sorted(

        filtered_items["AttractionAddress"]

        .dropna()

        .unique()
    )
)

selected_item = filtered_items[

    filtered_items["AttractionAddress"]

    == selected_location

].iloc[0]

selected_attraction_id = selected_item["AttractionId"]

st.write(

    "Selected Attraction ID:",

    selected_attraction_id
)

# =====================================================
# PREDICT BUTTON
# =====================================================

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

    st.write("Continent:", selected_continent)

    st.write("Region:", selected_region)

    st.write("Country:", selected_country)

    st.write("City:", selected_city)

    st.write("Visit Year:", selected_year)

    st.write("Visit Month:", selected_month)

    st.write("Visit Mode:", selected_mode)

    st.write("Attraction Type:", selected_type)

    st.write("Location:", selected_location)