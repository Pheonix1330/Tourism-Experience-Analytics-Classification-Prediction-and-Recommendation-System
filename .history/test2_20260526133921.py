import streamlit as st
import pandas as pd

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Attraction Attribute Testing",
    layout="wide"
)

st.title("Attraction Attribute Testing")

# =========================================
# LOAD FILES
# =========================================

type_df = pd.read_excel("DATA SET/Type.xlsx")

updated_item_df = pd.read_excel("DATA SET/Updated_Item.xlsx")

transaction_df = pd.read_excel("Transaction.xlsx")

# =========================================
# CLEAN COLUMN NAMES
# =========================================

type_df.columns = type_df.columns.str.strip()

updated_item_df.columns = (
    updated_item_df.columns.str.strip()
)

transaction_df.columns = (
    transaction_df.columns.str.strip()
)

# =========================================
# CONVERT IDS TO STRING
# =========================================

updated_item_df["AttractionId"] = (
    updated_item_df["AttractionId"]
    .astype(str)
)

transaction_df["AttractionId"] = (
    transaction_df["AttractionId"]
    .astype(str)
)

# =========================================
# TYPE OPTIONS
# =========================================

type_options = sorted(

    updated_item_df["AttractionType"]

    .dropna()

    .unique()

    .tolist()
)

selected_type = st.selectbox(

    "Select Attraction Type",

    type_options
)

# =========================================
# FILTER LOCATIONS BASED ON TYPE
# =========================================

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

    "Select Attraction Location",

    location_options
)

# =========================================
# GET SELECTED ATTRACTION ID
# =========================================

selected_row = filtered_type_df[

    filtered_type_df["AttractionAddress"]

    == selected_location
]

selected_attraction_id = str(

    selected_row.iloc[0]["AttractionId"]
)

# =========================================
# MATCH TRANSACTION USING ATTRACTION ID
# =========================================

matched_transaction_df = transaction_df[

    transaction_df["AttractionId"]

    == selected_attraction_id
]

# =========================================
# CALCULATE AVG RATING
# =========================================

if len(matched_transaction_df) > 0:

    avg_rating = round(

        matched_transaction_df["Rating"]

        .mean(),

        2
    )

else:

    avg_rating = 0

# =========================================
# RATING SLIDER
# =========================================

selected_avg_rating = st.slider(

    "Previous Average Rating",

    1.0,

    5.0,

    float(min(max(avg_rating, 1), 5))
)

# =========================================
# OUTPUT
# =========================================

st.subheader("Selected Details")

st.write("Selected Type:", selected_type)

st.write("Selected Location:", selected_location)

st.write("Attraction ID:", selected_attraction_id)

st.write("Calculated Average Rating:", avg_rating)

# =========================================
# SHOW MATCHED TRANSACTION DATA
# =========================================

st.subheader("Matched Transaction Rows")

st.dataframe(matched_transaction_df)