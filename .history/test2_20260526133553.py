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

type_df = pd.read_excel("Type.xlsx")

updated_item_df = pd.read_excel("Updated_Item.xlsx")

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

type_df["AttractionTypeId"] = (
    type_df["AttractionTypeId"]
    .astype(str)
    .str.zfill(3)
)

updated_item_df["AttractionTypeId"] = (
    updated_item_df["AttractionTypeId"]
    .astype(str)
    .str.zfill(3)
)

updated_item_df["AttractionId"] = (
    updated_item_df["AttractionId"]
    .astype(str)
)

transaction_df["AttractionId"] = (
    transaction_df["AttractionId"]
    .astype(str)
)

# =========================================
# MERGE TYPE NAMES
# =========================================

merged_attr_df = pd.merge(

    updated_item_df,

    type_df,

    on="AttractionTypeId",

    how="left"
)

# =========================================
# TYPE DROPDOWN
# =========================================

st.subheader("Attraction Attributes")

type_options = sorted(

    merged_attr_df["AttractionType"]

    .dropna()

    .unique()

    .tolist()
)

selected_type = st.selectbox(

    "Select Attraction Type",

    type_options
)

# =========================================
# FILTER LOCATION BASED ON TYPE
# =========================================

filtered_locations_df = merged_attr_df[

    merged_attr_df["AttractionType"]

    == selected_type
]

location_options = sorted(

    filtered_locations_df["AttractionAddress"]

    .dropna()

    .unique()

    .tolist()
)

selected_location = st.selectbox(

    "Select Attraction Location",

    location_options
)

# =========================================
# GET ATTRACTION ID
# =========================================

selected_attraction = filtered_locations_df[

    filtered_locations_df["AttractionAddress"]

    == selected_location
]

selected_attraction_id = str(

    selected_attraction.iloc[0]["AttractionId"]
)

# =========================================
# GET RATING DATA
# =========================================

rating_data = transaction_df[

    transaction_df["AttractionId"]

    == selected_attraction_id
]

# =========================================
# CALCULATE AVG RATING
# =========================================

if len(rating_data) > 0:

    avg_rating = round(

        rating_data["Rating"].mean(),

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

st.write("Attraction Type:", selected_type)

st.write("Attraction Location:", selected_location)

st.write("Attraction ID:", selected_attraction_id)

st.write("Calculated Average Rating:", avg_rating)

# =========================================
# SHOW MATCHED TRANSACTIONS
# =========================================

st.subheader("Matched Transaction Rows")

st.dataframe(rating_data)