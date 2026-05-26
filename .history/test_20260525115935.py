import streamlit as st
import pandas as pd

# ======================
# LOAD CCC DATA
# ======================

ccc_df = pd.read_excel("DATA SET/CCC.xlsx", sheet_name="Final_Hierarchy")

ccc_df = ccc_df.dropna(subset=["Continent", "Country", "City"])

ccc_df = ccc_df[
    (ccc_df["Continent"] != "-") &
    (ccc_df["Country"] != "-") &
    (ccc_df["City"] != "-")
]

# ======================
# CONTINENT DROPDOWN
# ======================

continents = sorted(ccc_df["Continent"].unique().tolist())

selected_continent = st.selectbox(
    "Select Continent",
    continents
)

# ======================
# COUNTRY DROPDOWN
# ======================

countries = sorted(
    ccc_df[
        ccc_df["Continent"] == selected_continent
    ]["Country"].unique().tolist()
)

selected_country = st.selectbox(
    "Select Country",
    countries
)

# ======================
# CITY DROPDOWN
# ======================

cities = sorted(
    ccc_df[
        (ccc_df["Continent"] == selected_continent) &
        (ccc_df["Country"] == selected_country)
    ]["City"].unique().tolist()
)

selected_city = st.selectbox(
    "Select City",
    cities
)

# ======================
# OUTPUT
# ======================

st.write("Continent:", selected_continent)
st.write("Country:", selected_country)
st.write("City:", selected_city)