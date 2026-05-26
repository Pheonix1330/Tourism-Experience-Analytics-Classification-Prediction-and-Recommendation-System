import streamlit as st
import pandas as pd

# ======================
# LOAD DATA
# ======================

country_df = pd.read_excel("DATA SET/Country.xlsx")
city_df = pd.read_excel("DATA SET/City.xlsx")

# ======================
# FIX CONTINENT MAPPING
# ======================

def get_continent(region_id):
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

# ======================
# MERGE CITY WITH COUNTRY
# ======================

merged_city = city_df.merge(
    country_df,
    on="CountryId",
    how="left"
)

# ======================
# CONTINENT DROPDOWN
# ======================

continents = sorted(
    country_df["Continent"]
    .dropna()
    .unique()
    .tolist()
)

selected_continent = st.selectbox(
    "Select Continent",
    continents
)

# ======================
# COUNTRY DROPDOWN
# ======================

countries = merged_city[
    merged_city["Continent"] == selected_continent
]["Country"].dropna().unique().tolist()

countries = sorted([c for c in countries if c != "-"])

selected_country = st.selectbox(
    "Select Country",
    countries
)

# ======================
# CITY DROPDOWN
# ======================

cities = merged_city[
    (merged_city["Continent"] == selected_continent) &
    (merged_city["Country"] == selected_country)
]["CityName"].dropna().unique().tolist()

cities = sorted([c for c in cities if c != "-"])

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