import streamlit as st
import pandas as pd

# ======================
# LOAD DATA
# ======================

continent_df = pd.read_excel("DATA SET/Continent.xlsx")
country_df = pd.read_excel("DATA SET/Country.xlsx")
city_df = pd.read_excel("DATA SET/City.xlsx")

# ======================
# MERGE TABLES
# ======================

merged_country = country_df.merge(
    continent_df,
    left_on="RegionId",
    right_on="ContinentId",
    how="left"
)

merged_city = city_df.merge(
    country_df,
    on="CountryId",
    how="left"
)

merged_city = merged_city.merge(
    continent_df,
    left_on="RegionId",
    right_on="ContinentId",
    how="left"
)

# ======================
# CONTINENT DROPDOWN
# ======================

continents = sorted(
    continent_df["Continent"]
    .dropna()
    .unique()
    .tolist()
)

continents = [c for c in continents if c != "-"]

selected_continent = st.selectbox(
    "Select Continent",
    continents
)

# ======================
# COUNTRY DROPDOWN
# ======================

countries = merged_country[
    merged_country["Continent"] == selected_continent
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