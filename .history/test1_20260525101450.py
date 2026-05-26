import pandas as pd
import streamlit as st

# LOAD FILES
transaction_df = pd.read_excel("DATA SET/Transaction.xlsx")
mode_df = pd.read_excel("DATA SET/Mode.xlsx")

# LINK VISIT MODE IDS
# Merge transaction and mode tables
merged_df = transaction_df.merge(
    mode_df,
    left_on="VisitMode",
    right_on="VisitModeId",
    how="left"
)

# VISIT DETAILS FILTERS

st.subheader("Visit Details")

# -------------------------
# YEAR OPTIONS
# -------------------------
# Get year range from Transaction.xlsx
min_year = int(merged_df["VisitYear"].min())
max_year = int(merged_df["VisitYear"].max())

year_options = list(range(min_year, max_year + 1))

selected_year = st.selectbox(
    "Select Visit Year",
    year_options
)

# -------------------------
# MONTH OPTIONS
# -------------------------
month_options = list(range(1, 13))

selected_month = st.selectbox(
    "Select Visit Month",
    month_options
)

# -------------------------
# VISIT MODE OPTIONS
# -------------------------
visit_mode_options = merged_df["VisitMode"].dropna().unique()

selected_mode = st.selectbox(
    "Select Visit Mode",
    sorted(visit_mode_options)
)

# FILTER DATA
filtered_df = merged_df[
    (merged_df["VisitYear"] == selected_year) &
    (merged_df["VisitMonth"] == selected_month) &
    (merged_df["VisitMode"] == selected_mode)
]

# SHOW RESULTS
st.write("Filtered Data")
st.dataframe(filtered_df)