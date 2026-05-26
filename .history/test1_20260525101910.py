import pandas as pd
import streamlit as st

# LOAD EXCEL FILES
transaction_df = pd.read_excel("DATA SET/Transaction.xlsx")
mode_df = pd.read_excel("Mode.xlsx")

# CLEAN COLUMN NAMES
transaction_df.columns = transaction_df.columns.str.strip()
mode_df.columns = mode_df.columns.str.strip()

# CONVERT VISIT MODE IDS TO STRING
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

# MERGE MODE NAMES
merged_df = pd.merge(
    transaction_df,
    mode_df,
    left_on="VisitMode",
    right_on="VisitModeId",
    how="left"
)

# STREAMLIT PAGE
st.title("Tourism Visit Details")

st.subheader("Visit Filters")

# YEAR OPTIONS
year_options = sorted(
    merged_df["VisitYear"].dropna().unique()
)

selected_year = st.selectbox(
    "Select Visit Year",
    year_options
)

# MONTH OPTIONS
month_options = list(range(1, 13))

selected_month = st.selectbox(
    "Select Visit Month",
    month_options
)

# VISIT MODE OPTIONS
visit_mode_options = sorted(
    merged_df["VisitMode_y"].dropna().unique()
)

selected_mode = st.selectbox(
    "Select Visit Mode",
    visit_mode_options
)

# FILTER DATA
filtered_df = merged_df[
    (merged_df["VisitYear"] == selected_year) &
    (merged_df["VisitMonth"] == selected_month) &
    (merged_df["VisitMode_y"] == selected_mode)
]

# DISPLAY FILTERED DATA
st.subheader("Filtered Transaction Data")

st.dataframe(filtered_df)

# OPTIONAL DISPLAY
st.subheader("Merged Columns Preview")

st.write(merged_df.columns.tolist())