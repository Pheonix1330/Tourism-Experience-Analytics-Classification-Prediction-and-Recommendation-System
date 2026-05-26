# Tourism Experience Analytics System

This project is a Streamlit-based tourism recommendation and analytics system. It uses tourism transaction data, attraction metadata, location data, and trained machine learning models to predict attraction ratings, predict user visit modes, and recommend tourist attractions.

## Project Overview

The application provides five main modules:

1. **Regression: Predicting Attraction Ratings**
   - Predicts the expected rating for a selected tourist attraction.
   - Uses user location, visit year, visit month, visit mode, and attraction type as inputs.

2. **Classification: User Visit Mode Prediction**
   - Predicts the likely visit mode for a user.
   - Uses demographic and attraction-related information such as continent, region, country, city, visit year, visit month, and attraction type.

3. **Collaborative Filtering**
   - Recommends attractions based on similar users and their rating behavior.
   - Uses user visit history and ratings from the transaction dataset.

4. **Content-Based Filtering**
   - Recommends attractions based on attraction type and location.
   - Ranks attractions using their average popularity rating.

5. **Hybrid Recommendation System**
   - Combines user history with attraction content features.
   - Produces ranked recommendations by considering both collaborative and content-based signals.

## Technologies Used

- Python
- Streamlit
- Pandas
- Joblib
- Scikit-learn models saved as `.pkl` files
- Excel datasets

## Project Structure

```text
Tourism Experience Analytics/
|-- app.py
|-- README.md
|-- DATA SET/
|   |-- Region.xlsx
|   |-- Continent.xlsx
|   |-- Country.xlsx
|   |-- City.xlsx
|   |-- Mode.xlsx
|   |-- Type.xlsx
|   |-- Transaction.xlsx
|   `-- Updated_Item.xlsx
|-- rating_prediction_model.pkl
|-- visitmode_prediction_model.pkl
|-- label_encoder.pkl
|-- feature_columns.pkl
|-- ML.ipynb
`-- MODEL_TRAINING.ipynb
```

## Installation Process

### 1. Install Python

Install Python 3.9 or later from:

```text
https://www.python.org/downloads/
```

Make sure Python is added to your system PATH.

### 2. Open the Project Folder

Open a terminal or command prompt inside the project directory:

```bash
cd "D:\Tourism Experience Analytics"
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

### 4. Install Required Packages

```bash
pip install streamlit pandas joblib scikit-learn openpyxl
```

### 5. Check Dataset and Model Files

Before running the app, make sure the following files exist:

- `DATA SET/Region.xlsx`
- `DATA SET/Continent.xlsx`
- `DATA SET/Country.xlsx`
- `DATA SET/City.xlsx`
- `DATA SET/Mode.xlsx`
- `DATA SET/Type.xlsx`
- `DATA SET/Transaction.xlsx`
- `DATA SET/Updated_Item.xlsx`
- `rating_prediction_model.pkl`
- `visitmode_prediction_model.pkl`
- `label_encoder.pkl`

The current application uses this dataset path inside `app.py`:

```python
DATA_PATH = r"D:\Tourism Experience Analytics\DATA SET"
```

If you move the project to another location, update this path in `app.py`.

## How to Run the Project

Run the Streamlit application with:

```bash
streamlit run app.py
```

After the command starts, Streamlit will open the app in your browser. If it does not open automatically, copy the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How the Application Works

### Data Loading

The app loads multiple Excel files from the `DATA SET` folder. These files contain information about regions, continents, countries, cities, visit modes, attraction types, user transactions, and attraction details.

### Data Preparation

The app cleans column names, standardizes IDs, and merges related datasets. For example:

- Country data is connected with region and continent data.
- City data is connected with country data.
- Transaction data is connected with visit mode data.
- Attraction IDs are converted to a consistent format for matching.

### Rating Prediction

In the regression module, the user selects demographic details, visit details, attraction type, and location. The app sends these values to the trained `rating_prediction_model.pkl` model and displays the predicted attraction rating out of 5.

### Visit Mode Prediction

In the classification module, the user selects location, visit time, and attraction type. The app uses `visitmode_prediction_model.pkl` to predict the visit mode. The encoded model output is converted back to a readable label using `label_encoder.pkl`.

### Collaborative Recommendations

The collaborative filtering module selects a user, checks their visit history, finds similar users who visited the same attractions, and recommends highly rated attractions that the selected user has not already visited.

### Content-Based Recommendations

The content-based module filters attractions by attraction type and location. It then ranks matching attractions using their average ratings from the transaction dataset.

### Hybrid Recommendations

The hybrid module combines the selected user's history with attraction type and location filters. It recommends attractions that match the chosen content features while excluding attractions already visited by the user.

## Notes

- Keep the trained model files in the same folder as `app.py`.
- Keep the Excel datasets inside the `DATA SET` folder.
- If dataset column names are changed, the code in `app.py` may also need to be updated.
- The recommendation quality depends on the available transaction data and the trained model accuracy.
