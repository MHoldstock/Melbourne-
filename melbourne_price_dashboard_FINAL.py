
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

@st.cache_data
def load_data():
    # Load raw Excel content without a header
    raw_df = pd.read_excel("melbourne property analysis.xlsx", sheet_name="Sheet1", header=None)

    # Try to find the header row where 'Price' is located
    header_row_index = None
    for i in range(10, 40):  # Search a reasonable range
        if "Price" in raw_df.iloc[i].values:
            header_row_index = i
            break

    if header_row_index is None:
        st.error("❌ Could not find 'Price' in any row. Please check the Excel structure.")
        return pd.DataFrame()

    # Load again using the detected header row
    df = pd.read_excel("melbourne property analysis.xlsx", sheet_name="Sheet1", header=header_row_index)

    # Drop completely empty columns or duplicated column names
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(axis=1, how="all")

    # Now drop rows where price is missing
    df = df.dropna(subset=["Price"])

    # Convert key columns to numeric
    numeric_cols = ['Rooms', 'Price', 'Distance', 'Bedroom2', 'Bathroom', 'Car',
                    'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that are mostly empty
    df = df.dropna(thresh=len(df.columns) - 3)
    return df


df = load_data()
st.title("Melbourne Property Price Dashboard")
st.write("📊 Preview of cleaned data:")
st.write(df.head())
