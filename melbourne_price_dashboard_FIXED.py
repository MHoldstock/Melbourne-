
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
    df = pd.read_excel("melbourne property analysis.xlsx", sheet_name="Sheet1", skiprows=25)

    # Show columns for debugging
    st.write("Columns before header fix:", df.columns.tolist())

    # If 'Price' isn't in columns, set the first row as header
    if 'Price' not in df.columns:
        df.columns = df.iloc[0]
        df = df[1:]

    # Show again after fixing
    st.write("Columns after header fix:", df.columns.tolist())

    # Now drop rows where price is missing
    df = df.dropna(subset=["Price"])

    numeric_cols = ['Rooms', 'Price', 'Distance', 'Bedroom2', 'Bathroom', 'Car',
                    'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(thresh=len(df.columns) - 3)
    return df


df = load_data()

st.title("Melbourne Property Price Dashboard")

# Simple check
st.write("Preview:", df.head())
