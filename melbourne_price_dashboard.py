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

# Load and preprocess data
@st.cache_data
def load_data():
    # Try loading with fewer rows skipped
    df = pd.read_excel("melbourne property analysis.xlsx", sheet_name="Sheet1", skiprows=25)

    # Show columns to debug
    st.write("🔍 Columns before fix:", df.columns.tolist())

    # If "Price" is not found, try setting the first row as header
    if 'Price' not in df.columns:
        df.columns = df.iloc[0]
        df = df[1:]

    st.write("✅ Columns after fix:", df.columns.tolist())

    # Clean up the data
    df = df.dropna(subset=["Price"])
    numeric_cols = ['Rooms', 'Price', 'Distance', 'Bedroom2', 'Bathroom', 'Car',
                    'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(thresh=len(df.columns) - 3)
    return df

df = load_data()

# Sidebar - Input features
st.sidebar.header("Filter Data")
suburbs = st.sidebar.multiselect("Select Suburbs", options=sorted(df["Suburb"].dropna().unique()), default=[])
prop_type = st.sidebar.multiselect("Property Type", options=sorted(df["Type"].dropna().unique()), default=[])

filtered_df = df.copy()
if suburbs:
    filtered_df = filtered_df[filtered_df["Suburb"].isin(suburbs)]
if prop_type:
    filtered_df = filtered_df[filtered_df["Type"].isin(prop_type)]

# Visualizations
st.title("Melbourne Property Price Dashboard")

st.subheader("Average Price by Region")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=filtered_df, x="Regionname", y="Price", estimator=np.mean, ci=None, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Price vs Rooms")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=filtered_df, x="Rooms", y="Price", ax=ax2)
st.pyplot(fig2)

st.subheader("Price vs Distance from CBD")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=filtered_df, x="Distance", y="Price", ax=ax3)
st.pyplot(fig3)

# Prediction model
def train_model(df):
    features = [
        'Suburb', 'Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 'Car',
        'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Regionname'
    ]
    target = 'Price'
    df_model = df[features + [target]].dropna()
    X = df_model.drop(columns=target)
    y = df_model[target]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])

    model.fit(X, y)
    return model, categorical_features + numeric_features

st.subheader("Predict Property Price")
model, input_features = train_model(df)

with st.form("prediction_form"):
    st.write("Input property details:")
    input_data = {}
    for feature in input_features:
        if df[feature].dtype == "object":
            input_data[feature] = st.selectbox(feature, sorted(df[feature].dropna().unique()))
        else:
            input_data[feature] = st.number_input(feature, value=float(df[feature].median()))

    submitted = st.form_submit_button("Predict Price")
    if submitted:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Property Price: AUD ${prediction:,.0f}")
