
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
    raw_df = pd.read_excel("melbourne property analysis.xlsx", sheet_name="Sheet1", header=None)
    header_row_index = None
    for i in range(10, 40):
        if "Price" in raw_df.iloc[i].values:
            header_row_index = i
            break
    if header_row_index is None:
        st.error("❌ Could not find 'Price' in any row. Please check the Excel structure.")
        return pd.DataFrame()
    df = pd.read_excel("melbourne property analysis.xlsx", sheet_name="Sheet1", header=header_row_index)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(axis=1, how="all")
    df = df.dropna(subset=["Price"])
    numeric_cols = ['Rooms', 'Price', 'Distance', 'Bedroom2', 'Bathroom', 'Car',
                    'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(thresh=len(df.columns) - 3)
    return df

df = load_data()

# Filters
st.sidebar.header("Filter Data")
suburbs = st.sidebar.multiselect("Suburb", sorted(df["Suburb"].dropna().unique()), [])
types = st.sidebar.multiselect("Property Type", sorted(df["Type"].dropna().unique()), [])

filtered_df = df.copy()
if suburbs:
    filtered_df = filtered_df[filtered_df["Suburb"].isin(suburbs)]
if types:
    filtered_df = filtered_df[filtered_df["Type"].isin(types)]

# Dashboard layout
st.title("🏡 Melbourne Property Price Dashboard")

st.subheader("📊 Average Price by Region")
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=filtered_df, x="Regionname", y="Price", estimator=np.mean, ci=None, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("💰 Price vs Rooms")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=filtered_df, x="Rooms", y="Price", ax=ax2)
st.pyplot(fig2)

st.subheader("📍 Price vs Distance from CBD")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=filtered_df, x="Distance", y="Price", ax=ax3)
st.pyplot(fig3)

# Model training
def train_model(data):
    features = ['Suburb', 'Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 'Car',
                'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Regionname']
    target = 'Price'
    data = data.dropna(subset=features + [target])
    X = data[features]
    y = data[target]

    num_features = X.select_dtypes(include=["number"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])

    model.fit(X, y)
    return model, num_features + cat_features

st.subheader("🔮 Predict Property Price")
model, input_features = train_model(df)

with st.form("prediction_form"):
    st.write("Enter property details:")
    user_input = {}
    for feature in input_features:
        if df[feature].dtype == "object":
            user_input[feature] = st.selectbox(feature, sorted(df[feature].dropna().unique()))
        else:
            user_input[feature] = st.number_input(feature, value=float(df[feature].median()))
    if st.form_submit_button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"💡 Estimated Property Price: AUD ${prediction:,.0f}")
