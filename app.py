import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Road Accident Analysis",
    layout="wide"
)

df = pd.read_csv("indian_roads_dataset.csv")
loaded_model = joblib.load("accident_severity_model.pkl")

st.title("Road Accident Hotspot Analysis and Prediction")
st.write("Road Accident Data Analytics Dashboard")

st.header("Dataset Overview")
st.write("Dataset Shape:", df.shape)
st.dataframe(df.head())

st.header("Accident Severity Prediction")

st.write("Enter accident details to predict severity.")

city = st.selectbox("City", sorted(df["city"].dropna().unique()))
state = st.selectbox("State", sorted(df["state"].dropna().unique()))
hour = st.slider("Accident Hour", 0, 23, 12)

road_type = st.selectbox("Road Type", sorted(df["road_type"].dropna().unique()))

weather = st.selectbox("Weather", sorted(df["weather"].dropna().unique()))

traffic_density = st.selectbox(
    "Traffic Density",
    sorted(df["traffic_density"].dropna().unique())
)

latitude = st.number_input("Latitude", value=float(df["latitude"].mean()))
longitude = st.number_input("Longitude", value=float(df["longitude"].mean()))

day_of_week = st.selectbox(
    "Day",
    ["Monday", "Tuesday", "Wednesday", "Thursday",
     "Friday", "Saturday", "Sunday"]
)

is_weekend = st.selectbox("Is Weekend?", [0, 1])
lanes = st.selectbox("Number of Lanes", sorted(df["lanes"].dropna().unique()))

traffic_signal = st.selectbox("Traffic Signal", sorted(df["traffic_signal"].dropna().unique()))

visibility = st.selectbox("Visibility", sorted(df["visibility"].dropna().unique()))

temperature = st.number_input(
    "Temperature",
    value=float(df["temperature"].mean())
)

cause = st.selectbox("Accident Cause", sorted(df["cause"].dropna().unique()))

vehicles_involved = st.number_input("Vehicles Involved", min_value=1, value=2)

casualties = st.number_input("Casualties", min_value=0, value=1)

is_peak_hour = st.selectbox("Is Peak Hour?", [0, 1])

festival = st.selectbox(
    "Festival",
    sorted(df["festival"].fillna("No Festival").unique())
)

risk_score = st.number_input(
    "Risk Score",
    value=float(df["risk_score"].mean())
)

if st.button("Predict Accident Severity"):

    input_data = pd.DataFrame([{
        "city": city,
        "state": state,
        "latitude": latitude,
        "longitude": longitude,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "road_type": road_type,
        "lanes": lanes,
        "traffic_signal": traffic_signal,
        "weather": weather,
        "visibility": visibility,
        "temperature": temperature,
        "traffic_density": traffic_density,
        "cause": cause,
        "vehicles_involved": vehicles_involved,
        "casualties": casualties,
        "is_peak_hour": is_peak_hour,
        "festival": festival,
        "risk_score": risk_score
    }])

    prediction = loaded_model.predict(input_data)

    st.success("Predicted Accident Severity: " + prediction[0])

import matplotlib.pyplot as plt

st.header("Accident Severity Analysis")

severity_count = df["accident_severity"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
severity_count.plot(kind="bar", ax=ax)

ax.set_title("Accidents by Severity")
ax.set_xlabel("Accident Severity")
ax.set_ylabel("Number of Accidents")
ax.tick_params(axis="x", rotation=0)

st.pyplot(fig)
plt.close(fig)

st.header("State-wise Accident Analysis")

state_count = df["state"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
state_count.plot(kind="bar", ax=ax)

ax.set_title("Accidents by State")
ax.set_xlabel("State")
ax.set_ylabel("Number of Accidents")
plt.xticks(rotation=45)

st.pyplot(fig)
plt.close(fig)

st.header("City-wise Accident Analysis")

city_count = df["city"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))
city_count.plot(kind="bar", ax=ax)

ax.set_title("Top 10 Cities by Accidents")
ax.set_xlabel("City")
ax.set_ylabel("Number of Accidents")
plt.xticks(rotation=45)

st.pyplot(fig)
plt.close(fig)

st.header("Road Type Analysis")

road_count = df["road_type"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
road_count.plot(kind="bar", ax=ax)

ax.set_title("Accidents by Road Type")
ax.set_xlabel("Road Type")
ax.set_ylabel("Number of Accidents")
plt.xticks(rotation=45)

st.pyplot(fig)
plt.close(fig)

st.header("Weather Analysis")

weather_count = df["weather"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
weather_count.plot(kind="bar", ax=ax)

ax.set_title("Accidents by Weather Condition")
ax.set_xlabel("Weather")
ax.set_ylabel("Number of Accidents")
plt.xticks(rotation=45)

st.pyplot(fig)
plt.close(fig)

st.header("Traffic Density Analysis")

traffic_count = df["traffic_density"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
traffic_count.plot(kind="bar", ax=ax)

ax.set_title("Accidents by Traffic Density")
ax.set_xlabel("Traffic Density")
ax.set_ylabel("Number of Accidents")
plt.xticks(rotation=0)

st.pyplot(fig)
plt.close(fig)

st.header("Accident Hotspot Map")

map_data = df[["latitude", "longitude"]].dropna()

st.map(map_data)

st.write("Accident locations displayed on the map.")
