
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from google import genai

# Page settings
st.set_page_config(
    page_title="Road Accident Analysis",
    page_icon="🚗",
    layout="wide"
)

# Load files
df = pd.read_csv("indian_roads_dataset.csv")
loaded_model = joblib.load("accident_severity_model.pkl")

# Title
st.title("🚗 Road Accident Hotspot Analysis and Prediction")
st.write("Road Accident Data Analytics Dashboard")

# Dashboard Summary
st.header("Dashboard Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Accidents", len(df))

with col2:
    st.metric("Total Cities", df["city"].nunique())

with col3:
    st.metric("Total States", df["state"].nunique())

with col4:
    st.metric("Total Casualties", int(df["casualties"].sum()))

# Dataset Overview
st.header("Dataset Overview")
st.write("Dataset Shape:", df.shape)
st.dataframe(df.head())

# ML Prediction
st.header("Accident Severity Prediction")
st.write("Enter accident details to predict severity.")

city = st.selectbox("City", sorted(df["city"].dropna().unique()))
state = st.selectbox("State", sorted(df["state"].dropna().unique()))
hour = st.slider("Accident Hour", 0, 23, 12)

road_type = st.selectbox(
    "Road Type", sorted(df["road_type"].dropna().unique())
)

weather = st.selectbox(
    "Weather", sorted(df["weather"].dropna().unique())
)

traffic_density = st.selectbox(
    "Traffic Density",
    sorted(df["traffic_density"].dropna().unique())
)

latitude = st.number_input(
    "Latitude", value=float(df["latitude"].mean())
)

longitude = st.number_input(
    "Longitude", value=float(df["longitude"].mean())
)

day_of_week = st.selectbox(
    "Day",
    ["Monday", "Tuesday", "Wednesday", "Thursday",
     "Friday", "Saturday", "Sunday"]
)

is_weekend = st.selectbox("Is Weekend?", [0, 1])

lanes = st.selectbox(
    "Number of Lanes", sorted(df["lanes"].dropna().unique())
)

traffic_signal = st.selectbox(
    "Traffic Signal",
    sorted(df["traffic_signal"].dropna().unique())
)

visibility = st.selectbox(
    "Visibility", sorted(df["visibility"].dropna().unique())
)

temperature = st.number_input(
    "Temperature", value=float(df["temperature"].mean())
)

cause = st.selectbox(
    "Accident Cause", sorted(df["cause"].dropna().unique())
)

vehicles_involved = st.number_input(
    "Vehicles Involved", min_value=1, value=2
)

casualties = st.number_input(
    "Casualties", min_value=0, value=1
)

is_peak_hour = st.selectbox("Is Peak Hour?", [0, 1])

festival = st.selectbox(
    "Festival",
    sorted(df["festival"].fillna("No Festival").unique())
)

risk_score = st.number_input(
    "Risk Score", value=float(df["risk_score"].mean())
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

    st.success(
        "Predicted Accident Severity: " + str(prediction[0])
    )

# Severity Analysis
st.header("Accident Severity Analysis")

severity_count = df["accident_severity"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
severity_count.plot(kind="bar", ax=ax)
ax.set_title("Accidents by Severity")
ax.set_xlabel("Accident Severity")
ax.set_ylabel("Number of Accidents")
plt.xticks(rotation=0)
st.pyplot(fig)
plt.close(fig)

# State Analysis
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

# City Analysis
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

# Road Type Analysis
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

# Weather Analysis
st.header("Weather-wise Accident Analysis")

weather_count = df["weather"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
weather_count.plot(kind="bar", ax=ax)
ax.set_title("Accidents by Weather Condition")
ax.set_xlabel("Weather")
ax.set_ylabel("Number of Accidents")
plt.xticks(rotation=45)
st.pyplot(fig)
plt.close(fig)

# Traffic Density Analysis
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

# Hotspot Map
st.header("Accident Hotspot Map")

map_data = df[["latitude", "longitude"]].dropna()
st.map(map_data)

st.write(
    "Map shows the geographical locations of recorded accidents."
)

# =====================================
# RAG AI CHATBOT
# =====================================

st.divider()

st.header("🤖 AI Road Accident Assistant")

st.write(
    "Ask questions about the accident dataset. "
    "The assistant retrieves relevant dataset information "
    "and uses Gemini AI to answer."
)

# Retrieve relevant information from dataset
def retrieve_data(question):

    q = question.lower()
    context = []

    context.append(
        f"Total accidents: {len(df)}"
    )

    context.append(
        f"Total cities: {df['city'].nunique()}"
    )

    context.append(
        f"Total states: {df['state'].nunique()}"
    )

    context.append(
        f"Total casualties: {int(df['casualties'].sum())}"
    )

    context.append(
        "\nAccidents by severity:\n" +
        df["accident_severity"].value_counts().to_string()
    )

    # Retrieve relevant categories
    columns = [
        "city",
        "state",
        "weather",
        "road_type",
        "traffic_density",
        "cause"
    ]

    matched_values = []

    for column in columns:
        for value in df[column].dropna().unique():
            value_text = str(value)

            if len(value_text) > 2 and value_text.lower() in q:
                matched_values.append((column, value))

    # Retrieve matching accident records
    if matched_values:
        mask = pd.Series(False, index=df.index)

        for column, value in matched_values:
            mask = mask | (df[column] == value)

        relevant_data = df[mask].head(10)

        context.append(
            "\nRelevant accident records:\n" +
            relevant_data[
                [
                    "city",
                    "state",
                    "weather",
                    "road_type",
                    "traffic_density",
                    "cause",
                    "accident_severity",
                    "casualties"
                ]
            ].to_string(index=False)
        )

    else:
        context.append(
            "\nTop accident cities:\n" +
            df["city"].value_counts().head(10).to_string()
        )

        context.append(
            "\nTop accident causes:\n" +
            df["cause"].value_counts().head(10).to_string()
        )

        context.append(
            "\nWeather-wise accidents:\n" +
            df["weather"].value_counts().to_string()
        )

        context.append(
            "\nTraffic density:\n" +
            df["traffic_density"].value_counts().to_string()
        )

    return "\n".join(context)


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User question
question = st.chat_input(
    "Ask about road accidents..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:
            api_key = st.secrets["GEMINI_API_KEY"]

            client = genai.Client(api_key=api_key)

            retrieved_context = retrieve_data(question)

            prompt = f"""
            You are an AI assistant for a Road Accident
            Analysis and Prediction project.

            Answer the user's question using the
            retrieved dataset information below.

            Do not invent statistics or facts.
            If the dataset does not contain the answer,
            clearly say that the information is unavailable.

            Retrieved dataset information:
            {retrieved_context}

            User question:
            {question}
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            answer = response.text

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(
                "Chatbot error: " + str(e)
            )
