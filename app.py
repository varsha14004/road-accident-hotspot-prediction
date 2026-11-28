
import streamlit as st
import pandas as pd
import joblib
from google import genai

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Road Accident Dashboard",
    page_icon="🚦",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("indian_roads_dataset.csv")

df = load_data()
model = joblib.load("accident_severity_model.pkl")

st.title("🚦 Road Accident Hotspot Analysis & Prediction")
st.caption("Interactive Dashboard with AI-powered RAG Assistant")

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔎 Dashboard Filters")

def add_filter(label, column):
    values = ["All"] + sorted(df[column].dropna().astype(str).unique().tolist())
    return st.sidebar.selectbox(label, values)

selected_city = add_filter("City", "city")
selected_state = add_filter("State", "state")
selected_severity = add_filter("Accident Severity", "accident_severity")
selected_weather = add_filter("Weather", "weather")
selected_road = add_filter("Road Type", "road_type")
selected_traffic = add_filter("Traffic Density", "traffic_density")

filtered = df.copy()

filters = {
    "city": selected_city,
    "state": selected_state,
    "accident_severity": selected_severity,
    "weather": selected_weather,
    "road_type": selected_road,
    "traffic_density": selected_traffic
}

for column, value in filters.items():
    if value != "All":
        filtered = filtered[
            filtered[column].astype(str) == value
        ]

# ---------------- SUMMARY ----------------
st.subheader("📊 Dashboard Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Accidents", f"{len(filtered):,}")
c2.metric("Total Cities", filtered["city"].nunique())
c3.metric("Total States", filtered["state"].nunique())
c4.metric("Total Casualties", f"{filtered['casualties'].sum():,}")

if filtered.empty:
    st.warning("No accidents found for the selected filters.")
else:
    # ---------------- HIGHEST STATE ----------------
    st.subheader("🏆 Highest Accident State")

    state_counts = filtered["state"].value_counts()

    if not state_counts.empty:
        highest_state = state_counts.idxmax()
        highest_count = int(state_counts.max())

        st.metric("Highest State", highest_state,
                  f"{highest_count:,} accidents")

        chosen_state = st.selectbox(
            "View a state's accident details",
            ["Select a state"] + sorted(filtered["state"].dropna().unique().tolist())
        )

        if chosen_state != "Select a state":
            state_data = filtered[filtered["state"] == chosen_state]
            st.write(f"Accident details for **{chosen_state}**")
            st.dataframe(state_data, use_container_width=True)

    # ---------------- CHARTS ----------------
    st.subheader("📈 Accident Analysis")

    left, right = st.columns(2)

    with left:
        st.write("**Accidents by State**")
        st.bar_chart(filtered["state"].value_counts())

        st.write("**Accidents by Severity**")
        st.bar_chart(filtered["accident_severity"].value_counts())

    with right:
        st.write("**Accidents by City**")
        st.bar_chart(filtered["city"].value_counts())

        st.write("**Accidents by Weather**")
        st.bar_chart(filtered["weather"].value_counts())

    left2, right2 = st.columns(2)

    with left2:
        st.write("**Accidents by Road Type**")
        st.bar_chart(filtered["road_type"].value_counts())

    with right2:
        st.write("**Accidents by Traffic Density**")
        st.bar_chart(filtered["traffic_density"].value_counts())

    # ---------------- MAP ----------------
    st.subheader("📍 Accident Hotspot Map")
    map_data = filtered[["latitude", "longitude"]].dropna()
    if not map_data.empty:
        st.map(map_data)
    else:
        st.info("No map locations available for these filters.")

# ---------------- PREDICTION ----------------
st.divider()
st.subheader("🚗 Accident Severity Prediction")

feature_columns = [
    "city", "state", "latitude", "longitude", "hour",
    "day_of_week", "is_weekend", "road_type", "lanes",
    "traffic_signal", "weather", "visibility", "temperature",
    "traffic_density", "cause", "vehicles_involved",
    "casualties", "is_peak_hour", "festival", "risk_score"
]

with st.form("prediction_form"):
    p1, p2, p3 = st.columns(3)

    with p1:
        city = st.selectbox("City", sorted(df["city"].dropna().unique()))
        state = st.selectbox("State", sorted(df["state"].dropna().unique()))
        road_type = st.selectbox("Road Type", sorted(df["road_type"].dropna().unique()))
        weather = st.selectbox("Weather", sorted(df["weather"].dropna().unique()))
        traffic_density = st.selectbox(
            "Traffic Density", sorted(df["traffic_density"].dropna().unique())
        )
        cause = st.selectbox("Accident Cause", sorted(df["cause"].dropna().unique()))
        festival = st.selectbox("Festival", sorted(df["festival"].dropna().unique()))

    with p2:
        latitude = st.number_input("Latitude", value=float(df["latitude"].median()))
        longitude = st.number_input("Longitude", value=float(df["longitude"].median()))
        hour = st.slider("Hour", 0, 23, 12)
        day_of_week = st.selectbox(
            "Day of Week", sorted(df["day_of_week"].dropna().unique())
        )
        is_weekend = st.selectbox("Weekend?", [0, 1])
        lanes = st.number_input("Number of Lanes", min_value=1, value=2)
        traffic_signal = st.selectbox(
            "Traffic Signal", sorted(df["traffic_signal"].dropna().unique())
        )

    with p3:
        visibility = st.number_input(
            "Visibility", value=float(df["visibility"].median())
        )
        temperature = st.number_input(
            "Temperature", value=float(df["temperature"].median())
        )
        vehicles_involved = st.number_input(
            "Vehicles Involved", min_value=1, value=2
        )
        casualties = st.number_input("Casualties", min_value=0, value=1)
        is_peak_hour = st.selectbox("Peak Hour?", [0, 1])
        risk_score = st.number_input(
            "Risk Score", value=float(df["risk_score"].median())
        )

    predict_button = st.form_submit_button("Predict Accident Severity")

if predict_button:
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
    }], columns=feature_columns)

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Accident Severity: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------- RAG AI CHATBOT ----------------
st.divider()
st.subheader("🤖 RAG - AI Road Accident Assistant")
st.write("Ask questions about the accident dataset.")

question = st.chat_input("Ask about road accidents...")

if question:
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = genai.Client(api_key=api_key)

            # Retrieve relevant dataset information
            search_text = question.lower()
            relevant = df[
                df.astype(str).apply(
                    lambda row: row.str.lower().str.contains(
                        search_text, regex=False
                    ).any(),
                    axis=1
                )
            ]

            if relevant.empty:
                relevant = df.sample(min(100, len(df)), random_state=42)

            context = (
                f"Dataset contains {len(df)} accident records.\n"
                f"Relevant records:\n{relevant.head(30).to_string(index=False)}\n"
                f"Overall severity counts:\n"
                f"{df['accident_severity'].value_counts().to_dict()}\n"
                f"Overall state counts:\n"
                f"{df['state'].value_counts().to_dict()}"
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=(
                    "You are a road accident data assistant. "
                    "Answer using the supplied dataset context. "
                    "Do not invent statistics. If the context does not "
                    "contain the answer, clearly say the data is insufficient.\n\n"
                    f"DATA CONTEXT:\n{context}\n\n"
                    f"USER QUESTION: {question}"
                )
            )

            st.write(response.text)

        except Exception as e:
            st.error(f"Chatbot error: {e}")
