import streamlit as st
import pandas as pd
import numpy as np
from pybaseball import statcast_batter, statcast_pitcher
from datetime import date, timedelta

st.set_page_config(page_title="MLB HR EDGE ENGINE", layout="wide")

# ----------------------------
# 🔧 FEATURES
# ----------------------------

def weather_boost(temp, wind, wind_out):
    score = 0
    if temp >= 27:
        score += 0.10
    if temp >= 32:
        score += 0.15
    if wind >= 10 and wind_out:
        score += 0.15
    return min(score, 0.30)


def build_score(h, p, weather):
    return (
        0.35 * h["barrel_rate"] +
        0.25 * (h["avg_ev"] / 100) +
        0.20 * h["hr_form"] +
        0.15 * p["hr_rate_allowed"] +
        0.05 * weather
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-8 * (x - 0.5)))


# ----------------------------
# 📊 DATA LOADER
# ----------------------------

@st.cache_data
def load_data(start, end):
    df = statcast_batter(start, end)

    df["is_hr"] = (df["events"] == "home_run").astype(int)

    hitters = df.groupby("player_name").agg({
        "launch_speed": "mean",
        "barrel": "mean",
        "is_hr": "sum"
    }).reset_index()

    hitters.rename(columns={
        "launch_speed": "avg_ev",
        "barrel": "barrel_rate",
        "is_hr": "hr_form"
    }, inplace=True)

    return hitters


@st.cache_data
def load_pitchers(start, end):
    df = statcast_pitcher(start, end)

    pitchers = df.groupby("player_name").agg({
        "launch_speed": "mean",
        "barrel": "mean",
        "events": lambda x: (x == "home_run").mean()
    }).reset_index()

    pitchers.rename(columns={
        "launch_speed": "ev_allowed",
        "barrel": "barrel_rate_allowed",
        "events": "hr_rate_allowed"
    }, inplace=True)

    return pitchers


# ----------------------------
# 🚀 UI
# ----------------------------

st.title("⚾ MLB HOME RUN EDGE ENGINE (Sweet Spot++ Betting Model)")

st.write("Built for HR props, barrels, matchup edges, and daily MLB slate targeting.")

# Date range
start = st.date_input("Start Date", date.today() - timedelta(days=14))
end = st.date_input("End Date", date.today())

temp = st.slider("Temperature (°C)", 0, 40, 25)
wind = st.slider("Wind Speed", 0, 30, 8)
wind_out = st.checkbox("Wind blowing OUT", value=True)

if st.button("RUN HR MODEL 🚀"):

    with st.spinner("Loading Statcast data..."):

        hitters = load_data(start, end)
        pitchers = load_pitchers(start, end)

        weather = weather_boost(temp, wind, wind_out)

        results = []

        for _, h in hitters.iterrows():
            for _, p in pitchers.iterrows():

                score = build_score(h, p, weather)
                prob = sigmoid(score)

                results.append({
                    "Hitter": h["player_name"],
                    "Pitcher": p["player_name"],
                    "HR Probability": round(prob * 100, 2),
                    "Barrel Rate": round(h["barrel_rate"], 3),
                    "EV": round(h["avg_ev"], 1),
                    "Pitcher HR Allowed": round(p["hr_rate_allowed"], 3)
                })

        df = pd.DataFrame(results)

        st.subheader("🔥 TOP HOME RUN PICKS")

        st.dataframe(
            df.sort_values("HR Probability", ascending=False).head(25),
            use_container_width=True
        )

        st.subheader("📊 HR DISTRIBUTION")

        st.bar_chart(df["HR Probability"].head(20))
