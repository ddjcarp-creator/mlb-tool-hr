import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from pybaseball import statcast_batter, statcast_pitcher

# -----------------------------
# ⚙️ CONFIG
# -----------------------------

st.set_page_config(page_title="MLB HR EDGE ENGINE", layout="wide")

# -----------------------------
# 🌍 PARK FACTORS
# -----------------------------

PARK_FACTORS = {
    "Coors Field": 1.25,
    "Yankee Stadium": 1.15,
    "Fenway Park": 1.10,
    "Dodger Stadium": 1.05,
    "T-Mobile Park": 0.85,
    "Petco Park": 0.85
}

def get_park_factor(park):
    return PARK_FACTORS.get(park, 1.0)

# -----------------------------
# 🌦 WEATHER BOOST
# -----------------------------

def weather_boost(temp, wind, wind_out):
    score = 0
    if temp >= 27:
        score += 0.10
    if temp >= 32:
        score += 0.15
    if wind >= 10 and wind_out:
        score += 0.15
    return min(score, 0.30)

# -----------------------------
# 📊 HR MODEL
# -----------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-8 * (x - 0.5)))

def build_score(h, p, weather, park):
    return (
        0.30 * h["barrel_rate"] +
        0.20 * (h["avg_ev"] / 100) +
        0.20 * h["hr_form"] +
        0.15 * p["hr_rate_allowed"] +
        0.10 * weather +
        0.05 * park
    )

def expected_value(prob, odds):
    implied = 1 / odds
    return (prob - implied) * 100

# -----------------------------
# 📥 DATA LOADERS
# -----------------------------

@st.cache_data
def load_hitters():
    end = date.today()
    start = end.replace(day=max(1, end.day - 14))

    df = statcast_batter(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d")
    )

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
def load_pitchers():
    end = date.today()
    start = end.replace(day=max(1, end.day - 14))

    df = statcast_pitcher(
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d")
    )

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

# -----------------------------
# 🚀 UI
# -----------------------------

st.title("⚾ MLB HOME RUN EDGE ENGINE v4 (BETTING MODEL)")

st.write("Automated HR prediction model using Statcast, weather, park factors, and matchup edges.")

# Inputs
col1, col2, col3 = st.columns(3)

with col1:
    temp = st.slider("Temperature (°C)", 0, 40, 25)

with col2:
    wind = st.slider("Wind Speed", 0, 30, 8)

with col3:
    wind_out = st.checkbox("Wind blowing OUT", value=True)

park = st.selectbox("Ballpark", list(PARK_FACTORS.keys()) + ["Other"])

odds_input = st.number_input("Average HR Odds (decimal, e.g. 5.50)", value=5.5)

# -----------------------------
# RUN MODEL
# -----------------------------

if st.button("RUN HR MODEL 🚀"):

    with st.spinner("Loading MLB Statcast data..."):

        hitters = load_hitters()
        pitchers = load_pitchers()

        weather = weather_boost(temp, wind, wind_out)
        park_factor = get_park_factor(park)

        results = []

        for _, h in hitters.iterrows():
            for _, p in pitchers.iterrows():

                score = build_score(h, p, weather, park_factor)
                prob = sigmoid(score)

                ev = expected_value(prob, odds_input)

                results.append({
                    "Hitter": h["player_name"],
                    "Pitcher": p["player_name"],
                    "HR Probability": round(prob * 100, 2),
                    "EV %": round(ev, 2),
                    "Barrel Rate": round(h["barrel_rate"], 3),
                    "EV (Exit Velocity)": round(h["avg_ev"], 1),
                    "Pitcher HR Rate": round(p["hr_rate_allowed"], 3)
                })

        df = pd.DataFrame(results)

        st.subheader("🔥 TOP HR PICKS (BY EXPECTED VALUE)")

        st.dataframe(
            df.sort_values("EV %", ascending=False).head(20),
            use_container_width=True
        )

        st.subheader("📊 HR PROBABILITY DISTRIBUTION")

        st.bar_chart(df["HR Probability"].head(20))

        st.subheader("💰 POSITIVE EV BETS ONLY")

        st.dataframe(
            df[df["EV %"] > 0].sort_values("EV %", ascending=False),
            use_container_width=True
        )
