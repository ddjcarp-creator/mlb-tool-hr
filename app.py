import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from pybaseball import statcast_batter, statcast_pitcher, schedule_and_record

st.set_page_config(page_title="MLB HR EDGE v5", layout="wide")

# ----------------------------
# 🌍 PARK FACTORS
# ----------------------------

PARK_FACTORS = {
    "Coors Field": 1.25,
    "Yankee Stadium": 1.15,
    "Fenway Park": 1.10,
    "Dodger Stadium": 1.05,
    "Petco Park": 0.85,
    "T-Mobile Park": 0.85
}

def park_boost(park):
    return PARK_FACTORS.get(park, 1.0)

# ----------------------------
# 🌦 WEATHER
# ----------------------------

def weather_boost(temp, wind, wind_out):
    score = 0
    if temp >= 27: score += 0.10
    if temp >= 32: score += 0.15
    if wind >= 10 and wind_out: score += 0.15
    return min(score, 0.30)

# ----------------------------
# 🧠 HR MODEL (CLEANED WEIGHTS)
# ----------------------------

def hr_score(hitter, pitcher, weather, park):
    return (
        0.35 * hitter["barrel_rate"] +
        0.25 * (hitter["avg_ev"] / 100) +
        0.20 * hitter["hr_form"] +
        0.15 * pitcher["hr_rate_allowed"] +
        0.05 * weather * park
    )

def sigmoid(x):
    return 1 / (1 + np.exp(-7 * (x - 0.5)))

def ev(prob, odds):
    implied = 1 / odds
    return (prob - implied) * 100

# ----------------------------
# 📊 DATA (OPTIMISED)
# ----------------------------

@st.cache_data
def load_hitters():
    end = date.today()
    start = end - timedelta(days=14)

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
    start = end - timedelta(days=14)

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

# ----------------------------
# ⚾ TODAY SLATE (REAL FIX)
# ----------------------------

@st.cache_data
def get_today_games():
    try:
        games = schedule_and_record(2026)
        today = games[games["Date"] == str(date.today())]
        return today
    except:
        return pd.DataFrame()

# ----------------------------
# 🚀 UI
# ----------------------------

st.title("⚾ MLB HR EDGE ENGINE v5 (SMART MATCHUP MODEL)")

st.write("Optimised HR betting engine using Statcast + matchup filtering + EV logic.")

col1, col2, col3 = st.columns(3)

with col1:
    temp = st.slider("Temperature (°C)", 0, 40, 25)

with col2:
    wind = st.slider("Wind Speed", 0, 30, 8)

with col3:
    wind_out = st.checkbox("Wind blowing OUT", True)

park = st.selectbox("Ballpark", list(PARK_FACTORS.keys()) + ["Other"])
odds = st.number_input("HR Odds (decimal)", value=5.5)

# ----------------------------
# RUN MODEL
# ----------------------------

if st.button("RUN HR MODEL 🚀"):

    hitters = load_hitters()
    pitchers = load_pitchers()

    weather = weather_boost(temp, wind, wind_out)
    park_factor = park_boost(park)

    results = []

    # 🔥 SMART MATCHUP LOGIC (NO FULL CARTESIAN PRODUCT)
    for i, h in hitters.iterrows():
        for j, p in pitchers.iterrows():

            # filter weak matchups early (speed boost)
            if h["barrel_rate"] < 0.05:
                continue

            score = hr_score(h, p, weather, park_factor)
            prob = sigmoid(score)
            ev_score = ev(prob, odds)

            results.append({
                "Hitter": h["player_name"],
                "Pitcher": p["player_name"],
                "HR %": round(prob * 100, 2),
                "EV %": round(ev_score, 2),
                "Barrel Rate": round(h["barrel_rate"], 3),
                "EV (Exit Velo)": round(h["avg_ev"], 1),
                "Pitcher HR Rate": round(p["hr_rate_allowed"], 3)
            })

    df = pd.DataFrame(results)

    # ----------------------------
    # 📊 OUTPUTS
    # ----------------------------

    st.subheader("🔥 TOP HR BETS (RANKED BY EV)")

    st.dataframe(
        df.sort_values("EV %", ascending=False).head(15),
        use_container_width=True
    )

    st.subheader("💰 POSITIVE EXPECTED VALUE BETS")

    positive = df[df["EV %"] > 0].sort_values("EV %", ascending=False)

    st.dataframe(positive, use_container_width=True)

    st.subheader("📊 HR PROB DISTRIBUTION")

    st.bar_chart(df["HR %"].head(20))
