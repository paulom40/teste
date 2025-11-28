import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Page config
st.set_page_config(
    page_title="NerdyTips Clone - AI Football Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS – NerdyTips style
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stApp {background-color: #0e1117; color: #fafafa;}
    .prediction-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #00ff9d;
        margin: 10px 0;
    }
    .banker {border-left: 6px solid #ffd700 !important; box-shadow: 0 0 15px rgba(255,215,0,0.4);}
    .high-conf {color: #00ff9d; font-weight: bold;}
    .medium-conf {color: #ffcc00;}
    .low-conf {color: #ff6b6b;}
    h1, h2, h3 {color: #00ff9d;}
    .stSelectbox > div > div {background-color: #1e1e2e; color: white;}
</style>
""", unsafe_allow_html=True)

# Sample data
LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Champions League", "Europa League"]
TEAMS = {
    "Premier League": ["Arsenal", "Man City", "Liverpool", "Chelsea", "Man United", "Tottenham", "Newcastle", "Aston Villa"],
    "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Girona", "Athletic Bilbao", "Real Sociedad"],
    "Bundesliga": ["Bayern Munich", "Dortmund", "Leverkusen", "RB Leipzig", "Stuttgart"],
    "Serie A": ["Inter", "AC Milan", "Juventus", "Napoli", "Atalanta", "Roma"],
    "Ligue 1": ["PSG", "Monaco", "Lille", "Nice", "Marseille"],
}

def generate_predictions(n=40):
    matches = []
    for _ in range(n):
        league = random.choice(LEAGUES)
        home = random.choice(TEAMS.get(league, ["Team A"]))
        away = random.choice([t for t in TEAMS.get(league, ["Team B"]) if t != home])
        
        home_prob = round(random.uniform(20, 75), 1)
        draw_prob = round(random.uniform(15, 35), 1)
        away_prob = round(100 - home_prob - draw_prob, 1)
        
        best_tip = max(["1", "X", "2"], key=lambda x: {"1": home_prob, "X": draw_prob, "2": away_prob}[x])
        confidence = random.choices([">90%", "85-90%", "80-85%", "75-80%"], weights=[10, 25, 40, 25])[0]
        
        over25 = round(random.uniform(35, 85), 1)
        btts = round(random.uniform(30, 78), 1)
        
        # Realistic score based on favorite
        if best_tip == "1":
            score = f"{random.randint(1,4)}-{random.randint(0,2)}"
        elif best_tip == "2":
            score = f"{random.randint(0,2)}-{random.randint(1,4)}"
        else:
            score = f"{random.randint(0,3)}-{random.randint(0,3)}"
        
        is_banker = confidence in [">90%", "85-90%"] and max(home_prob, away_prob) > 65

        matches.append({
            "Date": (datetime.now() + timedelta(days=random.randint(0,3))).strftime("%d %b %Y"),
            "Time": f"{random.randint(12,23):02d}:{random.choice(['00','30'])}",
            "League": league,
            "Home": home,
            "Away": away,
            "1": f"{home_prob}%",
            "X": f"{draw_prob}%",
            "2": f"{away_prob}%",
            "Best Tip": best_tip,
            "Confidence": confidence,
            "Over 2.5": f"{over25}%",
            "BTTS": f"{btts}%",
            "Predicted Score": score,
            "Banker": "Yes" if is_banker else "No"
        })
    return pd.DataFrame(matches)

# Sidebar filters
with st.sidebar:
    st.image("https://nerdytips.com/wp-content/uploads/2024/01/nerdytips-logo.png", width=200)
    st.markdown("## AI Football Predictor")
    st.markdown("**Powered by NT 4.0 Engine**")
    st.markdown("---")
    
    st.markdown("### Filters")
    league_filter = st.multiselect("League", options=LEAGUES, default=LEAGUES[:3])
    confidence_filter = st.multiselect("Min Confidence", options=[">90%", "85-90%", "80-85%", "75-80%"], default=["85-90%", ">90%"])
    banker_only = st.checkbox("Banker Tips Only", value=False)

# Title
st.markdown("<h1 style='text-align:center;'>NerdyTips AI - Football Predictions</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888;'>Over 173,000 matches analyzed • Updated every 5 minutes</p>", unsafe_allow_html=True)

# Load data
df = generate_predictions(40)

# Apply filters
filtered = df.copy()
if league_filter:
    filtered = filtered[filtered["League"].isin(league_filter)]
if confidence_filter:
    filtered = filtered[filtered["Confidence"].isin(confidence_filter)]
if banker_only:
    filtered = filtered[filtered["Banker"] == "Yes"]

# Display matches
for _, row in filtered.iterrows():
    is_banker = row["Banker"] == "Yes"
    card_class = "prediction-card banker" if is_banker else "prediction-card"
    conf_class = "high-conf" if row["Confidence"] in [">90%", "85-90%"] else "medium-conf"
    banker_badge = "BANKER TIP" if is_banker else ""

    col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
    
    with col1:
        st.markdown(f"""
        <div class='{card_class}'>
            <small>{row['Date']} • {row['Time']} • {row['League']}</small>
            <h3>{row['Home']} vs {row['Away']}</h3>
            <p><b>Predicted Score:</b> {row['Predicted Score']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='{card_class}'>
            <h4>1X2</h4>
            <p><b>1:</b> {row['1']} <b>X:</b> {row['X']} <b>2:</b> {row['2']}</p>
            <p class='high-conf'>Best Tip: <strong>{row['Best Tip']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='{card_class}'>
            <h4>Goals</h4>
            <p>Over 2.5: {row['Over 2.5']}</p>
            <p>BTTS: {row['BTTS']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='{card_class}'>
            <h4>AI Confidence</h4>
            <p class='{conf_class}'><strong>{row['Confidence']}</strong></p>
            <p style='color:#ffd700; font-size:1.4em;'>{banker_badge}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer – FIXED string
st.markdown("""
---
<p style='text-align:center; color:#666;'>
    This is a demo clone of NerdyTips • For real predictions: 
    <a href='https://nerdytips.com' style='color:#00ff9d;'>nerdytips.com</a><br>
    Made with ❤️ using Streamlit • Data simulated for demo purposes
</p>
""", unsafe_allow_html=True)
