import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="ATP Monte Carlo Predictor", layout="wide")

st.title("🎾 ATP Match Predictor with Monte Carlo Simulation")

# -------------------------------------------------------
# Upload Excel
# -------------------------------------------------------

uploaded_file = st.file_uploader("Upload ATP Excel file", type=["xlsx"])

if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)

# -------------------------------------------------------
# Prepare data
# -------------------------------------------------------

for s in ['1','2','3','4','5']:

    df[f'W{s}'] = pd.to_numeric(df.get(f'W{s}',0), errors='coerce').fillna(0)
    df[f'L{s}'] = pd.to_numeric(df.get(f'L{s}',0), errors='coerce').fillna(0)

df["Total_Games"] = (
    df["W1"]+df["L1"]+
    df["W2"]+df["L2"]+
    df["W3"]+df["L3"]+
    df["W4"]+df["L4"]+
    df["W5"]+df["L5"]
)

df = df[df["Total_Games"].between(10,80)]

players = sorted(set(df["Winner"]) | set(df["Loser"]))

# -------------------------------------------------------
# Match selection
# -------------------------------------------------------

col1,col2,col3 = st.columns(3)

with col1:
    player_a = st.selectbox("Player A", players)

with col2:
    player_b = st.selectbox("Player B", [p for p in players if p != player_a])

with col3:
    surface = st.selectbox("Surface", sorted(df["Surface"].dropna().unique()))

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def get_player_matches(player):

    m = df[(df["Winner"]==player) | (df["Loser"]==player)]

    return m.sort_values("Date", ascending=False).head(20)


def avg_games(player):

    m = get_player_matches(player)

    if len(m)==0:
        return 23

    return m["Total_Games"].mean()


def simple_elo():

    players = list(set(df["Winner"]) | set(df["Loser"]))

    elo = {p:1500 for p in players}

    for _,row in df.iterrows():

        w = row["Winner"]
        l = row["Loser"]

        exp = 1/(1+10**((elo[l]-elo[w])/400))

        k = 24

        elo[w] += k*(1-exp)
        elo[l] += k*(0-exp)

    return elo


elo = simple_elo()

# -------------------------------------------------------
# Monte Carlo simulation
# -------------------------------------------------------

def monte_carlo(player_a, player_b, sims=10000):

    base = (avg_games(player_a) + avg_games(player_b)) / 2

    elo_diff = abs(elo[player_a] - elo[player_b])

    balance = 1 - min(elo_diff/600,0.35)

    mean_games = base * balance

    std_games = 4.2

    simulations = np.random.normal(mean_games,std_games,sims)

    simulations = np.clip(simulations,12,70)

    return simulations


# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

if st.button("Run Prediction"):

    sims = monte_carlo(player_a, player_b)

    prediction = round(np.mean(sims),1)

    st.subheader("Predicted Total Games")

    st.markdown(f"# {prediction}")

    # ---------------------------------------------------
    # Tie-break probability
    # ---------------------------------------------------

    tb_prob = np.mean(sims > 12) * 0.35

    st.metric("Tie-break Probability", f"{round(tb_prob*100,1)}%")

    # ---------------------------------------------------
    # Over Under table
    # ---------------------------------------------------

    lines = [20.5,21.5,22.5,23.5,24.5]

    rows = []

    for line in lines:

        prob_over = np.mean(sims > line)

        prob_under = 1 - prob_over

        rows.append({
            "Line": line,
            "Over %": round(prob_over*100,2),
            "Under %": round(prob_under*100,2)
        })

    prob_df = pd.DataFrame(rows)

    st.subheader("Over / Under Probabilities")

    st.dataframe(prob_df)

    # ---------------------------------------------------
    # Histogram
    # ---------------------------------------------------

    fig, ax = plt.subplots()

    ax.hist(sims, bins=40)

    ax.set_title("Monte Carlo Distribution of Total Games")

    ax.set_xlabel("Total Games")

    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # ---------------------------------------------------
    # HTML REPORT
    # ---------------------------------------------------

    html = f"""
    <html>
    <head>
    <title>ATP Match Prediction</title>
    <style>
    body {{
        font-family: Arial;
        background:#f4f6ff;
        padding:40px;
    }}

    .container {{
        max-width:900px;
        margin:auto;
        background:white;
        padding:40px;
        border-radius:12px;
        box-shadow:0 10px 25px rgba(0,0,0,0.15);
    }}

    h1 {{
        text-align:center;
        color:#1a237e;
    }}

    .prediction {{
        font-size:80px;
        text-align:center;
        font-weight:bold;
        margin:30px;
        color:#3f51b5;
    }}

    table {{
        width:100%;
        border-collapse:collapse;
        margin-top:30px;
    }}

    th,td {{
        padding:12px;
        border-bottom:1px solid #ddd;
        text-align:center;
    }}

    </style>
    </head>

    <body>

    <div class="container">

    <h1>ATP Match Prediction</h1>

    <h2 style="text-align:center">
    {player_a} vs {player_b}
    </h2>

    <h3 style="text-align:center">
    Surface: {surface}
    </h3>

    <div class="prediction">{prediction}</div>

    <h3>Over / Under Probabilities</h3>

    {prob_df.to_html(index=False)}

    <p style="margin-top:40px">
    Generated: {datetime.now()}
    </p>

    </div>

    </body>
    </html>
    """

    st.download_button(
        "📥 Download HTML Report",
        html,
        file_name=f"{player_a}_vs_{player_b}_prediction.html",
        mime="text/html"
    )
