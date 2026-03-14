import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="WTA Monte Carlo Predictor", layout="wide")

st.title("🎾 WTA Advanced Monte Carlo Predictor")

# ---------------------------------------------------------
# Upload dataset
# ---------------------------------------------------------

uploaded_file = st.file_uploader("Upload WTA Excel dataset", type=["xlsx"])

if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)

# ---------------------------------------------------------
# Prepare dataset
# ---------------------------------------------------------

for s in ['1','2','3']:

    df[f'W{s}'] = pd.to_numeric(df.get(f'W{s}',0), errors='coerce').fillna(0)
    df[f'L{s}'] = pd.to_numeric(df.get(f'L{s}',0), errors='coerce').fillna(0)

df["Total_Games"] = (
    df["W1"]+df["L1"]+
    df["W2"]+df["L2"]+
    df["W3"]+df["L3"]
)

df = df[df["Total_Games"].between(10,80)]

players = sorted(set(df["Winner"]) | set(df["Loser"]))

# ---------------------------------------------------------
# Match selection
# ---------------------------------------------------------

c1,c2,c3 = st.columns(3)

with c1:
    player_a = st.selectbox("Player A", players)

with c2:
    player_b = st.selectbox("Player B", [p for p in players if p != player_a])

with c3:
    surface = st.selectbox("Surface", sorted(df["Surface"].dropna().unique()))

# ---------------------------------------------------------
# Player stats
# ---------------------------------------------------------

def get_matches(player):

    m = df[(df["Winner"]==player) | (df["Loser"]==player)]

    return m.sort_values("Date", ascending=False).head(25)


def service_hold_estimate(player):

    m = get_matches(player)

    if len(m)==0:
        return 0.65

    wins = (m["Winner"]==player).mean()

    return 0.60 + wins*0.20


# ---------------------------------------------------------
# Elo rating
# ---------------------------------------------------------

def compute_elo():

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

elo = compute_elo()

# ---------------------------------------------------------
# Game simulation
# ---------------------------------------------------------

def simulate_game(p_hold):

    return np.random.rand() < p_hold


def simulate_set(pA,pB):

    gA = 0
    gB = 0

    server = 0

    while True:

        if server==0:

            if simulate_game(pA):
                gA+=1
            else:
                gB+=1

        else:

            if simulate_game(pB):
                gB+=1
            else:
                gA+=1

        server = 1-server

        if (gA>=6 or gB>=6) and abs(gA-gB)>=2:
            break

        if gA==6 and gB==6:

            if np.random.rand()<0.5:
                gA+=1
            else:
                gB+=1

            break

    return gA,gB


def simulate_match(pA,pB):

    setsA=0
    setsB=0

    games=0
    tiebreak=False

    while setsA<2 and setsB<2:

        gA,gB = simulate_set(pA,pB)

        games += gA+gB

        if gA==7 or gB==7:
            tiebreak=True

        if gA>gB:
            setsA+=1
        else:
            setsB+=1

    return games, setsA, setsB, tiebreak


# ---------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------

def monte_carlo(player_a, player_b, sims=10000):

    pA = service_hold_estimate(player_a)
    pB = service_hold_estimate(player_b)

    results = []

    set20=0
    set21=0
    tb=0

    for _ in range(sims):

        games,sA,sB,tb_flag = simulate_match(pA,pB)

        results.append(games)

        if (sA==2 and sB==0) or (sB==2 and sA==0):
            set20+=1
        else:
            set21+=1

        if tb_flag:
            tb+=1

    return np.array(results), set20/sims, set21/sims, tb/sims


# ---------------------------------------------------------
# Betting odds input
# ---------------------------------------------------------

st.subheader("Betting Market")

c1,c2,c3 = st.columns(3)

with c1:
    line = st.number_input("Over/Under Line", value=21.5)

with c2:
    over_odds = st.number_input("Over Odds", value=1.90)

with c3:
    under_odds = st.number_input("Under Odds", value=1.90)

# ---------------------------------------------------------
# Run simulation
# ---------------------------------------------------------

if st.button("Run Monte Carlo Simulation"):

    sims,set20,set21,tb_prob = monte_carlo(player_a,player_b)

    prediction = round(np.mean(sims),1)

    st.subheader("Predicted Total Games")

    st.markdown(f"# {prediction}")

    # Set probabilities

    st.subheader("Set Probabilities")

    c1,c2 = st.columns(2)

    c1.metric("2-0 Sets", f"{round(set20*100,1)}%")
    c2.metric("2-1 Sets", f"{round(set21*100,1)}%")

    st.metric("Tie-break Probability", f"{round(tb_prob*100,1)}%")

    # Over under

    prob_over = np.mean(sims > line)
    prob_under = 1-prob_over

    edge_over = prob_over*over_odds - 1
    edge_under = prob_under*under_odds - 1

    st.subheader("Betting Edge")

    c1,c2 = st.columns(2)

    c1.metric("Over Probability", f"{round(prob_over*100,1)}%",
              f"Edge {round(edge_over*100,1)}%")

    c2.metric("Under Probability", f"{round(prob_under*100,1)}%",
              f"Edge {round(edge_under*100,1)}%")

    # Chart

    fig, ax = plt.subplots()

    ax.hist(sims, bins=40)

    ax.set_title("Monte Carlo Total Games Distribution")

    ax.set_xlabel("Total Games")

    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # ---------------------------------------------------------
    # HTML report
    # ---------------------------------------------------------

    html = f"""
    <html>
    <body style="font-family:Arial;background:#f4f6ff;padding:40px">

    <h1 style="text-align:center">WTA Monte Carlo Prediction</h1>

    <h2 style="text-align:center">{player_a} vs {player_b}</h2>

    <h3 style="text-align:center">Surface: {surface}</h3>

    <h1 style="text-align:center">{prediction} Total Games</h1>

    <h3>Set Probabilities</h3>

    <p>2-0: {round(set20*100,1)}%</p>
    <p>2-1: {round(set21*100,1)}%</p>

    <h3>Tie-break Probability</h3>

    <p>{round(tb_prob*100,1)}%</p>

    <h3>Over / Under</h3>

    <p>Line {line}</p>
    <p>Over {round(prob_over*100,1)}%</p>
    <p>Under {round(prob_under*100,1)}%</p>

    <p>Generated {datetime.now()}</p>

    </body>
    </html>
    """

    st.download_button(
        "📥 Download HTML Report",
        html,
        file_name=f"{player_a}_vs_{player_b}_WTA_report.html",
        mime="text/html"
    )
