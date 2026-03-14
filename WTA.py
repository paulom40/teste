import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="WTA Pro Predictor", layout="wide")

st.title("🎾 WTA Professional Monte Carlo Predictor")

# ---------------------------------------------------------
# Upload dataset
# ---------------------------------------------------------

uploaded_file = st.file_uploader("Upload WTA Excel dataset", type=["xlsx"])

if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)

# ---------------------------------------------------------
# Clean dataset
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
# Player match history
# ---------------------------------------------------------

def get_matches(player, surface=None):

    m = df[(df["Winner"]==player) | (df["Loser"]==player)]

    if surface:
        m = m[m["Surface"]==surface]

    return m.sort_values("Date", ascending=False).head(30)

# ---------------------------------------------------------
# Surface Elo
# ---------------------------------------------------------

def compute_surface_elo():

    elo = {}

    for surf in df["Surface"].dropna().unique():

        sub = df[df["Surface"]==surf]

        players = set(sub["Winner"]) | set(sub["Loser"])

        ratings = {p:1500 for p in players}

        for _,row in sub.iterrows():

            w=row["Winner"]
            l=row["Loser"]

            exp = 1/(1+10**((ratings[l]-ratings[w])/400))

            k=24

            ratings[w]+=k*(1-exp)
            ratings[l]+=k*(0-exp)

        elo[surf]=ratings

    return elo

surface_elo = compute_surface_elo()

def get_elo(player, surface):

    try:
        return surface_elo[surface][player]
    except:
        return 1500

# ---------------------------------------------------------
# Hold probability estimate
# ---------------------------------------------------------

def hold_estimate(player, surface):

    m = get_matches(player, surface)

    if len(m)==0:
        return 0.65

    wins = (m["Winner"]==player).mean()

    return 0.60 + wins*0.22

# ---------------------------------------------------------
# Game simulation
# ---------------------------------------------------------

def simulate_game(p_hold):

    return np.random.rand() < p_hold


def simulate_set(pA,pB):

    gA=gB=0
    server=0
    tiebreak=False

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

        server=1-server

        if (gA>=6 or gB>=6) and abs(gA-gB)>=2:
            break

        if gA==6 and gB==6:

            tiebreak=True

            if np.random.rand()<0.5:
                gA+=1
            else:
                gB+=1

            break

    return gA,gB,tiebreak


def simulate_match(pA,pB):

    setsA=setsB=0
    games=0
    tb=False

    while setsA<2 and setsB<2:

        gA,gB,tb_flag = simulate_set(pA,pB)

        games += gA+gB

        if tb_flag:
            tb=True

        if gA>gB:
            setsA+=1
        else:
            setsB+=1

    return games,setsA,setsB,tb

# ---------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------

def monte_carlo(player_a, player_b, surface, sims=10000):

    pA = hold_estimate(player_a, surface)
    pB = hold_estimate(player_b, surface)

    results=[]
    set20=set21=tb=0

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
# Betting inputs
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

if st.button("Run Simulation"):

    sims,set20,set21,tb_prob = monte_carlo(player_a,player_b,surface)

    pred = round(np.mean(sims),1)

    st.subheader("Predicted Total Games")

    st.markdown(f"# {pred}")

    st.subheader("Set Probabilities")

    c1,c2 = st.columns(2)

    c1.metric("2-0",f"{round(set20*100,1)}%")
    c2.metric("2-1",f"{round(set21*100,1)}%")

    st.metric("Tie-break Probability",f"{round(tb_prob*100,1)}%")

    # -----------------------------------------------------
    # Betting edge
    # -----------------------------------------------------

    prob_over = np.mean(sims>line)
    prob_under = 1-prob_over

    edge_over = prob_over*over_odds-1
    edge_under = prob_under*under_odds-1

    st.subheader("Betting Edge")

    c1,c2 = st.columns(2)

    c1.metric("Over",f"{round(prob_over*100,1)}%",
              f"Edge {round(edge_over*100,1)}%")

    c2.metric("Under",f"{round(prob_under*100,1)}%",
              f"Edge {round(edge_under*100,1)}%")

    # -----------------------------------------------------
    # Distribution chart
    # -----------------------------------------------------

    fig,ax = plt.subplots()

    ax.hist(sims,bins=40)

    ax.set_title("Monte Carlo Total Games Distribution")

    st.pyplot(fig)

    # -----------------------------------------------------
    # HTML REPORT
    # -----------------------------------------------------

    html=f"""
    <html>
    <body style="font-family:Arial;padding:40px;background:#f4f6ff">

    <h1 style="text-align:center">WTA Monte Carlo Prediction</h1>

    <h2 style="text-align:center">{player_a} vs {player_b}</h2>

    <h3 style="text-align:center">Surface: {surface}</h3>

    <h1 style="text-align:center">{pred} Total Games</h1>

    <p>2-0 probability: {round(set20*100,1)}%</p>
    <p>2-1 probability: {round(set21*100,1)}%</p>

    <p>Tie-break probability: {round(tb_prob*100,1)}%</p>

    <p>Over {line}: {round(prob_over*100,1)}%</p>
    <p>Under {line}: {round(prob_under*100,1)}%</p>

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
