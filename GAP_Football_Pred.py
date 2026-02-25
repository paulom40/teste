import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# --- PAGE CONFIG ---
st.set_page_config(page_title="GAP Pro: Goals, Corners & Fair Odds", layout="wide")

st.title("⚽ GAP Pro Predictor")
st.markdown("Predicting **Goals**, **Corners**, and **Fair Market Odds** using Generalised Attacking Performance.")

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Model Hyperparameters")
l_goals = st.sidebar.slider("Goals Learning Rate (λ)", 0.01, 0.20, 0.05, 0.01)
l_corners = st.sidebar.slider("Corners Learning Rate (λ)", 0.01, 0.50, 0.15, 0.01)
phi1 = st.sidebar.slider("Home Weight (φ1)", 0.50, 1.00, 0.70, 0.05)
phi2 = st.sidebar.slider("Away Weight (φ2)", 0.50, 1.00, 0.70, 0.05)

uploaded_file = st.sidebar.file_uploader("Upload E0.csv", type="csv")

# --- CORE MODEL LOGIC ---
def process_full_gap_model(df, lg, lc, p1, p2):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    # Ratings stored as: [Goal_Ha, Goal_Hd, Goal_Aa, Goal_Ad, Corn_Ha, Corn_Hd, Corn_Aa, Corn_Ad]
    # Starting Goals at 1.35, Corners at 5.0
    ratings = {team: [1.35, 1.35, 1.35, 1.35, 5.0, 5.0, 5.0, 5.0] for team in teams}
    
    for idx, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        # Actual Stats
        gh, ga = row['FTHG'], row['FTAG']
        ch, ca = row['HC'], row['AC']
        
        r_h, r_a = ratings[h], ratings[a]
        
        # 1. Update GOALS
        exp_gh = (r_h[0] + r_a[3]) / 2
        exp_ga = (r_a[2] + r_h[1]) / 2
        
        ratings[h][0] = max(r_h[0] + lg * p1 * (gh - exp_gh), 0.1)
        ratings[h][2] = max(r_h[2] + lg * (1-p1) * (gh - exp_gh), 0.1)
        ratings[h][1] = max(r_h[1] + lg * p1 * (ga - exp_ga), 0.1)
        ratings[h][3] = max(r_h[3] + lg * (1-p1) * (ga - exp_ga), 0.1)
        
        ratings[a][2] = max(r_a[2] + lg * p2 * (ga - exp_ga), 0.1)
        ratings[a][0] = max(r_a[0] + lg * (1-p2) * (ga - exp_ga), 0.1)
        ratings[a][3] = max(r_a[3] + lg * p2 * (gh - exp_gh), 0.1)
        ratings[a][1] = max(r_a[1] + lg * (1-p2) * (gh - exp_gh), 0.1)

        # 2. Update CORNERS
        exp_ch = (r_h[4] + r_a[7]) / 2
        exp_ca = (r_a[6] + r_h[5]) / 2
        
        ratings[h][4] = max(r_h[4] + lc * p1 * (ch - exp_ch), 0.5)
        ratings[h][6] = max(r_h[6] + lc * (1-p1) * (ch - exp_ch), 0.5)
        ratings[h][5] = max(r_h[5] + lc * p1 * (ca - exp_ca), 0.5)
        ratings[h][7] = max(r_h[7] + lc * (1-p1) * (ca - exp_ca), 0.5)
        
        ratings[a][6] = max(r_a[6] + lc * p2 * (ca - exp_ca), 0.5)
        ratings[a][4] = max(r_a[4] + lc * (1-p2) * (ca - exp_ca), 0.5)
        ratings[a][7] = max(r_a[7] + lc * p2 * (ch - exp_ch), 0.5)
        ratings[a][5] = max(r_a[5] + lc * (1-p2) * (ch - exp_ch), 0.5)

    return ratings

def get_fair_odds(prob):
    return round(1 / prob, 2) if prob > 0 else 0

# --- UI LOGIC ---
if uploaded_file:
    data = pd.read_csv(uploaded_file).sort_values('Date')
    final_ratings = process_full_gap_model(data, l_goals, l_corners, phi1, phi2)
    
    tab1, tab2 = st.tabs(["📊 Team Ratings", "🔮 Predictions & Fair Odds"])

    with tab1:
        st.subheader("Final Goal & Corner Strengths")
        rdf = pd.DataFrame.from_dict(final_ratings, orient='index', 
            columns=['G_Ha', 'G_Hd', 'G_Aa', 'G_Ad', 'C_Ha', 'C_Hd', 'C_Aa', 'C_Ad'])
        st.dataframe(rdf.style.background_gradient(cmap='Blues'))

    with tab2:
        col_a, col_b = st.columns(2)
        h_team = col_a.selectbox("Home Team", sorted(final_ratings.keys()))
        a_team = col_b.selectbox("Away Team", sorted(final_ratings.keys()), index=1)
        
        # Get Current Ratings
        rh, ra = final_ratings[h_team], final_ratings[a_team]
        
        # Predictions
        mu_gh, mu_ga = (rh[0] + ra[3])/2, (ra[2] + rh[1])/2
        mu_ch, mu_ca = (rh[4] + ra[7])/2, (ra[6] + rh[5])/2
        
        st.divider()
        
        # 1. GOAL MARKETS
        st.subheader("🥅 Goal Market Predictions")
        
        # Poisson for 1X2
        max_g = 10
        p_gh = poisson.pmf(np.arange(max_g), mu_gh)
        p_ga = poisson.pmf(np.arange(max_g), mu_ga)
        m = np.outer(p_gh, p_ga)
        
        win, draw, loss = np.sum(np.tril(m, -1)), np.sum(np.diag(m)), np.sum(np.triu(m, 1))
        p_u25 = m[0,0]+m[0,1]+m[0,2]+m[1,0]+m[1,1]+m[2,0]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{h_team} Fair Odds", get_fair_odds(win))
        c2.metric("Draw Fair Odds", get_fair_odds(draw))
        c3.metric(f"{a_team} Fair Odds", get_fair_odds(loss))
        c4.metric("Over 2.5 Fair Odds", get_fair_odds(1-p_u25))

        # 2. CORNER MARKETS
        st.subheader("🚩 Corner Market Predictions")
        st.write(f"Expected Corners: **{h_team} {mu_ch:.1f}** — **{mu_ca:.1f} {a_team}** (Total: **{mu_ch+mu_ca:.1f}**)")
        
        # Poisson for Corners Over/Under
        total_corners_mu = mu_ch + mu_ca
        p_u105_corners = poisson.cdf(10, total_corners_mu)
        
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Expected Total Corners", round(total_corners_mu, 1))
        cc2.metric("Fair Odds Over 10.5", get_fair_odds(1-p_u105_corners))
        cc3.metric("Fair Odds Under 10.5", get_fair_odds(p_u105_corners))

        st.info("💡 'Fair Odds' represent a 100% book. If the bookmaker offers higher odds than shown here, it is a 'Value' bet.")

else:
    st.info("Please upload your E0.csv file.")
