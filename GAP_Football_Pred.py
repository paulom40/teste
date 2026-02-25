import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# --- PAGE CONFIG ---
st.set_page_config(page_title="GAP Football Prediction Model", layout="wide")

st.title("⚽ GAP Football Model & Predictor")

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Model Parameters")
lambda_param = st.sidebar.slider("Learning Rate (λ)", 0.01, 0.20, 0.05, 0.01)
phi1 = st.sidebar.slider("Home Weight (φ1)", 0.50, 1.00, 0.70, 0.05)
phi2 = st.sidebar.slider("Away Weight (φ2)", 0.50, 1.00, 0.70, 0.05)
initial_rating = st.sidebar.number_input("Initial Rating (Avg Goals)", 0.5, 3.0, 1.35)

uploaded_file = st.sidebar.file_uploader("Upload your E0.csv file", type="csv")

# --- MODEL LOGIC ---
def process_gap_model(df, l, p1, p2, init):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    ratings = {team: [init, init, init, init] for team in teams} # [Ha, Hd, Aa, Ad]
    
    history = []
    for idx, row in df.iterrows():
        h_team, a_team = row['HomeTeam'], row['AwayTeam']
        sh, sa = row['FTHG'], row['FTAG']
        
        r_h, r_a = ratings[h_team], ratings[a_team]
        
        # Expected Goals (The GAP Criteria)
        exp_h = (r_h[0] + r_a[3]) / 2  
        exp_a = (r_a[2] + r_h[1]) / 2  
        
        history.append({
            'Home': h_team, 'Away': a_team, 'Score': f"{sh}-{sa}",
            'Exp_H': exp_h, 'Exp_A': exp_a, 'Total_Exp': exp_h + exp_a,
            'Actual_Total': sh + sa
        })
        
        # Updates
        ratings[h_team][0] = max(r_h[0] + l * p1 * (sh - exp_h), 0)
        ratings[h_team][2] = max(r_h[2] + l * (1 - p1) * (sh - exp_h), 0)
        ratings[h_team][1] = max(r_h[1] + l * p1 * (sa - exp_a), 0)
        ratings[h_team][3] = max(r_h[3] + l * (1 - p1) * (sa - exp_a), 0)
        
        ratings[a_team][2] = max(r_a[2] + l * p2 * (sa - exp_a), 0)
        ratings[a_team][0] = max(r_a[0] + l * (1 - p2) * (sa - exp_a), 0)
        ratings[a_team][3] = max(r_a[3] + l * p2 * (sh - exp_h), 0)
        ratings[a_team][1] = max(r_a[1] + l * (1 - p2) * (sh - exp_h), 0)

    return ratings, pd.DataFrame(history)

# --- APP TABS ---
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data = data.sort_values('Date')
    
    final_ratings, match_history = process_gap_model(data, lambda_param, phi1, phi2, initial_rating)
    
    tab1, tab2 = st.tabs(["📊 Model Analysis", "🔮 Predictions"])

    with tab1:
        st.subheader("Current League Ratings")
        rating_df = pd.DataFrame.from_dict(final_ratings, orient='index', 
                                         columns=['Home Attack', 'Home Defense', 'Away Attack', 'Away Defense'])
        rating_df['Strength'] = rating_df.mean(axis=1)
        st.dataframe(rating_df.sort_values('Strength', ascending=False).style.background_gradient(cmap='RdYlGn'))

    with tab2:
        st.subheader("Predict a Match")
        c1, c2 = st.columns(2)
        with c1:
            h_select = st.selectbox("Select Home Team", sorted(rating_df.index))
        with c2:
            a_select = st.selectbox("Select Away Team", sorted(rating_df.index), index=1)

        if h_select == a_select:
            st.warning("Please select two different teams.")
        else:
            # Get Ratings
            r_h = final_ratings[h_select]
            r_a = final_ratings[a_select]
            
            # Calculate Poisson Means (Criteria from Article)
            mu_h = (r_h[0] + r_a[3]) / 2
            mu_a = (r_a[2] + r_h[1]) / 2

            # Create Poisson Goal Matrix (0-7 goals)
            max_g = 8
            goals = np.arange(0, max_g)
            prob_h = poisson.pmf(goals, mu_h)
            prob_a = poisson.pmf(goals, mu_a)
            
            # Probability Matrix
            m = np.outer(prob_h, prob_a)
            
            p_hw = np.sum(np.tril(m, -1))
            p_d  = np.sum(np.diag(m))
            p_aw = np.sum(np.triu(m, 1))
            
            # Over/Under 2.5
            p_under = m[0,0] + m[0,1] + m[0,2] + m[1,0] + m[1,1] + m[2,0]
            p_over = 1 - p_under

            # Results Display
            st.info(f"Predicted Goals: **{h_select} {mu_h:.2f} - {mu_a:.2f} {a_select}**")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric(f"{h_select} Win", f"{p_hw:.1%}")
            res_col2.metric("Draw", f"{p_d:.1%}")
            res_col3.metric(f"{a_select} Win", f"{p_aw:.1%}")

            st.write(f"**Over/Under 2.5 Goals:** Over {p_over:.1%} | Under {p_under:.1%}")

            # Heatmap of scores
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(m, cmap='Blues')
            ax.set_xlabel(f"{a_select} Goals")
            ax.set_ylabel(f"{h_select} Goals")
            ax.set_title("Score Probability Matrix")
            plt.colorbar(im)
            st.pyplot(fig)
else:
    st.info("Upload CSV in the sidebar to view Ratings and Predictions.")
