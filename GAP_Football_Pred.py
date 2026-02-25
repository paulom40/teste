import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="GAP Football Prediction Model", layout="wide")

st.title("⚽ GAP Football Model Tester")
st.markdown("""
This app implements the **Generalised Attacking Performance (GAP)** model by Edward Wheatcroft.
It calculates 4 separate ratings for each team: **Home Attack, Home Defense, Away Attack, and Away Defense.**
""")

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Model Parameters")
lambda_param = st.sidebar.slider("Learning Rate (λ)", 0.01, 0.20, 0.05, 0.01)
phi1 = st.sidebar.slider("Home Weight (φ1)", 0.50, 1.00, 0.70, 0.05)
phi2 = st.sidebar.slider("Away Weight (φ2)", 0.50, 1.00, 0.70, 0.05)
initial_rating = st.sidebar.number_input("Initial Rating (Avg Goals)", 0.5, 3.0, 1.35)

uploaded_file = st.sidebar.file_uploader("Upload your E0.csv file", type="csv")

# --- CORE MODEL LOGIC ---
def process_gap_model(df, l, p1, p2, init):
    # Initialize ratings: Team Name -> [Ha, Hd, Aa, Ad]
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    ratings = {team: [init, init, init, init] for team in teams}
    
    history = []

    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        sh = row['FTHG'] # Home Goals
        sa = row['FTAG'] # Away Goals
        
        # Current Ratings before match
        r_h = ratings[home_team] # [Ha, Hd, Aa, Ad]
        r_a = ratings[away_team] # [Ha, Hd, Aa, Ad]
        
        # Expected Goals
        exp_h = (r_h[0] + r_a[3]) / 2  # (Home Attack + Away Defense) / 2
        exp_a = (r_a[2] + r_h[1]) / 2  # (Away Attack + Home Defense) / 2
        
        # Store for analysis
        history.append({
            'Date': row['Date'],
            'Home': home_team, 'Away': away_team,
            'Score': f"{sh}-{sa}",
            'Exp_H': exp_h, 'Exp_A': exp_a,
            'Total_Exp': exp_h + exp_a,
            'Actual_Total': sh + sa
        })
        
        # UPDATES (Based on Eq 1 and 2 from the paper)
        # Update Home Team (i)
        perf_h = sh - exp_h
        perf_d_h = sa - exp_a
        
        ratings[home_team][0] = max(r_h[0] + l * p1 * perf_h, 0)      # New Ha
        ratings[home_team][2] = max(r_h[2] + l * (1 - p1) * perf_h, 0) # New Aa
        ratings[home_team][1] = max(r_h[1] + l * p1 * perf_d_h, 0)    # New Hd
        ratings[home_team][3] = max(r_h[3] + l * (1 - p1) * perf_d_h, 0) # New Ad
        
        # Update Away Team (j)
        perf_a = sa - exp_a
        perf_d_a = sh - exp_h
        
        ratings[away_team][2] = max(r_a[2] + l * p2 * perf_a, 0)      # New Aa
        ratings[away_team][0] = max(r_a[0] + l * (1 - p2) * perf_a, 0) # New Ha
        ratings[away_team][3] = max(r_a[3] + l * p2 * perf_d_a, 0)    # New Ad
        ratings[away_team][1] = max(r_a[1] + l * (1 - p2) * perf_d_a, 0) # New Hd

    return ratings, pd.DataFrame(history)

# --- MAIN APP UI ---
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data = data.sort_values('Date')
    
    # Run the model
    final_ratings, match_history = process_gap_model(data, lambda_param, phi1, phi2, initial_rating)
    
    # Display 1: Final League Strength
    st.subheader("📊 Current Team Ratings")
    rating_df = pd.DataFrame.from_dict(final_ratings, orient='index', 
                                     columns=['Home Attack', 'Home Defense', 'Away Attack', 'Away Defense'])
    rating_df['Overall Strength'] = rating_df.mean(axis=1)
    st.dataframe(rating_df.sort_values('Overall Strength', ascending=False).style.background_gradient(cmap='RdYlGn'))

    # Display 2: Backtest Analysis
    st.subheader("📈 Backtest: Expected vs Actual Goals")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Last 10 Matches Predictions")
        st.table(match_history.tail(10))
        
    with col2:
        # Simple Over/Under 2.5 Accuracy
        match_history['Pred_Over'] = match_history['Total_Exp'] > 2.5
        match_history['Actual_Over'] = match_history['Actual_Total'] > 2.5
        accuracy = (match_history['Pred_Over'] == match_history['Actual_Over']).mean()
        st.metric("O/U 2.5 Accuracy", f"{accuracy:.2%}")
        
        # Chart
        fig, ax = plt.subplots()
        ax.scatter(match_history['Total_Exp'], match_history['Actual_Total'], alpha=0.5)
        ax.set_xlabel("Model Expected Total Goals")
        ax.set_ylabel("Actual Total Goals")
        ax.axhline(2.5, color='red', linestyle='--')
        ax.axvline(2.5, color='red', linestyle='--')
        st.pyplot(fig)

    # Display 3: Team Search
    st.subheader("🔍 Team Evolution")
    selected_team = st.selectbox("Select a team to see their history", rating_df.index)
    # Note: To show history properly, we'd need to store ratings after every match. 
    # For now, we show their final stats.
    st.write(f"Final Profile for {selected_team}:")
    st.bar_chart(rating_df.loc[selected_team, ['Home Attack', 'Home Defense', 'Away Attack', 'Away Defense']])

else:
    st.info("Please upload the E0.csv file to begin.")
