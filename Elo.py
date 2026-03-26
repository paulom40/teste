import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ------------------------------
# ELO System Implementation
# ------------------------------

def expected_score(rating_a, rating_b, scale=400):
    """Calculate expected score for player A against player B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / scale))

def update_elo(rating_a, rating_b, score_a, k_factor=32, scale=400):
    """Update ELO ratings after a match."""
    expected_a = expected_score(rating_a, rating_b, scale)
    expected_b = 1 - expected_a
    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - score_a) - expected_b)
    return new_rating_a, new_rating_b

def get_k_factor(player_rating, match_count, base_k=32, min_k=16, max_k=32, experience_threshold=50):
    """Adjust K-factor based on player experience and rating."""
    # Reduce K-factor for experienced players
    if match_count > experience_threshold:
        k = base_k * 0.5
    else:
        k = base_k

    # Optional: Cap K-factor for very high-rated players
    if player_rating > 2400:
        k = min(k, min_k)
    elif player_rating < 1500:
        k = max(k, max_k)

    return max(min_k, min(k, max_k))

# ------------------------------
# Data Preparation
# ------------------------------

def parse_date(date_str):
    """Convert Excel date (as string or number) to datetime."""
    if pd.isna(date_str):
        return None
    try:
        # If it's already a datetime
        return pd.to_datetime(date_str)
    except:
        try:
            # Try parsing as Excel serial number
            if isinstance(date_str, (int, float)):
                return pd.to_datetime('1899-12-30') + pd.Timedelta(days=date_str)
            else:
                # Try to parse string date (e.g., 20250602)
                return pd.to_datetime(str(int(date_str)), format='%Y%m%d')
        except:
            return None

def prepare_matches(df):
    """Extract matches from the dataframe."""
    matches = []
    required_cols = ['tourney_date', 'surface', 'winner_id', 'winner_name', 'loser_id', 'loser_name', 'score']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Found: {df.columns.tolist()}")
        return pd.DataFrame()

    # Filter out rows with missing IDs
    valid = df.dropna(subset=['winner_id', 'loser_id'])
    valid = valid[valid['winner_id'].astype(str).str.strip() != '']
    valid = valid[valid['loser_id'].astype(str).str.strip() != '']

    # Parse dates
    dates = valid['tourney_date'].apply(parse_date)
    valid = valid[dates.notna()]
    valid['tourney_date'] = dates[dates.notna()]

    matches = valid[['tourney_date', 'surface', 'winner_id', 'winner_name', 'loser_id', 'loser_name', 'score']].copy()
    return matches

def initialize_ratings(matches):
    """Initialize ratings for all players."""
    all_players = set(matches['winner_id'].unique()) | set(matches['loser_id'].unique())
    # Start all players at 1500
    return {pid: 1500 for pid in all_players}

def simulate_elo(matches, initial_ratings, k_factor_func, scale=400, surface_specific=True):
    """Run ELO simulation."""
    ratings = initial_ratings.copy()
    player_matches = {pid: 0 for pid in ratings.keys()}
    history = []

    # Sort matches by date
    matches_sorted = matches.sort_values('tourney_date')

    for idx, row in matches_sorted.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        surface = row['surface']

        # Get current ratings
        rating_winner = ratings[winner]
        rating_loser = ratings[loser]

        # Determine K-factors
        k_winner = k_factor_func(rating_winner, player_matches[winner])
        k_loser = k_factor_func(rating_loser, player_matches[loser])

        # Update ratings
        new_winner, new_loser = update_elo(rating_winner, rating_loser, 1.0, k_winner, scale)

        # Update history for surface-specific tracking
        history.append({
            'date': row['tourney_date'],
            'surface': surface,
            'winner_id': winner,
            'winner_name': row['winner_name'],
            'loser_id': loser,
            'loser_name': row['loser_name'],
            'winner_rating_before': rating_winner,
            'loser_rating_before': rating_loser,
            'winner_rating_after': new_winner,
            'loser_rating_after': new_loser,
            'k_winner': k_winner,
            'k_loser': k_loser
        })

        # Apply updates
        ratings[winner] = new_winner
        ratings[loser] = new_loser
        player_matches[winner] += 1
        player_matches[loser] += 1

    return ratings, history

def simulate_surface_elo(matches, initial_ratings, k_factor_func, scale=400):
    """Run ELO simulation separately for each surface."""
    surfaces = matches['surface'].unique()
    surface_ratings = {}
    surface_histories = {}

    for surface in surfaces:
        surface_matches = matches[matches['surface'] == surface].copy()
        if len(surface_matches) > 0:
            # Initialize fresh ratings for each surface
            surf_ratings = {pid: 1500 for pid in initial_ratings.keys()}
            surf_ratings, surf_history = simulate_elo(surface_matches, surf_ratings, k_factor_func, scale, surface_specific=True)
            surface_ratings[surface] = surf_ratings
            surface_histories[surface] = surf_history

    return surface_ratings, surface_histories

# ------------------------------
# Streamlit App
# ------------------------------

st.set_page_config(page_title="Tennis ELO Rating System", layout="wide")
st.title("🏆 Tennis ELO Rating System by Surface")

st.markdown("""
This app computes ELO ratings for tennis players based on match results, with separate ratings per surface (Hard, Clay, Grass).
Adjust the parameters on the left to fine-tune the system.
""")

with st.sidebar:
    st.header("⚙️ ELO Parameters")
    st.markdown("""
    - **Base K-Factor**: Higher values make ratings more volatile.
    - **Experience Threshold**: Players with more matches get reduced K-factor (stabilize).
    - **Minimum/Maximum K**: Caps to prevent extreme changes.
    - **Scale**: Typical value 400 (controls rating spread).
    """)
    base_k = st.slider("Base K-Factor", 16, 48, 32, step=2)
    min_k = st.slider("Minimum K-Factor", 8, 24, 16)
    max_k = st.slider("Maximum K-Factor", 24, 48, 32)
    exp_thresh = st.slider("Experience Threshold (matches)", 20, 100, 50)
    scale = st.slider("Scale Factor", 200, 600, 400, step=50)

    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded_file is None:
        st.info("Please upload your Excel file to begin.")
        st.stop()

def custom_k_factor(rating, matches):
    """Custom K-factor function using the parameters set in sidebar."""
    # Reduce for experienced players
    if matches > exp_thresh:
        k = base_k * 0.5
    else:
        k = base_k

    # Cap based on rating
    if rating > 2400:
        k = min(k, min_k)
    elif rating < 1500:
        k = max(k, max_k)
    return max(min_k, min(k, max_k))

@st.cache_data
def load_and_process(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name=0, dtype=str)
    # Convert numeric-like columns
    for col in ['winner_id', 'loser_id']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    matches = prepare_matches(df)
    if matches.empty:
        st.error("No valid matches found after processing.")
        return None, None, None
    initial_ratings = initialize_ratings(matches)
    # Global ELO
    global_ratings, global_history = simulate_elo(matches, initial_ratings, custom_k_factor, scale)
    # Surface-specific ELO
    surface_ratings, surface_histories = simulate_surface_elo(matches, initial_ratings, custom_k_factor, scale)
    return global_ratings, surface_ratings, (global_history, surface_histories)

# Load data
with st.spinner("Processing data and computing ELO ratings..."):
    global_ratings, surface_ratings, histories = load_and_process(uploaded_file)

if global_ratings is None:
    st.stop()

global_history, surface_histories = histories

# ------------------------------
# Display Rankings
# ------------------------------
st.header("📊 Player Rankings")

# Global Rankings
st.subheader("Overall ELO Rankings")
global_df = pd.DataFrame(list(global_ratings.items()), columns=['Player ID', 'ELO'])
global_df = global_df.sort_values('ELO', ascending=False).reset_index(drop=True)
# Add player names (if available)
# Try to map names from matches
name_map = {}
for _, row in matches.iterrows():
    name_map[row['winner_id']] = row['winner_name']
    name_map[row['loser_id']] = row['loser_name']
global_df['Player'] = global_df['Player ID'].map(name_map).fillna(global_df['Player ID'])
global_df = global_df[['Player', 'ELO']]
st.dataframe(global_df.head(50), use_container_width=True)

# Surface Rankings
tabs = st.tabs([f"{surface} Rankings" for surface in surface_ratings.keys()])
for tab, surface in zip(tabs, surface_ratings.keys()):
    with tab:
        df_surf = pd.DataFrame(list(surface_ratings[surface].items()), columns=['Player ID', 'ELO'])
        df_surf = df_surf.sort_values('ELO', ascending=False).reset_index(drop=True)
        df_surf['Player'] = df_surf['Player ID'].map(name_map).fillna(df_surf['Player ID'])
        df_surf = df_surf[['Player', 'ELO']]
        st.dataframe(df_surf.head(50), use_container_width=True)

# ------------------------------
# ELO Evolution Chart
# ------------------------------
st.header("📈 ELO Evolution Over Time")
players_to_plot = st.multiselect("Select players to plot (by ID or Name)", 
                                  options=global_df['Player'].tolist(),
                                  default=global_df['Player'].head(5).tolist() if len(global_df) > 5 else global_df['Player'].tolist())

if players_to_plot:
    # Prepare time series data from global history
    hist_df = pd.DataFrame(global_history)
    if not hist_df.empty:
        # Build player name mapping
        pid_to_name = dict(zip(global_df['Player ID'], global_df['Player']))
        # Create a list of events for each player
        player_events = {}
        for _, row in hist_df.iterrows():
            for pid, name, rating_before, rating_after in [
                (row['winner_id'], row['winner_name'], row['winner_rating_before'], row['winner_rating_after']),
                (row['loser_id'], row['loser_name'], row['loser_rating_before'], row['loser_rating_after'])
            ]:
                player_name = pid_to_name.get(pid, name)
                if player_name not in players_to_plot:
                    continue
                if player_name not in player_events:
                    player_events[player_name] = []
                player_events[player_name].append((row['date'], rating_after))
        # Plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        for player, events in player_events.items():
            events.sort(key=lambda x: x[0])
            dates = [e[0] for e in events]
            ratings = [e[1] for e in events]
            ax.plot(dates, ratings, marker='.', linestyle='-', label=player)
        ax.set_xlabel("Date")
        ax.set_ylabel("ELO Rating")
        ax.set_title("ELO Rating Evolution")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("No historical data available for plotting.")

# ------------------------------
# Export Results
# ------------------------------
st.header("💾 Export Data")
col1, col2 = st.columns(2)
with col1:
    # Export global rankings
    csv_global = global_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Global Rankings (CSV)", csv_global, "global_elo_rankings.csv", "text/csv")
with col2:
    # Export surface rankings
    if surface_ratings:
        surface_dfs = {}
        for surface, ratings in surface_ratings.items():
            df = pd.DataFrame(list(ratings.items()), columns=['Player ID', 'ELO'])
            df['Player'] = df['Player ID'].map(name_map).fillna(df['Player ID'])
            df = df[['Player', 'ELO']].sort_values('ELO', ascending=False)
            surface_dfs[surface] = df
        # Combine into one Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            global_df.to_excel(writer, sheet_name='Global', index=False)
            for surface, df in surface_dfs.items():
                df.to_excel(writer, sheet_name=surface, index=False)
        st.download_button("Download All Rankings (Excel)", output.getvalue(), "elo_rankings.xlsx", 
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------------------
# About ELO Parameters
# ------------------------------
with st.expander("ℹ️ About ELO System and Parameters"):
    st.markdown("""
    **ELO Rating System**:
    - Each player starts with 1500 points.
    - After each match, points are transferred based on expected outcome.
    - Expected score formula: E = 1 / (1 + 10^((rating_opponent - rating_self)/scale))
    - Update: new_rating = old_rating + K * (actual_score - expected_score)

    **K-Factor (K)**:
    - Controls how much ratings change after a match.
    - Higher K → more volatility.
    - In this app, K is reduced for players with many matches (stabilization).
    - Also capped for very high/low ratings.

    **Scale (S)**:
    - Typical value is 400.
    - Determines the spread of ratings: a difference of S points implies the higher-rated player is about 10 times more likely to win.

    **Surface-Specific Ratings**:
    - Separate ratings are computed for Hard, Clay, and Grass.
    - Only matches on that surface are used.
    - This gives a more accurate measure of a player's ability on a given surface.
    """)

st.markdown("---")
st.caption("ELO system implementation for tennis. Data provided in the uploaded Excel file.")
