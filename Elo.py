import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

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
    if match_count > experience_threshold:
        k = base_k * 0.5
    else:
        k = base_k

    if player_rating > 2400:
        k = min(k, min_k)
    elif player_rating < 1500:
        k = max(k, max_k)

    return max(min_k, min(k, max_k))

# ------------------------------
# Match Prediction Functions
# ------------------------------

def predict_match_winner(rating_a, rating_b, surface_ratings_a=None, surface_ratings_b=None, surface=None, scale=400):
    """Predict probability of player A winning."""
    if surface and surface_ratings_a and surface_ratings_b and surface in surface_ratings_a:
        # Use surface-specific ratings if available
        rating_a_use = surface_ratings_a[surface]
        rating_b_use = surface_ratings_b[surface]
    else:
        # Use overall ratings
        rating_a_use = rating_a
        rating_b_use = rating_b
    
    prob_a_wins = expected_score(rating_a_use, rating_b_use, scale)
    return prob_a_wins

def calculate_expected_games(prob_win, over_under_line=21.5):
    """Calculate expected total games and over/under probability."""
    # In tennis best-of-3 sets, total games typically range from 12 to 36+
    # Expected games based on probability of winning
    # This is a simplified model assuming correlation between set dominance and total games
    
    # Expected number of sets (2 or 3)
    prob_3_sets = 2 * prob_win * (1 - prob_win)  # Probability of going to 3 sets
    
    # Expected games based on win probability
    # If player is strong favorite, games will be fewer
    # If evenly matched, more games expected
    if prob_win > 0.7:
        # Dominant win - fewer games
        expected_games = 18 + (1 - prob_win) * 10
    elif prob_win < 0.3:
        # Dominant loss - fewer games
        expected_games = 18 + prob_win * 10
    else:
        # Competitive match - more games
        expected_games = 22 + abs(0.5 - prob_win) * 4
    
    # Add additional games for 3-set matches
    expected_games += prob_3_sets * 4
    
    # Probability of over 21.5 games
    if expected_games > over_under_line:
        prob_over = 0.6
        if prob_win > 0.65 or prob_win < 0.35:
            prob_over = 0.4
        elif 0.45 < prob_win < 0.55:
            prob_over = 0.7
    else:
        prob_over = 0.4
    
    return expected_games, prob_over, prob_3_sets

def get_betting_recommendation(prob_win, prob_over, expected_games, line=21.5):
    """Get betting recommendations based on predictions."""
    recommendations = []
    
    # Moneyline recommendation
    if prob_win > 0.55:
        recommendations.append(f"✅ FAVORITE: {prob_win:.1%} chance to win")
        if prob_win > 0.65:
            recommendations.append("   → Strong favorite, good value if odds > 1.5")
        else:
            recommendations.append("   → Slight favorite, look for odds > 1.8")
    elif prob_win < 0.45:
        recommendations.append(f"⚠️ UNDERDOG: {prob_win:.1%} chance to win")
        recommendations.append("   → Value play if odds > 2.5")
    else:
        recommendations.append(f"🤝 COIN FLIP: {prob_win:.1%} chance to win")
        recommendations.append("   → Look for close odds, avoid heavy favorites")
    
    # Over/Under recommendation
    if prob_over > 0.55:
        recommendations.append(f"📈 OVER {line}: {prob_over:.1%} probability")
        if expected_games > line + 1:
            recommendations.append(f"   → Expected: {expected_games:.1f} games")
    elif prob_over < 0.45:
        recommendations.append(f"📉 UNDER {line}: {(1-prob_over):.1%} probability")
        recommendations.append(f"   → Expected: {expected_games:.1f} games")
    else:
        recommendations.append(f"⚖️ TOO CLOSE: Avoid over/under bets")
    
    return recommendations

# ------------------------------
# Data Preparation
# ------------------------------

def parse_date(date_str):
    """Convert Excel date to datetime."""
    if pd.isna(date_str):
        return None
    try:
        return pd.to_datetime(date_str)
    except:
        try:
            if isinstance(date_str, (int, float)):
                return pd.to_datetime('1899-12-30') + pd.Timedelta(days=date_str)
            else:
                return pd.to_datetime(str(int(date_str)), format='%Y%m%d')
        except:
            return None

def prepare_matches(df):
    """Extract matches from the dataframe."""
    required_cols = ['tourney_date', 'surface', 'winner_id', 'winner_name', 'loser_id', 'loser_name', 'score']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Found: {df.columns.tolist()}")
        return pd.DataFrame()

    valid = df.dropna(subset=['winner_id', 'loser_id'])
    valid = valid[valid['winner_id'].astype(str).str.strip() != '']
    valid = valid[valid['loser_id'].astype(str).str.strip() != '']

    dates = valid['tourney_date'].apply(parse_date)
    valid = valid[dates.notna()]
    valid = valid.copy()
    valid['tourney_date'] = dates[dates.notna()]

    matches = valid[['tourney_date', 'surface', 'winner_id', 'winner_name', 'loser_id', 'loser_name', 'score']].copy()
    return matches

def initialize_ratings(matches):
    """Initialize ratings for all players."""
    all_players = set(matches['winner_id'].unique()) | set(matches['loser_id'].unique())
    return {pid: 1500 for pid in all_players}

def simulate_elo(matches, initial_ratings, k_factor_func, scale=400):
    """Run ELO simulation."""
    ratings = initial_ratings.copy()
    player_matches = {pid: 0 for pid in ratings.keys()}
    history = []

    matches_sorted = matches.sort_values('tourney_date')

    for idx, row in matches_sorted.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        
        rating_winner = ratings[winner]
        rating_loser = ratings[loser]

        k_winner = k_factor_func(rating_winner, player_matches[winner])
        k_loser = k_factor_func(rating_loser, player_matches[loser])

        new_winner, new_loser = update_elo(rating_winner, rating_loser, 1.0, k_winner, scale)

        history.append({
            'date': row['tourney_date'],
            'surface': row['surface'],
            'winner_id': winner,
            'winner_name': row['winner_name'],
            'loser_id': loser,
            'loser_name': row['loser_name'],
            'winner_rating_before': rating_winner,
            'loser_rating_before': rating_loser,
            'winner_rating_after': new_winner,
            'loser_rating_after': new_loser,
        })

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
            surf_ratings = {pid: 1500 for pid in initial_ratings.keys()}
            surf_ratings, surf_history = simulate_elo(surface_matches, surf_ratings, k_factor_func, scale)
            surface_ratings[surface] = surf_ratings
            surface_histories[surface] = surf_history

    return surface_ratings, surface_histories

# ------------------------------
# Streamlit App
# ------------------------------

st.set_page_config(page_title="Tennis ELO Rating System", layout="wide")
st.title("🏆 Tennis ELO Rating System by Surface")

st.markdown("""
This app computes ELO ratings for tennis players based on match results, with separate ratings per surface.
Use the **Match Predictor** tab to analyze potential matchups and get betting insights.
""")

with st.sidebar:
    st.header("⚙️ ELO Parameters")
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
    if matches > exp_thresh:
        k = base_k * 0.5
    else:
        k = base_k
    
    if rating > 2400:
        k = min(k, min_k)
    elif rating < 1500:
        k = max(k, max_k)
    return max(min_k, min(k, max_k))

@st.cache_data
def load_and_process(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name=0, dtype=str)
    for col in ['winner_id', 'loser_id']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    matches = prepare_matches(df)
    if matches.empty:
        return None, None, None, None, None
    
    initial_ratings = initialize_ratings(matches)
    global_ratings, global_history = simulate_elo(matches, initial_ratings, custom_k_factor, scale)
    surface_ratings, surface_histories = simulate_surface_elo(matches, initial_ratings, custom_k_factor, scale)
    
    # Calculate player stats
    player_stats = {}
    for _, row in matches.iterrows():
        for player_id, player_name in [(row['winner_id'], row['winner_name']), (row['loser_id'], row['loser_name'])]:
            if player_id not in player_stats:
                player_stats[player_id] = {'name': player_name, 'matches': 0, 'surfaces': {}}
            player_stats[player_id]['matches'] += 1
            surface = row['surface']
            if surface not in player_stats[player_id]['surfaces']:
                player_stats[player_id]['surfaces'][surface] = 0
            player_stats[player_id]['surfaces'][surface] += 1
    
    return matches, global_ratings, surface_ratings, (global_history, surface_histories), player_stats

with st.spinner("Processing data and computing ELO ratings..."):
    result = load_and_process(uploaded_file)
    if result[0] is None:
        st.stop()
    matches, global_ratings, surface_ratings, histories, player_stats = result

global_history, surface_histories = histories

# Create name mapping
name_map = {}
if matches is not None and not matches.empty:
    for _, row in matches.iterrows():
        name_map[row['winner_id']] = row['winner_name']
        name_map[row['loser_id']] = row['loser_name']

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Rankings", "📈 ELO Evolution", "🎯 Match Predictor", "ℹ️ About"])

# ------------------------------
# Tab 1: Rankings (Fixed column names)
# ------------------------------
with tab1:
    st.header("📊 Player Rankings")
    
    # Global Rankings
    st.subheader("Overall ELO Rankings")
    global_df = pd.DataFrame(list(global_ratings.items()), columns=['player_id', 'elo'])
    global_df = global_df.sort_values('elo', ascending=False).reset_index(drop=True)
    global_df['player_name'] = global_df['player_id'].map(name_map).fillna(global_df['player_id'])
    global_df['matches'] = global_df['player_id'].map(lambda x: player_stats.get(x, {}).get('matches', 0))
    global_df = global_df[['player_name', 'elo', 'matches']]
    global_df.columns = ['Player', 'ELO', 'Matches']  # Rename columns for display
    st.dataframe(global_df.head(50), use_container_width=True)
    
    # Surface Rankings
    if surface_ratings:
        st.subheader("Surface-Specific Rankings")
        surf_tabs = st.tabs([f"{surface}" for surface in surface_ratings.keys()])
        for tab, surface in zip(surf_tabs, surface_ratings.keys()):
            with tab:
                df_surf = pd.DataFrame(list(surface_ratings[surface].items()), columns=['player_id', 'elo'])
                df_surf = df_surf.sort_values('elo', ascending=False).reset_index(drop=True)
                df_surf['player_name'] = df_surf['player_id'].map(name_map).fillna(df_surf['player_id'])
                df_surf['matches'] = df_surf['player_id'].map(lambda x: player_stats.get(x, {}).get('surfaces', {}).get(surface, 0))
                df_surf = df_surf[['player_name', 'elo', 'matches']]
                df_surf.columns = ['Player', 'ELO', 'Matches']
                st.dataframe(df_surf.head(50), use_container_width=True)
# ------------------------------
# Tab 2: ELO Evolution
# ------------------------------
with tab2:
    st.header("📈 ELO Evolution Over Time")
    players_to_plot = st.multiselect("Select players to plot", 
                                      options=global_df['Player'].tolist(),
                                      default=global_df['Player'].head(5).tolist() if len(global_df) > 5 else global_df['Player'].tolist())
    
    if players_to_plot and global_history:
        hist_df = pd.DataFrame(global_history)
        if not hist_df.empty:
            player_events = {}
            for _, row in hist_df.iterrows():
                for pid, name, rating_after in [
                    (row['winner_id'], row['winner_name'], row['winner_rating_after']),
                    (row['loser_id'], row['loser_name'], row['loser_rating_after'])
                ]:
                    player_name = name_map.get(pid, name)
                    if player_name not in players_to_plot:
                        continue
                    if player_name not in player_events:
                        player_events[player_name] = []
                    player_events[player_name].append((row['date'], rating_after))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            for player, events in player_events.items():
                events.sort(key=lambda x: x[0])
                dates = [e[0] for e in events]
                ratings = [e[1] for e in events]
                ax.plot(dates, ratings, marker='.', linestyle='-', label=player, linewidth=2, markersize=4)
            ax.set_xlabel("Date")
            ax.set_ylabel("ELO Rating")
            ax.set_title("ELO Rating Evolution")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No historical data available for plotting.")
    else:
        if not players_to_plot:
            st.info("Please select players to view their rating evolution.")

# ------------------------------
# Tab 3: Match Predictor (FIXED)
# ------------------------------
with tab3:
    st.header("🎯 Match Predictor - 21 Line Analysis")
    st.markdown("""
    Compare two players and get predictions for:
    - Win probability
    - Expected total games
    - Over/Under 21.5 games analysis
    - Betting recommendations
    """)
    
    # Get list of unique player names
    all_players = sorted(global_df['Player'].tolist())
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1_name = st.selectbox("Select Player 1", options=all_players, key="player1")
        if player1_name:
            # Get player info safely
            player1_row = global_df[global_df['Player'] == player1_name]
            if not player1_row.empty:
                player1_id = player1_row['Player ID'].iloc[0]
                player1_elo = global_ratings.get(player1_id, 1500)
                player1_matches = player_stats.get(player1_id, {}).get('matches', 0)
                st.info(f"**{player1_name}**\n\nOverall ELO: {player1_elo:.0f}\nTotal Matches: {player1_matches}")
            else:
                st.error(f"Could not find player: {player1_name}")
                player1_id = None
                player1_elo = 1500
    
    with col2:
        # Filter out player1 from player2 options
        player2_options = [p for p in all_players if p != player1_name]
        player2_name = st.selectbox("Select Player 2", options=player2_options, key="player2")
        if player2_name:
            player2_row = global_df[global_df['Player'] == player2_name]
            if not player2_row.empty:
                player2_id = player2_row['Player ID'].iloc[0]
                player2_elo = global_ratings.get(player2_id, 1500)
                player2_matches = player_stats.get(player2_id, {}).get('matches', 0)
                st.info(f"**{player2_name}**\n\nOverall ELO: {player2_elo:.0f}\nTotal Matches: {player2_matches}")
            else:
                st.error(f"Could not find player: {player2_name}")
                player2_id = None
                player2_elo = 1500
    
    if player1_name and player2_name and player1_name != player2_name and player1_id and player2_id:
        st.divider()
        
        # Surface selection
        surface_options = ['Overall']
        if surface_ratings:
            surface_options.extend(sorted(surface_ratings.keys()))
        
        selected_surface = st.selectbox("Select Surface", surface_options)
        
        # Get surface-specific ratings
        player1_surface_rating = None
        player2_surface_rating = None
        surface_rating_diff = None
        
        if selected_surface != 'Overall' and surface_ratings and selected_surface in surface_ratings:
            player1_surface_rating = surface_ratings[selected_surface].get(player1_id, 1500)
            player2_surface_rating = surface_ratings[selected_surface].get(player2_id, 1500)
            surface_rating_diff = player1_surface_rating - player2_surface_rating
            
            # Check if players have matches on this surface
            player1_surface_matches = player_stats.get(player1_id, {}).get('surfaces', {}).get(selected_surface, 0)
            player2_surface_matches = player_stats.get(player2_id, {}).get('surfaces', {}).get(selected_surface, 0)
            
            st.info(f"""
            **Surface Ratings ({selected_surface})**
            - {player1_name}: {player1_surface_rating:.0f} ({player1_surface_matches} matches)
            - {player2_name}: {player2_surface_rating:.0f} ({player2_surface_matches} matches)
            - Difference: {abs(surface_rating_diff):.0f} points
            """)
        
        # Calculate win probability
        prob_player1_wins = predict_match_winner(
            player1_elo, player2_elo,
            surface_ratings, surface_ratings,
            selected_surface if selected_surface != 'Overall' else None,
            scale
        )
        prob_player2_wins = 1 - prob_player1_wins
        
        # Calculate expected games
        expected_games, prob_over_21_5, prob_3_sets = calculate_expected_games(prob_player1_wins)
        
        # Display predictions
        st.subheader("📊 Match Prediction")
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                f"{player1_name} Win Probability",
                f"{prob_player1_wins:.1%}",
                delta=f"vs {prob_player2_wins:.1%}"
            )
        
        with col2:
            st.metric(
                "Expected Total Games",
                f"{expected_games:.1f}",
                delta=f"Line: 21.5"
            )
        
        with col3:
            st.metric(
                "Over 21.5 Probability",
                f"{prob_over_21_5:.1%}",
                delta=f"3-set: {prob_3_sets:.1%}"
            )
        
        # Progress bar visualization
        st.subheader("Win Probability Visualization")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(prob_player1_wins, text=f"{player1_name}: {prob_player1_wins:.1%}")
        with col2:
            st.write(f"**{prob_player2_wins:.1%}** {player2_name}")
        
        # Display surface-specific advantages
        if selected_surface != 'Overall' and surface_ratings:
            st.subheader("🎾 Surface Advantage Analysis")
            overall_diff = player1_elo - player2_elo
            surface_diff = player1_surface_rating - player2_surface_rating
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Rating Difference", f"{overall_diff:+.0f}", 
                         help=f"{player1_name} is {abs(overall_diff):.0f} points {'higher' if overall_diff > 0 else 'lower'} overall")
            with col2:
                st.metric(f"Surface Rating Difference ({selected_surface})", f"{surface_diff:+.0f}",
                         help=f"{player1_name} is {abs(surface_diff):.0f} points {'higher' if surface_diff > 0 else 'lower'} on {selected_surface}")
            
            if abs(surface_diff) > abs(overall_diff) + 50:
                st.success(f"⚠️ **Surface Advantage**: {player1_name if surface_diff > 0 else player2_name} performs significantly better on {selected_surface}!")
        
        # Betting recommendations
        st.subheader("💡 Betting Recommendations (21 Line)")
        recommendations = get_betting_recommendation(prob_player1_wins, prob_over_21_5, expected_games, 21.5)
        
        for rec in recommendations:
            st.write(rec)
        
        # Detailed analysis
        with st.expander("📈 Detailed Analysis"):
            st.markdown(f"""
            ### Match Analysis: {player1_name} vs {player2_name}
            
            **ELO Ratings:**
            - {player1_name}: {player1_elo:.0f}
            - {player2_name}: {player2_elo:.0f}
            - Difference: {abs(player1_elo - player2_elo):.0f} points
            
            **Win Probability Factors:**
            - {player1_name} wins {prob_player1_wins:.1%} of the time
            - {player2_name} wins {prob_player2_wins:.1%} of the time
            
            **Games Analysis:**
            - Expected total games: {expected_games:.1f}
            - 3-set probability: {prob_3_sets:.1%}
            - Over 21.5 probability: {prob_over_21_5:.1%}
            
            **Key Insights:**
            """)
            
            if prob_player1_wins > 0.7:
                st.markdown(f"- ✅ {player1_name} is a strong favorite")
            elif prob_player1_wins < 0.3:
                st.markdown(f"- ✅ {player2_name} is a strong favorite")
            else:
                st.markdown("- ⚖️ This is expected to be a competitive match")
            
            if expected_games > 21.5:
                st.markdown(f"- 📈 Expect a longer match with {expected_games:.0f}+ total games")
            else:
                st.markdown(f"- 📉 Expect a relatively quick match with {expected_games:.0f} total games")
            
            if prob_over_21_5 > 0.6:
                st.markdown("- 🎯 OVER 21.5 games looks promising")
            elif prob_over_21_5 < 0.4:
                st.markdown("- 🎯 UNDER 21.5 games looks promising")
            else:
                st.markdown("- ⚠️ Over/Under is too close to call")
            
            # Additional surface-specific insights
            if selected_surface != 'Overall' and surface_ratings:
                st.markdown(f"""
                **Surface-Specific Insights ({selected_surface}):**
                - {player1_name} surface ELO: {player1_surface_rating:.0f}
                - {player2_name} surface ELO: {player2_surface_rating:.0f}
                - Surface advantage: {abs(surface_rating_diff):.0f} points
                """)
                
                if player1_surface_matches < 10 or player2_surface_matches < 10:
                    st.warning(f"⚠️ Limited data on {selected_surface} surface for one or both players. Predictions may be less reliable.")
    
    elif player1_name == player2_name and player1_name:
        st.warning("⚠️ Please select two different players to compare")
    elif not player1_name or not player2_name:
        st.info("👈 Select two players from the dropdown menus to begin analysis")

# ------------------------------
# Helper function to get player info safely
# ------------------------------
def get_player_info(player_name, global_df, global_ratings, player_stats):
    """Safely get player information."""
    player_row = global_df[global_df['Player'] == player_name]
    if player_row.empty:
        return None, None, None
    player_id = player_row['Player ID'].iloc[0]
    elo = global_ratings.get(player_id, 1500)
    matches = player_stats.get(player_id, {}).get('matches', 0)
    return player_id, elo, matches
