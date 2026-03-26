import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------------
# Data Preparation Functions
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
    
    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.error(f"Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    # Filter out rows with missing IDs
    valid = df.dropna(subset=['winner_id', 'loser_id'])
    valid = valid[valid['winner_id'].astype(str).str.strip() != '']
    valid = valid[valid['loser_id'].astype(str).str.strip() != '']

    # Parse dates
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

def expected_score_basic(rating_a, rating_b, scale=400):
    """Basic expected score function."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / scale))

def update_elo(rating_a, rating_b, score_a, k_factor=32, scale=400):
    """Update ELO ratings after a match."""
    expected_a = expected_score_basic(rating_a, rating_b, scale)
    expected_b = 1 - expected_a
    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - score_a) - expected_b)
    return new_rating_a, new_rating_b

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

def calculate_game_statistics(matches_df):
    """Calculate advanced statistics from match data."""
    game_stats = {}
    
    for _, row in matches_df.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        surface = row['surface']
        
        # Initialize stats for players
        for player in [winner, loser]:
            if player not in game_stats:
                game_stats[player] = {
                    'matches': 0,
                    'wins': 0,
                    'surface_stats': {},
                    'recent_results': [],
                    'win_streak': 0,
                    'avg_games_won': 0,
                    'avg_games_lost': 0,
                    'tiebreak_wins': 0,
                    'three_set_matches': 0
                }
        
        # Update match statistics
        game_stats[winner]['matches'] += 1
        game_stats[winner]['wins'] += 1
        game_stats[winner]['recent_results'].append(1)
        game_stats[loser]['matches'] += 1
        game_stats[loser]['recent_results'].append(0)
        
        # Limit recent results
        if len(game_stats[winner]['recent_results']) > 20:
            game_stats[winner]['recent_results'] = game_stats[winner]['recent_results'][-20:]
        if len(game_stats[loser]['recent_results']) > 20:
            game_stats[loser]['recent_results'] = game_stats[loser]['recent_results'][-20:]
        
        # Parse score to get game statistics
        try:
            score = str(row['score'])
            sets = score.split()
            total_games = 0
            for s in sets:
                if '-' in s and not s.startswith('RET'):
                    parts = s.split('-')
                    if len(parts) == 2:
                        try:
                            g1 = int(parts[0].replace('(', ''))
                            g2 = int(parts[1].replace(')', '').replace('(', ''))
                            total_games += g1 + g2
                        except:
                            pass
            
            game_stats[winner]['avg_games_won'] += total_games / 2
            game_stats[loser]['avg_games_lost'] += total_games / 2
            
            # Check for tiebreaks
            if '(' in score and ')' in score:
                game_stats[winner]['tiebreak_wins'] += 1
            
            # Check for 3-set matches
            if len(sets) >= 3:
                game_stats[winner]['three_set_matches'] += 1
                game_stats[loser]['three_set_matches'] += 1
                
        except:
            pass
        
        # Update surface statistics
        for player in [winner, loser]:
            if surface not in game_stats[player]['surface_stats']:
                game_stats[player]['surface_stats'][surface] = {'wins': 0, 'losses': 0, 'games_won': 0, 'games_lost': 0}
            
            if player == winner:
                game_stats[player]['surface_stats'][surface]['wins'] += 1
            else:
                game_stats[player]['surface_stats'][surface]['losses'] += 1
    
    # Calculate derived statistics
    for player in game_stats:
        if game_stats[player]['matches'] > 0:
            game_stats[player]['avg_games_won'] /= game_stats[player]['matches']
            game_stats[player]['avg_games_lost'] /= game_stats[player]['matches']
        
        for surface in game_stats[player]['surface_stats']:
            stats = game_stats[player]['surface_stats'][surface]
            total = stats['wins'] + stats['losses']
            if total > 0:
                stats['win_pct'] = stats['wins'] / total
                stats['rating_diff'] = (stats['win_pct'] - 0.5) * 400
    
    return game_stats

def predict_match_winner(rating_a, rating_b, surface_ratings_a=None, surface_ratings_b=None, surface=None, scale=400):
    """Predict probability of player A winning."""
    try:
        if surface and surface_ratings_a is not None and surface_ratings_b is not None and surface in surface_ratings_a:
            rating_a_use = surface_ratings_a.get(surface, rating_a)
            rating_b_use = surface_ratings_b.get(surface, rating_b)
        else:
            rating_a_use = rating_a
            rating_b_use = rating_b
        
        rating_a_use = float(rating_a_use) if rating_a_use is not None else 1500.0
        rating_b_use = float(rating_b_use) if rating_b_use is not None else 1500.0
        
        prob_a_wins = expected_score_basic(rating_a_use, rating_b_use, scale)
        return prob_a_wins
    except Exception as e:
        return 0.5

def calculate_expected_games(prob_win, over_under_line=21.5):
    """Calculate expected total games and over/under probability."""
    prob_3_sets = 2 * prob_win * (1 - prob_win)
    
    if prob_win > 0.7:
        expected_games = 18 + (1 - prob_win) * 10
    elif prob_win < 0.3:
        expected_games = 18 + prob_win * 10
    else:
        expected_games = 22 + abs(0.5 - prob_win) * 4
    
    expected_games += prob_3_sets * 4
    
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

def get_k_factor_simple(rating, matches, base_k=32, min_k=16, max_k=32, exp_thresh=50):
    """Simple K-factor function."""
    if matches > exp_thresh:
        k = base_k * 0.5
    else:
        k = base_k
    
    if rating > 2400:
        k = min(k, min_k)
    elif rating < 1500:
        k = max(k, max_k)
    return max(min_k, min(k, max_k))

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
    
    # Calculate advanced game statistics
    game_stats = calculate_game_statistics(matches)
    
    # Initialize and run ELO
    initial_ratings = initialize_ratings(matches)
    global_ratings, global_history = simulate_elo(matches, initial_ratings, custom_k_factor, scale)
    
    # Calculate surface-specific ratings
    surface_ratings, surface_histories = simulate_surface_elo(matches, initial_ratings, custom_k_factor, scale)
    
    # Calculate accuracy
    correct_predictions = 0
    for i in range(len(global_history)):
        match = global_history[i]
        winner_rating = match['winner_rating_before']
        loser_rating = match['loser_rating_before']
        prob = expected_score_basic(winner_rating, loser_rating, scale)
        if (prob > 0.5 and match['winner_id'] == match['winner_id']) or (prob < 0.5 and match['loser_id'] == match['loser_id']):
            correct_predictions += 1
    
    accuracy = correct_predictions / len(global_history) if global_history else 0
    
    return matches, global_ratings, surface_ratings, game_stats, accuracy, global_history

with st.spinner("Processing data and computing ELO ratings..."):
    result = load_and_process(uploaded_file)
    if result[0] is None:
        st.stop()
    matches, global_ratings, surface_ratings, game_stats, accuracy, global_history = result

# Create name mapping
name_map = {}
if matches is not None and not matches.empty:
    for _, row in matches.iterrows():
        name_map[row['winner_id']] = row['winner_name']
        name_map[row['loser_id']] = row['loser_name']

# Create display dataframe
global_df = pd.DataFrame(list(global_ratings.items()), columns=['player_id', 'elo'])
global_df = global_df.sort_values('elo', ascending=False).reset_index(drop=True)
global_df['player_name'] = global_df['player_id'].map(name_map).fillna(global_df['player_id'])
display_df = global_df[['player_name', 'elo']].copy()
display_df.columns = ['Player', 'ELO']

# Display accuracy metric
st.metric("Model Accuracy", f"{accuracy:.1%}", 
          delta="Target: 60-65%",
          help="Percentage of correctly predicted match outcomes")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Rankings", "📈 ELO Evolution", "🎯 Match Predictor", "ℹ️ About"])

# ------------------------------
# Tab 1: Rankings
# ------------------------------
with tab1:
    st.header("📊 Player Rankings")
    
    st.subheader("Overall ELO Rankings")
    st.dataframe(display_df.head(50), use_container_width=True)
    
    if surface_ratings:
        st.subheader("Surface-Specific Rankings")
        surf_tabs = st.tabs([f"{surface}" for surface in surface_ratings.keys()])
        for tab, surface in zip(surf_tabs, surface_ratings.keys()):
            with tab:
                df_surf = pd.DataFrame(list(surface_ratings[surface].items()), columns=['player_id', 'elo'])
                df_surf = df_surf.sort_values('elo', ascending=False).reset_index(drop=True)
                df_surf['player_name'] = df_surf['player_id'].map(name_map).fillna(df_surf['player_id'])
                df_surf = df_surf[['player_name', 'elo']]
                df_surf.columns = ['Player', 'ELO']
                st.dataframe(df_surf.head(50), use_container_width=True)

# ------------------------------
# Tab 2: ELO Evolution
# ------------------------------
with tab2:
    st.header("📈 ELO Evolution Over Time")
    players_to_plot = st.multiselect("Select players to plot", 
                                      options=display_df['Player'].tolist(),
                                      default=display_df['Player'].head(5).tolist() if len(display_df) > 5 else display_df['Player'].tolist())
    
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
# Tab 3: Match Predictor
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
    
    all_players = sorted(display_df['Player'].tolist())
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1_name = st.selectbox("Select Player 1", options=all_players, key="player1")
        if player1_name:
            player1_row = global_df[global_df['player_name'] == player1_name]
            if not player1_row.empty:
                player1_id = player1_row['player_id'].iloc[0]
                player1_elo = float(player1_row['elo'].iloc[0])
                stats1 = game_stats.get(player1_id, {})
                st.info(f"**{player1_name}**\n\nOverall ELO: {player1_elo:.0f}\nMatches: {stats1.get('matches', 0)}")
    
    with col2:
        player2_options = [p for p in all_players if p != player1_name]
        player2_name = st.selectbox("Select Player 2", options=player2_options, key="player2")
        if player2_name:
            player2_row = global_df[global_df['player_name'] == player2_name]
            if not player2_row.empty:
                player2_id = player2_row['player_id'].iloc[0]
                player2_elo = float(player2_row['elo'].iloc[0])
                stats2 = game_stats.get(player2_id, {})
                st.info(f"**{player2_name}**\n\nOverall ELO: {player2_elo:.0f}\nMatches: {stats2.get('matches', 0)}")
    
    if player1_name and player2_name and player1_name != player2_name:
        st.divider()
        
        surface_options = ['Overall']
        if surface_ratings:
            surface_options.extend(sorted(surface_ratings.keys()))
        
        selected_surface = st.selectbox("Select Surface", surface_options)
        
        # Get surface-specific ratings
        player1_surface_rating = None
        player2_surface_rating = None
        
        if selected_surface != 'Overall' and surface_ratings and selected_surface in surface_ratings:
            player1_surface_rating = surface_ratings[selected_surface].get(player1_id, player1_elo)
            player2_surface_rating = surface_ratings[selected_surface].get(player2_id, player2_elo)
        
        # Calculate win probability
        try:
            if selected_surface != 'Overall' and player1_surface_rating and player2_surface_rating:
                prob_player1_wins = expected_score_basic(player1_surface_rating, player2_surface_rating, scale)
            else:
                prob_player1_wins = expected_score_basic(player1_elo, player2_elo, scale)
        except:
            prob_player1_wins = 0.5
        
        prob_player2_wins = 1 - prob_player1_wins
        
        # Calculate expected games
        expected_games, prob_over_21_5, prob_3_sets = calculate_expected_games(prob_player1_wins)
        
        # Display predictions
        st.subheader("📊 Match Prediction")
        
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
        
        # Display surface-specific info
        if selected_surface != 'Overall' and player1_surface_rating and player2_surface_rating:
            st.subheader("🎾 Surface Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{player1_name} on {selected_surface}", f"{player1_surface_rating:.0f}")
            with col2:
                st.metric(f"{player2_name} on {selected_surface}", f"{player2_surface_rating:.0f}")
        
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
    
    elif player1_name == player2_name and player1_name:
        st.warning("⚠️ Please select two different players to compare")
    elif not player1_name or not player2_name:
        st.info("👈 Select two players from the dropdown menus to begin analysis")

# ------------------------------
# Tab 4: About
# ------------------------------
with tab4:
    st.header("ℹ️ About the ELO System and Match Predictor")
    st.markdown("""
    ### ELO Rating System
    - Each player starts with 1500 points
    - After each match, points are transferred based on expected outcome
    - Expected score formula: E = 1 / (1 + 10^((rating_opponent - rating_self)/scale))
    - Update: new_rating = old_rating + K * (actual_score - expected_score)
    
    ### K-Factor (K)
    - Controls how much ratings change after a match
    - Higher K → more volatility
    - In this app, K is reduced for players with many matches (stabilization)
    - Also capped for very high/low ratings
    
    ### Scale (S)
    - Typical value is 400
    - Determines the spread of ratings: a difference of S points implies the higher-rated player is about 10 times more likely to win
    
    ### Surface-Specific Ratings
    - Separate ratings are computed for Hard, Clay, and Grass
    - Only matches on that surface are used
    - This gives a more accurate measure of a player's ability on a given surface
    
    ### Match Predictor (21 Line Analysis)
    - Uses both overall and surface-specific ELO ratings
    - Calculates expected total games based on match competitiveness
    - Provides over/under 21.5 games probability
    - Includes betting recommendations for moneyline and totals
    """)

# ------------------------------
# Export Functionality
# ------------------------------
with st.expander("💾 Export Data"):
    col1, col2 = st.columns(2)
    with col1:
        csv_global = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Global Rankings (CSV)", csv_global, "global_elo_rankings.csv", "text/csv")
    with col2:
        if surface_ratings:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                display_df.to_excel(writer, sheet_name='Global', index=False)
                for surface, ratings in surface_ratings.items():
                    df = pd.DataFrame(list(ratings.items()), columns=['player_id', 'elo'])
                    df['player_name'] = df['player_id'].map(name_map).fillna(df['player_id'])
                    df = df[['player_name', 'elo']].sort_values('elo', ascending=False)
                    df.columns = ['Player', 'ELO']
                    df.to_excel(writer, sheet_name=surface, index=False)
            st.download_button("Download All Rankings (Excel)", output.getvalue(), "elo_rankings.xlsx", 
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------------------
# Statistics Summary
# ------------------------------
with st.expander("📊 Statistics Summary"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(global_ratings))
    with col2:
        st.metric("Total Matches", len(global_history) if global_history else 0)
    with col3:
        top_player = display_df.iloc[0]['Player'] if not display_df.empty else "N/A"
        top_elo = display_df.iloc[0]['ELO'] if not display_df.empty else 0
        st.metric("Top Player", f"{top_player} ({top_elo:.0f})")
    with col4:
        avg_elo = display_df['ELO'].mean() if not display_df.empty else 0
        st.metric("Average ELO", f"{avg_elo:.0f}")

st.markdown("---")
st.caption(f"ELO System | Accuracy: {accuracy:.1%} | {len(global_history)} matches analyzed")
