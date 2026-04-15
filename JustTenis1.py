import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Tennis O/U 21.5 Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-over {
        background-color: #ff6b6b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-under {
        background-color: #51cf66;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# SURFACE-SPECIFIC MATCH LENGTH BASELINE
# ==============================================================================

def compute_surface_length_stats(df):
    """Calculate baseline match length statistics by surface."""
    length_stats = {
        'Hard': {
            'avg_games': 22.0,
            'median_games': 22.0,
            'std': 5.0,
            'p_over_21_5': 0.45,
            'q1': 19,
            'q3': 25,
        },
        'Clay': {
            'avg_games': 26.0,
            'median_games': 25.0,
            'std': 6.5,
            'p_over_21_5': 0.62,
            'q1': 21,
            'q3': 31,
        },
        'Grass': {
            'avg_games': 20.0,
            'median_games': 19.0,
            'std': 4.0,
            'p_over_21_5': 0.28,
            'q1': 17,
            'q3': 23,
        },
    }
    
    # Compute from actual data if available
    if df is not None and len(df) > 0:
        for surface in ['Hard', 'Clay', 'Grass']:
            surf_matches = df[df['surface'] == surface]
            
            if len(surf_matches) > 10:
                games = surf_matches['total_games'].values
                
                length_stats[surface] = {
                    'avg_games': float(np.mean(games)),
                    'median_games': float(np.median(games)),
                    'std': float(np.std(games)) if np.std(games) > 0 else 1.0,
                    'p_over_21_5': float((games > 21.5).mean()),
                    'q1': float(np.percentile(games, 25)),
                    'q3': float(np.percentile(games, 75)),
                    'min_games': float(np.min(games)),
                    'max_games': float(np.max(games)),
                    'count': len(surf_matches),
                }
    
    return length_stats


# ==============================================================================
# SERVE/RETURN DOMINANCE METRICS
# ==============================================================================

def calculate_serve_return_stats(df):
    """Extract serve dominance and return weakness metrics."""
    stats = defaultdict(lambda: {
        'straight_set_win_rate': 0.35,
        'three_set_loss_rate': 0.35,
        'serve_dominance': 0.35,
        'return_weakness': 0.35,
        'total_wins': 0,
        'total_losses': 0,
    })
    
    if df is None or len(df) == 0:
        return stats
    
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())
    
    for player in players:
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(matches) == 0:
            continue
        
        wins = df[df['winner'] == player]
        losses = df[df['loser'] == player]
        
        straight_set_wins = 0
        total_wins = len(wins)
        
        if total_wins > 0:
            for _, row in wins.iterrows():
                score = str(row.get('score', ''))
                sets = len([x for x in score.split() if x])
                if sets == 2:
                    straight_set_wins += 1
        
        three_set_losses = 0
        total_losses = len(losses)
        
        if total_losses > 0:
            for _, row in losses.iterrows():
                score = str(row.get('score', ''))
                sets = len([x for x in score.split() if x])
                if sets >= 3:
                    three_set_losses += 1
        
        total_matches = total_wins + total_losses
        
        stats[player] = {
            'straight_set_win_rate': straight_set_wins / max(total_wins, 1),
            'three_set_loss_rate': three_set_losses / max(total_losses, 1),
            'serve_dominance': straight_set_wins / max(total_matches, 1),
            'return_weakness': three_set_losses / max(total_matches, 1),
            'total_wins': total_wins,
            'total_losses': total_losses,
        }
    
    return stats


# ==============================================================================
# PLAYER MATCH STYLE & GRIND FACTOR
# ==============================================================================

def calculate_player_match_style(df, player_stats):
    """Identify 'grinders' (extend matches) vs 'closers' (finish quickly)."""
    if df is None or len(df) == 0:
        return player_stats
    
    for player in player_stats:
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(matches) == 0:
            player_stats[player]['grind_factor'] = 1.0
            player_stats[player]['finish_factor'] = 1.0
            player_stats[player]['match_volatility'] = 0.0
            continue
        
        wins = df[df['winner'] == player]
        losses = df[df['loser'] == player]
        
        avg_games_winning = wins['total_games'].mean() if len(wins) > 0 else 22.0
        avg_games_losing = losses['total_games'].mean() if len(losses) > 0 else 22.0
        
        grind_factor = avg_games_losing / 22.0
        finish_factor = avg_games_winning / 22.0
        
        match_volatility = 0.0
        if len(wins) > 1:
            match_volatility = wins['total_games'].std() / avg_games_winning if avg_games_winning > 0 else 0
        
        player_stats[player]['grind_factor'] = grind_factor
        player_stats[player]['finish_factor'] = finish_factor
        player_stats[player]['match_volatility'] = match_volatility
    
    return player_stats


# ==============================================================================
# RECENT MATCH LENGTH TREND
# ==============================================================================

def extract_recent_length_trend(df, player, surface, window=5):
    """Extract recent match length trend on specific surface."""
    if df is None or len(df) == 0:
        return {
            'avg_recent_games': 22.0,
            'max_recent_games': 22.0,
            'min_recent_games': 22.0,
            'recent_games_increasing': 0.0,
            'recent_count': 0,
        }
    
    matches = df[(df['winner'] == player) | (df['loser'] == player)]
    matches = matches[matches['surface'] == surface]
    
    if 'date' in matches.columns:
        matches = matches.sort_values('date', ascending=False)
    else:
        matches = matches.iloc[::-1]  # Assume chronological order
    
    recent = matches.head(window)
    
    if len(recent) == 0:
        return {
            'avg_recent_games': 22.0,
            'max_recent_games': 22.0,
            'min_recent_games': 22.0,
            'recent_games_increasing': 0.0,
            'recent_count': 0,
        }
    
    games = recent['total_games'].values
    
    trend = 0.0
    if len(recent) >= 3:
        if 'date' in recent.columns:
            recent_sorted = recent.sort_values('date')
        else:
            recent_sorted = recent
        earlier_avg = recent_sorted.iloc[:-2]['total_games'].mean()
        latest_avg = recent_sorted.iloc[-2:]['total_games'].mean()
        trend = (latest_avg - earlier_avg) / 22.0
    
    return {
        'avg_recent_games': float(np.mean(games)),
        'max_recent_games': float(np.max(games)),
        'min_recent_games': float(np.min(games)),
        'recent_games_increasing': trend,
        'recent_count': len(recent),
    }


# ==============================================================================
# COMPETITIVENESS INDEX
# ==============================================================================

def build_competitiveness_features(p1, p2, player_stats, h2h_surface, surface):
    """Build competitiveness index - close matches go longer."""
    s1 = player_stats[p1]
    s2 = player_stats[p2]
    
    # ELO GAP (40% weight)
    elo_gap = abs(s1['surface_elo'][surface] - s2['surface_elo'][surface])
    elo_gap_normalized = min(elo_gap / 400.0, 2.0)
    elo_competitiveness = max(1.0 - elo_gap_normalized, 0.0)
    
    # H2H COMPETITIVENESS (30% weight)
    h2h_p1_wins = h2h_surface.get((p1, p2), {}).get(surface, 0)
    h2h_p2_wins = h2h_surface.get((p2, p1), {}).get(surface, 0)
    total_h2h = h2h_p1_wins + h2h_p2_wins
    
    if total_h2h >= 3:
        h2h_win_ratio = h2h_p1_wins / total_h2h
        h2h_competitiveness = 1.0 - abs(h2h_win_ratio - 0.5) * 2
    else:
        h2h_competitiveness = 0.5
    
    # WIN RATE PARITY (30% weight)
    surf_wr_gap = abs(s1['surface_win_rate'][surface] - s2['surface_win_rate'][surface])
    wr_competitiveness = max(1.0 - surf_wr_gap * 2, 0.0)
    
    # COMBINE
    overall_competitiveness = (
        elo_competitiveness * 0.40 +
        h2h_competitiveness * 0.30 +
        wr_competitiveness * 0.30
    )
    
    return {
        'overall_competitiveness': overall_competitiveness,
        'elo_competitiveness': elo_competitiveness,
        'h2h_competitiveness': h2h_competitiveness,
        'wr_competitiveness': wr_competitiveness,
        'elo_gap_normalized': elo_gap_normalized,
        'surf_wr_gap': surf_wr_gap,
    }


# ==============================================================================
# MASTER FEATURE BUILDER
# ==============================================================================

def build_ou_features(p1, p2, surface, player_stats, h2h, h2h_surface, 
                      surface_length_stats, df, match=None):
    """Build specialized feature set for Over/Under 21.5 prediction."""
    if p1 not in player_stats or p2 not in player_stats:
        return None
    
    s1 = player_stats[p1]
    s2 = player_stats[p2]
    surf = surface if surface in ['Hard', 'Clay', 'Grass'] else 'Hard'
    
    # SURFACE-SPECIFIC BASELINE
    surf_stats = surface_length_stats[surf]
    surface_over_21_5_baseline = surf_stats['p_over_21_5']
    surface_avg_games = surf_stats['avg_games']
    surface_std = surf_stats['std']
    
    player_avg_p1 = s1['avg_games']
    player_avg_p2 = s2['avg_games']
    
    game_dev_p1 = (player_avg_p1 - surface_avg_games) / max(surface_std, 1.0)
    game_dev_p2 = (player_avg_p2 - surface_avg_games) / max(surface_std, 1.0)
    
    avg_deviations = (abs(game_dev_p1) + abs(game_dev_p2)) / 2
    
    # COMPETITIVENESS INDEX
    comp_data = build_competitiveness_features(p1, p2, player_stats, h2h_surface, surf)
    overall_comp = comp_data['overall_competitiveness']
    elo_gap_norm = comp_data['elo_gap_normalized']
    
    # GRIND FACTOR
    grind_p1 = s1.get('grind_factor', 1.0)
    grind_p2 = s2.get('grind_factor', 1.0)
    finish_p1 = s1.get('finish_factor', 1.0)
    finish_p2 = s2.get('finish_factor', 1.0)
    
    grind_combined = grind_p1 + grind_p2
    finish_combined = finish_p1 * finish_p2
    style_mismatch = abs(grind_p1 - grind_p2)
    
    # RECENT TREND
    recent_p1 = extract_recent_length_trend(df, p1, surf, window=5)
    recent_p2 = extract_recent_length_trend(df, p2, surf, window=5)
    
    recent_avg_p1 = recent_p1['avg_recent_games'] / 22.0
    recent_avg_p2 = recent_p2['avg_recent_games'] / 22.0
    recent_combined = (recent_avg_p1 + recent_avg_p2) / 2
    
    trend_p1 = recent_p1['recent_games_increasing']
    trend_p2 = recent_p2['recent_games_increasing']
    trend_combined = (trend_p1 + trend_p2) / 2
    
    # SERVE DOMINANCE
    serve_dom_p1 = s1.get('serve_dominance', 0.35)
    serve_dom_p2 = s2.get('serve_dominance', 0.35)
    serve_dom_diff = serve_dom_p1 - serve_dom_p2
    serve_dom_avg = (serve_dom_p1 + serve_dom_p2) / 2
    
    # CONTEXT
    surf_exp_p1 = s1['surface_match_count'].get(surf, 5)
    surf_exp_p2 = s2['surface_match_count'].get(surf, 5)
    exp_ratio = (surf_exp_p1 + 5) / (surf_exp_p2 + 5)
    exp_parity = min(exp_ratio, 1.0/exp_ratio) if exp_ratio > 0 else 0.5
    
    recent_form_p1 = s1['recent_form']
    recent_form_p2 = s2['recent_form']
    form_parity = 1.0 - abs(recent_form_p1 - recent_form_p2)
    
    # FEATURE VECTOR
    features = [
        surface_over_21_5_baseline,
        surface_over_21_5_baseline,
        game_dev_p1,
        game_dev_p2,
        overall_comp,
        overall_comp,
        overall_comp,
        elo_gap_norm,
        comp_data['h2h_competitiveness'],
        comp_data['wr_competitiveness'],
        grind_combined,
        grind_combined,
        finish_combined,
        style_mismatch,
        recent_avg_p1,
        recent_avg_p2,
        recent_combined,
        trend_combined,
        abs(trend_p1 - trend_p2),
        serve_dom_diff,
        serve_dom_avg,
        serve_dom_p1,
        serve_dom_p2,
        exp_parity,
        form_parity,
        avg_deviations,
        (grind_p1 + grind_p2) / 2 - (finish_p1 + finish_p2) / 2,
        overall_comp * grind_combined,
    ]
    
    return features


# ==============================================================================
# PREDICTION FUNCTION (Simplified model)
# ==============================================================================

def predict_over_under(features):
    """
    Simplified prediction model.
    In production, replace with trained GradientBoosting model.
    """
    if features is None:
        return 0.50, "UNDER"
    
    # Weighted scoring based on feature importance
    weights = {
        'surface_baseline': 0.20,
        'competitiveness': 0.25,
        'grind_factor': 0.15,
        'recent_trend': 0.15,
        'serve_dominance': 0.15,
        'context': 0.10
    }
    
    # Surface baseline (features 0-1)
    surface_score = (features[0] + features[1]) / 2
    
    # Competitiveness (features 4-9)
    comp_score = np.mean(features[4:10])
    
    # Grind factor (features 10-13)
    grind_score = np.mean(features[10:14])
    
    # Recent trend (features 14-18)
    recent_score = np.mean(features[14:19])
    
    # Serve dominance (features 19-22)
    serve_score = 1 - np.mean(features[19:23])  # Lower serve dominance = longer matches
    
    # Context (features 23-27)
    context_score = np.mean(features[23:28])
    
    # Combine scores
    final_score = (
        surface_score * weights['surface_baseline'] +
        comp_score * weights['competitiveness'] +
        grind_score * weights['grind_factor'] +
        recent_score * weights['recent_trend'] +
        serve_score * weights['serve_dominance'] +
        context_score * weights['context']
    )
    
    # Add slight random variation for demo (remove in production)
    final_score = np.clip(final_score + np.random.normal(0, 0.03), 0.2, 0.85)
    
    prediction = "OVER" if final_score > 0.5 else "UNDER"
    confidence = abs(final_score - 0.5) * 2
    
    return final_score, prediction, confidence


# ==============================================================================
# DEMO DATA GENERATION
# ==============================================================================

def generate_demo_data():
    """Generate demo player data for demonstration."""
    players = {
        'Novak Djokovic': {
            'surface_elo': {'Hard': 2450, 'Clay': 2400, 'Grass': 2480},
            'surface_win_rate': {'Hard': 0.85, 'Clay': 0.80, 'Grass': 0.88},
            'surface_match_count': {'Hard': 150, 'Clay': 120, 'Grass': 60},
            'avg_games': 21.5,
            'recent_form': 0.85,
            'serve_dominance': 0.65,
            'grind_factor': 0.95,
            'finish_factor': 1.05,
        },
        'Rafael Nadal': {
            'surface_elo': {'Hard': 2400, 'Clay': 2550, 'Grass': 2350},
            'surface_win_rate': {'Hard': 0.78, 'Clay': 0.92, 'Grass': 0.75},
            'surface_match_count': {'Hard': 130, 'Clay': 160, 'Grass': 50},
            'avg_games': 25.5,
            'recent_form': 0.78,
            'serve_dominance': 0.45,
            'grind_factor': 1.25,
            'finish_factor': 0.85,
        },
        'Carlos Alcaraz': {
            'surface_elo': {'Hard': 2350, 'Clay': 2380, 'Grass': 2300},
            'surface_win_rate': {'Hard': 0.75, 'Clay': 0.82, 'Grass': 0.70},
            'surface_match_count': {'Hard': 60, 'Clay': 70, 'Grass': 20},
            'avg_games': 23.0,
            'recent_form': 0.82,
            'serve_dominance': 0.55,
            'grind_factor': 1.10,
            'finish_factor': 0.95,
        },
        'Jannik Sinner': {
            'surface_elo': {'Hard': 2380, 'Clay': 2320, 'Grass': 2350},
            'surface_win_rate': {'Hard': 0.80, 'Clay': 0.72, 'Grass': 0.78},
            'surface_match_count': {'Hard': 80, 'Clay': 65, 'Grass': 25},
            'avg_games': 22.0,
            'recent_form': 0.88,
            'serve_dominance': 0.70,
            'grind_factor': 0.90,
            'finish_factor': 1.10,
        },
        'Daniil Medvedev': {
            'surface_elo': {'Hard': 2420, 'Clay': 2280, 'Grass': 2320},
            'surface_win_rate': {'Hard': 0.82, 'Clay': 0.65, 'Grass': 0.72},
            'surface_match_count': {'Hard': 110, 'Clay': 85, 'Grass': 30},
            'avg_games': 24.0,
            'recent_form': 0.72,
            'serve_dominance': 0.60,
            'grind_factor': 1.15,
            'finish_factor': 0.90,
        },
        'Alexander Zverev': {
            'surface_elo': {'Hard': 2320, 'Clay': 2350, 'Grass': 2250},
            'surface_win_rate': {'Hard': 0.72, 'Clay': 0.75, 'Grass': 0.65},
            'surface_match_count': {'Hard': 100, 'Clay': 95, 'Grass': 35},
            'avg_games': 24.5,
            'recent_form': 0.70,
            'serve_dominance': 0.58,
            'grind_factor': 1.20,
            'finish_factor': 0.88,
        }
    }
    
    # Add default stats for any missing players
    default_stats = {
        'surface_elo': {'Hard': 2200, 'Clay': 2200, 'Grass': 2200},
        'surface_win_rate': {'Hard': 0.50, 'Clay': 0.50, 'Grass': 0.50},
        'surface_match_count': {'Hard': 20, 'Clay': 20, 'Grass': 20},
        'avg_games': 22.0,
        'recent_form': 0.50,
        'serve_dominance': 0.35,
        'grind_factor': 1.0,
        'finish_factor': 1.0,
    }
    
    return players, default_stats


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    st.markdown('<h1 class="main-header">🎾 Tennis Over/Under 21.5 Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Tournament/Surface selection
        tournament_type = st.selectbox(
            "Tournament Type",
            ["Grand Slam", "ATP Masters 1000", "ATP 500", "ATP 250"]
        )
        
        surface = st.selectbox(
            "Surface",
            ["Hard", "Clay", "Grass"],
            help="Different surfaces have different match length characteristics"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info(
            "This model predicts whether a tennis match will go OVER or UNDER 21.5 total games.\n\n"
            "**Key Factors:**\n"
            "• Surface baseline statistics\n"
            "• Match competitiveness index\n"
            "• Player grind factors\n"
            "• Recent form trends\n"
            "• Serve/return dominance"
        )
    
    # Generate demo data
    players_data, default_stats = generate_demo_data()
    player_list = sorted(players_data.keys())
    
    # Player selection
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.selectbox("Player 1", player_list, index=0)
    
    with col2:
        player2 = st.selectbox("Player 2", player_list, index=1)
    
    if player1 == player2:
        st.warning("⚠️ Please select two different players")
        return
    
    # Predict button
    if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
        with st.spinner("Analyzing match factors..."):
            # Prepare player stats
            player_stats = {}
            for player in [player1, player2]:
                stats = players_data.get(player, default_stats.copy())
                player_stats[player] = stats
            
            # Prepare empty data structures
            df = None  # In production, load actual match data
            h2h_surface = defaultdict(lambda: defaultdict(int))
            surface_length_stats = compute_surface_length_stats(df)
            
            # Build features
            features = build_ou_features(
                player1, player2, surface, player_stats, 
                h2h_surface, h2h_surface, surface_length_stats, df
            )
            
            # Make prediction
            probability, prediction, confidence = predict_over_under(features)
            
            # Display prediction
            st.markdown("---")
            
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown(f"### 🎾 {player1}")
                st.metric("Surface ELO", f"{player_stats[player1]['surface_elo'][surface]:.0f}")
                st.metric("Surface Win Rate", f"{player_stats[player1]['surface_win_rate'][surface]:.1%}")
                st.metric("Avg Games/Match", f"{player_stats[player1]['avg_games']:.1f}")
                st.metric("Grind Factor", f"{player_stats[player1]['grind_factor']:.2f}")
            
            with col2:
                st.markdown("### 📊 Prediction")
                
                if prediction == "OVER":
                    st.markdown(f'<div class="prediction-over"><h2>OVER 21.5</h2><p>Probability: {probability:.1%}</p><p>Confidence: {confidence:.1%}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-under"><h2>UNDER 21.5</h2><p>Probability: {1-probability:.1%}</p><p>Confidence: {confidence:.1%}</p></div>', unsafe_allow_html=True)
                
                # Surface baseline
                surf_stats = surface_length_stats[surface]
                st.metric("Surface Baseline", f"{surf_stats['p_over_21_5']:.1%}")
                st.metric("Surface Avg Games", f"{surf_stats['avg_games']:.1f}")
            
            with col3:
                st.markdown(f"### 🎾 {player2}")
                st.metric("Surface ELO", f"{player_stats[player2]['surface_elo'][surface]:.0f}")
                st.metric("Surface Win Rate", f"{player_stats[player2]['surface_win_rate'][surface]:.1%}")
                st.metric("Avg Games/Match", f"{player_stats[player2]['avg_games']:.1f}")
                st.metric("Grind Factor", f"{player_stats[player2]['grind_factor']:.2f}")
            
            # Detailed analysis
            st.markdown("---")
            st.markdown("### 🔍 Detailed Analysis")
            
            # Competitiveness analysis
            comp_data = build_competitiveness_features(
                player1, player2, player_stats, h2h_surface, surface
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Match Competitiveness")
                
                # Gauge chart for competitiveness
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=comp_data['overall_competitiveness'] * 100,
                    title={'text': "Competitiveness Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                **Competitiveness Components:**
                - ELO Gap: {comp_data['elo_competitiveness']:.2%}
                - H2H Record: {comp_data['h2h_competitiveness']:.2%}
                - Win Rate Parity: {comp_data['wr_competitiveness']:.2%}
                """)
            
            with col2:
                st.markdown("#### Match Style Analysis")
                
                # Radar chart for player styles
                categories = ['Grind Factor', 'Finish Factor', 'Serve Dominance', 'Surface Experience']
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=[
                        player_stats[player1]['grind_factor'],
                        player_stats[player1]['finish_factor'],
                        player_stats[player1]['serve_dominance'],
                        min(player_stats[player1]['surface_match_count'].get(surface, 20) / 100, 1)
                    ],
                    theta=categories,
                    fill='toself',
                    name=player1
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=[
                        player_stats[player2]['grind_factor'],
                        player_stats[player2]['finish_factor'],
                        player_stats[player2]['serve_dominance'],
                        min(player_stats[player2]['surface_match_count'].get(surface, 20) / 100, 1)
                    ],
                    theta=categories,
                    fill='toself',
                    name=player2
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1.5]
                        )),
                    showlegend=True,
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction factors
            st.markdown("#### Key Prediction Factors")
            
            factors_data = {
                "Factor": ["Surface Baseline", "Competitiveness", "Grind Factor", "Recent Trend", "Serve Dominance"],
                "Impact on OVER": [
                    f"{surf_stats['p_over_21_5']:.1%}",
                    f"{comp_data['overall_competitiveness']:.1%}",
                    f"{(player_stats[player1]['grind_factor'] + player_stats[player2]['grind_factor']) / 2:.2f}",
                    "Increasing" if comp_data['overall_competitiveness'] > 0.5 else "Decreasing",
                    f"{1 - (player_stats[player1]['serve_dominance'] + player_stats[player2]['serve_dominance']) / 2:.1%}"
                ]
            }
            
            factors_df = pd.DataFrame(factors_data)
            st.dataframe(factors_df, use_container_width=True, hide_index=True)
            
            # Recommendation
            st.markdown("---")
            if prediction == "OVER":
                st.success(f"🎯 **Recommendation: Bet OVER 21.5 games**\n\n"
                          f"This match shows high competitiveness ({comp_data['overall_competitiveness']:.1%}) "
                          f"on {surface} courts where matches average {surf_stats['avg_games']:.1f} games. "
                          f"Both players tend to extend matches, suggesting value on the OVER.")
            else:
                st.info(f"🎯 **Recommendation: Bet UNDER 21.5 games**\n\n"
                       f"This match shows lower competitiveness ({comp_data['overall_competitiveness']:.1%}) "
                       f"on {surface} courts where matches average {surf_stats['avg_games']:.1f} games. "
                       f"One or both players tend to finish matches quickly, suggesting value on the UNDER.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>🎾 Tennis O/U 21.5 Predictor | Model Accuracy Target: 65-70%</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
