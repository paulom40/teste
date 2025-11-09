# app.py - WITH SHOTS CONCEDED & HTML EXPORT
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, nbinom
import io
from typing import Dict, Any, Tuple, List
import requests
from PIL import Image
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
import base64
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Football Predictor - Complete Analysis", layout="wide")
st.title("⚽ Football Predictor Pro - Complete Analysis")
st.markdown("""
**Complete Match Analysis**  
- **Last 5 Home/Away Games Analysis**  
- **Shot Predictions & Analysis**  
- **Expected Shots Conceded**  
- **League Average Comparisons**  
- **HTML Export**  
""")

# ================================
# ENHANCED SHOT PREDICTION FUNCTIONS
# ================================
def predict_shots(home_team: str, away_team: str, stats: Dict[str, Any], league_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict shots for both teams based on recent form and league averages including shots conceded
    """
    predictions = {
        'home_shots': 0,
        'away_shots': 0,
        'home_shots_conceded': 0,
        'away_shots_conceded': 0,
        'total_shots': 0,
        'home_shot_efficiency': 0.0,
        'away_shot_efficiency': 0.0
    }
    
    # Get team stats
    home_team_stats = league_stats['team_home_stats'].get(home_team, {})
    away_team_stats = league_stats['team_away_stats'].get(away_team, {})
    league_home_avg = league_stats.get('league_home_avg', {})
    league_away_avg = league_stats.get('league_away_avg', {})
    
    # Predict home team shots FOR (attacking)
    if home_team_stats.get('shots_for') is not None and league_home_avg.get('shots_for') is not None:
        # Use team's home shooting tendency vs league average
        home_shot_factor = home_team_stats['shots_for'] / league_home_avg['shots_for']
        # Consider opponent's away defensive shooting tendency
        away_defense_factor = away_team_stats.get('shots_against', league_away_avg.get('shots_against', 1)) / league_away_avg.get('shots_against', 1)
        
        predicted_home_shots = league_home_avg['shots_for'] * home_shot_factor * (2 - away_defense_factor) / 2
        predictions['home_shots'] = max(round(predicted_home_shots), 1)
    
    # Predict home team shots CONCEDED (defensive)
    if home_team_stats.get('shots_against') is not None and league_home_avg.get('shots_against') is not None:
        # Use team's home defensive tendency vs league average
        home_defense_factor = home_team_stats['shots_against'] / league_home_avg['shots_against']
        # Consider opponent's away attacking shooting tendency
        away_attack_factor = away_team_stats.get('shots_for', league_away_avg.get('shots_for', 1)) / league_away_avg.get('shots_for', 1)
        
        predicted_home_conceded = league_home_avg['shots_against'] * home_defense_factor * away_attack_factor
        predictions['home_shots_conceded'] = max(round(predicted_home_conceded), 1)
    
    # Predict away team shots FOR (attacking)
    if away_team_stats.get('shots_for') is not None and league_away_avg.get('shots_for') is not None:
        # Use team's away shooting tendency vs league average
        away_shot_factor = away_team_stats['shots_for'] / league_away_avg['shots_for']
        # Consider opponent's home defensive shooting tendency
        home_defense_factor = home_team_stats.get('shots_against', league_home_avg.get('shots_against', 1)) / league_home_avg.get('shots_against', 1)
        
        predicted_away_shots = league_away_avg['shots_for'] * away_shot_factor * (2 - home_defense_factor) / 2
        predictions['away_shots'] = max(round(predicted_away_shots), 1)
    
    # Predict away team shots CONCEDED (defensive)
    if away_team_stats.get('shots_against') is not None and league_away_avg.get('shots_against') is not None:
        # Use team's away defensive tendency vs league average
        away_defense_factor = away_team_stats['shots_against'] / league_away_avg['shots_against']
        # Consider opponent's home attacking shooting tendency
        home_attack_factor = home_team_stats.get('shots_for', league_home_avg.get('shots_for', 1)) / league_home_avg.get('shots_for', 1)
        
        predicted_away_conceded = league_away_avg['shots_against'] * away_defense_factor * home_attack_factor
        predictions['away_shots_conceded'] = max(round(predicted_away_conceded), 1)
    
    predictions['total_shots'] = predictions['home_shots'] + predictions['away_shots']
    
    # Predict shot efficiency
    if home_team_stats.get('shot_efficiency') is not None:
        predictions['home_shot_efficiency'] = home_team_stats['shot_efficiency']
    if away_team_stats.get('shot_efficiency') is not None:
        predictions['away_shot_efficiency'] = away_team_stats['shot_efficiency']
    
    return predictions

def calculate_shot_probabilities(home_shots: int, away_shots: int, home_shots_conceded: int, away_shots_conceded: int, 
                               home_efficiency: float, away_efficiency: float) -> Dict[str, Any]:
    """
    Calculate probabilities for shot-related markets including defensive metrics
    """
    # Expected goals from shots
    home_expected_goals = home_shots * home_efficiency
    away_expected_goals = away_shots * away_efficiency
    
    # Shot probability matrix (simplified)
    max_shots = 25
    home_shot_probs = poisson.pmf(np.arange(max_shots), home_shots)
    away_shot_probs = poisson.pmf(np.arange(max_shots), away_shots)
    
    # Most likely shot counts
    home_most_likely = np.argmax(home_shot_probs)
    away_most_likely = np.argmax(away_shot_probs)
    
    # Defensive shot probabilities
    home_conceded_probs = poisson.pmf(np.arange(max_shots), home_shots_conceded)
    away_conceded_probs = poisson.pmf(np.arange(max_shots), away_shots_conceded)
    
    return {
        'home_expected_goals_from_shots': home_expected_goals,
        'away_expected_goals_from_shots': away_expected_goals,
        'home_most_likely_shots': home_most_likely,
        'away_most_likely_shots': away_most_likely,
        'home_most_likely_shots_conceded': np.argmax(home_conceded_probs),
        'away_most_likely_shots_conceded': np.argmax(away_conceded_probs),
        'total_expected_goals': home_expected_goals + away_expected_goals,
        'shot_advantage': home_shots - away_shots,
        'defensive_shot_advantage': home_shots_conceded - away_shots_conceded,
        'both_teams_5_plus_shots_prob': (1 - poisson.cdf(4, home_shots)) * (1 - poisson.cdf(4, away_shots)),
        'over_25_total_shots_prob': 1 - poisson.cdf(25, home_shots + away_shots),
        'home_under_shots_prob': poisson.cdf(home_shots_conceded - 1, home_shots_conceded),
        'away_under_shots_prob': poisson.cdf(away_shots_conceded - 1, away_shots_conceded)
    }

# ================================
# HTML EXPORT FUNCTION
# ================================
def generate_html_export(pred: Dict[str, Any], home_team: str, away_team: str, 
                        stats: Dict[str, Any], logos: Dict[str, str], 
                        league_stats: Dict[str, Any] = None) -> str:
    """Generate professional HTML export with all predictions including shots"""
    
    p = pred["predictions"]
    
    # Embed logos as base64
    def embed_logo(team: str) -> str:
        url = logos.get(team)
        if not url:
            return ""
        try:
            img = load_image(url)
            if img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                b64 = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/png;base64,{b64}"
        except:
            pass
        return ""
    
    home_logo_b64 = embed_logo(home_team)
    away_logo_b64 = embed_logo(away_team)
    
    # Team form analysis
    g = stats["goals"]
    home_attack = g["home_attack"].get(home_team, 1.0)
    home_defence = g["home_defence"].get(home_team, 1.0)
    away_attack = g["away_attack"].get(away_team, 1.0)
    away_defence = g["away_defence"].get(away_team, 1.0)
    
    # Shot predictions section
    shot_section = ""
    if p['shots']['home_shots'] > 0 and p['shots']['away_shots'] > 0:
        shot_section = f"""
        <div class="shot-predictions">
            <h3>🎯 Shot Predictions</h3>
            <div class="shot-grid">
                <div class="shot-team">
                    <h4>{home_team}</h4>
                    <p><strong>Expected Shots:</strong> {p['shots']['home_shots']}</p>
                    <p><strong>Expected Shots Conceded:</strong> {p['shots']['home_shots_conceded']}</p>
                    <p><strong>Shot Efficiency:</strong> {p['shots']['home_shot_efficiency']:.1%}</p>
                </div>
                <div class="shot-team">
                    <h4>{away_team}</h4>
                    <p><strong>Expected Shots:</strong> {p['shots']['away_shots']}</p>
                    <p><strong>Expected Shots Conceded:</strong> {p['shots']['away_shots_conceded']}</p>
                    <p><strong>Shot Efficiency:</strong> {p['shots']['away_shot_efficiency']:.1%}</p>
                </div>
                <div class="shot-totals">
                    <h4>Match Totals</h4>
                    <p><strong>Total Expected Shots:</strong> {p['shots']['total_shots']}</p>
                    <p><strong>Shot Advantage:</strong> {p['shots']['home_shots'] - p['shots']['away_shots']:+.0f}</p>
                </div>
            </div>
        </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{home_team} vs {away_team} - Prediction Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50, #34495e);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .teams {{
                display: flex;
                justify-content: space-around;
                align-items: center;
                padding: 30px;
                background: #f8f9fa;
            }}
            .team {{
                text-align: center;
                flex: 1;
            }}
            .logo {{
                width: 100px;
                height: 100px;
                object-fit: contain;
                margin-bottom: 15px;
            }}
            .team-name {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .vs {{
                font-size: 36px;
                font-weight: bold;
                color: #e74c3c;
                margin: 0 40px;
            }}
            .prediction-section {{
                padding: 30px;
                background: white;
            }}
            .score {{
                text-align: center;
                font-size: 48px;
                font-weight: bold;
                color: #2c3e50;
                margin: 20px 0;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .metric-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                border-left: 4px solid #3498db;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin: 10px 0;
            }}
            .metric-label {{
                font-size: 14px;
                color: #7f8c8d;
                text-transform: uppercase;
            }}
            .shot-predictions {{
                background: #e8f4f8;
                padding: 25px;
                margin: 20px 0;
                border-radius: 10px;
                border-left: 4px solid #3498db;
            }}
            .shot-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 20px;
                margin-top: 15px;
            }}
            .shot-team {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            .shot-totals {{
                background: #2c3e50;
                color: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            .form-analysis {{
                background: #ecf0f1;
                padding: 30px;
                margin: 20px 0;
                border-radius: 10px;
            }}
            .form-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
            }}
            .form-team {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .injury-section {{
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            .footer {{
                background: #2c3e50;
                color: white;
                text-align: center;
                padding: 20px;
                font-size: 14px;
            }}
            .section-title {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }}
            .stats-team {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Football Match Prediction Report</h1>
                <p>Complete Analysis | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="teams">
                <div class="team">
                    {f'<img src="{home_logo_b64}" class="logo" alt="{home_team}">' if home_logo_b64 else ''}
                    <div class="team-name">{home_team}</div>
                    <div>Last {p['games_used']['home']} home games analyzed</div>
                </div>
                <div class="vs">VS</div>
                <div class="team">
                    {f'<img src="{away_logo_b64}" class="logo" alt="{away_team}">' if away_logo_b64 else ''}
                    <div class="team-name">{away_team}</div>
                    <div>Last {p['games_used']['away']} away games analyzed</div>
                </div>
            </div>
            
            <div class="prediction-section">
                <div class="score">{p['goals']['score']}</div>
                <div style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">
                    Most Likely Score Based on Recent Form
                </div>
                
                <h2 class="section-title">Match Probabilities</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Home Win</div>
                        <div class="metric-value">{p['goals']['home_win']:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Draw</div>
                        <div class="metric-value">{p['goals']['draw']:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Away Win</div>
                        <div class="metric-value">{p['goals']['away_win']:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Both Teams Score</div>
                        <div class="metric-value">{p['goals']['btts_yes']:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Over 2.5 Goals</div>
                        <div class="metric-value">{p['goals']['over_25']:.1%}</div>
                    </div>
                </div>
                
                <h2 class="section-title">Expected Match Statistics</h2>
                <div class="stats-grid">
                    <div class="stats-team">
                        <h4>{home_team}</h4>
                        <p><strong>Expected Goals (xG):</strong> {p['xg']['home']:.2f}</p>
                        <p><strong>Expected Corners:</strong> {p['corners']['home']}</p>
                        <p><strong>Form Strength:</strong> {home_attack:.2f}× attack | {1/home_defence:.2f}× defense</p>
                    </div>
                    <div class="stats-team">
                        <h4>{away_team}</h4>
                        <p><strong>Expected Goals (xG):</strong> {p['xg']['away']:.2f}</p>
                        <p><strong>Expected Corners:</strong> {p['corners']['away']}</p>
                        <p><strong>Form Strength:</strong> {away_attack:.2f}× attack | {1/away_defence:.2f}× defense</p>
                    </div>
                </div>
                
                {shot_section}
                
                <div class="form-analysis">
                    <h2 class="section-title">Form Analysis</h2>
                    <div class="form-grid">
                        <div class="form-team">
                            <h3>{home_team} - Home Form</h3>
                            <p><strong>Attack Strength:</strong> {home_attack:.2f}× league average</p>
                            <p><strong>Defense Strength:</strong> {1/home_defence:.2f}× league average</p>
                            <p><strong>Games Analyzed:</strong> {p['games_used']['home']} recent home games</p>
                        </div>
                        <div class="form-team">
                            <h3>{away_team} - Away Form</h3>
                            <p><strong>Attack Strength:</strong> {away_attack:.2f}× league average</p>
                            <p><strong>Defense Strength:</strong> {1/away_defence:.2f}× league average</p>
                            <p><strong>Games Analyzed:</strong> {p['games_used']['away']} recent away games</p>
                        </div>
                    </div>
                </div>
                
                {f'''
                <div class="injury-section">
                    <h2 class="section-title">Injury Impact</h2>
                    <p><strong>{p['injury_summary']}</strong></p>
                </div>
                ''' if p.get('injury_summary') else ''}
            </div>
            
            <div class="footer">
                <p>Generated by Football Predictor Pro | Complete Analysis with Shot Predictions</p>
                <p>© {datetime.now().year} - All predictions based on statistical models and recent form analysis</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

# ================================
# ENHANCED DISPLAY WITH SHOTS CONCEDED
# ================================
def display_form_based_predictions(pred: Dict[str, Any], home_team: str, away_team: str, 
                                 stats: Dict[str, Any], league_stats: Dict[str, Any]):
    p = pred["predictions"]
    
    st.markdown(f"### **{home_team} vs {away_team}**")
    st.markdown("#### 🎯 Last 5 Games Form Analysis")
    
    logos = {home_team: get_team_logo(home_team), away_team: get_team_logo(away_team)}
    colA, colB, colC = st.columns([1,2,1])
    
    with colA:
        if logos[home_team]: 
            img = load_image(logos[home_team])
            if img: st.image(img, width=80)
        st.write(f"**{home_team}**")
        st.caption(f"Last {p['games_used']['home']} home games")
        
    with colC:
        if logos[away_team]: 
            img = load_image(logos[away_team])
            if img: st.image(img, width=80)
        st.write(f"**{away_team}**")
        st.caption(f"Last {p['games_used']['away']} away games")
        
    with colB:
        st.markdown(f"<h2 style='text-align:center'>{p['goals']['score']}</h2>", unsafe_allow_html=True)
        st.caption("Most likely score based on recent form")

    # Match probabilities
    colW1, colW2, colW3 = st.columns(3)
    colW1.metric("Home Win", f"{p['goals']['home_win']:.1%}")
    colW2.metric("Draw", f"{p['goals']['draw']:.1%}")
    colW3.metric("Away Win", f"{p['goals']['away_win']:.1%}")

    colB1, colB2 = st.columns(2)
    colB1.metric("Both Teams to Score", f"{p['goals']['btts_yes']:.1%}")
    colB2.metric("Over 2.5 Goals", f"{p['goals']['over_25']:.1%}")

    # ===== ENHANCED SHOT PREDICTIONS SECTION =====
    st.markdown("---")
    st.markdown("#### 🎯 Shot Predictions")
    
    if p['shots']['home_shots'] > 0 and p['shots']['away_shots'] > 0:
        col_shot1, col_shot2, col_shot3 = st.columns(3)
        
        with col_shot1:
            st.metric(f"{home_team} Expected Shots", f"{p['shots']['home_shots']}")
            st.metric(f"{home_team} Expected Shots Conceded", f"{p['shots']['home_shots_conceded']}")
            st.metric("Shot Efficiency", f"{p['shots']['home_shot_efficiency']:.1%}")
            if p['shot_probabilities']:
                st.metric("Expected Goals from Shots", f"{p['shot_probabilities']['home_expected_goals_from_shots']:.2f}")
        
        with col_shot2:
            st.metric(f"{away_team} Expected Shots", f"{p['shots']['away_shots']}")
            st.metric(f"{away_team} Expected Shots Conceded", f"{p['shots']['away_shots_conceded']}")
            st.metric("Shot Efficiency", f"{p['shots']['away_shot_efficiency']:.1%}")
            if p['shot_probabilities']:
                st.metric("Expected Goals from Shots", f"{p['shot_probabilities']['away_expected_goals_from_shots']:.2f}")
        
        with col_shot3:
            st.metric("Total Expected Shots", f"{p['shots']['total_shots']}")
            st.metric("Shot Advantage", f"{p['shots']['home_shots'] - p['shots']['away_shots']:+.0f}")
            st.metric("Defensive Shot Advantage", f"{p['shots']['home_shots_conceded'] - p['shots']['away_shots_conceded']:+.0f}")
            if p['shot_probabilities']:
                st.metric("Both Teams 5+ Shots", f"{p['shot_probabilities']['both_teams_5_plus_shots_prob']:.1%}")
        
        # Enhanced shot probability insights
        if p['shot_probabilities']:
            st.markdown("##### 📊 Shot Market Insights")
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.metric("Most Likely Home Shots", f"{p['shot_probabilities']['home_most_likely_shots']}")
                st.metric("Most Likely Away Shots", f"{p['shot_probabilities']['away_most_likely_shots']}")
                st.metric("Home Most Likely Conceded", f"{p['shot_probabilities']['home_most_likely_shots_conceded']}")
            
            with col_insight2:
                st.metric("Away Most Likely Conceded", f"{p['shot_probabilities']['away_most_likely_shots_conceded']}")
                st.metric("Over 25.5 Total Shots", f"{p['shot_probabilities']['over_25_total_shots_prob']:.1%}")
                st.metric("Total Expected Goals", f"{p['shot_probabilities']['total_expected_goals']:.2f}")
    else:
        st.info("📊 Shot data not available in uploaded dataset. Include 'HS' (Home Shots) and 'AS' (Away Shots) columns for shot predictions.")

    # ===== LEAGUE COMPARISON SECTION =====
    st.markdown("---")
    st.markdown("#### 📊 League Performance Comparison")
    
    # Create comparison tables
    home_comparison_df, away_comparison_df = create_comparison_tables(league_stats, home_team, away_team)
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        if not home_comparison_df.empty:
            st.markdown(f"##### 🏠 {home_team} Home Performance vs League Average")
            st.dataframe(home_comparison_df, use_container_width=True, hide_index=True)
    
    with col_comp2:
        if not away_comparison_df.empty:
            st.markdown(f"##### 🚌 {away_team} Away Performance vs League Average")
            st.dataframe(away_comparison_df, use_container_width=True, hide_index=True)
    
    # League rankings
    st.markdown("#### 📈 League Rankings (Last 5 Games)")
    
    home_rankings_df, away_rankings_df = create_league_ranking_tables(league_stats)
    
    col_rank1, col_rank2 = st.columns(2)
    
    with col_rank1:
        if not home_rankings_df.empty:
            st.markdown("##### 🏠 Home Performance Rankings")
            
            def highlight_home_team(row):
                if row['Team'] == home_team:
                    return ['background-color: #ffeb3b; color: #000000; font-weight: bold'] * len(row)
                return [''] * len(row)
            
            styled_home_df = home_rankings_df.style.apply(highlight_home_team, axis=1)
            st.dataframe(styled_home_df, use_container_width=True, height=400)
    
    with col_rank2:
        if not away_rankings_df.empty:
            st.markdown("##### 🚌 Away Performance Rankings")
            
            def highlight_away_team(row):
                if row['Team'] == away_team:
                    return ['background-color: #ffeb3b; color: #000000; font-weight: bold'] * len(row)
                return [''] * len(row)
            
            styled_away_df = away_rankings_df.style.apply(highlight_away_team, axis=1)
            st.dataframe(styled_away_df, use_container_width=True, height=400)

    # ===== LEAGUE AVERAGES SUMMARY =====
    st.markdown("#### 📋 League Averages Summary")
    
    col_avg1, col_avg2 = st.columns(2)
    
    with col_avg1:
        st.markdown("##### 🏠 Home Games Averages")
        if league_stats.get('league_home_avg'):
            home_avg = league_stats['league_home_avg']
            st.metric("Goals Scored", f"{home_avg['goals_scored']:.2f}")
            st.metric("Goals Conceded", f"{home_avg['goals_conceded']:.2f}")
            st.metric("Win Rate", f"{home_avg['win_rate']:.1%}")
            if 'shots_for' in home_avg:
                st.metric("Shots For", f"{home_avg['shots_for']:.1f}")
                st.metric("Shots Against", f"{home_avg['shots_against']:.1f}")
    
    with col_avg2:
        st.markdown("##### 🚌 Away Games Averages")
        if league_stats.get('league_away_avg'):
            away_avg = league_stats['league_away_avg']
            st.metric("Goals Scored", f"{away_avg['goals_scored']:.2f}")
            st.metric("Goals Conceded", f"{away_avg['goals_conceded']:.2f}")
            st.metric("Win Rate", f"{away_avg['win_rate']:.1%}")
            if 'shots_for' in away_avg:
                st.metric("Shots For", f"{away_avg['shots_for']:.1f}")
                st.metric("Shots Against", f"{away_avg['shots_against']:.1f}")

    # Rest of the display
    st.markdown("#### ⚽ Expected Match Stats")
    colX1, colX2 = st.columns(2)
    with colX1:
        st.write(f"**Expected Goals (xG)**")
        st.write(f"{home_team}: **{p['xg']['home']}**")
        st.write(f"{away_team}: **{p['xg']['away']}**")
    with colX2:
        st.write(f"**Expected Corners**")
        st.write(f"{home_team}: **{p['corners']['home']}**")
        st.write(f"{away_team}: **{p['corners']['away']}**")
        st.write(f"**Total**: **{p['corners']['total']}**")

    if p["injury_summary"]:
        st.markdown(f"#### 🏥 Injury Impact")
        st.markdown(f"<span style='color:red'>{p['injury_summary']}</span>", unsafe_allow_html=True)

    # ===== HTML EXPORT BUTTON =====
    st.markdown("---")
    st.markdown("#### 📤 Export Report")
    
    # HTML Export Button at the bottom
    html_content = generate_html_export(pred, home_team, away_team, stats, logos, league_stats)
    st.download_button(
        label="🌐 Download HTML Report", 
        data=html_content,
        file_name=f"{home_team}_vs_{away_team}_prediction_report.html",
        mime="text/html",
        use_container_width=True,
        type="primary"
    )
    st.caption("Professional HTML report with all predictions, team logos, and detailed analysis")

# ================================
# MAIN APP (KEEP EXISTING IMPLEMENTATION)
# ================================
# ... (keep all the existing main app code from previous implementation)
# The main app structure remains the same, we've just enhanced the shot predictions and added HTML export

# Note: Make sure to include all the other existing functions like:
# - calculate_team_form
# - compute_form_based_stats  
# - apply_injury_adjustment
# - predict_form_based_match
# - And all the league comparison functions
