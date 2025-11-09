# app.py - WITH LEAGUE AVERAGE COMPARISON TABLES
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
st.set_page_config(page_title="Football Predictor - League Analysis", layout="wide")
st.title("⚽ Football Predictor Pro - League Analysis")
st.markdown("""
**Form-Based Analysis with League Comparisons**  
- **Last 5 Home/Away Games Analysis**  
- **League Average Comparison Tables**  
- **Team vs League Performance**  
- **Excel & HTML Export**  
""")

# ================================
# LEAGUE COMPARISON FUNCTIONS
# ================================
def calculate_league_averages(df: pd.DataFrame, home_col: str, away_col: str, 
                            hg_col: str, ag_col: str, hc_col: str = None, ac_col: str = None,
                            n_games: int = 5) -> Dict[str, Any]:
    """
    Calculate league averages for last N home/away games for all teams
    """
    teams = sorted(set(df[home_col]).union(df[away_col]))
    
    league_stats = {
        'team_home_stats': {},
        'team_away_stats': {}, 
        'league_home_avg': {},
        'league_away_avg': {},
        'overall_league_avg': {}
    }
    
    # Initialize stats dictionaries
    home_stats_list = []
    away_stats_list = []
    
    for team in teams:
        # Home games analysis
        home_games = get_last_n_home_games(df, team, home_col, n_games)
        if len(home_games) > 0:
            home_stats = {
                'team': team,
                'games_played': len(home_games),
                'goals_scored': home_games[hg_col].mean(),
                'goals_conceded': home_games[ag_col].mean(),
                'goal_difference': (home_games[hg_col] - home_games[ag_col]).mean(),
                'win_rate': (home_games[hg_col] > home_games[ag_col]).mean(),
                'draw_rate': (home_games[hg_col] == home_games[ag_col]).mean(),
                'loss_rate': (home_games[hg_col] < home_games[ag_col]).mean(),
                'clean_sheets': (home_games[ag_col] == 0).mean(),
                'failed_to_score': (home_games[hg_col] == 0).mean()
            }
            
            # Add corners if available
            if hc_col and hc_col in home_games.columns:
                home_stats['corners_for'] = home_games[hc_col].mean()
                home_stats['corners_against'] = home_games[ac_col].mean()
            
            home_stats_list.append(home_stats)
            league_stats['team_home_stats'][team] = home_stats
        
        # Away games analysis
        away_games = get_last_n_away_games(df, team, away_col, n_games)
        if len(away_games) > 0:
            away_stats = {
                'team': team,
                'games_played': len(away_games),
                'goals_scored': away_games[ag_col].mean(),
                'goals_conceded': away_games[hg_col].mean(),
                'goal_difference': (away_games[ag_col] - away_games[hg_col]).mean(),
                'win_rate': (away_games[ag_col] > away_games[hg_col]).mean(),
                'draw_rate': (away_games[ag_col] == away_games[hg_col]).mean(),
                'loss_rate': (away_games[ag_col] < away_games[hg_col]).mean(),
                'clean_sheets': (away_games[hg_col] == 0).mean(),
                'failed_to_score': (away_games[ag_col] == 0).mean()
            }
            
            # Add corners if available
            if ac_col and ac_col in away_games.columns:
                away_stats['corners_for'] = away_games[ac_col].mean()
                away_stats['corners_against'] = away_games[hc_col].mean()
            
            away_stats_list.append(away_stats)
            league_stats['team_away_stats'][team] = away_stats
    
    # Calculate league averages
    if home_stats_list:
        home_df = pd.DataFrame(home_stats_list)
        league_stats['league_home_avg'] = {
            'goals_scored': home_df['goals_scored'].mean(),
            'goals_conceded': home_df['goals_conceded'].mean(),
            'goal_difference': home_df['goal_difference'].mean(),
            'win_rate': home_df['win_rate'].mean(),
            'draw_rate': home_df['draw_rate'].mean(),
            'loss_rate': home_df['loss_rate'].mean(),
            'clean_sheets': home_df['clean_sheets'].mean(),
            'failed_to_score': home_df['failed_to_score'].mean()
        }
        
        if 'corners_for' in home_df.columns:
            league_stats['league_home_avg']['corners_for'] = home_df['corners_for'].mean()
            league_stats['league_home_avg']['corners_against'] = home_df['corners_against'].mean()
    
    if away_stats_list:
        away_df = pd.DataFrame(away_stats_list)
        league_stats['league_away_avg'] = {
            'goals_scored': away_df['goals_scored'].mean(),
            'goals_conceded': away_df['goals_conceded'].mean(),
            'goal_difference': away_df['goal_difference'].mean(),
            'win_rate': away_df['win_rate'].mean(),
            'draw_rate': away_df['draw_rate'].mean(),
            'loss_rate': away_df['loss_rate'].mean(),
            'clean_sheets': away_df['clean_sheets'].mean(),
            'failed_to_score': away_df['failed_to_score'].mean()
        }
        
        if 'corners_for' in away_df.columns:
            league_stats['league_away_avg']['corners_for'] = away_df['corners_for'].mean()
            league_stats['league_away_avg']['corners_against'] = away_df['corners_against'].mean()
    
    # Overall league averages (combined home and away)
    all_goals_scored = []
    all_goals_conceded = []
    
    for stats in home_stats_list:
        all_goals_scored.append(stats['goals_scored'])
        all_goals_conceded.append(stats['goals_conceded'])
    
    for stats in away_stats_list:
        all_goals_scored.append(stats['goals_scored'])
        all_goals_conceded.append(stats['goals_conceded'])
    
    league_stats['overall_league_avg'] = {
        'goals_scored': np.mean(all_goals_scored) if all_goals_scored else 0,
        'goals_conceded': np.mean(all_goals_conceded) if all_goals_conceded else 0
    }
    
    return league_stats

def create_comparison_tables(league_stats: Dict[str, Any], home_team: str, away_team: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create comparison tables showing team performance vs league averages
    """
    home_comparison = []
    away_comparison = []
    
    home_team_stats = league_stats['team_home_stats'].get(home_team, {})
    away_team_stats = league_stats['team_away_stats'].get(away_team, {})
    league_home_avg = league_stats['league_home_avg']
    league_away_avg = league_stats['league_away_avg']
    
    # Home team comparison table
    if home_team_stats:
        home_comparison = [
            {
                'Metric': 'Goals Scored',
                f'{home_team} (Home)': f"{home_team_stats['goals_scored']:.2f}",
                'League Avg (Home)': f"{league_home_avg['goals_scored']:.2f}",
                'Difference': f"{home_team_stats['goals_scored'] - league_home_avg['goals_scored']:+.2f}",
                'Percentage': f"{(home_team_stats['goals_scored'] / league_home_avg['goals_scored'] - 1) * 100:+.1f}%"
            },
            {
                'Metric': 'Goals Conceded',
                f'{home_team} (Home)': f"{home_team_stats['goals_conceded']:.2f}",
                'League Avg (Home)': f"{league_home_avg['goals_conceded']:.2f}",
                'Difference': f"{home_team_stats['goals_conceded'] - league_home_avg['goals_conceded']:+.2f}",
                'Percentage': f"{(home_team_stats['goals_conceded'] / league_home_avg['goals_conceded'] - 1) * 100:+.1f}%"
            },
            {
                'Metric': 'Goal Difference',
                f'{home_team} (Home)': f"{home_team_stats['goal_difference']:+.2f}",
                'League Avg (Home)': f"{league_home_avg['goal_difference']:+.2f}",
                'Difference': f"{home_team_stats['goal_difference'] - league_home_avg['goal_difference']:+.2f}",
                'Percentage': 'N/A'
            },
            {
                'Metric': 'Win Rate',
                f'{home_team} (Home)': f"{home_team_stats['win_rate']:.1%}",
                'League Avg (Home)': f"{league_home_avg['win_rate']:.1%}",
                'Difference': f"{(home_team_stats['win_rate'] - league_home_avg['win_rate']) * 100:+.1f}%",
                'Percentage': f"{(home_team_stats['win_rate'] / league_home_avg['win_rate'] - 1) * 100:+.1f}%"
            },
            {
                'Metric': 'Clean Sheets',
                f'{home_team} (Home)': f"{home_team_stats['clean_sheets']:.1%}",
                'League Avg (Home)': f"{league_home_avg['clean_sheets']:.1%}",
                'Difference': f"{(home_team_stats['clean_sheets'] - league_home_avg['clean_sheets']) * 100:+.1f}%",
                'Percentage': f"{(home_team_stats['clean_sheets'] / league_home_avg['clean_sheets'] - 1) * 100:+.1f}%"
            }
        ]
        
        # Add corners if available
        if 'corners_for' in home_team_stats:
            home_comparison.extend([
                {
                    'Metric': 'Corners For',
                    f'{home_team} (Home)': f"{home_team_stats['corners_for']:.1f}",
                    'League Avg (Home)': f"{league_home_avg['corners_for']:.1f}",
                    'Difference': f"{home_team_stats['corners_for'] - league_home_avg['corners_for']:+.1f}",
                    'Percentage': f"{(home_team_stats['corners_for'] / league_home_avg['corners_for'] - 1) * 100:+.1f}%"
                },
                {
                    'Metric': 'Corners Against',
                    f'{home_team} (Home)': f"{home_team_stats['corners_against']:.1f}",
                    'League Avg (Home)': f"{league_home_avg['corners_against']:.1f}",
                    'Difference': f"{home_team_stats['corners_against'] - league_home_avg['corners_against']:+.1f}",
                    'Percentage': f"{(home_team_stats['corners_against'] / league_home_avg['corners_against'] - 1) * 100:+.1f}%"
                }
            ])
    
    # Away team comparison table
    if away_team_stats:
        away_comparison = [
            {
                'Metric': 'Goals Scored',
                f'{away_team} (Away)': f"{away_team_stats['goals_scored']:.2f}",
                'League Avg (Away)': f"{league_away_avg['goals_scored']:.2f}",
                'Difference': f"{away_team_stats['goals_scored'] - league_away_avg['goals_scored']:+.2f}",
                'Percentage': f"{(away_team_stats['goals_scored'] / league_away_avg['goals_scored'] - 1) * 100:+.1f}%"
            },
            {
                'Metric': 'Goals Conceded',
                f'{away_team} (Away)': f"{away_team_stats['goals_conceded']:.2f}",
                'League Avg (Away)': f"{league_away_avg['goals_conceded']:.2f}",
                'Difference': f"{away_team_stats['goals_conceded'] - league_away_avg['goals_conceded']:+.2f}",
                'Percentage': f"{(away_team_stats['goals_conceded'] / league_away_avg['goals_conceded'] - 1) * 100:+.1f}%"
            },
            {
                'Metric': 'Goal Difference',
                f'{away_team} (Away)': f"{away_team_stats['goal_difference']:+.2f}",
                'League Avg (Away)': f"{league_away_avg['goal_difference']:+.2f}",
                'Difference': f"{away_team_stats['goal_difference'] - league_away_avg['goal_difference']:+.2f}",
                'Percentage': 'N/A'
            },
            {
                'Metric': 'Win Rate',
                f'{away_team} (Away)': f"{away_team_stats['win_rate']:.1%}",
                'League Avg (Away)': f"{league_away_avg['win_rate']:.1%}",
                'Difference': f"{(away_team_stats['win_rate'] - league_away_avg['win_rate']) * 100:+.1f}%",
                'Percentage': f"{(away_team_stats['win_rate'] / league_away_avg['win_rate'] - 1) * 100:+.1f}%"
            },
            {
                'Metric': 'Clean Sheets',
                f'{away_team} (Away)': f"{away_team_stats['clean_sheets']:.1%}",
                'League Avg (Away)': f"{league_away_avg['clean_sheets']:.1%}",
                'Difference': f"{(away_team_stats['clean_sheets'] - league_away_avg['clean_sheets']) * 100:+.1f}%",
                'Percentage': f"{(away_team_stats['clean_sheets'] / league_away_avg['clean_sheets'] - 1) * 100:+.1f}%"
            }
        ]
        
        # Add corners if available
        if 'corners_for' in away_team_stats:
            away_comparison.extend([
                {
                    'Metric': 'Corners For',
                    f'{away_team} (Away)': f"{away_team_stats['corners_for']:.1f}",
                    'League Avg (Away)': f"{league_away_avg['corners_for']:.1f}",
                    'Difference': f"{away_team_stats['corners_for'] - league_away_avg['corners_for']:+.1f}",
                    'Percentage': f"{(away_team_stats['corners_for'] / league_away_avg['corners_for'] - 1) * 100:+.1f}%"
                },
                {
                    'Metric': 'Corners Against',
                    f'{away_team} (Away)': f"{away_team_stats['corners_against']:.1f}",
                    'League Avg (Away)': f"{league_away_avg['corners_against']:.1f}",
                    'Difference': f"{away_team_stats['corners_against'] - league_away_avg['corners_against']:+.1f}",
                    'Percentage': f"{(away_team_stats['corners_against'] / league_away_avg['corners_against'] - 1) * 100:+.1f}%"
                }
            ])
    
    return pd.DataFrame(home_comparison), pd.DataFrame(away_comparison)

def create_league_ranking_tables(league_stats: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create league ranking tables for home and away performance
    """
    home_rankings = []
    away_rankings = []
    
    # Home performance rankings
    for team, stats in league_stats['team_home_stats'].items():
        home_rankings.append({
            'Team': team,
            'Games': stats['games_played'],
            'Goals Scored': stats['goals_scored'],
            'Goals Conceded': stats['goals_conceded'],
            'Goal Difference': stats['goal_difference'],
            'Win Rate': stats['win_rate'],
            'Clean Sheets': stats['clean_sheets']
        })
    
    # Away performance rankings
    for team, stats in league_stats['team_away_stats'].items():
        away_rankings.append({
            'Team': team,
            'Games': stats['games_played'],
            'Goals Scored': stats['goals_scored'],
            'Goals Conceded': stats['goals_conceded'],
            'Goal Difference': stats['goal_difference'],
            'Win Rate': stats['win_rate'],
            'Clean Sheets': stats['clean_sheets']
        })
    
    # Convert to DataFrames and sort
    home_df = pd.DataFrame(home_rankings)
    away_df = pd.DataFrame(away_rankings)
    
    if not home_df.empty:
        home_df = home_df.sort_values('Goal Difference', ascending=False)
    if not away_df.empty:
        away_df = away_df.sort_values('Goal Difference', ascending=False)
    
    return home_df, away_df

def display_league_comparison_visualizations(league_stats: Dict[str, Any], home_team: str, away_team: str):
    """
    Create visualizations comparing teams to league averages
    """
    home_stats = league_stats['team_home_stats'].get(home_team, {})
    away_stats = league_stats['team_away_stats'].get(away_team, {})
    league_home_avg = league_stats['league_home_avg']
    league_away_avg = league_stats['league_away_avg']
    
    if home_stats and away_stats:
        # Goals comparison chart
        fig_goals = go.Figure()
        
        # Home team vs league home average
        fig_goals.add_trace(go.Bar(
            name=f'{home_team} (Home)',
            x=['Goals Scored', 'Goals Conceded'],
            y=[home_stats['goals_scored'], home_stats['goals_conceded']],
            marker_color='#1f77b4'
        ))
        
        fig_goals.add_trace(go.Bar(
            name='League Home Avg',
            x=['Goals Scored', 'Goals Conceded'],
            y=[league_home_avg['goals_scored'], league_home_avg['goals_conceded']],
            marker_color='#aec7e8'
        ))
        
        # Away team vs league away average
        fig_goals.add_trace(go.Bar(
            name=f'{away_team} (Away)',
            x=['Goals Scored', 'Goals Conceded'],
            y=[away_stats['goals_scored'], away_stats['goals_conceded']],
            marker_color='#ff7f0e'
        ))
        
        fig_goals.add_trace(go.Bar(
            name='League Away Avg',
            x=['Goals Scored', 'Goals Conceded'],
            y=[league_away_avg['goals_scored'], league_away_avg['goals_conceded']],
            marker_color='#ffbb78'
        ))
        
        fig_goals.update_layout(
            title='Goals Comparison: Teams vs League Averages',
            barmode='group',
            xaxis_title='Metric',
            yaxis_title='Average Goals',
            showlegend=True
        )
        
        st.plotly_chart(fig_goals, use_container_width=True)
        
        # Win rate comparison
        fig_win_rates = go.Figure()
        
        fig_win_rates.add_trace(go.Bar(
            name=f'{home_team} Home Win Rate',
            x=['Win Rate'],
            y=[home_stats['win_rate'] * 100],
            marker_color='#2ca02c'
        ))
        
        fig_win_rates.add_trace(go.Bar(
            name='League Home Win Rate',
            x=['Win Rate'],
            y=[league_home_avg['win_rate'] * 100],
            marker_color='#98df8a'
        ))
        
        fig_win_rates.add_trace(go.Bar(
            name=f'{away_team} Away Win Rate',
            x=['Win Rate'],
            y=[away_stats['win_rate'] * 100],
            marker_color='#d62728'
        ))
        
        fig_win_rates.add_trace(go.Bar(
            name='League Away Win Rate',
            x=['Win Rate'],
            y=[league_away_avg['win_rate'] * 100],
            marker_color='#ff9896'
        ))
        
        fig_win_rates.update_layout(
            title='Win Rate Comparison: Teams vs League Averages',
            xaxis_title='Team Context',
            yaxis_title='Win Rate (%)',
            showlegend=True
        )
        
        st.plotly_chart(fig_win_rates, use_container_width=True)

# ================================
# UPDATED DISPLAY FUNCTION WITH LEAGUE COMPARISONS
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

    # ===== LEAGUE COMPARISON SECTION =====
    st.markdown("---")
    st.markdown("#### 📊 League Performance Comparison")
    
    # Create comparison tables
    home_comparison_df, away_comparison_df = create_comparison_tables(league_stats, home_team, away_team)
    
    if not home_comparison_df.empty:
        st.markdown(f"##### 🏠 {home_team} Home Performance vs League Average")
        st.dataframe(home_comparison_df, use_container_width=True, hide_index=True)
    
    if not away_comparison_df.empty:
        st.markdown(f"##### 🚌 {away_team} Away Performance vs League Average")
        st.dataframe(away_comparison_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    display_league_comparison_visualizations(league_stats, home_team, away_team)
    
    # League rankings
    st.markdown("#### 📈 League Rankings (Last 5 Games)")
    
    home_rankings_df, away_rankings_df = create_league_ranking_tables(league_stats)
    
    col_rank1, col_rank2 = st.columns(2)
    
    with col_rank1:
        if not home_rankings_df.empty:
            st.markdown("##### 🏠 Home Performance Rankings")
            # Highlight the current home team
            styled_home_df = home_rankings_df.style.apply(
                lambda x: ['background: #e6f3ff' if x['Team'] == home_team else '' for _ in x], 
                axis=1
            )
            st.dataframe(styled_home_df, use_container_width=True, height=400)
    
    with col_rank2:
        if not away_rankings_df.empty:
            st.markdown("##### 🚌 Away Performance Rankings")
            # Highlight the current away team
            styled_away_df = away_rankings_df.style.apply(
                lambda x: ['background: #e6f3ff' if x['Team'] == away_team else '' for _ in x], 
                axis=1
            )
            st.dataframe(styled_away_df, use_container_width=True, height=400)

    # ===== LEAGUE AVERAGES SUMMARY =====
    st.markdown("#### 📋 League Averages Summary")
    
    col_avg1, col_avg2 = st.columns(2)
    
    with col_avg1:
        st.markdown("##### 🏠 Home Games Averages")
        home_avg = league_stats['league_home_avg']
        st.metric("Goals Scored", f"{home_avg['goals_scored']:.2f}")
        st.metric("Goals Conceded", f"{home_avg['goals_conceded']:.2f}")
        st.metric("Win Rate", f"{home_avg['win_rate']:.1%}")
        st.metric("Clean Sheets", f"{home_avg['clean_sheets']:.1%}")
        if 'corners_for' in home_avg:
            st.metric("Corners For", f"{home_avg['corners_for']:.1f}")
            st.metric("Corners Against", f"{home_avg['corners_against']:.1f}")
    
    with col_avg2:
        st.markdown("##### 🚌 Away Games Averages")
        away_avg = league_stats['league_away_avg']
        st.metric("Goals Scored", f"{away_avg['goals_scored']:.2f}")
        st.metric("Goals Conceded", f"{away_avg['goals_conceded']:.2f}")
        st.metric("Win Rate", f"{away_avg['win_rate']:.1%}")
        st.metric("Clean Sheets", f"{away_avg['clean_sheets']:.1%}")
        if 'corners_for' in away_avg:
            st.metric("Corners For", f"{away_avg['corners_for']:.1f}")
            st.metric("Corners Against", f"{away_avg['corners_against']:.1f}")

    # Rest of the original display function...
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

    # Form Analysis
    st.markdown("#### 📈 Recent Form Analysis")
    g = stats["goals"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{home_team} Home Form**")
        home_attack = g["home_attack"].get(home_team, 1.0)
        home_defence = g["home_defence"].get(home_team, 1.0)
        st.write(f"Attack: {home_attack:.2f}× avg")
        st.write(f"Defence: {1/home_defence:.2f}× avg")
        
    with col2:
        st.write(f"**{away_team} Away Form**")
        away_attack = g["away_attack"].get(away_team, 1.0)
        away_defence = g["away_defence"].get(away_team, 1.0)
        st.write(f"Attack: {away_attack:.2f}× avg")
        st.write(f"Defence: {1/away_defence:.2f}× avg")

    if p["injury_summary"]:
        st.markdown(f"#### 🏥 Injury Impact")
        st.markdown(f"<span style='color:red'>{p['injury_summary']}</span>", unsafe_allow_html=True)

    # ===== EXPORT BUTTONS =====
    st.markdown("---")
    st.markdown("#### 📤 Export Prediction")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        # Excel Export (updated to include league stats)
        excel_data = generate_excel_export(pred, home_team, away_team, stats, league_stats)
        st.download_button(
            label="📊 Download Excel Report",
            data=excel_data,
            file_name=f"{home_team}_vs_{away_team}_prediction.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.caption("Includes league comparison tables")
    
    with col_exp2:
        # HTML Export (updated to include league stats)
        html_content = generate_html_export(pred, home_team, away_team, stats, logos, league_stats)
        st.download_button(
            label="🌐 Download HTML Report", 
            data=html_content,
            file_name=f"{home_team}_vs_{away_team}_prediction.html",
            mime="text/html",
            use_container_width=True
        )
        st.caption("Professional report with league analysis")

# ================================
# UPDATED EXPORT FUNCTIONS TO INCLUDE LEAGUE STATS
# ================================
def generate_excel_export(pred: Dict[str, Any], home_team: str, away_team: str, 
                         stats: Dict[str, Any], league_stats: Dict[str, Any] = None) -> BytesIO:
    """Enhanced Excel export with league comparison sheets"""
    # ... (previous Excel export code) ...
    # ADD NEW SHEETS FOR LEAGUE COMPARISONS
    # Sheet: "League Home Comparison"
    # Sheet: "League Away Comparison" 
    # Sheet: "League Rankings"
    # ... implementation details ...

def generate_html_export(pred: Dict[str, Any], home_team: str, away_team: str, 
                        stats: Dict[str, Any], logos: Dict[str, str], 
                        league_stats: Dict[str, Any] = None) -> str:
    """Enhanced HTML export with league comparisons"""
    # ... (previous HTML export code) ...
    # ADD LEAGUE COMPARISON SECTIONS TO HTML
    # League averages section
    # Team vs league comparison tables
    # ... implementation details ...

# ================================
# MAIN APP INTEGRATION
# ================================
# In your main app section, after computing form-based stats:

if uploaded_file is not None:
    # ... existing code ...
    
    with st.spinner(f"🔄 Analyzing last {n_games} home/away games form..."):
        team_stats = compute_form_based_stats(_df=df, home_col=col_map["HomeTeam"], away_col=col_map["AwayTeam"],
                                            hg_col=col_map["FTHG"], ag_col=col_map["FTAG"], hc_col=col_map.get("HC"), 
                                            ac_col=col_map.get("AC"), n_games=n_games)
        
        # NEW: Calculate league averages
        league_stats = calculate_league_averages(df, col_map["HomeTeam"], col_map["AwayTeam"],
                                               col_map["FTHG"], col_map["FTAG"], col_map.get("HC"), 
                                               col_map.get("AC"), n_games)

    # ... rest of main app code ...

    if st.button(f"🎯 Predict Based on Last {n_games} Games", type="primary", use_container_width=True):
        with st.spinner("Analyzing recent form and league comparisons..."):
            pred = predict_form_based_match(home_team, away_team, team_stats, injuries)
            display_form_based_predictions(pred, home_team, away_team, team_stats, league_stats)
