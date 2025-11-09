# app.py - COMPLETE WORKING VERSION WITH SHOT PREDICTIONS
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
- **League Average Comparisons**  
- **Excel & HTML Export**  
""")

# ================================
# CORE FUNCTIONS
# ================================
@st.cache_data(ttl=3600)
def get_team_logo(team_name: str) -> str:
    team_clean = team_name.strip().lower().replace(" ", "_").replace(".", "").replace("'", "")
    replacements = {
        "man_utd": "Manchester_United_F.C.", "man_city": "Manchester_City_F.C.",
        "arsenal": "Arsenal_F.C.", "chelsea": "Chelsea_F.C.", "liverpool": "Liverpool_F.C.",
        "nottm_forest": "Nottingham_Forest_F.C.", "leeds": "Leeds_United_F.C.",
        "spurs": "Tottenham_Hotspur_F.C.", "newcastle": "Newcastle_United_F.C.",
        "brighton": "Brighton_&_Hove_Albion_F.C.", "west_ham": "West_Ham_United_F.C."
    }
    wiki_name = replacements.get(team_clean, team_name.replace(" ", "_") + "_F.C.")
    url = f"https://en.wikipedia.org/wiki/File:{wiki_name}_logo.svg"
    try:
        if requests.head(url, timeout=5).status_code == 200:
            return url
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def load_image(url: str):
    try:
        response = requests.get(url, timeout=10)
        return Image.open(BytesIO(response.content)).convert("RGBA")
    except:
        return None

@st.cache_data(show_spinner="Loading CSV...")
def load_csv(uploaded_file_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="utf-8")
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        return df
    except:
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="latin1")

@st.cache_data(show_spinner=False)
def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "")
        if "home" in lower and "team" in lower: mapping["HomeTeam"] = col
        elif "away" in lower and "team" in lower: mapping["AwayTeam"] = col
        elif lower in ["fthg", "hgoals"]: mapping["FTHG"] = col
        elif lower in ["ftag", "agoals"]: mapping["FTAG"] = col
        elif lower in ["hc", "homecorners"]: mapping["HC"] = col
        elif lower in ["ac", "awaycorners"]: mapping["AC"] = col
        elif lower in ["hs", "homeshotsontarget"]: mapping["HS"] = col
        elif lower in ["as", "awayshotsontarget"]: mapping["AS"] = col
        elif lower in ["hxg", "home_xg"]: mapping["HxG"] = col
        elif lower in ["axg", "away_xg"]: mapping["AxG"] = col
        elif "date" in lower: mapping["Date"] = col
    return mapping

def parse_injuries(injury_str: str) -> Dict[str, Dict[str, float]]:
    injuries = {}
    if not injury_str.strip(): return injuries
    for line in injury_str.split('\n'):
        match = re.match(r'(\w+):\s*(\w+)\s*\(role:(\w+),\s*impact:(\d+)%\)', line.strip())
        if match:
            team, player, role, impact = match.groups()
            impact = float(impact) / 100
            if team not in injuries: injuries[team] = {}
            injuries[team][player] = {"role": role, "impact": impact}
    return injuries

def get_last_n_home_games(df: pd.DataFrame, team: str, home_col: str, n: int = 5) -> pd.DataFrame:
    home_games = df[df[home_col] == team].copy()
    if 'Date' in home_games.columns:
        home_games = home_games.sort_values('Date', ascending=False)
    return home_games.head(n)

def get_last_n_away_games(df: pd.DataFrame, team: str, away_col: str, n: int = 5) -> pd.DataFrame:
    away_games = df[df[away_col] == team].copy()
    if 'Date' in away_games.columns:
        away_games = away_games.sort_values('Date', ascending=False)
    return away_games.head(n)

# ================================
# LEAGUE COMPARISON FUNCTIONS WITH SHOTS
# ================================
def calculate_league_averages(df: pd.DataFrame, home_col: str, away_col: str, 
                            hg_col: str, ag_col: str, hc_col: str = None, ac_col: str = None,
                            hs_col: str = None, as_col: str = None, n_games: int = 5) -> Dict[str, Any]:
    """
    Calculate league averages for last N home/away games for all teams including shots
    """
    teams = sorted(set(df[home_col]).union(df[away_col]))
    
    league_stats = {
        'team_home_stats': {},
        'team_away_stats': {}, 
        'league_home_avg': {},
        'league_away_avg': {},
        'overall_league_avg': {}
    }
    
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
                'clean_sheets': (home_games[ag_col] == 0).mean(),
                'shots_for': home_games[hs_col].mean() if hs_col and hs_col in home_games.columns else None,
                'shots_against': home_games[as_col].mean() if as_col and as_col in home_games.columns else None,
            }
            
            # Calculate shot efficiency if shots data available
            if hs_col and hs_col in home_games.columns and home_games[hs_col].sum() > 0:
                home_stats['shot_efficiency'] = (home_games[hg_col] / home_games[hs_col]).mean()
            
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
                'clean_sheets': (away_games[hg_col] == 0).mean(),
                'shots_for': away_games[as_col].mean() if as_col and as_col in away_games.columns else None,
                'shots_against': away_games[hs_col].mean() if hs_col and hs_col in away_games.columns else None,
            }
            
            # Calculate shot efficiency if shots data available
            if as_col and as_col in away_games.columns and away_games[as_col].sum() > 0:
                away_stats['shot_efficiency'] = (away_games[ag_col] / away_games[as_col]).mean()
            
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
            'clean_sheets': home_df['clean_sheets'].mean(),
        }
        
        if 'corners_for' in home_df.columns:
            league_stats['league_home_avg']['corners_for'] = home_df['corners_for'].mean()
            league_stats['league_home_avg']['corners_against'] = home_df['corners_against'].mean()
        
        if 'shots_for' in home_df.columns and home_df['shots_for'].notna().any():
            league_stats['league_home_avg']['shots_for'] = home_df['shots_for'].mean()
            league_stats['league_home_avg']['shots_against'] = home_df['shots_against'].mean()
            if 'shot_efficiency' in home_df.columns:
                league_stats['league_home_avg']['shot_efficiency'] = home_df['shot_efficiency'].mean()
    
    if away_stats_list:
        away_df = pd.DataFrame(away_stats_list)
        league_stats['league_away_avg'] = {
            'goals_scored': away_df['goals_scored'].mean(),
            'goals_conceded': away_df['goals_conceded'].mean(),
            'goal_difference': away_df['goal_difference'].mean(),
            'win_rate': away_df['win_rate'].mean(),
            'clean_sheets': away_df['clean_sheets'].mean(),
        }
        
        if 'corners_for' in away_df.columns:
            league_stats['league_away_avg']['corners_for'] = away_df['corners_for'].mean()
            league_stats['league_away_avg']['corners_against'] = away_df['corners_against'].mean()
        
        if 'shots_for' in away_df.columns and away_df['shots_for'].notna().any():
            league_stats['league_away_avg']['shots_for'] = away_df['shots_for'].mean()
            league_stats['league_away_avg']['shots_against'] = away_df['shots_against'].mean()
            if 'shot_efficiency' in away_df.columns:
                league_stats['league_away_avg']['shot_efficiency'] = away_df['shot_efficiency'].mean()
    
    return league_stats

def create_comparison_tables(league_stats: Dict[str, Any], home_team: str, away_team: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create comparison tables showing team performance vs league averages including shots
    """
    home_comparison = []
    away_comparison = []
    
    home_team_stats = league_stats['team_home_stats'].get(home_team, {})
    away_team_stats = league_stats['team_away_stats'].get(away_team, {})
    league_home_avg = league_stats['league_home_avg']
    league_away_avg = league_stats['league_away_avg']
    
    # Home team comparison table
    if home_team_stats and league_home_avg:
        home_comparison = [
            {
                'Metric': 'Goals Scored',
                'Team': f"{home_team_stats['goals_scored']:.2f}",
                'League Avg': f"{league_home_avg['goals_scored']:.2f}",
                'Difference': f"{home_team_stats['goals_scored'] - league_home_avg['goals_scored']:+.2f}",
                'Percentage': f"{(home_team_stats['goals_scored'] / league_home_avg['goals_scored'] - 1) * 100:+.1f}%"
            },
            {
                'Metric': 'Goals Conceded',
                'Team': f"{home_team_stats['goals_conceded']:.2f}",
                'League Avg': f"{league_home_avg['goals_conceded']:.2f}",
                'Difference': f"{home_team_stats['goals_conceded'] - league_home_avg['goals_conceded']:+.2f}",
                'Percentage': f"{(home_team_stats['goals_conceded'] / league_home_avg['goals_conceded'] - 1) * 100:+.1f}%"
            }
        ]
        
        # Add shots data if available
        if home_team_stats.get('shots_for') is not None and league_home_avg.get('shots_for') is not None:
            home_comparison.extend([
                {
                    'Metric': 'Shots For',
                    'Team': f"{home_team_stats['shots_for']:.1f}",
                    'League Avg': f"{league_home_avg['shots_for']:.1f}",
                    'Difference': f"{home_team_stats['shots_for'] - league_home_avg['shots_for']:+.1f}",
                    'Percentage': f"{(home_team_stats['shots_for'] / league_home_avg['shots_for'] - 1) * 100:+.1f}%"
                },
                {
                    'Metric': 'Shots Against',
                    'Team': f"{home_team_stats['shots_against']:.1f}",
                    'League Avg': f"{league_home_avg['shots_against']:.1f}",
                    'Difference': f"{home_team_stats['shots_against'] - league_home_avg['shots_against']:+.1f}",
                    'Percentage': f"{(home_team_stats['shots_against'] / league_home_avg['shots_against'] - 1) * 100:+.1f}%"
                }
            ])
            
            if home_team_stats.get('shot_efficiency') is not None and league_home_avg.get('shot_efficiency') is not None:
                home_comparison.append({
                    'Metric': 'Shot Efficiency',
                    'Team': f"{home_team_stats['shot_efficiency']:.1%}",
                    'League Avg': f"{league_home_avg['shot_efficiency']:.1%}",
                    'Difference': f"{(home_team_stats['shot_efficiency'] - league_home_avg['shot_efficiency']) * 100:+.1f}%",
                    'Percentage': f"{(home_team_stats['shot_efficiency'] / league_home_avg['shot_efficiency'] - 1) * 100:+.1f}%"
                })
        
        home_comparison.extend([
            {
                'Metric': 'Win Rate',
                'Team': f"{home_team_stats['win_rate']:.1%}",
                'League Avg': f"{league_home_avg['win_rate']:.1%}",
                'Difference': f"{(home_team_stats['win_rate'] - league_home_avg['win_rate']) * 100:+.1f}%",
                'Percentage': f"{(home_team_stats['win_rate'] / league_home_avg['win_rate'] - 1) * 100:+.1f}%"
            }
        ])
    
    # Away team comparison table
    if away_team_stats and league_away_avg:
        away_comparison = [
            {
                'Metric': 'Goals Scored',
                'Team': f"{away_team_stats['goals_scored']:.2f}",
                'League Avg': f"{league_away_avg['goals_scored']:.2f}",
                'Difference': f"{away_team_stats['goals_scored'] - league_away_avg['goals_scored']:+.2f}",
                'Percentage': f"{(away_team_stats['goals_scored'] / league_away_avg['goals_scored'] - 1) * 100:+.1f}%"
            },
            {
                'Metric': 'Goals Conceded',
                'Team': f"{away_team_stats['goals_conceded']:.2f}",
                'League Avg': f"{league_away_avg['goals_conceded']:.2f}",
                'Difference': f"{away_team_stats['goals_conceded'] - league_away_avg['goals_conceded']:+.2f}",
                'Percentage': f"{(away_team_stats['goals_conceded'] / league_away_avg['goals_conceded'] - 1) * 100:+.1f}%"
            }
        ]
        
        # Add shots data if available
        if away_team_stats.get('shots_for') is not None and league_away_avg.get('shots_for') is not None:
            away_comparison.extend([
                {
                    'Metric': 'Shots For',
                    'Team': f"{away_team_stats['shots_for']:.1f}",
                    'League Avg': f"{league_away_avg['shots_for']:.1f}",
                    'Difference': f"{away_team_stats['shots_for'] - league_away_avg['shots_for']:+.1f}",
                    'Percentage': f"{(away_team_stats['shots_for'] / league_away_avg['shots_for'] - 1) * 100:+.1f}%"
                },
                {
                    'Metric': 'Shots Against',
                    'Team': f"{away_team_stats['shots_against']:.1f}",
                    'League Avg': f"{league_away_avg['shots_against']:.1f}",
                    'Difference': f"{away_team_stats['shots_against'] - league_away_avg['shots_against']:+.1f}",
                    'Percentage': f"{(away_team_stats['shots_against'] / league_away_avg['shots_against'] - 1) * 100:+.1f}%"
                }
            ])
            
            if away_team_stats.get('shot_efficiency') is not None and league_away_avg.get('shot_efficiency') is not None:
                away_comparison.append({
                    'Metric': 'Shot Efficiency',
                    'Team': f"{away_team_stats['shot_efficiency']:.1%}",
                    'League Avg': f"{league_away_avg['shot_efficiency']:.1%}",
                    'Difference': f"{(away_team_stats['shot_efficiency'] - league_away_avg['shot_efficiency']) * 100:+.1f}%",
                    'Percentage': f"{(away_team_stats['shot_efficiency'] / league_away_avg['shot_efficiency'] - 1) * 100:+.1f}%"
                })
        
        away_comparison.extend([
            {
                'Metric': 'Win Rate',
                'Team': f"{away_team_stats['win_rate']:.1%}",
                'League Avg': f"{league_away_avg['win_rate']:.1%}",
                'Difference': f"{(away_team_stats['win_rate'] - league_away_avg['win_rate']) * 100:+.1f}%",
                'Percentage': f"{(away_team_stats['win_rate'] / league_away_avg['win_rate'] - 1) * 100:+.1f}%"
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
            'Goals Scored': f"{stats['goals_scored']:.2f}",
            'Goals Conceded': f"{stats['goals_conceded']:.2f}",
            'Goal Difference': f"{stats['goal_difference']:+.2f}",
            'Win Rate': f"{stats['win_rate']:.1%}",
            'Clean Sheets': f"{stats['clean_sheets']:.1%}"
        })
    
    # Away performance rankings
    for team, stats in league_stats['team_away_stats'].items():
        away_rankings.append({
            'Team': team,
            'Games': stats['games_played'],
            'Goals Scored': f"{stats['goals_scored']:.2f}",
            'Goals Conceded': f"{stats['goals_conceded']:.2f}",
            'Goal Difference': f"{stats['goal_difference']:+.2f}",
            'Win Rate': f"{stats['win_rate']:.1%}",
            'Clean Sheets': f"{stats['clean_sheets']:.1%}"
        })
    
    # Convert to DataFrames and sort
    home_df = pd.DataFrame(home_rankings)
    away_df = pd.DataFrame(away_rankings)
    
    if not home_df.empty:
        home_df = home_df.sort_values('Goal Difference', ascending=False)
    if not away_df.empty:
        away_df = away_df.sort_values('Goal Difference', ascending=False)
    
    return home_df, away_df

# ================================
# SHOT PREDICTION FUNCTIONS
# ================================
def predict_shots(home_team: str, away_team: str, stats: Dict[str, Any], league_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict shots for both teams based on recent form and league averages
    """
    predictions = {
        'home_shots': 0,
        'away_shots': 0,
        'total_shots': 0,
        'home_shot_efficiency': 0.0,
        'away_shot_efficiency': 0.0
    }
    
    # Get team stats
    home_team_stats = league_stats['team_home_stats'].get(home_team, {})
    away_team_stats = league_stats['team_away_stats'].get(away_team, {})
    league_home_avg = league_stats.get('league_home_avg', {})
    league_away_avg = league_stats.get('league_away_avg', {})
    
    # Predict home team shots
    if home_team_stats.get('shots_for') is not None and league_home_avg.get('shots_for') is not None:
        # Use team's home shooting tendency vs league average
        home_shot_factor = home_team_stats['shots_for'] / league_home_avg['shots_for']
        # Consider opponent's away defensive shooting tendency
        away_defense_factor = away_team_stats.get('shots_against', league_away_avg.get('shots_against', 1)) / league_away_avg.get('shots_against', 1)
        
        predicted_home_shots = league_home_avg['shots_for'] * home_shot_factor * (2 - away_defense_factor) / 2
        predictions['home_shots'] = max(round(predicted_home_shots), 1)
    
    # Predict away team shots  
    if away_team_stats.get('shots_for') is not None and league_away_avg.get('shots_for') is not None:
        # Use team's away shooting tendency vs league average
        away_shot_factor = away_team_stats['shots_for'] / league_away_avg['shots_for']
        # Consider opponent's home defensive shooting tendency
        home_defense_factor = home_team_stats.get('shots_against', league_home_avg.get('shots_against', 1)) / league_home_avg.get('shots_against', 1)
        
        predicted_away_shots = league_away_avg['shots_for'] * away_shot_factor * (2 - home_defense_factor) / 2
        predictions['away_shots'] = max(round(predicted_away_shots), 1)
    
    predictions['total_shots'] = predictions['home_shots'] + predictions['away_shots']
    
    # Predict shot efficiency
    if home_team_stats.get('shot_efficiency') is not None:
        predictions['home_shot_efficiency'] = home_team_stats['shot_efficiency']
    if away_team_stats.get('shot_efficiency') is not None:
        predictions['away_shot_efficiency'] = away_team_stats['shot_efficiency']
    
    return predictions

def calculate_shot_probabilities(home_shots: int, away_shots: int, home_efficiency: float, away_efficiency: float) -> Dict[str, Any]:
    """
    Calculate probabilities for shot-related markets
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
    
    return {
        'home_expected_goals_from_shots': home_expected_goals,
        'away_expected_goals_from_shots': away_expected_goals,
        'home_most_likely_shots': home_most_likely,
        'away_most_likely_shots': away_most_likely,
        'total_expected_goals': home_expected_goals + away_expected_goals,
        'shot_advantage': home_shots - away_shots,
        'both_teams_5_plus_shots_prob': (1 - poisson.cdf(4, home_shots)) * (1 - poisson.cdf(4, away_shots)),
        'over_25_total_shots_prob': 1 - poisson.cdf(25, home_shots + away_shots)
    }

# ================================
# PREDICTION FUNCTIONS
# ================================
def calculate_team_form(df: pd.DataFrame, home_col: str, away_col: str, hg_col: str, ag_col: str, 
                       teams: List[str], n_games: int = 5) -> Dict[str, Any]:
    form_stats = {
        'home_attack': {}, 'home_defence': {}, 'away_attack': {}, 'away_defence': {},
        'home_games_used': {}, 'away_games_used': {}
    }
    
    recent_home_goals = []
    recent_away_goals = []
    
    for team in teams:
        home_games = get_last_n_home_games(df, team, home_col, n_games)
        form_stats['home_games_used'][team] = len(home_games)
        
        if len(home_games) > 0:
            home_goals_scored = home_games[hg_col].mean()
            home_goals_conceded = home_games[ag_col].mean()
            recent_home_goals.extend(home_games[hg_col].tolist())
            form_stats['home_attack'][team] = home_goals_scored
            form_stats['home_defence'][team] = home_goals_conceded
        else:
            form_stats['home_attack'][team] = 1.0
            form_stats['home_defence'][team] = 1.0
        
        away_games = get_last_n_away_games(df, team, away_col, n_games)
        form_stats['away_games_used'][team] = len(away_games)
        
        if len(away_games) > 0:
            away_goals_scored = away_games[ag_col].mean()
            away_goals_conceded = away_games[hg_col].mean()
            recent_away_goals.extend(away_games[ag_col].tolist())
            form_stats['away_attack'][team] = away_goals_scored
            form_stats['away_defence'][team] = away_goals_conceded
        else:
            form_stats['away_attack'][team] = 1.0
            form_stats['away_defence'][team] = 1.0
    
    form_stats['league_avg_home'] = np.mean(recent_home_goals) if recent_home_goals else 1.5
    form_stats['league_avg_away'] = np.mean(recent_away_goals) if recent_away_goals else 1.2
    
    return form_stats

@st.cache_data(show_spinner="Analyzing last 5 games form...")
def compute_form_based_stats(_df: pd.DataFrame, home_col: str, away_col: str, hg_col: str, ag_col: str,
                           hc_col=None, ac_col=None, hs_col=None, as_col=None, n_games: int = 5) -> Dict[str, Any]:
    df = _df.copy()
    for col in [hg_col, ag_col, hc_col, ac_col, hs_col, as_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    stats = {}
    teams = sorted(set(df[home_col]).union(df[away_col]))
    form_stats = calculate_team_form(df, home_col, away_col, hg_col, ag_col, teams, n_games)
    
    home_attack = {}; home_defence = {}; away_attack = {}; away_defence = {}
    
    for team in teams:
        home_attack[team] = form_stats['home_attack'][team] / form_stats['league_avg_home']
        home_defence[team] = form_stats['home_defence'][team] / form_stats['league_avg_away']
        away_attack[team] = form_stats['away_attack'][team] / form_stats['league_avg_away']
        away_defence[team] = form_stats['away_defence'][team] / form_stats['league_avg_home']

    stats["goals"] = {
        "league_avg_home": form_stats['league_avg_home'], "league_avg_away": form_stats['league_avg_away'],
        "home_attack": home_attack, "away_attack": away_attack, "home_defence": home_defence, "away_defence": away_defence,
        "games_used": form_stats['home_games_used'], "away_games_used": form_stats['away_games_used']
    }

    if hc_col and ac_col and hc_col in df.columns and ac_col in df.columns:
        corner_stats = calculate_team_form(df, home_col, away_col, hc_col, ac_col, teams, n_games)
        stats["corners"] = {
            "league_avg_home": corner_stats['league_avg_home'], "league_avg_away": corner_stats['league_avg_away'],
            "home_attack": {t: corner_stats['home_attack'][t] / corner_stats['league_avg_home'] for t in teams},
            "away_attack": {t: corner_stats['away_attack'][t] / corner_stats['league_avg_away'] for t in teams},
            "home_defence": {t: corner_stats['home_defence'][t] / corner_stats['league_avg_away'] for t in teams},
            "away_defence": {t: corner_stats['away_defence'][t] / corner_stats['league_avg_home'] for t in teams}
        }
    else:
        stats["corners"] = {
            "league_avg_home": 5.5, "league_avg_away": 4.8,
            "home_attack": {t: 1.0 for t in teams}, "away_attack": {t: 1.0 for t in teams},
            "home_defence": {t: 1.0 for t in teams}, "away_defence": {t: 1.0 for t in teams},
        }

    return stats

def apply_injury_adjustment(stats: Dict[str, Any], injuries: Dict[str, Dict[str, float]]) -> str:
    summary = ""
    for team, players in injuries.items():
        attack_reduction = defence_reduction = 0
        for p, data in players.items():
            if data["role"] in ["forward", "midfielder", "winger", "striker"]:
                attack_reduction += data["impact"]
            elif data["role"] in ["defender", "goalkeeper"]:
                defence_reduction += data["impact"]
        attack_reduction = min(attack_reduction, 0.3); defence_reduction = min(defence_reduction, 0.3)
        if attack_reduction > 0: summary += f"{team} Attack -{attack_reduction*100:.0f}% | "
        if defence_reduction > 0: summary += f"{team} Defence -{defence_reduction*100:.0f}% | "
    return summary.strip(" | ")

@st.cache_data(show_spinner=False)
def predict_form_based_match(home: str, away: str, stats: Dict[str, Any], injuries: Dict = None, league_stats: Dict[str, Any] = None) -> Dict[str, Any]:
    injury_summary = apply_injury_adjustment(stats, injuries) if injuries else ""

    predictions = {
        "goals": {"score": "N/A", "home_win": 0, "draw": 0, "away_win": 0, "btts_yes": 0, "over_25": 0},
        "xg": {"home": 0.0, "away": 0.0}, 
        "corners": {"home": 0, "away": 0, "total": 0},
        "shots": {"home": 0, "away": 0, "total": 0, "home_efficiency": 0.0, "away_efficiency": 0.0},
        "shot_probabilities": {},
        "form_based": True, 
        "injury_summary": injury_summary,
        "games_used": {"home": stats["goals"]["games_used"].get(home, 0), "away": stats["goals"]["away_games_used"].get(away, 0)}
    }

    # Original goals prediction
    g = stats.get("goals", {})
    if g:
        l_home = g["league_avg_home"]; l_away = g["league_avg_away"]
        att_h = g["home_attack"].get(home, 1.0); def_a = g["away_defence"].get(away, 1.0)
        att_a = g["away_attack"].get(away, 1.0); def_h = g["home_defence"].get(home, 1.0)
        lambda_h = att_h * def_a * l_home; lambda_a = att_a * def_h * l_away

        max_g = 8
        prob_matrix = np.zeros((max_g + 1, max_g + 1))
        for h in range(max_g + 1):
            for a in range(max_g + 1):
                p = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
                prob_matrix[h, a] = p
        prob_matrix /= prob_matrix.sum()
        
        h_idx, a_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
        predictions["goals"]["score"] = f"{h_idx}–{a_idx}"
        predictions["goals"]["home_win"] = (prob_matrix[1:, :].sum() - prob_matrix.diagonal()[1:].sum())
        predictions["goals"]["away_win"] = (prob_matrix[:, 1:].sum() - prob_matrix.diagonal()[1:].sum())
        predictions["goals"]["draw"] = prob_matrix.diagonal().sum()
        predictions["goals"]["btts_yes"] = (prob_matrix[1:, 1:]).sum()
        predictions["goals"]["over_25"] = (prob_matrix[3:, :].sum() + prob_matrix[:, 3:].sum() - prob_matrix[3:, 3:].sum())
        predictions["xg"]["home"] = max(round(lambda_h, 2), 0.1)
        predictions["xg"]["away"] = max(round(lambda_a, 2), 0.1)

    # Corners prediction
    c = stats.get("corners")
    if c:
        mu_hc = c["home_attack"].get(home, 1.0) * c["away_defence"].get(away, 1.0) * c["league_avg_home"]
        mu_ac = c["away_attack"].get(away, 1.0) * c["home_defence"].get(home, 1.0) * c["league_avg_away"]
        predictions["corners"]["home"] = max(int(np.round(mu_hc)), 1)
        predictions["corners"]["away"] = max(int(np.round(mu_ac)), 1)
        predictions["corners"]["total"] = predictions["corners"]["home"] + predictions["corners"]["away"]

    # Shot predictions if league stats available
    if league_stats:
        shot_predictions = predict_shots(home, away, stats, league_stats)
        predictions["shots"].update(shot_predictions)
        
        # Calculate shot probabilities
        if shot_predictions['home_shots'] > 0 and shot_predictions['away_shots'] > 0:
            shot_probs = calculate_shot_probabilities(
                shot_predictions['home_shots'], 
                shot_predictions['away_shots'],
                shot_predictions['home_shot_efficiency'],
                shot_predictions['away_shot_efficiency']
            )
            predictions["shot_probabilities"] = shot_probs

    return {"predictions": predictions}

# ================================
# ENHANCED DISPLAY WITH SHOT PREDICTIONS
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

    # ===== SHOT PREDICTIONS SECTION =====
    st.markdown("---")
    st.markdown("#### 🎯 Shot Predictions")
    
    if p['shots']['home_shots'] > 0 and p['shots']['away_shots'] > 0:
        col_shot1, col_shot2, col_shot3 = st.columns(3)
        
        with col_shot1:
            st.metric(f"{home_team} Expected Shots", f"{p['shots']['home_shots']}")
            st.metric("Shot Efficiency", f"{p['shots']['home_shot_efficiency']:.1%}")
            if p['shot_probabilities']:
                st.metric("Expected Goals from Shots", f"{p['shot_probabilities']['home_expected_goals_from_shots']:.2f}")
        
        with col_shot2:
            st.metric(f"{away_team} Expected Shots", f"{p['shots']['away_shots']}")
            st.metric("Shot Efficiency", f"{p['shots']['away_shot_efficiency']:.1%}")
            if p['shot_probabilities']:
                st.metric("Expected Goals from Shots", f"{p['shot_probabilities']['away_expected_goals_from_shots']:.2f}")
        
        with col_shot3:
            st.metric("Total Expected Shots", f"{p['shots']['total_shots']}")
            st.metric("Shot Advantage", f"{p['shots']['home_shots'] - p['shots']['away_shots']:+.0f}")
            if p['shot_probabilities']:
                st.metric("Both Teams 5+ Shots", f"{p['shot_probabilities']['both_teams_5_plus_shots_prob']:.1%}")
        
        # Shot probability insights
        if p['shot_probabilities']:
            st.markdown("##### 📊 Shot Market Insights")
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.metric("Most Likely Home Shots", f"{p['shot_probabilities']['home_most_likely_shots']}")
                st.metric("Most Likely Away Shots", f"{p['shot_probabilities']['away_most_likely_shots']}")
            
            with col_insight2:
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
    
    # League rankings - FIXED HIGHLIGHTING
    st.markdown("#### 📈 League Rankings (Last 5 Games)")
    
    home_rankings_df, away_rankings_df = create_league_ranking_tables(league_stats)
    
    col_rank1, col_rank2 = st.columns(2)
    
    with col_rank1:
        if not home_rankings_df.empty:
            st.markdown("##### 🏠 Home Performance Rankings")
            
            # FIXED: Proper highlighting with dark text
            def highlight_home_team(row):
                if row['Team'] == home_team:
                    return ['background-color: #ffeb3b; color: #000000; font-weight: bold'] * len(row)
                return [''] * len(row)
            
            styled_home_df = home_rankings_df.style.apply(highlight_home_team, axis=1)
            st.dataframe(styled_home_df, use_container_width=True, height=400)
    
    with col_rank2:
        if not away_rankings_df.empty:
            st.markdown("##### 🚌 Away Performance Rankings")
            
            # FIXED: Proper highlighting with dark text
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
            st.metric("Clean Sheets", f"{home_avg['clean_sheets']:.1%}")
            if 'shots_for' in home_avg:
                st.metric("Shots For", f"{home_avg['shots_for']:.1f}")
    
    with col_avg2:
        st.markdown("##### 🚌 Away Games Averages")
        if league_stats.get('league_away_avg'):
            away_avg = league_stats['league_away_avg']
            st.metric("Goals Scored", f"{away_avg['goals_scored']:.2f}")
            st.metric("Goals Conceded", f"{away_avg['goals_conceded']:.2f}")
            st.metric("Win Rate", f"{away_avg['win_rate']:.1%}")
            st.metric("Clean Sheets", f"{away_avg['clean_sheets']:.1%}")
            if 'shots_for' in away_avg:
                st.metric("Shots For", f"{away_avg['shots_for']:.1f}")

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

# ================================
# MAIN APP
# ================================
st.sidebar.header("📁 Upload Match Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"])

if uploaded_file is not None:
    df = load_csv(uploaded_file.read())
    if df.empty:
        st.error("Empty CSV file.")
    else:
        st.success(f"✅ Loaded {len(df):,} matches")
        
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head(8))
        
        mapping = detect_columns(df)
        
        st.sidebar.subheader("🔧 Column Mapping")
        col_map = {}
        # ADD SHOT COLUMNS TO MAPPING
        for label in ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC", "HS", "AS", "Date"]:
            detected = mapping.get(label)
            options = [""] + list(df.columns)
            default_idx = options.index(detected) if detected in options else 0
            col_map[label] = st.sidebar.selectbox(f"**{label}**", options=options, index=default_idx)

        missing = [r for r in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"] if not col_map[r]]
        if missing:
            st.error(f"❌ Map required fields: {', '.join(missing)}")
            st.stop()

        st.sidebar.subheader("⚙️ Form Analysis Settings")
        n_games = st.sidebar.slider("Number of games for form analysis", 3, 10, 5)
        require_dates = st.sidebar.toggle("Require date column", value=True)

        if require_dates and not col_map.get("Date"):
            st.warning("⚠️ Date column not mapped. Form analysis may be less accurate.")
            df = df.sort_index(ascending=False)

        with st.spinner(f"🔄 Analyzing last {n_games} home/away games form..."):
            team_stats = compute_form_based_stats(_df=df, home_col=col_map["HomeTeam"], away_col=col_map["AwayTeam"],
                                                hg_col=col_map["FTHG"], ag_col=col_map["FTAG"], hc_col=col_map.get("HC"), 
                                                ac_col=col_map.get("AC"), hs_col=col_map.get("HS"), as_col=col_map.get("AS"),
                                                n_games=n_games)
            
            # Calculate league averages WITH SHOTS
            league_stats = calculate_league_averages(df, col_map["HomeTeam"], col_map["AwayTeam"],
                                                   col_map["FTHG"], col_map["FTAG"], col_map.get("HC"), 
                                                   col_map.get("AC"), col_map.get("HS"), col_map.get("AS"), n_games)

        teams = sorted(set(df[col_map["HomeTeam"]]).union(df[col_map["AwayTeam"]]))

        st.sidebar.subheader("🏥 Current Injuries")
        injury_input = st.sidebar.text_area("Injured Players", placeholder="Arsenal: Saka (role:forward, impact:15%)", height=100)
        injuries = parse_injuries(injury_input)

        st.markdown("---")
        st.subheader("🔮 Form-Based Match Prediction")
        
        col1, col2 = st.columns(2)
        home_team = col1.selectbox("Home Team", teams, key="home_select")
        away_team = col2.selectbox("Away Team", teams, key="away_select")

        if st.button(f"🎯 Predict Based on Last {n_games} Games", type="primary", use_container_width=True):
            with st.spinner("Analyzing recent form, shots, and league comparisons..."):
                # UPDATE: Pass league_stats to prediction function
                pred = predict_form_based_match(home_team, away_team, team_stats, injuries, league_stats)
                display_form_based_predictions(pred, home_team, away_team, team_stats, league_stats)

else:
    st.info("📁 Please upload CSV data to get started")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Form Analysis")
st.sidebar.info(f"Using last {n_games if 'n_games' in locals() else 5} home/away games")
