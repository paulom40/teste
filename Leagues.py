# app.py - COMPLETE VERSION WITH CSV UPLOAD
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
# LEAGUE COMPARISON FUNCTIONS
# ================================
def calculate_league_averages(df: pd.DataFrame, home_col: str, away_col: str, 
                            hg_col: str, ag_col: str, hc_col: str = None, ac_col: str = None,
                            hs_col: str = None, as_col: str = None, n_games: int = 5) -> Dict[str, Any]:
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
        home_shot_factor = home_team_stats['shots_for'] / league_home_avg['shots_for']
        away_defense_factor = away_team_stats.get('shots_against', league_away_avg.get('shots_against', 1)) / league_away_avg.get('shots_against', 1)
        predicted_home_shots = league_home_avg['shots_for'] * home_shot_factor * (2 - away_defense_factor) / 2
        predictions['home_shots'] = max(round(predicted_home_shots), 1)
    
    # Predict home team shots CONCEDED (defensive)
    if home_team_stats.get('shots_against') is not None and league_home_avg.get('shots_against') is not None:
        home_defense_factor = home_team_stats['shots_against'] / league_home_avg['shots_against']
        away_attack_factor = away_team_stats.get('shots_for', league_away_avg.get('shots_for', 1)) / league_away_avg.get('shots_for', 1)
        predicted_home_conceded = league_home_avg['shots_against'] * home_defense_factor * away_attack_factor
        predictions['home_shots_conceded'] = max(round(predicted_home_conceded), 1)
    
    # Predict away team shots FOR (attacking)
    if away_team_stats.get('shots_for') is not None and league_away_avg.get('shots_for') is not None:
        away_shot_factor = away_team_stats['shots_for'] / league_away_avg['shots_for']
        home_defense_factor = home_team_stats.get('shots_against', league_home_avg.get('shots_against', 1)) / league_home_avg.get('shots_against', 1)
        predicted_away_shots = league_away_avg['shots_for'] * away_shot_factor * (2 - home_defense_factor) / 2
        predictions['away_shots'] = max(round(predicted_away_shots), 1)
    
    # Predict away team shots CONCEDED (defensive)
    if away_team_stats.get('shots_against') is not None and league_away_avg.get('shots_against') is not None:
        away_defense_factor = away_team_stats['shots_against'] / league_away_avg['shots_against']
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
                shot_predictions['home_shots_conceded'],
                shot_predictions['away_shots_conceded'],
                shot_predictions['home_shot_efficiency'],
                shot_predictions['away_shot_efficiency']
            )
            predictions["shot_probabilities"] = shot_probs

    return {"predictions": predictions}

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
# MAIN APP WITH CSV UPLOAD
# ================================
st.sidebar.header("📁 Upload Match Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"], 
                                        help="Upload your football match data CSV file")

if uploaded_file is not None:
    df = load_csv(uploaded_file.read())
    if df.empty:
        st.error("Empty CSV file.")
    else:
        st.success(f"✅ Loaded {len(df):,} matches")
        
        # Show preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head(8))
            st.write(f"**Teams in dataset**: {len(set(df['HomeTeam'].dropna()) | set(df['AwayTeam'].dropna()))}")
        
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

        # Form analysis settings
        st.sidebar.subheader("⚙️ Form Analysis Settings")
        n_games = st.sidebar.slider("Number of games for form analysis", 3, 10, 5,
                                   help="Analyze last N home/away games for each team")
        require_dates = st.sidebar.toggle("Require date column", value=True,
                                         help="Date column needed for accurate recent form")

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

        # Injury Input
        st.sidebar.subheader("🏥 Current Injuries")
        injury_input = st.sidebar.text_area("Injured Players", 
                                          placeholder="Example:\nArsenal: Saka (role:forward, impact:15%)\nChelsea: James (role:defender, impact:20%)",
                                          height=100,
                                          help="Enter one injury per line in the format shown")
        injuries = parse_injuries(injury_input)

        # Prediction Section
        st.markdown("---")
        st.subheader("🔮 Form-Based Match Prediction")
        
        col1, col2 = st.columns(2)
        home_team = col1.selectbox("Home Team", teams, key="home_select")
        away_team = col2.selectbox("Away Team", teams, key="away_select")

        if st.button(f"🎯 Predict Based on Last {n_games} Games", type="primary", use_container_width=True):
            with st.spinner("Analyzing recent form, shots, and league comparisons..."):
                # Pass league_stats to prediction function
                pred = predict_form_based_match(home_team, away_team, team_stats, injuries, league_stats)
                display_form_based_predictions(pred, home_team, away_team, team_stats, league_stats)

else:
    st.info("📁 Please upload a CSV file to get started")
    
    with st.expander("💡 CSV Format Guide"):
        st.markdown("""
        **Required Columns:**
        - Home Team (e.g., 'HomeTeam', 'Home')
        - Away Team (e.g., 'AwayTeam', 'Away')  
        - Home Goals (e.g., 'FTHG', 'HG', 'HomeGoals')
        - Away Goals (e.g., 'FTAG', 'AG', 'AwayGoals')
        
        **Optional Columns:**
        - Home Corners, Away Corners ('HC', 'AC')
        - Home Shots, Away Shots ('HS', 'AS') - for shot predictions
        - Date (for accurate form analysis)
        
        **Example CSV structure:**
        ```
        Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC,HS,AS
        2025-01-15,Arsenal,Chelsea,2,1,6,4,15,10
        2025-01-14,Man Utd,Liverpool,1,1,5,7,12,16
        2025-01-13,Man City,Tottenham,3,0,8,2,18,6
        ```
        """)

# Season info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Form Analysis")
st.sidebar.info(f"Using last {n_games if 'n_games' in locals() else 5} home/away games for predictions")
