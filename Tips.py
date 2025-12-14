import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import poisson, skellam
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Football Analytics & Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("⚽ Advanced Football Analytics & Predictions Dashboard")
st.markdown("AI-powered predictions with corners, shots on target, and multiple statistical models")

# Sidebar configuration
st.sidebar.header("Data Configuration")

# Expanded leagues
leagues = {
    "England Premier League": "E0",
    "England Championship": "E1",
    "England League One": "E2",
    "Germany Bundesliga": "D1",
    "Spain La Liga": "SP1",
    "Italy Serie A": "I1",
    "France Ligue 1": "F1",
    "Netherlands Eredivisie": "N1",
    "Portugal Primeira Liga": "P1",
}

selected_league = st.sidebar.selectbox("Select League", list(leagues.keys()))
season = st.sidebar.text_input("Enter Season (e.g., 2425 for 2024/25)", value="2425")

# Function to fetch data from football-data.co.uk
@st.cache_data
def fetch_football_data(league_code, season_code):
    """Fetch CSV data from football-data.co.uk"""
    season_short = season_code[-4:] if len(season_code) == 6 else season_code
    
    url = f"https://www.football-data.co.uk/mmz4281/{season_short}/{league_code}.csv"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df
        else:
            st.warning(f"Status code: {response.status_code}. Data may not be available yet for this season.")
            return None
    except Exception as e:
        st.warning(f"Error: {e}")
        return None

# ============================================================================
# ADVANCED PREDICTION MODELS WITH CORNERS AND SHOTS
# ============================================================================

class AdvancedFootballPredictor:
    """Advanced football prediction models including corners and shots on target"""
    
    def __init__(self, df):
        self.df = df
        self.teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        self.team_stats = self._calculate_advanced_stats()
        self.league_stats = self._calculate_league_stats()
        
    def _calculate_league_stats(self):
        """Calculate league-wide statistics for corners and shots"""
        stats = {}
        
        # Try to find corner columns
        corner_cols = [col for col in self.df.columns if 'corner' in col.lower() or 'HC' in col or 'AC' in col]
        
        if corner_cols:
            # Usually HC = Home Corners, AC = Away Corners
            home_corner_col = next((col for col in corner_cols if 'HC' in col or 'home' in col.lower()), corner_cols[0])
            away_corner_col = next((col for col in corner_cols if 'AC' in col or 'away' in col.lower()), corner_cols[0])
            
            stats['avg_home_corners'] = self.df[home_corner_col].mean() if home_corner_col in self.df.columns else 5.0
            stats['avg_away_corners'] = self.df[away_corner_col].mean() if away_corner_col in self.df.columns else 4.0
            stats['avg_total_corners'] = stats['avg_home_corners'] + stats['avg_away_corners']
        else:
            # Default values if no corner data
            stats['avg_home_corners'] = 5.0
            stats['avg_away_corners'] = 4.0
            stats['avg_total_corners'] = 9.0
        
        # Shots data
        if 'HS' in self.df.columns and 'AS' in self.df.columns:
            stats['avg_home_shots'] = self.df['HS'].mean()
            stats['avg_away_shots'] = self.df['AS'].mean()
            stats['avg_total_shots'] = stats['avg_home_shots'] + stats['avg_away_shots']
        else:
            stats['avg_home_shots'] = 12.0
            stats['avg_away_shots'] = 10.0
            stats['avg_total_shots'] = 22.0
        
        # Shots on target (if available, otherwise estimate)
        if 'HST' in self.df.columns and 'AST' in self.df.columns:
            stats['avg_home_sot'] = self.df['HST'].mean()
            stats['avg_away_sot'] = self.df['AST'].mean()
            stats['avg_total_sot'] = stats['avg_home_sot'] + stats['avg_away_sot']
        else:
            # Estimate SOT as ~33% of total shots
            stats['avg_home_sot'] = stats['avg_home_shots'] * 0.33
            stats['avg_away_sot'] = stats['avg_away_shots'] * 0.33
            stats['avg_total_sot'] = stats['avg_home_sot'] + stats['avg_away_sot']
        
        # Goals
        stats['avg_home_goals'] = self.df['FTHG'].mean()
        stats['avg_away_goals'] = self.df['FTAG'].mean()
        stats['avg_total_goals'] = stats['avg_home_goals'] + stats['avg_away_goals']
        
        return stats
    
    def _calculate_advanced_stats(self):
        """Calculate comprehensive team statistics including corners and shots"""
        stats = {}
        
        for team in self.teams:
            # Home matches
            home_matches = self.df[self.df['HomeTeam'] == team]
            # Away matches
            away_matches = self.df[self.df['AwayTeam'] == team]
            
            # Basic stats
            home_games = len(home_matches)
            away_games = len(away_matches)
            total_games = home_games + away_games
            
            if total_games == 0:
                stats[team] = self._get_default_stats()
                continue
            
            # Goals
            home_gf = home_matches['FTHG'].sum() if not home_matches.empty else 0
            home_ga = home_matches['FTAG'].sum() if not home_matches.empty else 0
            away_gf = away_matches['FTAG'].sum() if not away_matches.empty else 0
            away_ga = away_matches['FTHG'].sum() if not away_matches.empty else 0
            
            total_gf = home_gf + away_gf
            total_ga = home_ga + away_ga
            
            # Averages
            avg_gf = total_gf / total_games if total_games > 0 else 0
            avg_ga = total_ga / total_games if total_games > 0 else 0
            
            # League averages for normalization
            league_avg_gf = (self.df['FTHG'].mean() + self.df['FTAG'].mean()) / 2
            
            # Advanced metrics
            attacking_strength = avg_gf / league_avg_gf if league_avg_gf > 0 else 1.0
            defensive_strength = avg_ga / league_avg_gf if league_avg_gf > 0 else 1.0
            
            # CORNERS statistics
            corner_stats = self._calculate_corner_stats(team, home_matches, away_matches)
            
            # SHOTS statistics
            shots_stats = self._calculate_shot_stats(team, home_matches, away_matches)
            
            # Form (last 5 games)
            form_rating = self._calculate_form_rating(team, home_matches, away_matches)
            
            # Consistency
            consistency = self._calculate_consistency(team, home_matches, away_matches)
            
            # Combine all stats
            stats[team] = {
                'attacking_strength': attacking_strength,
                'defensive_strength': defensive_strength,
                'avg_gf': avg_gf,
                'avg_ga': avg_ga,
                'form_rating': form_rating,
                'consistency': consistency,
                'home_advantage': 1.15,  # 15% home advantage
                'total_games': total_games,
                **corner_stats,  # Add corner statistics
                **shots_stats   # Add shots statistics
            }
        
        return stats
    
    def _calculate_corner_stats(self, team, home_matches, away_matches):
        """Calculate corner statistics for a team"""
        # Try to find corner columns
        corner_cols = [col for col in self.df.columns if 'corner' in col.lower() or 'HC' in col or 'AC' in col]
        
        if not corner_cols:
            return {
                'home_corners_for': 5.0,
                'home_corners_against': 4.0,
                'away_corners_for': 4.0,
                'away_corners_against': 5.0,
                'avg_corners_for': 4.5,
                'avg_corners_against': 4.5,
                'corner_attack_factor': 1.0,
                'corner_defense_factor': 1.0
            }
        
        # Usually HC = Home Corners, AC = Away Corners
        home_corner_col = next((col for col in corner_cols if 'HC' in col or 'home' in col.lower()), corner_cols[0])
        away_corner_col = next((col for col in corner_cols if 'AC' in col or 'away' in col.lower()), corner_cols[0])
        
        # Home corners (when team is home)
        home_corners_for = home_matches[home_corner_col].mean() if not home_matches.empty and home_corner_col in home_matches.columns else 5.0
        home_corners_against = home_matches[away_corner_col].mean() if not home_matches.empty and away_corner_col in home_matches.columns else 4.0
        
        # Away corners (when team is away)
        away_corners_for = away_matches[away_corner_col].mean() if not away_matches.empty and away_corner_col in away_matches.columns else 4.0
        away_corners_against = away_matches[home_corner_col].mean() if not away_matches.empty and home_corner_col in away_matches.columns else 5.0
        
        # Averages
        avg_corners_for = (home_corners_for + away_corners_for) / 2
        avg_corners_against = (home_corners_against + away_corners_against) / 2
        
        # League averages for normalization
        league_avg_corners_for = (self.df[home_corner_col].mean() + self.df[away_corner_col].mean()) / 2
        
        # Factors
        corner_attack_factor = avg_corners_for / league_avg_corners_for if league_avg_corners_for > 0 else 1.0
        corner_defense_factor = avg_corners_against / league_avg_corners_for if league_avg_corners_for > 0 else 1.0
        
        return {
            'home_corners_for': home_corners_for,
            'home_corners_against': home_corners_against,
            'away_corners_for': away_corners_for,
            'away_corners_against': away_corners_against,
            'avg_corners_for': avg_corners_for,
            'avg_corners_against': avg_corners_against,
            'corner_attack_factor': corner_attack_factor,
            'corner_defense_factor': corner_defense_factor
        }
    
    def _calculate_shot_stats(self, team, home_matches, away_matches):
        """Calculate shot statistics for a team"""
        stats = {}
        
        # Shots
        if 'HS' in self.df.columns and 'AS' in self.df.columns:
            home_shots_for = home_matches['HS'].mean() if not home_matches.empty else 12.0
            home_shots_against = home_matches['AS'].mean() if not home_matches.empty else 10.0
            away_shots_for = away_matches['AS'].mean() if not away_matches.empty else 10.0
            away_shots_against = away_matches['HS'].mean() if not away_matches.empty else 12.0
            
            avg_shots_for = (home_shots_for + away_shots_for) / 2
            avg_shots_against = (home_shots_against + away_shots_against) / 2
            
            league_avg_shots_for = (self.df['HS'].mean() + self.df['AS'].mean()) / 2
            
            stats.update({
                'home_shots_for': home_shots_for,
                'home_shots_against': home_shots_against,
                'away_shots_for': away_shots_for,
                'away_shots_against': away_shots_against,
                'avg_shots_for': avg_shots_for,
                'avg_shots_against': avg_shots_against,
                'shot_attack_factor': avg_shots_for / league_avg_shots_for if league_avg_shots_for > 0 else 1.0,
                'shot_defense_factor': avg_shots_against / league_avg_shots_for if league_avg_shots_for > 0 else 1.0
            })
        
        # Shots on target
        if 'HST' in self.df.columns and 'AST' in self.df.columns:
            home_sot_for = home_matches['HST'].mean() if not home_matches.empty else 4.0
            home_sot_against = home_matches['AST'].mean() if not home_matches.empty else 3.5
            away_sot_for = away_matches['AST'].mean() if not away_matches.empty else 3.5
            away_sot_against = away_matches['HST'].mean() if not away_matches.empty else 4.0
            
            avg_sot_for = (home_sot_for + away_sot_for) / 2
            avg_sot_against = (home_sot_against + away_sot_against) / 2
            
            league_avg_sot_for = (self.df['HST'].mean() + self.df['AST'].mean()) / 2
            
            stats.update({
                'home_sot_for': home_sot_for,
                'home_sot_against': home_sot_against,
                'away_sot_for': away_sot_for,
                'away_sot_against': away_sot_against,
                'avg_sot_for': avg_sot_for,
                'avg_sot_against': avg_sot_against,
                'sot_attack_factor': avg_sot_for / league_avg_sot_for if league_avg_sot_for > 0 else 1.0,
                'sot_defense_factor': avg_sot_against / league_avg_sot_for if league_avg_sot_for > 0 else 1.0
            })
        else:
            # Estimate SOT if data not available
            stats.update({
                'avg_sot_for': 3.8,
                'avg_sot_against': 3.8,
                'sot_attack_factor': 1.0,
                'sot_defense_factor': 1.0
            })
        
        return stats
    
    def _calculate_form_rating(self, team, home_matches, away_matches):
        """Calculate form rating based on last 5 games"""
        # Get last 5 matches
        last_5_home = home_matches.tail(5) if len(home_matches) >= 5 else home_matches
        last_5_away = away_matches.tail(5) if len(away_matches) >= 5 else away_matches
        
        form_points = 0
        form_games = 0
        
        for _, match in pd.concat([last_5_home, last_5_away]).iterrows():
            if match['HomeTeam'] == team:
                if match['FTR'] == 'H':
                    form_points += 3
                elif match['FTR'] == 'D':
                    form_points += 1
            else:
                if match['FTR'] == 'A':
                    form_points += 3
                elif match['FTR'] == 'D':
                    form_points += 1
            form_games += 1
        
        return form_points / (form_games * 3) if form_games > 0 else 0.5
    
    def _calculate_consistency(self, team, home_matches, away_matches):
        """Calculate performance consistency"""
        all_goals_scored = []
        for _, match in home_matches.iterrows():
            all_goals_scored.append(match['FTHG'])
        for _, match in away_matches.iterrows():
            all_goals_scored.append(match['FTAG'])
        
        if len(all_goals_scored) > 1:
            std_dev = np.std(all_goals_scored)
            # Convert to consistency score (higher = more consistent)
            consistency = 1 / (1 + std_dev)
        else:
            consistency = 0.7
        
        return consistency
    
    def _get_default_stats(self):
        """Return default stats for teams with no data"""
        return {
            'attacking_strength': 1.0,
            'defensive_strength': 1.0,
            'avg_gf': 1.5,
            'avg_ga': 1.5,
            'form_rating': 0.5,
            'consistency': 0.7,
            'home_advantage': 1.15,
            'total_games': 0,
            'corner_attack_factor': 1.0,
            'corner_defense_factor': 1.0,
            'shot_attack_factor': 1.0,
            'shot_defense_factor': 1.0,
            'sot_attack_factor': 1.0,
            'sot_defense_factor': 1.0,
            'avg_corners_for': 4.5,
            'avg_corners_against': 4.5,
            'avg_shots_for': 11.0,
            'avg_shots_against': 11.0,
            'avg_sot_for': 3.8,
            'avg_sot_against': 3.8
        }
    
    # ============================================================================
    # PREDICT CORNERS
    # ============================================================================
    def predict_corners(self, home_team, away_team):
        """Predict corners for a match"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None
        
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        
        # League averages
        league_avg_home_corners = self.league_stats['avg_home_corners']
        league_avg_away_corners = self.league_stats['avg_away_corners']
        
        # Base prediction using team factors
        home_corners = (league_avg_home_corners * 
                       home_stats['corner_attack_factor'] * 
                       away_stats['corner_defense_factor'] * 
                       home_stats['home_advantage'])
        
        away_corners = (league_avg_away_corners * 
                       away_stats['corner_attack_factor'] * 
                       home_stats['corner_defense_factor'])
        
        # Apply form adjustment
        form_adjustment = 0.1
        home_corners *= (1 + (home_stats['form_rating'] - 0.5) * form_adjustment)
        away_corners *= (1 + (away_stats['form_rating'] - 0.5) * form_adjustment)
        
        # Add randomness (football is unpredictable!)
        home_corners += np.random.uniform(-0.5, 0.5)
        away_corners += np.random.uniform(-0.5, 0.5)
        
        # Ensure minimum values
        home_corners = max(home_corners, 1.0)
        away_corners = max(away_corners, 1.0)
        
        total_corners = home_corners + away_corners
        
        # Calculate ranges
        home_range = (max(1, int(home_corners - 1.5)), int(home_corners + 1.5))
        away_range = (max(1, int(away_corners - 1.5)), int(away_corners + 1.5))
        total_range = (max(2, int(total_corners - 2.5)), int(total_corners + 2.5))
        
        return {
            'home_corners': round(home_corners, 1),
            'away_corners': round(away_corners, 1),
            'total_corners': round(total_corners, 1),
            'home_range': home_range,
            'away_range': away_range,
            'total_range': total_range,
            'home_corner_factor': home_stats['corner_attack_factor'],
            'away_corner_factor': away_stats['corner_attack_factor']
        }
    
    # ============================================================================
    # PREDICT SHOTS ON TARGET
    # ============================================================================
    def predict_shots_on_target(self, home_team, away_team):
        """Predict shots on target for a match"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None
        
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        
        # League averages
        league_avg_home_sot = self.league_stats['avg_home_sot']
        league_avg_away_sot = self.league_stats['avg_away_sot']
        
        # Base prediction
        home_sot = (league_avg_home_sot * 
                   home_stats['sot_attack_factor'] * 
                   away_stats['sot_defense_factor'] * 
                   home_stats['home_advantage'])
        
        away_sot = (league_avg_away_sot * 
                   away_stats['sot_attack_factor'] * 
                   home_stats['sot_defense_factor'])
        
        # Apply form adjustment
        form_adjustment = 0.15
        home_sot *= (1 + (home_stats['form_rating'] - 0.5) * form_adjustment)
        away_sot *= (1 + (away_stats['form_rating'] - 0.5) * form_adjustment)
        
        # Add randomness
        home_sot += np.random.uniform(-0.3, 0.3)
        away_sot += np.random.uniform(-0.3, 0.3)
        
        # Ensure minimum values
        home_sot = max(home_sot, 0.5)
        away_sot = max(away_sot, 0.5)
        
        total_sot = home_sot + away_sot
        
        # Calculate ranges
        home_range = (max(0, int(home_sot - 1.0)), int(home_sot + 1.0))
        away_range = (max(0, int(away_sot - 1.0)), int(away_sot + 1.0))
        
        return {
            'home_sot': round(home_sot, 1),
            'away_sot': round(away_sot, 1),
            'total_sot': round(total_sot, 1),
            'home_range': home_range,
            'away_range': away_range,
            'home_sot_factor': home_stats['sot_attack_factor'],
            'away_sot_factor': away_stats['sot_attack_factor']
        }
    
    # ============================================================================
    # PREDICT SHOTS (TOTAL)
    # ============================================================================
    def predict_total_shots(self, home_team, away_team):
        """Predict total shots for a match"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None
        
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        
        # League averages
        league_avg_home_shots = self.league_stats['avg_home_shots']
        league_avg_away_shots = self.league_stats['avg_away_shots']
        
        # Base prediction
        home_shots = (league_avg_home_shots * 
                     home_stats['shot_attack_factor'] * 
                     away_stats['shot_defense_factor'] * 
                     home_stats['home_advantage'])
        
        away_shots = (league_avg_away_shots * 
                     away_stats['shot_attack_factor'] * 
                     home_stats['shot_defense_factor'])
        
        # Apply form adjustment
        form_adjustment = 0.1
        home_shots *= (1 + (home_stats['form_rating'] - 0.5) * form_adjustment)
        away_shots *= (1 + (away_stats['form_rating'] - 0.5) * form_adjustment)
        
        # Add randomness
        home_shots += np.random.uniform(-1.0, 1.0)
        away_shots += np.random.uniform(-1.0, 1.0)
        
        # Ensure minimum values
        home_shots = max(home_shots, 3.0)
        away_shots = max(away_shots, 3.0)
        
        total_shots = home_shots + away_shots
        
        # Calculate SOT conversion (typically 30-35%)
        sot_conversion_rate = 0.33
        
        return {
            'home_shots': round(home_shots, 1),
            'away_shots': round(away_shots, 1),
            'total_shots': round(total_shots, 1),
            'sot_conversion_rate': sot_conversion_rate,
            'estimated_home_sot': round(home_shots * sot_conversion_rate, 1),
            'estimated_away_sot': round(away_shots * sot_conversion_rate, 1)
        }
    
    # ============================================================================
    # ENHANCED POISSON MODEL FOR GOALS
    # ============================================================================
    def predict_enhanced_poisson(self, home_team, away_team):
        """Enhanced Poisson model with form and consistency adjustments"""
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None
        
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        
        # League averages
        league_avg_home = self.df['FTHG'].mean()
        league_avg_away = self.df['FTAG'].mean()
        
        # Base expected goals
        base_home_xg = (league_avg_home * home_stats['attacking_strength'] / 
                        away_stats['defensive_strength']) * home_stats['home_advantage']
        base_away_xg = (league_avg_away * away_stats['attacking_strength'] / 
                       home_stats['defensive_strength'])
        
        # Apply form adjustment
        form_adjustment = 0.15
        home_form_factor = 1 + (home_stats['form_rating'] - 0.5) * form_adjustment
        away_form_factor = 1 + (away_stats['form_rating'] - 0.5) * form_adjustment
        
        # Apply consistency adjustment
        consistency_adjustment = 0.1
        home_consistency_factor = home_stats['consistency']
        away_consistency_factor = away_stats['consistency']
        
        # Final expected goals
        home_xg = base_home_xg * home_form_factor * home_consistency_factor
        away_xg = base_away_xg * away_form_factor * away_consistency_factor
        
        # Ensure minimum values
        home_xg = max(home_xg, 0.1)
        away_xg = max(away_xg, 0.1)
        
        return self._calculate_poisson_probabilities(home_xg, away_xg, home_team, away_team)
    
    def _calculate_poisson_probabilities(self, home_xg, away_xg, home_team, away_team):
        """Calculate Poisson probabilities for all scorelines"""
        max_goals = 7
        
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        scorelines = {}
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                scorelines[f"{i}-{j}"] = prob
                
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        # Get top 5 most likely scorelines
        top_scorelines = dict(sorted(scorelines.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Calculate confidence
        max_prob = max(home_win_prob, draw_prob, away_win_prob)
        confidence = max_prob * 100
        
        # Determine predicted winner
        if home_win_prob > away_win_prob and home_win_prob > draw_prob:
            predicted_winner = home_team
        elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
            predicted_winner = away_team
        else:
            predicted_winner = "Draw"
        
        return {
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'top_scorelines': top_scorelines,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'model': 'Enhanced Poisson'
        }
    
    # ============================================================================
    # COMPREHENSIVE MATCH PREDICTION
    # ============================================================================
    def predict_match_comprehensive(self, home_team, away_team):
        """Comprehensive match prediction with all statistics"""
        
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None
        
        # Get goal prediction
        goal_prediction = self.predict_enhanced_poisson(home_team, away_team)
        
        if not goal_prediction:
            return None
        
        # Get corner prediction
        corner_prediction = self.predict_corners(home_team, away_team)
        
        # Get shots on target prediction
        sot_prediction = self.predict_shots_on_target(home_team, away_team)
        
        # Get total shots prediction
        shots_prediction = self.predict_total_shots(home_team, away_team)
        
        # Combine all predictions
        comprehensive_prediction = {
            **goal_prediction,
            'home_team': home_team,
            'away_team': away_team,
            'home_stats': self.team_stats[home_team],
            'away_stats': self.team_stats[away_team]
        }
        
        # Add corner predictions if available
        if corner_prediction:
            comprehensive_prediction.update({
                'home_corners': corner_prediction['home_corners'],
                'away_corners': corner_prediction['away_corners'],
                'total_corners': corner_prediction['total_corners'],
                'home_corners_range': corner_prediction['home_range'],
                'away_corners_range': corner_prediction['away_range'],
                'total_corners_range': corner_prediction['total_range']
            })
        
        # Add shots on target predictions if available
        if sot_prediction:
            comprehensive_prediction.update({
                'home_sot': sot_prediction['home_sot'],
                'away_sot': sot_prediction['away_sot'],
                'total_sot': sot_prediction['total_sot'],
                'home_sot_range': sot_prediction['home_range'],
                'away_sot_range': sot_prediction['away_range']
            })
        
        # Add total shots predictions if available
        if shots_prediction:
            comprehensive_prediction.update({
                'home_shots': shots_prediction['home_shots'],
                'away_shots': shots_prediction['away_shots'],
                'total_shots': shots_prediction['total_shots'],
                'sot_conversion_rate': shots_prediction['sot_conversion_rate']
            })
        
        return comprehensive_prediction

# ============================================================================
# STREAMLIT APP
# ============================================================================

# Load data button
if st.sidebar.button("Load Data", type="primary"):
    league_code = leagues[selected_league]
    df = fetch_football_data(league_code, season)
    
    if df is not None:
        st.session_state.df = df
        
        # Initialize advanced predictor
        st.session_state.predictor = AdvancedFootballPredictor(df)
        
        st.success(f"✅ Data loaded successfully for {selected_league} ({season})")
    else:
        st.warning("Could not load data. Please check the season code.")

# Main dashboard
if 'df' in st.session_state:
    df = st.session_state.df
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎯 Match Predictions", "📈 Team Analysis"])
    
    with tab1:
        st.subheader("📈 League Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_matches = len(df.dropna(subset=['FTR']))
            st.metric("Total Matches", total_matches)
        
        with col2:
            avg_goals = df[['FTHG', 'FTAG']].sum().sum() / total_matches if total_matches > 0 else 0
            st.metric("Avg Goals/Match", f"{avg_goals:.2f}")
        
        with col3:
            home_wins = (df['FTR'] == 'H').sum()
            st.metric("Home Wins", f"{home_wins} ({100*home_wins/total_matches:.1f}%)")
        
        with col4:
            away_wins = (df['FTR'] == 'A').sum()
            st.metric("Away Wins", f"{away_wins} ({100*away_wins/total_matches:.1f}%)")
        
        # League statistics
        if 'predictor' in st.session_state:
            predictor = st.session_state.predictor
            league_stats = predictor.league_stats
            
            st.subheader("📊 League Averages")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Home Corners", f"{league_stats['avg_home_corners']:.1f}")
            
            with col2:
                st.metric("Avg Away Corners", f"{league_stats['avg_away_corners']:.1f}")
            
            with col3:
                st.metric("Avg Home SOT", f"{league_stats['avg_home_sot']:.1f}")
            
            with col4:
                st.metric("Avg Away SOT", f"{league_stats['avg_away_sot']:.1f}")
    
    with tab2:
        st.subheader("🎯 Comprehensive Match Prediction")
        st.info("Advanced predictions including corners, shots on target, and expected goals")
        
        if 'predictor' not in st.session_state:
            st.warning("Please load data first using the 'Load Data' button in the sidebar.")
        else:
            predictor = st.session_state.predictor
            teams = predictor.teams
            
            # Team selection
            col1, col2 = st.columns(2)
            
            with col1:
                home_team = st.selectbox("Select Home Team", teams, key="pred_home")
            
            with col2:
                away_options = [t for t in teams if t != home_team]
                away_team = st.selectbox("Select Away Team", away_options, key="pred_away")
            
            if home_team and away_team:
                # Get comprehensive prediction
                with st.spinner("Calculating predictions..."):
                    prediction = predictor.predict_match_comprehensive(home_team, away_team)
                
                if prediction:
                    # Display match header
                    st.markdown("---")
                    col1, col2, col3 = st.columns([3, 1, 3])
                    with col1:
                        st.markdown(f"### 🏠 {home_team}")
                    with col2:
                        st.markdown("### vs")
                    with col3:
                        st.markdown(f"### 🚌 {away_team}")
                    
                    # Row 1: Match Outcome
                    st.markdown("### 📊 Match Outcome Probabilities")
                    
                    prob_col1, prob_col2, prob_col3 = st.columns(3)
                    
                    with prob_col1:
                        home_prob = prediction['home_win_prob'] * 100
                        st.metric(f"{home_team} Win", f"{home_prob:.1f}%")
                    
                    with prob_col2:
                        draw_prob = prediction['draw_prob'] * 100
                        st.metric("Draw", f"{draw_prob:.1f}%")
                    
                    with prob_col3:
                        away_prob = prediction['away_win_prob'] * 100
                        st.metric(f"{away_team} Win", f"{away_prob:.1f}%")
                    
                    # Final prediction
                    pred_winner = prediction['predicted_winner']
                    confidence = prediction['confidence']
                    
                    if pred_winner == "Draw":
                        st.success(f"**🎯 Prediction: MATCH LIKELY TO END IN A DRAW** ({confidence:.1f}% confidence)")
                    else:
                        st.success(f"**🎯 Prediction: {pred_winner} TO WIN** ({confidence:.1f}% confidence)")
                    
                    # Row 2: Expected Goals
                    st.markdown("### 🥅 Expected Goals (xG)")
                    
                    xg_col1, xg_col2, xg_col3 = st.columns(3)
                    
                    with xg_col1:
                        st.metric(f"{home_team} xG", f"{prediction['home_xg']:.2f}")
                    
                    with xg_col2:
                        total_xg = prediction['home_xg'] + prediction['away_xg']
                        st.metric("Total xG", f"{total_xg:.2f}")
                    
                    with xg_col3:
                        st.metric(f"{away_team} xG", f"{prediction['away_xg']:.2f}")
                    
                    # Row 3: CORNERS Prediction
                    st.markdown("### 🎯 Corners Prediction")
                    
                    if 'home_corners' in prediction:
                        corner_col1, corner_col2, corner_col3, corner_col4 = st.columns(4)
                        
                        with corner_col1:
                            st.metric(
                                f"{home_team} Corners",
                                f"{prediction['home_corners']:.1f}",
                                f"Range: {prediction['home_corners_range'][0]}-{prediction['home_corners_range'][1]}"
                            )
                        
                        with corner_col2:
                            st.metric(
                                f"{away_team} Corners",
                                f"{prediction['away_corners']:.1f}",
                                f"Range: {prediction['away_corners_range'][0]}-{prediction['away_corners_range'][1]}"
                            )
                        
                        with corner_col3:
                            st.metric(
                                "Total Corners",
                                f"{prediction['total_corners']:.1f}",
                                f"Range: {prediction['total_corners_range'][0]}-{prediction['total_corners_range'][1]}"
                            )
                        
                        with corner_col4:
                            # Corner advantage
                            corner_diff = prediction['home_corners'] - prediction['away_corners']
                            diff_label = "Home Favored" if corner_diff > 0 else "Away Favored" if corner_diff < 0 else "Even"
                            st.metric(
                                "Corner Advantage",
                                f"{abs(corner_diff):.1f}",
                                diff_label
                            )
                    
                    # Row 4: SHOTS ON TARGET Prediction
                    st.markdown("### 🎯 Shots on Target Prediction")
                    
                    if 'home_sot' in prediction:
                        sot_col1, sot_col2, sot_col3, sot_col4 = st.columns(4)
                        
                        with sot_col1:
                            st.metric(
                                f"{home_team} SOT",
                                f"{prediction['home_sot']:.1f}",
                                f"Range: {prediction['home_sot_range'][0]}-{prediction['home_sot_range'][1]}"
                            )
                        
                        with sot_col2:
                            st.metric(
                                f"{away_team} SOT",
                                f"{prediction['away_sot']:.1f}",
                                f"Range: {prediction['away_sot_range'][0]}-{prediction['away_sot_range'][1]}"
                            )
                        
                        with sot_col3:
                            st.metric(
                                "Total SOT",
                                f"{prediction['total_sot']:.1f}"
                            )
                        
                        with sot_col4:
                            # SOT advantage
                            sot_diff = prediction['home_sot'] - prediction['away_sot']
                            diff_label = "Home Favored" if sot_diff > 0 else "Away Favored" if sot_diff < 0 else "Even"
                            st.metric(
                                "SOT Advantage",
                                f"{abs(sot_diff):.1f}",
                                diff_label
                            )
                    
                    # Row 5: TOTAL SHOTS Prediction
                    st.markdown("### 🎯 Total Shots Prediction")
                    
                    if 'home_shots' in prediction:
                        shots_col1, shots_col2, shots_col3 = st.columns(3)
                        
                        with shots_col1:
                            st.metric(f"{home_team} Shots", f"{prediction['home_shots']:.1f}")
                        
                        with shots_col2:
                            st.metric(f"{away_team} Shots", f"{prediction['away_shots']:.1f}")
                        
                        with shots_col3:
                            st.metric("Total Shots", f"{prediction['total_shots']:.1f}")
                        
                        # SOT conversion info
                        if 'sot_conversion_rate' in prediction:
                            st.info(f"**SOT Conversion Rate:** ~{prediction['sot_conversion_rate']*100:.0f}% of shots become shots on target")
                    
                    # Row 6: Most Likely Scorelines
                    st.markdown("### 📋 Most Likely Scorelines")
                    
                    if 'top_scorelines' in prediction and prediction['top_scorelines']:
                        scoreline_data = []
                        for score, prob in prediction['top_scorelines'].items():
                            scoreline_data.append({
                                'Score': score,
                                'Probability': f"{prob*100:.2f}%"
                            })
                        
                        scoreline_df = pd.DataFrame(scoreline_data)
                        st.dataframe(scoreline_df, use_container_width=True, hide_index=True)
                    
                    # Visualizations
                    with st.expander("📊 View Detailed Charts"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Corners comparison chart
                            if 'home_corners' in prediction:
                                corners_data = pd.DataFrame({
                                    'Team': [home_team, away_team],
                                    'Predicted Corners': [prediction['home_corners'], prediction['away_corners']]
                                })
                                
                                fig_corners = px.bar(
                                    corners_data,
                                    x='Team',
                                    y='Predicted Corners',
                                    title="Predicted Corners",
                                    color='Team',
                                    color_discrete_sequence=['blue', 'red']
                                )
                                st.plotly_chart(fig_corners, use_container_width=True)
                        
                        with col2:
                            # Shots on target comparison chart
                            if 'home_sot' in prediction:
                                sot_data = pd.DataFrame({
                                    'Team': [home_team, away_team],
                                    'Predicted SOT': [prediction['home_sot'], prediction['away_sot']]
                                })
                                
                                fig_sot = px.bar(
                                    sot_data,
                                    x='Team',
                                    y='Predicted SOT',
                                    title="Predicted Shots on Target",
                                    color='Team',
                                    color_discrete_sequence=['blue', 'red']
                                )
                                st.plotly_chart(fig_sot, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Team Analysis & Statistics")
        
        if 'predictor' not in st.session_state:
            st.warning("Please load data first using the 'Load Data' button in the sidebar.")
        else:
            predictor = st.session_state.predictor
            teams = predictor.teams
            
            # Team selection for analysis
            selected_team = st.selectbox("Select Team for Analysis", teams, key="team_analysis")
            
            if selected_team and selected_team in predictor.team_stats:
                team_stats = predictor.team_stats[selected_team]
                
                st.markdown(f"### 📊 {selected_team} Statistics")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Attacking Strength", f"{team_stats['attacking_strength']:.2f}")
                
                with col2:
                    st.metric("Defensive Strength", f"{team_stats['defensive_strength']:.2f}")
                
                with col3:
                    st.metric("Form Rating", f"{team_stats['form_rating']:.2f}")
                
                with col4:
                    st.metric("Consistency", f"{team_stats['consistency']:.2f}")
                
                # Corner statistics
                st.markdown("#### 🎯 Corner Statistics")
                corner_col1, corner_col2, corner_col3 = st.columns(3)
                
                with corner_col1:
                    st.metric("Avg Corners For", f"{team_stats.get('avg_corners_for', 0):.1f}")
                
                with corner_col2:
                    st.metric("Avg Corners Against", f"{team_stats.get('avg_corners_against', 0):.1f}")
                
                with corner_col3:
                    corner_factor = team_stats.get('corner_attack_factor', 1.0)
                    factor_label = "Above Avg" if corner_factor > 1.0 else "Below Avg" if corner_factor < 1.0 else "Average"
                    st.metric("Corner Attack Factor", f"{corner_factor:.2f}", factor_label)
                
                # Shot statistics
                st.markdown("#### 🎯 Shot Statistics")
                shot_col1, shot_col2, shot_col3 = st.columns(3)
                
                with shot_col1:
                    st.metric("Avg Shots For", f"{team_stats.get('avg_shots_for', 0):.1f}")
                
                with shot_col2:
                    sot = team_stats.get('avg_sot_for', 0)
                    st.metric("Avg Shots on Target", f"{sot:.1f}")
                
                with shot_col3:
                    sot_factor = team_stats.get('sot_attack_factor', 1.0)
                    factor_label = "Above Avg" if sot_factor > 1.0 else "Below Avg" if sot_factor < 1.0 else "Average"
                    st.metric("SOT Attack Factor", f"{sot_factor:.2f}", factor_label)

else:
    st.info("👈 Select a league and season, then click 'Load Data' to begin analysis")
