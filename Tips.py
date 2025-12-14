import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import poisson, skellam
import warnings
warnings.filterwarnings('ignore')

try:
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Football Analytics 2025/26",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("⚽ Football Analytics 2025/26 Season")
st.markdown("Predictions based exclusively on 2025/26 season data for maximum accuracy")

# Sidebar configuration
st.sidebar.header("Data Configuration - 2025/26 Season ONLY")

# Leagues for 2025/26 season
leagues = {
    "England Premier League 2025/26": "E0",
    "England Championship 2025/26": "E1", 
    "England League One 2025/26": "E2",
    "England League Two 2025/26": "E3",
    "Germany Bundesliga 2025/26": "D1",
    "Spain La Liga 2025/26": "SP1",
    "Italy Serie A 2025/26": "I1",
    "France Ligue 1 2025/26": "F1",
    "Netherlands Eredivisie 2025/26": "N1",
    "Portugal Primeira Liga 2025/26": "P1",
}

selected_league = st.sidebar.selectbox("Select League", list(leagues.keys()))
season = "2526"

# ============================================================================
# LIVE FIXTURES DATA
# ============================================================================

def get_live_fixtures():
    """Fetch today's fixtures from API-Football or similar source"""
    try:
        # Using a free football API endpoint (RapidAPI alternative)
        # You can replace with your own API key from api-football.com
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Simulated fixtures for demonstration
        # In production, integrate with: https://www.api-football.com/
        fixtures = {
            'fixtures': [
                {
                    'id': 1,
                    'homeTeam': 'Manchester City',
                    'awayTeam': 'Liverpool',
                    'league': 'England Premier League 2025/26',
                    'date': datetime.now(),
                    'time': '17:30',
                    'status': 'scheduled'
                },
                {
                    'id': 2,
                    'homeTeam': 'Arsenal',
                    'awayTeam': 'Chelsea',
                    'league': 'England Premier League 2025/26',
                    'date': datetime.now(),
                    'time': '20:00',
                    'status': 'scheduled'
                },
                {
                    'id': 3,
                    'homeTeam': 'Barcelona',
                    'awayTeam': 'Real Madrid',
                    'league': 'Spain La Liga 2025/26',
                    'date': datetime.now(),
                    'time': '21:00',
                    'status': 'scheduled'
                },
                {
                    'id': 4,
                    'homeTeam': 'Bayern Munich',
                    'awayTeam': 'Borussia Dortmund',
                    'league': 'Germany Bundesliga 2025/26',
                    'date': datetime.now(),
                    'time': '18:30',
                    'status': 'scheduled'
                },
            ]
        }
        return fixtures
    except Exception as e:
        st.warning(f"Could not fetch live fixtures: {e}")
        return {'fixtures': []}

# Function to fetch CURRENT 2025/26 data from football-data.co.uk
@st.cache_data(ttl=3600)
def fetch_current_season_data(league_code):
    """Fetch ONLY 2025/26 season data"""
    url = f"https://www.football-data.co.uk/mmz4281/2526/{league_code}.csv"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            st.sidebar.success(f"✅ 2025/26 data loaded: {len(df)} matches")
            return df
        else:
            alt_url = f"https://www.football-data.co.uk/mmz4281/25-26/{league_code}.csv"
            response = requests.get(alt_url, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                st.sidebar.success(f"✅ 2025/26 data loaded: {len(df)} matches")
                return df
            else:
                st.sidebar.warning(f"⚠️ Status {response.status_code}: 2025/26 data not available yet")
                st.sidebar.info("Using simulated 2025/26 data for demonstration")
                return create_simulated_2025_data(league_code)
    except Exception as e:
        st.sidebar.warning(f"Error: {e}")
        st.sidebar.info("Using simulated 2025/26 data for demonstration")
        return create_simulated_2025_data(league_code)

def create_simulated_2025_data(league_code):
    """Create simulated 2025/26 season data when real data isn't available"""
    today = datetime.now()
    start_date = datetime(2025, 8, 1)
    end_date = datetime(2025, 12, 31) if today > datetime(2025, 12, 31) else today
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    if 'E0' in league_code:
        teams = [
            'Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 'Tottenham',
            'Manchester United', 'Newcastle', 'Aston Villa', 'West Ham', 'Brighton',
            'Crystal Palace', 'Brentford', 'Fulham', 'Wolves', 'Everton',
            'Nottingham Forest', 'Leicester', 'Southampton', 'Leeds', 'Ipswich'
        ]
    elif 'D1' in league_code:
        teams = ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 
                'Eintracht Frankfurt', 'Wolfsburg', 'Borussia Mönchengladbach', 'Stuttgart',
                'Hoffenheim', 'Mainz', 'Augsburg', 'Hertha Berlin', 'Bochum', 'Schalke',
                'Werder Bremen', 'FC Köln', 'Freiburg', 'Union Berlin']
    elif 'SP1' in league_code:
        teams = ['Barcelona', 'Real Madrid', 'Atletico Madrid', 'Sevilla', 'Real Sociedad',
                'Villarreal', 'Athletic Bilbao', 'Valencia', 'Real Betis', 'Osasuna',
                'Celta Vigo', 'Getafe', 'Girona', 'Mallorca', 'Rayo Vallecano',
                'Almeria', 'Cadiz', 'Valladolid', 'Espanyol', 'Elche']
    else:
        teams = [f'Team {i+1}' for i in range(20)]
    
    matches = []
    match_id = 1
    
    for date in dates:
        if np.random.random() < 0.3:
            num_matches = np.random.randint(2, 6)
            selected_teams = np.random.choice(teams, size=num_matches*2, replace=False)
            
            for i in range(0, len(selected_teams), 2):
                home_team = selected_teams[i]
                away_team = selected_teams[i+1]
                
                home_strength = teams.index(home_team) / len(teams)
                away_strength = teams.index(away_team) / len(teams)
                home_advantage = 0.3
                
                home_xg = 1.2 + (home_strength * 0.8) - (away_strength * 0.4) + home_advantage
                away_xg = 0.8 + (away_strength * 0.8) - (home_strength * 0.4)
                
                home_goals = np.random.poisson(home_xg)
                away_goals = np.random.poisson(away_xg)
                
                ftr = 'H' if home_goals > away_goals else 'A' if home_goals < away_goals else 'D'
                
                home_shots = np.random.randint(8, 20)
                away_shots = np.random.randint(6, 18)
                home_sot = max(0, int(home_shots * np.random.uniform(0.25, 0.4)))
                away_sot = max(0, int(away_shots * np.random.uniform(0.25, 0.4)))
                home_corners = np.random.randint(3, 10)
                away_corners = np.random.randint(2, 8)
                
                matches.append({
                    'Date': date.strftime('%d/%m/%Y'),
                    'HomeTeam': home_team,
                    'AwayTeam': away_team,
                    'FTHG': home_goals,
                    'FTAG': away_goals,
                    'FTR': ftr,
                    'HS': home_shots,
                    'AS': away_shots,
                    'HST': home_sot,
                    'AST': away_sot,
                    'HC': home_corners,
                    'AC': away_corners,
                    'MatchID': f'2025_{match_id:04d}'
                })
                match_id += 1
    
    df = pd.DataFrame(matches)
    return df

# ============================================================================
# 2025/26 SEASON PREDICTOR
# ============================================================================

class CurrentSeasonPredictor:
    """Prediction model using ONLY 2025/26 season data"""
    
    def __init__(self, df):
        self.df = df
        self.season = "2025/26"
        self.teams = self._extract_current_teams()
        self.team_stats = self._calculate_current_season_stats()
        self.league_stats = self._calculate_league_stats()
        
    def _extract_current_teams(self):
        """Extract teams playing in the 2025/26 season"""
        home_teams = set(self.df['HomeTeam'].unique())
        away_teams = set(self.df['AwayTeam'].unique())
        current_teams = sorted(home_teams.union(away_teams))
        
        valid_teams = []
        for team in current_teams:
            matches = len(self.df[(self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)])
            if matches >= 3:
                valid_teams.append(team)
        
        return valid_teams
    
    def _calculate_current_season_stats(self):
        """Calculate statistics from 2025/26 season only"""
        stats = {}
        
        for team in self.teams:
            home_matches = self.df[self.df['HomeTeam'] == team]
            away_matches = self.df[self.df['AwayTeam'] == team]
            
            total_matches = len(home_matches) + len(away_matches)
            
            if total_matches == 0:
                continue
            
            home_gf = home_matches['FTHG'].sum() if not home_matches.empty else 0
            home_ga = home_matches['FTAG'].sum() if not home_matches.empty else 0
            away_gf = away_matches['FTAG'].sum() if not away_matches.empty else 0
            away_ga = away_matches['FTHG'].sum() if not away_matches.empty else 0
            
            total_gf = home_gf + away_gf
            total_ga = home_ga + away_ga
            
            avg_gf = total_gf / total_matches
            avg_ga = total_ga / total_matches
            
            league_avg_gf = (self.df['FTHG'].mean() + self.df['FTAG'].mean()) / 2
            
            attacking_strength = avg_gf / league_avg_gf if league_avg_gf > 0 else 1.0
            defensive_strength = avg_ga / league_avg_gf if league_avg_gf > 0 else 1.0
            
            form_rating = self._calculate_current_form(team, home_matches, away_matches)
            corner_stats = self._calculate_current_corner_stats(team, home_matches, away_matches)
            shot_stats = self._calculate_current_shot_stats(team, home_matches, away_matches)
            
            stats[team] = {
                'season': '2025/26',
                'matches_played': total_matches,
                'attacking_strength': attacking_strength,
                'defensive_strength': defensive_strength,
                'avg_gf_2025': avg_gf,
                'avg_ga_2025': avg_ga,
                'form_2025': form_rating,
                **corner_stats,
                **shot_stats
            }
        
        return stats
    
    def _calculate_current_form(self, team, home_matches, away_matches):
        """Calculate form based on last 5 matches of 2025/26"""
        all_matches = []
        
        for _, match in home_matches.iterrows():
            all_matches.append({
                'date': match.get('Date', ''),
                'team': team,
                'is_home': True,
                'goals_for': match['FTHG'],
                'goals_against': match['FTAG'],
                'result': 'H' if match['FTHG'] > match['FTAG'] else 'A' if match['FTHG'] < match['FTAG'] else 'D'
            })
        
        for _, match in away_matches.iterrows():
            all_matches.append({
                'date': match.get('Date', ''),
                'team': team,
                'is_home': False,
                'goals_for': match['FTAG'],
                'goals_against': match['FTHG'],
                'result': 'A' if match['FTAG'] > match['FTHG'] else 'H' if match['FTAG'] < match['FTHG'] else 'D'
            })
        
        all_matches.sort(key=lambda x: x['date'], reverse=True)
        last_5 = all_matches[:5]
        
        if not last_5:
            return 0.5
        
        form_points = 0
        for match in last_5:
            if (match['is_home'] and match['result'] == 'H') or (not match['is_home'] and match['result'] == 'A'):
                form_points += 3
            elif match['result'] == 'D':
                form_points += 1
        
        return form_points / (len(last_5) * 3)
    
    def _calculate_current_corner_stats(self, team, home_matches, away_matches):
        """Calculate corner statistics from 2025/26 season"""
        stats = {}
        
        corner_cols = [col for col in self.df.columns if any(x in col for x in ['HC', 'AC', 'Corners', 'corners'])]
        
        if corner_cols:
            home_corner_col = next((col for col in corner_cols if 'HC' in col or 'home' in col.lower()), corner_cols[0])
            away_corner_col = next((col for col in corner_cols if 'AC' in col or 'away' in col.lower()), corner_cols[0])
            
            if not home_matches.empty and home_corner_col in home_matches.columns:
                home_corners_for = home_matches[home_corner_col].mean()
                home_corners_against = home_matches[away_corner_col].mean() if away_corner_col in home_matches.columns else home_corners_for
            else:
                home_corners_for = 5.0
                home_corners_against = 4.0
            
            if not away_matches.empty and away_corner_col in away_matches.columns:
                away_corners_for = away_matches[away_corner_col].mean()
                away_corners_against = away_matches[home_corner_col].mean() if home_corner_col in away_matches.columns else away_corners_for
            else:
                away_corners_for = 4.0
                away_corners_against = 5.0
            
            avg_corners_for = (home_corners_for + away_corners_for) / 2
            avg_corners_against = (home_corners_against + away_corners_against) / 2
            league_avg_corners = (self.df[home_corner_col].mean() + self.df[away_corner_col].mean()) / 2
            
            stats.update({
                'corners_for_2025': avg_corners_for,
                'corners_against_2025': avg_corners_against,
                'corner_factor_2025': avg_corners_for / league_avg_corners if league_avg_corners > 0 else 1.0
            })
        
        return stats
    
    def _calculate_current_shot_stats(self, team, home_matches, away_matches):
        """Calculate shot statistics from 2025/26 season"""
        stats = {}
        
        if 'HS' in self.df.columns and 'AS' in self.df.columns:
            home_shots_for = home_matches['HS'].mean() if not home_matches.empty else 12.0
            away_shots_for = away_matches['AS'].mean() if not away_matches.empty else 10.0
            avg_shots_for = (home_shots_for + away_shots_for) / 2
            league_avg_shots = (self.df['HS'].mean() + self.df['AS'].mean()) / 2
            
            stats['shots_for_2025'] = avg_shots_for
            stats['shot_factor_2025'] = avg_shots_for / league_avg_shots if league_avg_shots > 0 else 1.0
        
        if 'HST' in self.df.columns and 'AST' in self.df.columns:
            home_sot_for = home_matches['HST'].mean() if not home_matches.empty else 4.0
            away_sot_for = away_matches['AST'].mean() if not away_matches.empty else 3.5
            avg_sot_for = (home_sot_for + away_sot_for) / 2
            league_avg_sot = (self.df['HST'].mean() + self.df['AST'].mean()) / 2
            
            stats['sot_for_2025'] = avg_sot_for
            stats['sot_factor_2025'] = avg_sot_for / league_avg_sot if league_avg_sot > 0 else 1.0
        
        return stats
    
    def _calculate_league_stats(self):
        """Calculate league averages from 2025/26 season"""
        stats = {}
        
        stats['avg_home_goals_2025'] = self.df['FTHG'].mean() if 'FTHG' in self.df.columns else 1.5
        stats['avg_away_goals_2025'] = self.df['FTAG'].mean() if 'FTAG' in self.df.columns else 1.2
        stats['avg_total_goals_2025'] = stats['avg_home_goals_2025'] + stats['avg_away_goals_2025']
        
        corner_cols = [col for col in self.df.columns if any(x in col for x in ['HC', 'AC', 'Corners'])]
        if corner_cols:
            home_corner_col = next((col for col in corner_cols if 'HC' in col or 'home' in col.lower()), corner_cols[0])
            away_corner_col = next((col for col in corner_cols if 'AC' in col or 'away' in col.lower()), corner_cols[0])
            
            stats['avg_home_corners_2025'] = self.df[home_corner_col].mean() if home_corner_col in self.df.columns else 5.0
            stats['avg_away_corners_2025'] = self.df[away_corner_col].mean() if away_corner_col in self.df.columns else 4.0
            stats['avg_total_corners_2025'] = stats['avg_home_corners_2025'] + stats['avg_away_corners_2025']
        
        if 'HS' in self.df.columns:
            stats['avg_home_shots_2025'] = self.df['HS'].mean()
            stats['avg_away_shots_2025'] = self.df['AS'].mean() if 'AS' in self.df.columns else 10.0
            stats['avg_total_shots_2025'] = stats['avg_home_shots_2025'] + stats['avg_away_shots_2025']
        
        if 'HST' in self.df.columns:
            stats['avg_home_sot_2025'] = self.df['HST'].mean()
            stats['avg_away_sot_2025'] = self.df['AST'].mean() if 'AST' in self.df.columns else 3.5
            stats['avg_total_sot_2025'] = stats['avg_home_sot_2025'] + stats['avg_away_sot_2025']
        
        return stats
    
    def predict_current_season_match(self, home_team, away_team):
        """Predict match using ONLY 2025/26 season data"""
        
        if home_team not in self.team_stats or away_team not in self.team_stats:
            return None
        
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        
        home_xg = (self.league_stats['avg_home_goals_2025'] * 
                  home_stats['attacking_strength'] / 
                  away_stats['defensive_strength'] * 
                  1.15)
        
        away_xg = (self.league_stats['avg_away_goals_2025'] * 
                  away_stats['attacking_strength'] / 
                  home_stats['defensive_strength'])
        
        form_adjustment = 0.1
        home_xg *= (1 + (home_stats['form_2025'] - 0.5) * form_adjustment)
        away_xg *= (1 + (away_stats['form_2025'] - 0.5) * form_adjustment)
        
        home_xg = max(home_xg, 0.1)
        away_xg = max(away_xg, 0.1)
        
        outcome_probs = self._calculate_poisson_outcomes(home_xg, away_xg, home_team, away_team)
        corners = self._predict_current_corners(home_team, away_team, home_stats, away_stats)
        shots = self._predict_current_shots(home_team, away_team, home_stats, away_stats)
        
        prediction = {
            **outcome_probs,
            **corners,
            **shots,
            'home_team': home_team,
            'away_team': away_team,
            'season': '2025/26',
            'data_source': 'Current Season Data Only',
            'home_matches_played': home_stats['matches_played'],
            'away_matches_played': away_stats['matches_played'],
            'home_form_2025': home_stats['form_2025'],
            'away_form_2025': away_stats['form_2025']
        }
        
        return prediction
    
    def _calculate_poisson_outcomes(self, home_xg, away_xg, home_team, away_team):
        """Calculate match outcome probabilities using Poisson"""
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
        
        top_scorelines = dict(sorted(scorelines.items(), key=lambda x: x[1], reverse=True)[:5])
        
        if home_win_prob > away_win_prob and home_win_prob > draw_prob:
            predicted_winner = home_team
            confidence = home_win_prob * 100
        elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
            predicted_winner = away_team
            confidence = away_win_prob * 100
        else:
            predicted_winner = "Draw"
            confidence = draw_prob * 100
        
        return {
            'home_xg_2025': round(home_xg, 2),
            'away_xg_2025': round(away_xg, 2),
            'home_win_prob_2025': home_win_prob,
            'draw_prob_2025': draw_prob,
            'away_win_prob_2025': away_win_prob,
            'top_scorelines_2025': top_scorelines,
            'predicted_winner_2025': predicted_winner,
            'confidence_2025': round(confidence, 1)
        }
    
    def _predict_current_corners(self, home_team, away_team, home_stats, away_stats):
        """Predict corners using 2025/26 data only"""
        
        base_home_corners = self.league_stats.get('avg_home_corners_2025', 5.0)
        base_away_corners = self.league_stats.get('avg_away_corners_2025', 4.0)
        
        home_corner_factor = home_stats.get('corner_factor_2025', 1.0)
        away_corner_factor = away_stats.get('corner_factor_2025', 1.0)
        
        home_corners = base_home_corners * home_corner_factor * 1.1
        away_corners = base_away_corners * away_corner_factor
        
        home_corners *= (1 + (home_stats['form_2025'] - 0.5) * 0.05)
        away_corners *= (1 + (away_stats['form_2025'] - 0.5) * 0.05)
        
        home_corners += np.random.uniform(-0.5, 0.5)
        away_corners += np.random.uniform(-0.5, 0.5)
        
        home_corners = max(min(home_corners, 12), 1)
        away_corners = max(min(away_corners, 10), 1)
        
        return {
            'home_corners_2025': round(home_corners, 1),
            'away_corners_2025': round(away_corners, 1),
            'total_corners_2025': round(home_corners + away_corners, 1)
        }
    
    def _predict_current_shots(self, home_team, away_team, home_stats, away_stats):
        """Predict shots on target using 2025/26 data only"""
        
        base_home_sot = self.league_stats.get('avg_home_sot_2025', 4.0)
        base_away_sot = self.league_stats.get('avg_away_sot_2025', 3.5)
        
        home_sot_factor = home_stats.get('sot_factor_2025', 1.0)
        away_sot_factor = away_stats.get('sot_factor_2025', 1.0)
        
        home_sot = base_home_sot * home_sot_factor * 1.1
        away_sot = base_away_sot * away_sot_factor
        
        home_sot *= (1 + (home_stats['form_2025'] - 0.5) * 0.08)
        away_sot *= (1 + (away_stats['form_2025'] - 0.5) * 0.08)
        
        home_sot += np.random.uniform(-0.3, 0.3)
        away_sot += np.random.uniform(-0.3, 0.3)
        
        home_sot = max(min(home_sot, 10), 0.5)
        away_sot = max(min(away_sot, 8), 0.5)
        
        return {
            'home_sot_2025': round(home_sot, 1),
            'away_sot_2025': round(away_sot, 1),
            'total_sot_2025': round(home_sot + away_sot, 1)
        }

# ============================================================================
# EXPORT TO EXCEL FUNCTION
# ============================================================================

def export_predictions_to_excel(predictions_list, league_name):
    """Export predictions to Excel with formatting"""
    
    if not EXCEL_AVAILABLE or not predictions_list:
        return None
    
    try:
        df_export = pd.DataFrame([
            {
                'Date': datetime.now().strftime('%d/%m/%Y'),
                'Time': 'TBD',
                'Home Team': p['home_team'],
                'Away Team': p['away_team'],
                'League': league_name,
                'Home xG': p['home_xg_2025'],
                'Away xG': p['away_xg_2025'],
                'Home Win %': f"{p['home_win_prob_2025']*100:.1f}%",
                'Draw %': f"{p['draw_prob_2025']*100:.1f}%",
                'Away Win %': f"{p['away_win_prob_2025']*100:.1f}%",
                'Prediction': p['predicted_winner_2025'],
                'Confidence': f"{p['confidence_2025']:.1f}%",
                'Home Corners': p['home_corners_2025'],
                'Away Corners': p['away_corners_2025'],
                'Total
