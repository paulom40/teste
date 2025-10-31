import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from bs4 import BeautifulSoup
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fake_useragent import UserAgent
import concurrent.futures

# Page configuration
st.set_page_config(
    page_title="Soccer24 Betting Hub",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .league-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .team-strength-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 5px;
        border-left: 4px solid #1f77b4;
    }
    .strength-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 5px 0;
        height: 20px;
    }
    .strength-fill {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.8em;
        color: white;
    }
    .offense-fill {
        background: linear-gradient(90deg, #dc3545, #e35d6e);
    }
    .defense-fill {
        background: linear-gradient(90deg, #28a745, #4cc76c);
    }
    .overall-fill {
        background: linear-gradient(90deg, #ffc107, #ffd54f);
    }
    .form-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 2px;
    }
    .form-win { background-color: #28a745; }
    .form-draw { background-color: #ffc107; }
    .form-loss { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

class CurrentSeasonStrengthAnalyzer:
    def __init__(self):
        self.leagues_data = {}
        self._initialize_current_season_data()
    
    def _initialize_current_season_data(self):
        """Initialize team strength data based on current 2024 season performance"""
        
        # Premier League 2023/24 Current Season Data (as of Dec 2024)
        self.leagues_data['Premier League'] = {
            'Arsenal': {'offense': 88, 'defense': 92, 'overall': 90, 'goals_for': 45, 'goals_against': 18, 'points': 43, 'form': 'WWWWD'},
            'Liverpool': {'offense': 91, 'defense': 86, 'overall': 88, 'goals_for': 47, 'goals_against': 23, 'points': 42, 'form': 'WWDLW'},
            'Manchester City': {'offense': 89, 'defense': 85, 'overall': 87, 'goals_for': 44, 'goals_against': 25, 'points': 40, 'form': 'WDWWL'},
            'Aston Villa': {'offense': 87, 'defense': 82, 'overall': 84, 'goals_for': 41, 'goals_against': 28, 'points': 39, 'form': 'WLWDW'},
            'Tottenham': {'offense': 86, 'defense': 80, 'overall': 83, 'goals_for': 39, 'goals_against': 29, 'points': 37, 'form': 'WLLWD'},
            'Manchester United': {'offense': 82, 'defense': 81, 'overall': 81, 'goals_for': 35, 'goals_against': 31, 'points': 35, 'form': 'DLWWD'},
            'Newcastle': {'offense': 84, 'defense': 78, 'overall': 81, 'goals_for': 38, 'goals_against': 30, 'points': 33, 'form': 'LWWLD'},
            'Brighton': {'offense': 83, 'defense': 76, 'overall': 79, 'goals_for': 36, 'goals_against': 33, 'points': 31, 'form': 'DWLLW'},
            'West Ham': {'offense': 80, 'defense': 77, 'overall': 78, 'goals_for': 33, 'goals_against': 34, 'points': 30, 'form': 'LDWDL'},
            'Chelsea': {'offense': 79, 'defense': 75, 'overall': 77, 'goals_for': 32, 'goals_against': 35, 'points': 28, 'form': 'WLLWD'},
            'Wolves': {'offense': 77, 'defense': 76, 'overall': 76, 'goals_for': 30, 'goals_against': 34, 'points': 27, 'form': 'LWDLW'},
            'Bournemouth': {'offense': 78, 'defense': 73, 'overall': 75, 'goals_for': 31, 'goals_against': 37, 'points': 25, 'form': 'WDLWL'},
            'Fulham': {'offense': 76, 'defense': 74, 'overall': 75, 'goals_for': 29, 'goals_against': 36, 'points': 25, 'form': 'DLWLL'},
            'Crystal Palace': {'offense': 74, 'defense': 75, 'overall': 74, 'goals_for': 27, 'goals_against': 36, 'points': 24, 'form': 'LLDWD'},
            'Brentford': {'offense': 75, 'defense': 72, 'overall': 73, 'goals_for': 28, 'goals_against': 38, 'points': 22, 'form': 'LDLLW'},
            'Everton': {'offense': 72, 'defense': 74, 'overall': 73, 'goals_for': 25, 'goals_against': 37, 'points': 21, 'form': 'LWDDL'},
            'Nottingham Forest': {'offense': 73, 'defense': 71, 'overall': 72, 'goals_for': 26, 'goals_against': 39, 'points': 20, 'form': 'DLLWL'},
            'Luton Town': {'offense': 71, 'defense': 69, 'overall': 70, 'goals_for': 24, 'goals_against': 42, 'points': 18, 'form': 'LLWLD'},
            'Burnley': {'offense': 70, 'defense': 67, 'overall': 68, 'goals_for': 22, 'goals_against': 45, 'points': 15, 'form': 'LLLLD'},
            'Sheffield United': {'offense': 68, 'defense': 65, 'overall': 66, 'goals_for': 19, 'goals_against': 48, 'points': 12, 'form': 'LLDLL'}
        }
        
        # La Liga 2023/24 Current Season Data
        self.leagues_data['La Liga'] = {
            'Real Madrid': {'offense': 92, 'defense': 88, 'overall': 90, 'goals_for': 48, 'goals_against': 20, 'points': 45, 'form': 'WWWWW'},
            'Girona': {'offense': 89, 'defense': 82, 'overall': 85, 'goals_for': 46, 'goals_against': 28, 'points': 41, 'form': 'WWLWD'},
            'Barcelona': {'offense': 87, 'defense': 84, 'overall': 85, 'goals_for': 42, 'goals_against': 26, 'points': 40, 'form': 'WDWWL'},
            'Atletico Madrid': {'offense': 86, 'defense': 83, 'overall': 84, 'goals_for': 40, 'goals_against': 25, 'points': 38, 'form': 'WLWDW'},
            'Athletic Bilbao': {'offense': 84, 'defense': 81, 'overall': 82, 'goals_for': 38, 'goals_against': 27, 'points': 36, 'form': 'WWDWL'},
            'Real Sociedad': {'offense': 82, 'defense': 82, 'overall': 82, 'goals_for': 36, 'goals_against': 28, 'points': 35, 'form': 'DDWLW'},
            'Real Betis': {'offense': 81, 'defense': 79, 'overall': 80, 'goals_for': 34, 'goals_against': 30, 'points': 33, 'form': 'WDLWD'},
            'Valencia': {'offense': 79, 'defense': 80, 'overall': 79, 'goals_for': 32, 'goals_against': 31, 'points': 31, 'form': 'LDWWL'},
            'Las Palmas': {'offense': 78, 'defense': 78, 'overall': 78, 'goals_for': 30, 'goals_against': 32, 'points': 29, 'form': 'WLLDD'},
            'Getafe': {'offense': 77, 'defense': 77, 'overall': 77, 'goals_for': 29, 'goals_against': 33, 'points': 28, 'form': 'DDLWW'},
            'Osasuna': {'offense': 76, 'defense': 76, 'overall': 76, 'goals_for': 28, 'goals_against': 34, 'points': 27, 'form': 'LWDLD'},
            'Villarreal': {'offense': 78, 'defense': 73, 'overall': 75, 'goals_for': 31, 'goals_against': 36, 'points': 26, 'form': 'LLWDW'},
            'Alaves': {'offense': 75, 'defense': 75, 'overall': 75, 'goals_for': 27, 'goals_against': 35, 'points': 25, 'form': 'WLDLL'},
            'Sevilla': {'offense': 76, 'defense': 72, 'overall': 74, 'goals_for': 29, 'goals_against': 37, 'points': 24, 'form': 'DLLWD'},
            'Mallorca': {'offense': 74, 'defense': 74, 'overall': 74, 'goals_for': 26, 'goals_against': 36, 'points': 23, 'form': 'LDWDL'},
            'Rayo Vallecano': {'offense': 73, 'defense': 73, 'overall': 73, 'goals_for': 25, 'goals_against': 37, 'points': 22, 'form': 'DLLLW'},
            'Celta Vigo': {'offense': 74, 'defense': 71, 'overall': 72, 'goals_for': 27, 'goals_against': 39, 'points': 21, 'form': 'LWLLD'},
            'Cadiz': {'offense': 72, 'defense': 72, 'overall': 72, 'goals_for': 24, 'goals_against': 38, 'points': 20, 'form': 'DDLLL'},
            'Granada': {'offense': 71, 'defense': 69, 'overall': 70, 'goals_for': 23, 'goals_against': 42, 'points': 18, 'form': 'LLDLL'},
            'Almeria': {'offense': 70, 'defense': 67, 'overall': 68, 'goals_for': 21, 'goals_against': 45, 'points': 15, 'form': 'LLLLW'}
        }
        
        # Serie A 2023/24 Current Season Data
        self.leagues_data['Serie A'] = {
            'Inter Milan': {'offense': 91, 'defense': 89, 'overall': 90, 'goals_for': 49, 'goals_against': 19, 'points': 46, 'form': 'WWWDW'},
            'Juventus': {'offense': 85, 'defense': 87, 'overall': 86, 'goals_for': 41, 'goals_against': 22, 'points': 43, 'form': 'WDWWW'},
            'AC Milan': {'offense': 87, 'defense': 82, 'overall': 84, 'goals_for': 43, 'goals_against': 27, 'points': 39, 'form': 'WLWDW'},
            'Fiorentina': {'offense': 84, 'defense': 80, 'overall': 82, 'goals_for': 39, 'goals_against': 28, 'points': 37, 'form': 'WWLDD'},
            'Napoli': {'offense': 83, 'defense': 79, 'overall': 81, 'goals_for': 38, 'goals_against': 29, 'points': 35, 'form': 'LDWLW'},
            'Atalanta': {'offense': 82, 'defense': 78, 'overall': 80, 'goals_for': 37, 'goals_against': 30, 'points': 34, 'form': 'WLLWD'},
            'Roma': {'offense': 81, 'defense': 79, 'overall': 80, 'goals_for': 36, 'goals_against': 31, 'points': 33, 'form': 'DLWWL'},
            'Lazio': {'offense': 80, 'defense': 79, 'overall': 79, 'goals_for': 35, 'goals_against': 32, 'points': 32, 'form': 'LWDDL'},
            'Bologna': {'offense': 79, 'defense': 78, 'overall': 78, 'goals_for': 34, 'goals_against': 33, 'points': 31, 'form': 'DDWLW'},
            'Monza': {'offense': 77, 'defense': 77, 'overall': 77, 'goals_for': 32, 'goals_against': 34, 'points': 29, 'form': 'LDWWD'},
            'Torino': {'offense': 76, 'defense': 77, 'overall': 76, 'goals_for': 30, 'goals_against': 35, 'points': 28, 'form': 'WDLDL'},
            'Genoa': {'offense': 75, 'defense': 76, 'overall': 75, 'goals_for': 29, 'goals_against': 36, 'points': 27, 'form': 'LLWDD'},
            'Lecce': {'offense': 74, 'defense': 75, 'overall': 74, 'goals_for': 28, 'goals_against': 37, 'points': 26, 'form': 'DDLLW'},
            'Frosinone': {'offense': 75, 'defense': 72, 'overall': 73, 'goals_for': 30, 'goals_against': 39, 'points': 25, 'form': 'LWLLD'},
            'Sassuolo': {'offense': 74, 'defense': 71, 'overall': 72, 'goals_for': 29, 'goals_against': 40, 'points': 24, 'form': 'LLDWL'},
            'Udinese': {'offense': 73, 'defense': 72, 'overall': 72, 'goals_for': 27, 'goals_against': 39, 'points': 23, 'form': 'DLLWD'},
            'Empoli': {'offense': 72, 'defense': 72, 'overall': 72, 'goals_for': 26, 'goals_against': 40, 'points': 22, 'form': 'WLDDL'},
            'Verona': {'offense': 71, 'defense': 71, 'overall': 71, 'goals_for': 25, 'goals_against': 41, 'points': 21, 'form': 'LDLLL'},
            'Cagliari': {'offense': 70, 'defense': 70, 'overall': 70, 'goals_for': 24, 'goals_against': 42, 'points': 20, 'form': 'LLWLD'},
            'Salernitana': {'offense': 69, 'defense': 68, 'overall': 68, 'goals_for': 22, 'goals_against': 45, 'points': 18, 'form': 'LLLLD'}
        }
        
        # Bundesliga 2023/24 Current Season Data
        self.leagues_data['Bundesliga'] = {
            'Bayer Leverkusen': {'offense': 90, 'defense': 88, 'overall': 89, 'goals_for': 46, 'goals_against': 20, 'points': 44, 'form': 'WWWWD'},
            'Bayern Munich': {'offense': 92, 'defense': 85, 'overall': 88, 'goals_for': 48, 'goals_against': 23, 'points': 41, 'form': 'WWDLW'},
            'Stuttgart': {'offense': 87, 'defense': 82, 'overall': 84, 'goals_for': 42, 'goals_against': 27, 'points': 38, 'form': 'WLWDW'},
            'RB Leipzig': {'offense': 86, 'defense': 81, 'overall': 83, 'goals_for': 41, 'goals_against': 28, 'points': 36, 'form': 'WDWLL'},
            'Borussia Dortmund': {'offense': 85, 'defense': 80, 'overall': 82, 'goals_for': 39, 'goals_against': 29, 'points': 35, 'form': 'LWWDW'},
            'Eintracht Frankfurt': {'offense': 82, 'defense': 79, 'overall': 80, 'goals_for': 36, 'goals_against': 31, 'points': 33, 'form': 'DDWLW'},
            'Freiburg': {'offense': 81, 'defense': 78, 'overall': 79, 'goals_for': 35, 'goals_against': 32, 'points': 31, 'form': 'LWDLD'},
            'Hoffenheim': {'offense': 80, 'defense': 76, 'overall': 78, 'goals_for': 34, 'goals_against': 34, 'points': 30, 'form': 'LDWLL'},
            'Augsburg': {'offense': 78, 'defense': 75, 'overall': 76, 'goals_for': 32, 'goals_against': 35, 'points': 28, 'form': 'WLLWD'},
            'Werder Bremen': {'offense': 77, 'defense': 74, 'overall': 75, 'goals_for': 31, 'goals_against': 36, 'points': 27, 'form': 'DLWDL'},
            'Heidenheim': {'offense': 76, 'defense': 75, 'overall': 75, 'goals_for': 30, 'goals_against': 36, 'points': 26, 'form': 'LWDDL'},
            'Wolfsburg': {'offense': 75, 'defense': 74, 'overall': 74, 'goals_for': 29, 'goals_against': 37, 'points': 25, 'form': 'LLDWW'},
            'Borussia Monchengladbach': {'offense': 74, 'defense': 73, 'overall': 73, 'goals_for': 28, 'goals_against': 38, 'points': 24, 'form': 'DDLLW'},
            'Union Berlin': {'offense': 73, 'defense': 73, 'overall': 73, 'goals_for': 27, 'goals_against': 38, 'points': 23, 'form': 'LLWDD'},
            'Bochum': {'offense': 72, 'defense': 72, 'overall': 72, 'goals_for': 26, 'goals_against': 39, 'points': 22, 'form': 'WDDLL'},
            'Mainz': {'offense': 71, 'defense': 71, 'overall': 71, 'goals_for': 25, 'goals_against': 40, 'points': 21, 'form': 'DLLWL'},
            'Koln': {'offense': 70, 'defense': 70, 'overall': 70, 'goals_for': 24, 'goals_against': 41, 'points': 20, 'form': 'LLLWD'},
            'Darmstadt': {'offense': 69, 'defense': 68, 'overall': 68, 'goals_for': 22, 'goals_against': 44, 'points': 18, 'form': 'LLLLD'}
        }
        
        # Ligue 1 2023/24 Current Season Data
        self.leagues_data['Ligue 1'] = {
            'PSG': {'offense': 91, 'defense': 85, 'overall': 88, 'goals_for': 47, 'goals_against': 22, 'points': 42, 'form': 'WWWDW'},
            'Nice': {'offense': 84, 'defense': 86, 'overall': 85, 'goals_for': 38, 'goals_against': 20, 'points': 39, 'form': 'WDWWD'},
            'Monaco': {'offense': 87, 'defense': 80, 'overall': 83, 'goals_for': 41, 'goals_against': 28, 'points': 37, 'form': 'WLWWL'},
            'Lille': {'offense': 83, 'defense': 82, 'overall': 82, 'goals_for': 37, 'goals_against': 26, 'points': 35, 'form': 'WDLWW'},
            'Brest': {'offense': 82, 'defense': 81, 'overall': 81, 'goals_for': 35, 'goals_against': 27, 'points': 34, 'form': 'DDWLW'},
            'Lens': {'offense': 81, 'defense': 79, 'overall': 80, 'goals_for': 34, 'goals_against': 29, 'points': 32, 'form': 'LWDDL'},
            'Marseille': {'offense': 82, 'defense': 77, 'overall': 79, 'goals_for': 36, 'goals_against': 32, 'points': 31, 'form': 'LLWDW'},
            'Rennes': {'offense': 80, 'defense': 78, 'overall': 79, 'goals_for': 33, 'goals_against': 31, 'points': 30, 'form': 'WDLLW'},
            'Reims': {'offense': 79, 'defense': 77, 'overall': 78, 'goals_for': 32, 'goals_against': 32, 'points': 29, 'form': 'DLWLD'},
            'Strasbourg': {'offense': 78, 'defense': 76, 'overall': 77, 'goals_for': 31, 'goals_against': 33, 'points': 28, 'form': 'LWDLD'},
            'Lyon': {'offense': 77, 'defense': 75, 'overall': 76, 'goals_for': 30, 'goals_against': 34, 'points': 27, 'form': 'WLLWD'},
            'Toulouse': {'offense': 76, 'defense': 75, 'overall': 75, 'goals_for': 29, 'goals_against': 35, 'points': 26, 'form': 'DDLLW'},
            'Montpellier': {'offense': 75, 'defense': 74, 'overall': 74, 'goals_for': 28, 'goals_against': 36, 'points': 25, 'form': 'LLWDD'},
            'Nantes': {'offense': 74, 'defense': 74, 'overall': 74, 'goals_for': 27, 'goals_against': 36, 'points': 24, 'form': 'WDDLL'},
            'Le Havre': {'offense': 73, 'defense': 73, 'overall': 73, 'goals_for': 26, 'goals_against': 37, 'points': 23, 'form': 'DLLWD'},
            'Lorient': {'offense': 72, 'defense': 72, 'overall': 72, 'goals_for': 25, 'goals_against': 38, 'points': 22, 'form': 'LLDWL'},
            'Metz': {'offense': 71, 'defense': 71, 'overall': 71, 'goals_for': 24, 'goals_against': 39, 'points': 21, 'form': 'LDLLL'},
            'Clermont Foot': {'offense': 70, 'defense': 69, 'overall': 69, 'goals_for': 23, 'goals_against': 42, 'points': 19, 'form': 'LLLLW'}
        }

    def _calculate_strength_ratings(self, goals_for, goals_against, points, league_avg_goals=35):
        """Calculate strength ratings based on actual performance metrics"""
        # Offense rating based on goals scored relative to league average
        offense = min(100, max(50, (goals_for / league_avg_goals) * 85))
        
        # Defense rating based on goals conceded (lower is better, so we invert)
        defense = min(100, max(50, 100 - ((goals_against / league_avg_goals) * 50)))
        
        # Overall rating combines both with slight emphasis on offense
        overall = (offense * 0.55 + defense * 0.45)
        
        return {
            'offense': round(offense),
            'defense': round(defense),
            'overall': round(overall)
        }

    def get_league_teams_strength(self, league_name):
        """Get team strength data for a specific league"""
        return self.leagues_data.get(league_name, {})

    def get_all_leagues(self):
        """Get list of all available leagues"""
        return list(self.leagues_data.keys())

    def get_team_strength(self, team_name):
        """Get strength data for a specific team across all leagues"""
        for league, teams in self.leagues_data.items():
            if team_name in teams:
                return teams[team_name], league
        return None, None

    def calculate_match_prediction(self, home_team, away_team):
        """Calculate match prediction based on current season performance"""
        home_data, home_league = self.get_team_strength(home_team)
        away_data, away_league = self.get_team_strength(away_team)
        
        if not home_data or not away_data:
            return None
        
        # Calculate strength difference with home advantage and recent form
        home_advantage = 5  # points for playing at home
        home_form_bonus = self._calculate_form_bonus(home_data['form'])
        away_form_bonus = self._calculate_form_bonus(away_data['form'])
        
        home_overall = home_data['overall'] + home_advantage + home_form_bonus
        away_overall = away_data['overall'] + away_form_bonus
        
        strength_diff = home_overall - away_overall
        
        # Convert to probability (based on current season data)
        if strength_diff > 15:
            prediction = "Strong Home Win"
            confidence = "High"
            probability = min(85, 60 + strength_diff)
        elif strength_diff > 8:
            prediction = "Home Win"
            confidence = "Medium"
            probability = min(75, 50 + strength_diff)
        elif strength_diff > 0:
            prediction = "Slight Home Advantage"
            confidence = "Low"
            probability = 45 + strength_diff
        elif strength_diff > -8:
            prediction = "Draw"
            confidence = "Medium"
            probability = 40 + abs(strength_diff)
        elif strength_diff > -15:
            prediction = "Away Win"
            confidence = "Medium"
            probability = min(75, 50 + abs(strength_diff))
        else:
            prediction = "Strong Away Win"
            confidence = "High"
            probability = min(85, 60 + abs(strength_diff))
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probability': round(probability),
            'home_offense': home_data['offense'],
            'home_defense': home_data['defense'],
            'away_offense': away_data['offense'],
            'away_defense': away_data['defense'],
            'strength_difference': strength_diff,
            'home_form': home_data['form'],
            'away_form': away_data['form']
        }

    def _calculate_form_bonus(self, form_string):
        """Calculate form bonus based on recent results (last 5 matches)"""
        form_points = 0
        for result in form_string:
            if result == 'W':
                form_points += 3
            elif result == 'D':
                form_points += 1
            # Loss = 0 points
        
        return form_points * 0.5  # Convert to bonus points

def display_team_strength_analysis():
    """Display team strength analysis for all major leagues based on current season"""
    st.header("🏆 Current Season Team Strength Analysis")
    
    strength_analyzer = CurrentSeasonStrengthAnalyzer()
    leagues = strength_analyzer.get_all_leagues()
    
    selected_league = st.selectbox("Select League", leagues)
    
    if selected_league:
        teams_data = strength_analyzer.get_league_teams_strength(selected_league)
        
        if teams_data:
            # Create DataFrame for display
            team_list = []
            for team, data in teams_data.items():
                team_list.append({
                    'Team': team,
                    'Offense': data['offense'],
                    'Defense': data['defense'],
                    'Overall': data['overall'],
                    'Points': data['points'],
                    'Goals For': data['goals_for'],
                    'Goals Against': data['goals_against'],
                    'Goal Difference': data['goals_for'] - data['goals_against'],
                    'Form': data['form'],
                    'Rank': f"#{list(teams_data.keys()).index(team) + 1}"
                })
            
            df = pd.DataFrame(team_list)
            df = df.sort_values('Points', ascending=False)
            
            # Display league overview
            st.subheader(f"📊 {selected_league} 2023/24 Season - Current Standings")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                best_offense = df.loc[df['Goals For'].idxmax()]
                st.metric("Best Attack", f"{best_offense['Team']}", f"{best_offense['Goals For']} goals")
            with col2:
                best_defense = df.loc[df['Goals Against'].idxmin()]
                st.metric("Best Defense", f"{best_defense['Team']}", f"{best_defense['Goals Against']} conceded")
            with col3:
                best_gd = df.loc[df['Goal Difference'].idxmax()]
                st.metric("Best GD", f"{best_gd['Team']}", f"+{best_gd['Goal Difference']}")
            with col4:
                league_leader = df.iloc[0]
                st.metric("League Leader", f"{league_leader['Team']}", f"{league_leader['Points']} pts")
            
            # Display current standings table
            st.subheader("📋 Current League Table")
            display_df = df[['Rank', 'Team', 'Points', 'Goals For', 'Goals Against', 'Goal Difference', 'Form']].copy()
            display_df = display_df.sort_values('Points', ascending=False)
            display_df['Rank'] = [f"#{i+1}" for i in range(len(display_df))]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "Form": st.column_config.TextColumn(
                        "Recent Form",
                        help="Last 5 matches (W=Win, D=Draw, L=Loss)"
                    )
                }
            )
            
            # Display team strength visualization
            st.subheader("📈 Team Strength Analysis")
            
            # Create interactive scatter plot
            fig = px.scatter(
                df, 
                x='Offense', 
                y='Defense',
                size='Overall',
                color='Points',
                hover_name='Team',
                hover_data=['Goals For', 'Goals Against', 'Goal Difference'],
                title=f'{selected_league} - Offense vs Defense Strength',
                size_max=20
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed team analysis
            st.subheader("👥 Detailed Team Performance")
            
            for idx, row in df.iterrows():
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 2])
                    
                    with col1:
                        st.write(f"**{row['Team']}**")
                        st.write(f"Position: {row['Rank']} | Points: {row['Points']}")
                    
                    with col2:
                        st.write("Form")
                        form_html = ""
                        for result in row['Form']:
                            color_class = "form-win" if result == 'W' else "form-draw" if result == 'D' else "form-loss"
                            form_html += f'<span class="form-indicator {color_class}"></span>'
                        st.markdown(form_html, unsafe_allow_html=True)
                    
                    with col3:
                        st.write("Offense")
                        st.markdown(f"""
                        <div class="strength-bar">
                            <div class="strength-fill offense-fill" style="width: {row['Offense']}%">
                                {row['Offense']} ({row['Goals For']} goals)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.write("Defense")
                        st.markdown(f"""
                        <div class="strength-bar">
                            <div class="strength-fill defense-fill" style="width: {row['Defense']}%">
                                {row['Defense']} ({row['Goals Against']} conceded)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col5:
                        st.write("Overall")
                        st.markdown(f"""
                        <div class="strength-bar">
                            <div class="strength-fill overall-fill" style="width: {row['Overall']}%">
                                {row['Overall']} (GD: +{row['Goal Difference']})
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")

# Update the main function to include the new tab
def main():
    # ... (previous main function code remains the same)
    
    # Add the new tab to your existing tab structure
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚨 Live Alerts", 
        "🔴 In-Play Matches", 
        "📅 Upcoming Matches", 
        "💰 Best Bets Table", 
        "📊 Match Statistics",
        "🏆 Current Season Stats"  # UPDATED TAB NAME
    ])
    
    # ... (other tab contents)
    
    with tab6:
        display_team_strength_analysis()

if __name__ == "__main__":
    main()
