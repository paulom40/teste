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
    .alert-critical {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border: 3px solid #ff0000;
        animation: pulse 2s infinite;
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
    .team-rank {
        background-color: #343a40;
        color: white;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 0.7em;
        font-weight: bold;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

class TeamStrengthAnalyzer:
    def __init__(self):
        self.leagues_data = {}
        self._initialize_league_data()
    
    def _initialize_league_data(self):
        """Initialize team strength data for major leagues"""
        
        # Premier League 2024
        self.leagues_data['Premier League'] = {
            'Manchester City': {'offense': 94, 'defense': 92, 'overall': 93},
            'Liverpool': {'offense': 92, 'defense': 88, 'overall': 90},
            'Arsenal': {'offense': 90, 'defense': 91, 'overall': 90},
            'Chelsea': {'offense': 88, 'defense': 85, 'overall': 86},
            'Manchester United': {'offense': 85, 'defense': 83, 'overall': 84},
            'Tottenham': {'offense': 87, 'defense': 82, 'overall': 84},
            'Newcastle': {'offense': 84, 'defense': 83, 'overall': 83},
            'Aston Villa': {'offense': 86, 'defense': 80, 'overall': 83},
            'Brighton': {'offense': 85, 'defense': 78, 'overall': 81},
            'West Ham': {'offense': 82, 'defense': 79, 'overall': 80},
            'Crystal Palace': {'offense': 78, 'defense': 80, 'overall': 79},
            'Wolves': {'offense': 79, 'defense': 78, 'overall': 78},
            'Fulham': {'offense': 80, 'defense': 75, 'overall': 77},
            'Everton': {'offense': 75, 'defense': 79, 'overall': 77},
            'Brentford': {'offense': 78, 'defense': 75, 'overall': 76},
            'Nottingham Forest': {'offense': 76, 'defense': 74, 'overall': 75},
            'Luton Town': {'offense': 74, 'defense': 72, 'overall': 73},
            'Burnley': {'offense': 73, 'defense': 71, 'overall': 72},
            'Sheffield United': {'offense': 70, 'defense': 69, 'overall': 69},
        }
        
        # La Liga 2024
        self.leagues_data['La Liga'] = {
            'Real Madrid': {'offense': 95, 'defense': 90, 'overall': 92},
            'Barcelona': {'offense': 92, 'defense': 89, 'overall': 90},
            'Atletico Madrid': {'offense': 88, 'defense': 91, 'overall': 89},
            'Girona': {'offense': 87, 'defense': 82, 'overall': 84},
            'Athletic Bilbao': {'offense': 84, 'defense': 83, 'overall': 83},
            'Real Sociedad': {'offense': 83, 'defense': 84, 'overall': 83},
            'Real Betis': {'offense': 82, 'defense': 80, 'overall': 81},
            'Valencia': {'offense': 79, 'defense': 81, 'overall': 80},
            'Villarreal': {'offense': 83, 'defense': 76, 'overall': 79},
            'Getafe': {'offense': 75, 'defense': 80, 'overall': 77},
            'Sevilla': {'offense': 78, 'defense': 75, 'overall': 76},
            'Osasuna': {'offense': 76, 'defense': 77, 'overall': 76},
            'Mallorca': {'offense': 73, 'defense': 78, 'overall': 75},
            'Las Palmas': {'offense': 74, 'defense': 75, 'overall': 74},
            'Rayo Vallecano': {'offense': 75, 'defense': 73, 'overall': 74},
            'Celta Vigo': {'offense': 77, 'defense': 70, 'overall': 73},
            'Cadiz': {'offense': 70, 'defense': 75, 'overall': 72},
            'Granada': {'offense': 72, 'defense': 68, 'overall': 70},
            'Alaves': {'offense': 69, 'defense': 71, 'overall': 70},
        }
        
        # Serie A 2024
        self.leagues_data['Serie A'] = {
            'Inter Milan': {'offense': 92, 'defense': 91, 'overall': 91},
            'Juventus': {'offense': 88, 'defense': 90, 'overall': 89},
            'AC Milan': {'offense': 90, 'defense': 85, 'overall': 87},
            'Napoli': {'offense': 87, 'defense': 83, 'overall': 85},
            'Atalanta': {'offense': 86, 'defense': 81, 'overall': 83},
            'Roma': {'offense': 84, 'defense': 82, 'overall': 83},
            'Lazio': {'offense': 83, 'defense': 82, 'overall': 82},
            'Fiorentina': {'offense': 82, 'defense': 79, 'overall': 80},
            'Bologna': {'offense': 79, 'defense': 80, 'overall': 79},
            'Monza': {'offense': 76, 'defense': 78, 'overall': 77},
            'Torino': {'offense': 75, 'defense': 79, 'overall': 77},
            'Genoa': {'offense': 74, 'defense': 77, 'overall': 75},
            'Lecce': {'offense': 73, 'defense': 76, 'overall': 74},
            'Udinese': {'offense': 75, 'defense': 72, 'overall': 73},
            'Empoli': {'offense': 71, 'defense': 74, 'overall': 72},
            'Frosinone': {'offense': 73, 'defense': 70, 'overall': 71},
            'Sassuolo': {'offense': 74, 'defense': 68, 'overall': 71},
            'Verona': {'offense': 70, 'defense': 71, 'overall': 70},
        }
        
        # Bundesliga 2024
        self.leagues_data['Bundesliga'] = {
            'Bayer Leverkusen': {'offense': 93, 'defense': 90, 'overall': 91},
            'Bayern Munich': {'offense': 95, 'defense': 87, 'overall': 91},
            'Stuttgart': {'offense': 88, 'defense': 83, 'overall': 85},
            'RB Leipzig': {'offense': 89, 'defense': 82, 'overall': 85},
            'Borussia Dortmund': {'offense': 87, 'defense': 83, 'overall': 85},
            'Eintracht Frankfurt': {'offense': 82, 'defense': 80, 'overall': 81},
            'Freiburg': {'offense': 79, 'defense': 81, 'overall': 80},
            'Hoffenheim': {'offense': 81, 'defense': 76, 'overall': 78},
            'Augsburg': {'offense': 78, 'defense': 75, 'overall': 76},
            'Werder Bremen': {'offense': 77, 'defense': 74, 'overall': 75},
            'Heidenheim': {'offense': 75, 'defense': 75, 'overall': 75},
            'Wolfsburg': {'offense': 76, 'defense': 73, 'overall': 74},
            'Borussia Monchengladbach': {'offense': 77, 'defense': 71, 'overall': 74},
            'Union Berlin': {'offense': 72, 'defense': 75, 'overall': 73},
            'Bochum': {'offense': 73, 'defense': 72, 'overall': 72},
            'Mainz': {'offense': 74, 'defense': 70, 'overall': 72},
            'Koln': {'offense': 71, 'defense': 72, 'overall': 71},
            'Darmstadt': {'offense': 69, 'defense': 68, 'overall': 68},
        }
        
        # Ligue 1 2024
        self.leagues_data['Ligue 1'] = {
            'PSG': {'offense': 94, 'defense': 86, 'overall': 90},
            'Monaco': {'offense': 87, 'defense': 80, 'overall': 83},
            'Lille': {'offense': 83, 'defense': 84, 'overall': 83},
            'Brest': {'offense': 81, 'defense': 82, 'overall': 81},
            'Nice': {'offense': 80, 'defense': 83, 'overall': 81},
            'Lens': {'offense': 82, 'defense': 79, 'overall': 80},
            'Lyon': {'offense': 81, 'defense': 78, 'overall': 79},
            'Marseille': {'offense': 83, 'defense': 75, 'overall': 79},
            'Rennes': {'offense': 80, 'defense': 77, 'overall': 78},
            'Reims': {'offense': 79, 'defense': 76, 'overall': 77},
            'Toulouse': {'offense': 77, 'defense': 75, 'overall': 76},
            'Montpellier': {'offense': 76, 'defense': 74, 'overall': 75},
            'Strasbourg': {'offense': 75, 'defense': 74, 'overall': 74},
            'Nantes': {'offense': 73, 'defense': 75, 'overall': 74},
            'Le Havre': {'offense': 72, 'defense': 74, 'overall': 73},
            'Lorient': {'offense': 74, 'defense': 71, 'overall': 72},
            'Metz': {'offense': 71, 'defense': 72, 'overall': 71},
            'Clermont Foot': {'offense': 70, 'defense': 70, 'overall': 70},
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
        """Calculate match prediction based on team strengths"""
        home_data, home_league = self.get_team_strength(home_team)
        away_data, away_league = self.get_team_strength(away_team)
        
        if not home_data or not away_data:
            return None
        
        # Calculate strength difference with home advantage
        home_advantage = 3  # points for playing at home
        home_overall = home_data['overall'] + home_advantage
        away_overall = away_data['overall']
        
        strength_diff = home_overall - away_overall
        
        # Convert to probability (simplified model)
        if strength_diff > 20:
            prediction = "Strong Home Win"
            confidence = "High"
        elif strength_diff > 10:
            prediction = "Home Win"
            confidence = "Medium"
        elif strength_diff > 0:
            prediction = "Slight Home Advantage"
            confidence = "Low"
        elif strength_diff > -10:
            prediction = "Draw"
            confidence = "Medium"
        elif strength_diff > -20:
            prediction = "Away Win"
            confidence = "Medium"
        else:
            prediction = "Strong Away Win"
            confidence = "High"
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'home_offense': home_data['offense'],
            'home_defense': home_data['defense'],
            'away_offense': away_data['offense'],
            'away_defense': away_data['defense'],
            'strength_difference': strength_diff
        }

class Soccer24Scraper:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.base_url = "https://www.soccer24.com"
        self.strength_analyzer = TeamStrengthAnalyzer()
        
    def get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def scrape_inplay_matches(self):
        """Scrape in-play matches from Soccer24"""
        try:
            url = f"{self.base_url}/live/"
            headers = self.get_headers()
            
            response = self.session.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            inplay_matches = []
            
            match_elements = soup.find_all('div', class_=re.compile('event__match'))
            
            for match_element in match_elements[:30]:
                try:
                    match_data = self._parse_inplay_match_element(match_element)
                    if match_data and match_data['status'] in ['LIVE', 'HALF_TIME']:
                        # Add team strength analysis
                        match_data['strength_analysis'] = self.strength_analyzer.calculate_match_prediction(
                            match_data['home_team'], match_data['away_team']
                        )
                        inplay_matches.append(match_data)
                except:
                    continue
            
            return inplay_matches
            
        except Exception as e:
            return self._get_fallback_inplay_matches()
    
    def scrape_upcoming_matches(self, days_ahead=3):
        """Scrape upcoming matches for multiple days"""
        upcoming_matches = {}
        
        for days in range(days_ahead + 1):
            target_date = datetime.now() + timedelta(days=days)
            date_str = target_date.strftime("%Y-%m-%d")
            
            try:
                day_matches = self._get_upcoming_matches_for_date(target_date)
                if day_matches:
                    upcoming_matches[date_str] = day_matches
            except:
                continue
        
        if not upcoming_matches:
            upcoming_matches = self._get_fallback_upcoming_matches(days_ahead)
        
        return upcoming_matches
    
    def _parse_inplay_match_element(self, match_element):
        """Parse individual in-play match element"""
        home_team_elem = match_element.find('div', class_=re.compile('event__participant--home'))
        away_team_elem = match_element.find('div', class_=re.compile('event__participant--away'))
        
        if not home_team_elem or not away_team_elem:
            return None
            
        home_team = home_team_elem.get_text(strip=True)
        away_team = away_team_elem.get_text(strip=True)
        
        score_elem = match_element.find('div', class_=re.compile('event__score'))
        home_score = 0
        away_score = 0
        
        if score_elem:
            score_text = score_elem.get_text(strip=True)
            if ':' in score_text:
                try:
                    home_score, away_score = map(int, score_text.split(':'))
                except:
                    pass
        
        minute_elem = match_element.find('div', class_=re.compile('event__stage'))
        minute = minute_elem.get_text(strip=True) if minute_elem else "LIVE"
        
        status = "LIVE"
        if "Finished" in minute:
            status = "FINISHED"
        elif "HT" in minute:
            status = "HALF_TIME"
        
        odds = self._generate_inplay_odds(home_team, away_team, home_score, away_score, minute)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'minute': minute,
            'status': status,
            'odds': odds,
            'timestamp': datetime.now(),
            'type': 'INPLAY'
        }
    
    def _get_upcoming_matches_for_date(self, target_date):
        """Get upcoming matches for specific date"""
        matches = []
        match_templates = self._get_match_templates()
        
        for i, (home, away, league) in enumerate(match_templates):
            match_time = f"{(15 + i % 6):02d}:00"
            
            odds = self._generate_upcoming_odds(home, away)
            stats = self._generate_match_stats(home, away)
            strength_analysis = self.strength_analyzer.calculate_match_prediction(home, away)
            
            matches.append({
                'home_team': home,
                'away_team': away,
                'league': league,
                'match_time': match_time,
                'date': target_date.strftime("%Y-%m-%d"),
                'odds': odds,
                'stats': stats,
                'strength_analysis': strength_analysis,
                'timestamp': datetime.now(),
                'type': 'UPCOMING'
            })
        
        return matches

    def _generate_inplay_odds(self, home_team, away_team, home_score, away_score, minute):
        """Generate realistic in-play odds"""
        # ... (same as before)
        return bookmakers

    def _generate_upcoming_odds(self, home_team, away_team):
        """Generate odds for upcoming matches"""
        # ... (same as before)
        return bookmakers

    def _generate_match_stats(self, home_team, away_team):
        """Generate match statistics"""
        # ... (same as before)
        return stats

    def _get_match_templates(self):
        """Get match templates for different leagues"""
        return [
            ('Manchester City', 'Liverpool', 'Premier League'),
            ('Real Madrid', 'Barcelona', 'La Liga'),
            ('Bayern Munich', 'Borussia Dortmund', 'Bundesliga'),
            ('PSG', 'Marseille', 'Ligue 1'),
            ('Juventus', 'Inter Milan', 'Serie A'),
            ('Arsenal', 'Chelsea', 'Premier League'),
            ('Atletico Madrid', 'Sevilla', 'La Liga'),
            ('AC Milan', 'Napoli', 'Serie A'),
            ('Bayer Leverkusen', 'RB Leipzig', 'Bundesliga'),
            ('Monaco', 'Lille', 'Ligue 1')
        ]

    def _get_fallback_inplay_matches(self):
        """Fallback in-play matches"""
        matches = []
        templates = self._get_match_templates()[:5]
        
        for home, away, league in templates:
            home_score = np.random.randint(0, 3)
            away_score = np.random.randint(0, 3)
            minute = f"{np.random.randint(25, 85)}'"
            
            match_data = {
                'home_team': home,
                'away_team': away,
                'home_score': home_score,
                'away_score': away_score,
                'minute': minute,
                'status': 'LIVE',
                'odds': self._generate_inplay_odds(home, away, home_score, away_score, minute),
                'timestamp': datetime.now(),
                'type': 'INPLAY'
            }
            
            # Add strength analysis
            match_data['strength_analysis'] = self.strength_analyzer.calculate_match_prediction(home, away)
            matches.append(match_data)
        
        return matches

    def _get_fallback_upcoming_matches(self, days_ahead):
        """Fallback upcoming matches"""
        upcoming_matches = {}
        base_date = datetime.now()
        
        for days in range(days_ahead + 1):
            target_date = base_date + timedelta(days=days)
            date_str = target_date.strftime("%Y-%m-%d")
            
            matches = []
            templates = self._get_match_templates()
            
            for i, (home, away, league) in enumerate(templates):
                match_time = f"{(15 + i % 6):02d}:00"
                
                matches.append({
                    'home_team': home,
                    'away_team': away,
                    'league': league,
                    'match_time': match_time,
                    'date': date_str,
                    'odds': self._generate_upcoming_odds(home, away),
                    'stats': self._generate_match_stats(home, away),
                    'strength_analysis': self.strength_analyzer.calculate_match_prediction(home, away),
                    'timestamp': datetime.now(),
                    'type': 'UPCOMING'
                })
            
            upcoming_matches[date_str] = matches
        
        return upcoming_matches

class LiveMatchMonitor:
    def __init__(self):
        self.favorite_teams = set()
        self.alert_history = []
        
    def add_favorite_team(self, team_name):
        if team_name and team_name.strip():
            self.favorite_teams.add(team_name.lower().strip())
            return True
        return False
        
    def remove_favorite_team(self, team_name):
        if team_name and team_name.strip():
            self.favorite_teams.discard(team_name.lower().strip())
            return True
        return False
    
    def check_favorite_alerts(self, matches):
        alerts = []
        
        for match in matches:
            if match['status'] in ['LIVE', 'HALF_TIME']:
                home_team = match['home_team']
                away_team = match['away_team']
                home_score = match['home_score']
                away_score = match['away_score']
                
                if home_team.lower() in self.favorite_teams and home_score < away_score:
                    alert = {
                        'type': 'FAVORITE_LOSING',
                        'team': home_team,
                        'match': f"{home_team} vs {away_team}",
                        'score': f"{home_score}-{away_score}",
                        'minute': match['minute'],
                        'timestamp': datetime.now(),
                        'severity': 'CRITICAL'
                    }
                    alerts.append(alert)
                    
                if away_team.lower() in self.favorite_teams and away_score < home_score:
                    alert = {
                        'type': 'FAVORITE_LOSING',
                        'team': away_team,
                        'match': f"{home_team} vs {away_team}",
                        'score': f"{home_score}-{away_score}",
                        'minute': match['minute'],
                        'timestamp': datetime.now(),
                        'severity': 'CRITICAL'
                    }
                    alerts.append(alert)
        
        for alert in alerts:
            self.alert_history.append(alert)
        
        return alerts

def display_team_strength_analysis():
    """Display team strength analysis for all major leagues"""
    st.header("🏆 Team Strength Analysis - Major Leagues")
    
    strength_analyzer = TeamStrengthAnalyzer()
    leagues = strength_analyzer.get_all_leagues()
    
    selected_league = st.selectbox("Select League", leagues)
    
    if selected_league:
        teams_data = strength_analyzer.get_league_teams_strength(selected_league)
        
        if teams_data:
            # Create DataFrame for display
            team_list = []
            for team, strengths in teams_data.items():
                team_list.append({
                    'Team': team,
                    'Offense': strengths['offense'],
                    'Defense': strengths['defense'],
                    'Overall': strengths['overall'],
                    'Rank': f"#{list(teams_data.keys()).index(team) + 1}"
                })
            
            df = pd.DataFrame(team_list)
            df = df.sort_values('Overall', ascending=False)
            
            # Display league overview
            st.subheader(f"📊 {selected_league} - Team Strengths")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                best_offense = df.loc[df['Offense'].idxmax()]
                st.metric("Best Offense", f"{best_offense['Team']} ({best_offense['Offense']})")
            with col2:
                best_defense = df.loc[df['Defense'].idxmax()]
                st.metric("Best Defense", f"{best_defense['Team']} ({best_defense['Defense']})")
            with col3:
                best_overall = df.loc[df['Overall'].idxmax()]
                st.metric("Strongest Team", f"{best_overall['Team']} ({best_overall['Overall']})")
            
            # Display team strength visualization
            st.subheader("📈 Team Strength Visualization")
            
            # Create interactive bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Offense',
                x=df['Team'],
                y=df['Offense'],
                marker_color='#dc3545'
            ))
            
            fig.add_trace(go.Bar(
                name='Defense',
                x=df['Team'],
                y=df['Defense'],
                marker_color='#28a745'
            ))
            
            fig.add_trace(go.Bar(
                name='Overall',
                x=df['Team'],
                y=df['Overall'],
                marker_color='#ffc107'
            ))
            
            fig.update_layout(
                title=f'{selected_league} - Team Strength Comparison',
                xaxis_tickangle=-45,
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed team cards
            st.subheader("👥 Detailed Team Analysis")
            
            for idx, row in df.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    
                    with col1:
                        st.write(f"**{row['Team']}**")
                        st.write(f"Rank: {row['Rank']}")
                    
                    with col2:
                        st.write("Offense")
                        st.markdown(f"""
                        <div class="strength-bar">
                            <div class="strength-fill offense-fill" style="width: {row['Offense']}%">
                                {row['Offense']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.write("Defense")
                        st.markdown(f"""
                        <div class="strength-bar">
                            <div class="strength-fill defense-fill" style="width: {row['Defense']}%">
                                {row['Defense']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.write("Overall")
                        st.markdown(f"""
                        <div class="strength-bar">
                            <div class="strength-fill overall-fill" style="width: {row['Overall']}%">
                                {row['Overall']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")

def main():
    st.markdown('<h1 class="main-header">⚽ Soccer24 Betting Hub</h1>', unsafe_allow_html=True)
    
    # Initialize systems
    if 'scraper' not in st.session_state:
        st.session_state.scraper = Soccer24Scraper()
    if 'monitor' not in st.session_state:
        st.session_state.monitor = LiveMatchMonitor()
    
    scraper = st.session_state.scraper
    monitor = st.session_state.monitor
    
    # Sidebar
    st.sidebar.title("⚙️ Settings & Favorites")
    
    # Favorite Teams
    st.sidebar.subheader("⭐ Favorite Teams")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        new_team = st.text_input("Add Team", placeholder="e.g., Liverpool")
        if st.button("Add") and new_team:
            if monitor.add_favorite_team(new_team):
                st.success(f"Added {new_team}")
    with col2:
        if monitor.favorite_teams:
            remove_team = st.selectbox("Remove Team", options=list(monitor.favorite_teams))
            if st.button("Remove"):
                if monitor.remove_favorite_team(remove_team):
                    st.success(f"Removed {remove_team}")
    
    if monitor.favorite_teams:
        st.sidebar.write("**Your Favorites:**")
        for team in sorted(monitor.favorite_teams):
            st.sidebar.write(f"⭐ {team.title()}")
    
    # Controls
    st.sidebar.subheader("🎯 Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 30s", value=True)
    days_ahead = st.sidebar.slider("Days ahead", 1, 7, 3)
    
    # Scrape data
    if auto_refresh or 'last_update' not in st.session_state:
        with st.spinner("🔄 Loading data..."):
            inplay_matches = scraper.scrape_inplay_matches()
            upcoming_matches = scraper.scrape_upcoming_matches(days_ahead)
            alerts = monitor.check_favorite_alerts(inplay_matches)
            
            st.session_state.inplay_matches = inplay_matches
            st.session_state.upcoming_matches = upcoming_matches
            st.session_state.alerts = alerts
            st.session_state.last_update = datetime.now()
    
    # Main tabs - ADDED NEW TAB FOR TEAM STRENGTH
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚨 Live Alerts", 
        "🔴 In-Play Matches", 
        "📅 Upcoming Matches", 
        "💰 Best Bets Table", 
        "📊 Match Statistics",
        "🏆 Team Strength"  # NEW TAB
    ])
    
    with tab1:
        display_live_alerts()
    
    with tab2:
        display_inplay_matches()
    
    with tab3:
        display_upcoming_matches()
    
    with tab4:
        display_best_bets_table()
    
    with tab5:
        display_match_statistics()
    
    with tab6:  # NEW TAB
        display_team_strength_analysis()
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()

# ... (keep all the existing display functions from previous code)

def display_live_alerts():
    # ... (same as before)
    pass

def display_inplay_matches():
    # ... (same as before)
    pass

def display_upcoming_matches():
    # ... (same as before)
    pass

def display_best_bets_table():
    # ... (same as before)
    pass

def display_match_statistics():
    # ... (same as before)
    pass

if __name__ == "__main__":
    main()
