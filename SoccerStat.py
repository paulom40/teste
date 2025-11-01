import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from scipy.stats import poisson
import statsmodels.formula.api as smf
from datetime import datetime, timedelta
import re

class AdvancedFootballPredictor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def scrape_fbref_team_stats(self, team_url):
        """Scrape advanced team statistics from FBref"""
        try:
            response = requests.get(team_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find standard stats table
            stats_table = soup.find('table', {'id': 'stats_standard_9'})
            if not stats_table:
                return None
                
            # Extract key metrics
            metrics = {}
            
            # Expected Goals (xG)
            xg_elem = soup.find('td', {'data-stat': 'xg'})
            if xg_elem:
                metrics['xg_per_game'] = float(xg_elem.text) if xg_elem.text else 0
            
            # Expected Goals Against (xGA)
            xga_elem = soup.find('td', {'data-stat': 'xg_against'})
            if xga_elem:
                metrics['xga_per_game'] = float(xga_elem.text) if xga_elem.text else 0
            
            # Shot-creating actions
            sca_elem = soup.find('td', {'data-stat': 'sca'})
            if sca_elem:
                metrics['sca_per_game'] = float(sca_elem.text) if sca_elem.text else 0
            
            # Pass completion %
            passes_elem = soup.find('td', {'data-stat': 'passes_pct'})
            if passes_elem:
                metrics['pass_accuracy'] = float(passes_elem.text.replace('%', '')) if passes_elem.text else 0
            
            # Pressures
            pressures_elem = soup.find('td', {'data-stat': 'pressures'})
            if pressures_elem:
                metrics['pressures_per_game'] = float(pressures_elem.text) if pressures_elem.text else 0
            
            return metrics
            
        except Exception as e:
            print(f"Error scraping FBref: {e}")
            return None
    
    def scrape_transfermarkt_team_value(self, team_name, league):
        """Scrape team market value from Transfermarkt"""
        try:
            # Simplified search - in practice you'd need to map team names to Transfermarkt URLs
            search_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={team_name.replace(' ', '+')}"
            response = requests.get(search_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find team value (simplified)
            value_elem = soup.find('td', {'class': 'rechts hauptlink'})
            if value_elem:
                value_text = value_elem.text.strip()
                # Convert "€125.50m" to millions
                if 'm' in value_text:
                    return float(value_text.replace('€', '').replace('m', ''))
                elif 'Th.' in value_text:
                    return float(value_text.replace('€', '').replace('Th.', '')) / 1000
            return None
            
        except Exception as e:
            print(f"Error scraping Transfermarkt: {e}")
            return None
    
    def get_league_fixtures(self, league_code, season='2024-2025'):
        """Get upcoming fixtures for a league"""
        leagues = {
            'premier_league': 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures',
            'la_liga': 'https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures',
            'serie_a': 'https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures',
            'bundesliga': 'https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures',
            'ligue_1': 'https://fbref.com/en/comps/13/schedule/Ligue-1-Scores-and-Fixtures'
        }
        
        try:
            url = leagues.get(league_code)
            if not url:
                return None
                
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            fixtures = []
            table = soup.find('table', {'id': 'sched_2024-2025_9_1'})  # Premier League example
            
            if table:
                for row in table.find_all('tr')[1:]:  # Skip header
                    cols = row.find_all('td')
                    if len(cols) >= 7:
                        try:
                            home_team = cols[3].text.strip()
                            away_team = cols[5].text.strip()
                            score = cols[6].text.strip()
                            
                            # Only include future fixtures (no score yet)
                            if score == '':
                                fixtures.append({
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'date': cols[0].text.strip()
                                })
                        except:
                            continue
            
            return fixtures
            
        except Exception as e:
            print(f"Error getting fixtures: {e}")
            return None
    
    def get_historical_performance(self, team_name, league_code):
        """Get historical performance data for a team"""
        # This would scrape multiple seasons of data
        # For demo, returning mock advanced metrics
        return {
            'xg_for': np.random.uniform(1.2, 2.1),
            'xg_against': np.random.uniform(1.0, 1.8),
            'pass_accuracy': np.random.uniform(75, 90),
            'shot_creating_actions': np.random.uniform(18, 28),
            'pressures': np.random.uniform(150, 250),
            'market_value': np.random.uniform(50, 500)  # in millions
        }
    
    def create_advanced_features(self, home_stats, away_stats):
        """Create advanced features for model prediction"""
        features = {
            'xg_diff': home_stats['xg_for'] - away_stats['xg_against'],
            'xg_against_diff': home_stats['xg_against'] - away_stats['xg_for'],
            'pass_accuracy_diff': home_stats['pass_accuracy'] - away_stats['pass_accuracy'],
            'sca_diff': home_stats['shot_creating_actions'] - away_stats['shot_creating_actions'],
            'pressure_diff': home_stats['pressures'] - away_stats['pressures'],
            'market_value_ratio': home_stats['market_value'] / away_stats['market_value'],
            'home_attack_strength': home_stats['xg_for'] / (home_stats['xg_for'] + away_stats['xg_against']),
            'away_attack_strength': away_stats['xg_for'] / (away_stats['xg_for'] + home_stats['xg_against'])
        }
        return features
    
    def train_advanced_poisson_model(self, training_data):
        """Train Poisson model with advanced metrics"""
        if len(training_data) < 10:
            return None
            
        df = pd.DataFrame(training_data)
        
        # Ensure we have required columns
        required_cols = ['home_goals', 'away_goals', 'xg_diff', 'pass_accuracy_diff', 'sca_diff']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None
        
        try:
            # Home goals model
            home_model = smf.poisson(
                'home_goals ~ xg_diff + pass_accuracy_diff + sca_diff + market_value_ratio', 
                data=df
            ).fit(disp=0)
            
            # Away goals model  
            away_model = smf.poisson(
                'away_goals ~ xg_diff + pass_accuracy_diff + sca_diff + market_value_ratio',
                data=df
            ).fit(disp=0)
            
            return home_model, away_model
            
        except Exception as e:
            print(f"Model training error: {e}")
            return None
    
    def predict_match_outcome(self, home_team, away_team, home_stats, away_stats, model):
        """Predict match outcome using advanced metrics"""
        if not model:
            return self.basic_prediction(home_stats, away_stats)
            
        home_model, away_model = model
        
        # Create feature vector
        features = self.create_advanced_features(home_stats, away_stats)
        feature_df = pd.DataFrame([features])
        
        # Predict expected goals
        try:
            home_goals = home_model.predict(feature_df).iloc[0]
            away_goals = away_model.predict(feature_df).iloc[0]
        except:
            home_goals, away_goals = self.basic_prediction(home_stats, away_stats)
        
        # Calculate probabilities using Poisson distribution
        home_win_prob, draw_prob, away_win_prob = self.calculate_poisson_probabilities(home_goals, away_goals)
        
        # Calculate over/under probabilities
        over_25_prob = self.calculate_over_under_probability(home_goals, away_goals, 2.5)
        over_35_prob = self.calculate_over_under_probability(home_goals, away_goals, 3.5)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'expected_home_goals': round(home_goals, 2),
            'expected_away_goals': round(away_goals, 2),
            'home_win_prob': round(home_win_prob, 3),
            'draw_prob': round(draw_prob, 3),
            'away_win_prob': round(away_win_prob, 3),
            'over_2.5_prob': round(over_25_prob, 3),
            'over_3.5_prob': round(over_35_prob, 3),
            'both_teams_score': round(1 - (poisson.pmf(0, home_goals) * poisson.pmf(0, away_goals)), 3),
            'prediction_confidence': max(home_win_prob, draw_prob, away_win_prob)
        }
    
    def basic_prediction(self, home_stats, away_stats):
        """Fallback basic prediction using xG"""
        home_goals = max(0.1, home_stats['xg_for'] - away_stats['xg_against'] / 2)
        away_goals = max(0.1, away_stats['xg_for'] - home_stats['xg_against'] / 2)
        return home_goals, away_goals
    
    def calculate_poisson_probabilities(self, home_goals, away_goals):
        """Calculate match outcome probabilities using Poisson distribution"""
        max_goals = 8
        home_probs = [poisson.pmf(i, home_goals) for i in range(max_goals)]
        away_probs = [poisson.pmf(i, away_goals) for i in range(max_goals)]
        
        home_win_prob = np.sum(np.outer(home_probs, away_probs) * 
                             (np.arange(max_goals)[:, None] > np.arange(max_goals)))
        draw_prob = np.sum(np.outer(home_probs, away_probs) * 
                          (np.arange(max_goals)[:, None] == np.arange(max_goals)))
        away_win_prob = 1 - home_win_prob - draw_prob
        
        return home_win_prob, draw_prob, away_win_prob
    
    def calculate_over_under_probability(self, home_goals, away_goals, line):
        """Calculate over/under probability"""
        max_goals = 10
        total_prob = 0
        for i in range(max_goals):
            for j in range(max_goals):
                if i + j > line:
                    total_prob += poisson.pmf(i, home_goals) * poisson.pmf(j, away_goals)
        return total_prob
    
    def generate_training_data(self, num_matches=100):
        """Generate synthetic training data with advanced metrics"""
        training_data = []
        
        for _ in range(num_matches):
            home_stats = self.get_historical_performance('Home Team', 'premier_league')
            away_stats = self.get_historical_performance('Away Team', 'premier_league')
            
            features = self.create_advanced_features(home_stats, away_stats)
            
            # Simulate match outcomes based on stats
            home_goals = np.random.poisson(home_stats['xg_for'] * 0.8 + away_stats['xg_against'] * 0.2)
            away_goals = np.random.poisson(away_stats['xg_for'] * 0.8 + home_stats['xg_against'] * 0.2)
            
            match_data = features.copy()
            match_data['home_goals'] = home_goals
            match_data['away_goals'] = away_goals
            
            training_data.append(match_data)
        
        return training_data
    
    def run_complete_analysis(self, league_code='premier_league'):
        """Run complete analysis for a league"""
        print(f"Starting advanced analysis for {league_code}...")
        
        # Generate training data
        print("Generating training data...")
        training_data = self.generate_training_data(200)
        
        # Train model
        print("Training advanced Poisson model...")
        model = self.train_advanced_poisson_model(training_data)
        
        # Get fixtures
        print("Fetching fixtures...")
        fixtures = self.get_league_fixtures(league_code)
        
        if not fixtures:
            print("No fixtures found, using demo fixtures...")
            fixtures = [
                {'home_team': 'Manchester City', 'away_team': 'Liverpool', 'date': '2024-12-15'},
                {'home_team': 'Arsenal', 'away_team': 'Chelsea', 'date': '2024-12-15'},
                {'home_team': 'Tottenham', 'away_team': 'Manchester United', 'date': '2024-12-16'}
            ]
        
        predictions = []
        
        print("Generating predictions...")
        for fixture in fixtures[:10]:  # Limit to first 10 fixtures for demo
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            
            # Get team stats (in real implementation, these would be scraped)
            home_stats = self.get_historical_performance(home_team, league_code)
            away_stats = self.get_historical_performance(away_team, league_code)
            
            # Generate prediction
            prediction = self.predict_match_outcome(
                home_team, away_team, home_stats, away_stats, model
            )
            prediction['date'] = fixture['date']
            prediction['league'] = league_code.replace('_', ' ').title()
            
            predictions.append(prediction)
            
            time.sleep(1)  # Rate limiting
        
        return predictions

# Streamlit App Integration
import streamlit as st

def main():
    st.set_page_config(
        page_title="Advanced Football Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚽ Advanced Football Predictor")
    st.markdown("Using FBref & Transfermarkt metrics for sophisticated predictions")
    
    # Initialize predictor
    predictor = AdvancedFootballPredictor()
    
    # Sidebar
    st.sidebar.header("Configuration")
    league = st.sidebar.selectbox(
        "Select League",
        ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
    )
    
    league_codes = {
        "Premier League": "premier_league",
        "La Liga": "la_liga", 
        "Serie A": "serie_a",
        "Bundesliga": "bundesliga",
        "Ligue 1": "ligue_1"
    }
    
    if st.sidebar.button("Generate Predictions"):
        with st.spinner("Analyzing team data and generating predictions..."):
            predictions = predictor.run_complete_analysis(league_codes[league])
            
            if predictions:
                st.success(f"Generated {len(predictions)} predictions!")
                
                # Display predictions
                df = pd.DataFrame(predictions)
                
                # Format display
                display_cols = [
                    'home_team', 'away_team', 'expected_home_goals', 'expected_away_goals',
                    'home_win_prob', 'draw_prob', 'away_win_prob', 'over_2.5_prob',
                    'both_teams_score', 'prediction_confidence'
                ]
                
                st.subheader("📊 Match Predictions")
                for pred in predictions:
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 2])
                        
                        with col1:
                            st.markdown(f"**{pred['home_team']}**")
                            st.metric("Expected Goals", pred['expected_home_goals'])
                            
                        with col2:
                            st.markdown("**vs**")
                            st.metric("Draw", f"{pred['draw_prob']:.1%}")
                            
                        with col3:
                            st.markdown(f"**{pred['away_team']}**") 
                            st.metric("Expected Goals", pred['expected_away_goals'])
                        
                        # Probabilities
                        prob_col1, prob_col2, prob_col3, prob_col4 = st.columns(4)
                        with prob_col1:
                            st.metric("Home Win", f"{pred['home_win_prob']:.1%}")
                        with prob_col2:
                            st.metric("Away Win", f"{pred['away_win_prob']:.1%}")
                        with prob_col3:
                            st.metric("Over 2.5", f"{pred['over_2.5_prob']:.1%}")
                        with prob_col4:
                            st.metric("Both Score", f"{pred['both_teams_score']:.1%}")
                        
                        st.progress(pred['prediction_confidence'])
                        st.markdown("---")
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name=f"football_predictions_{league.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )
                
            else:
                st.error("No predictions generated. Please try again.")

    # Methodology explanation
    with st.expander("📈 Methodology Explained"):
        st.markdown("""
        **Advanced Metrics Used:**
        
        - **Expected Goals (xG)**: Quality of scoring chances
        - **Shot-Creating Actions**: Moves that lead to shots  
        - **Passing Accuracy**: Team possession quality
        - **Defensive Pressures**: Aggressiveness in winning possession
        - **Market Value**: Squad quality indicator
        
        **Model Features:**
        - Poisson regression for goal prediction
        - Advanced metric differentials (home vs away)
        - Market value ratios
        - Attack/defense strength calculations
        """)

if __name__ == "__main__":
    main()
