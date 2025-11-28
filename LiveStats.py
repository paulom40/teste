# Leagues.py - FOOTBALL PREDICTOR PRO v9.0 (COMPLETE 1ST & 2ND DIVISION COVERAGE)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam, binom
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import base64

warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Predictor Pro v9.0", layout="wide")
st.markdown("""
# 🚀 Football Predictor Pro v9.0
**Complete European Coverage • 45-Minute Live Analysis • 1st & 2nd Division Models**
*Professional Grade Predictions for All Major Leagues*
""")

# ================================
# COMPREHENSIVE LEAGUE PROFILES - ALL MAJOR 1ST & 2ND DIVISIONS
# ================================
LEAGUE_PROFILES = {
    # === TOP TIER LEAGUES ===
    'Premier League': {
        'avg_goals_per_game': 2.82,
        'home_advantage': 1.35,
        'avg_shots': 24.5,
        'avg_sot': 8.2,
        'avg_corners': 10.8,
        'avg_dangerous_attacks': 85,
        'avg_cards': 3.8,
        'pace_factor': 1.15,
        'physicality': 1.20,
        'style': 'High intensity, direct play',
        'tier': 1,
        'over_25_goals_rate': 0.52,
        'btts_rate': 0.48,
        'second_half_goals_ratio': 0.55,
        'comeback_rate': 0.28,
        'fatigue_factor': 0.85,
        'volatility': 0.75  # Lower volatility in top tier
    },
    'La Liga': {
        'avg_goals_per_game': 2.65,
        'home_advantage': 1.28,
        'avg_shots': 22.8,
        'avg_sot': 7.5,
        'avg_corners': 9.5,
        'avg_dangerous_attacks': 78,
        'avg_cards': 4.2,
        'pace_factor': 1.05,
        'physicality': 0.95,
        'style': 'Technical, possession-focused',
        'tier': 1,
        'over_25_goals_rate': 0.45,
        'btts_rate': 0.42,
        'second_half_goals_ratio': 0.52,
        'comeback_rate': 0.25,
        'fatigue_factor': 0.88,
        'volatility': 0.70
    },
    'Serie A': {
        'avg_goals_per_game': 2.58,
        'home_advantage': 1.25,
        'avg_shots': 21.5,
        'avg_sot': 7.0,
        'avg_corners': 9.2,
        'avg_dangerous_attacks': 72,
        'avg_cards': 4.5,
        'pace_factor': 0.95,
        'physicality': 1.05,
        'style': 'Tactical, defensive discipline',
        'tier': 1,
        'over_25_goals_rate': 0.41,
        'btts_rate': 0.38,
        'second_half_goals_ratio': 0.48,
        'comeback_rate': 0.22,
        'fatigue_factor': 0.92,
        'volatility': 0.65
    },
    'Bundesliga': {
        'avg_goals_per_game': 3.05,
        'home_advantage': 1.32,
        'avg_shots': 26.2,
        'avg_sot': 8.8,
        'avg_corners': 11.2,
        'avg_dangerous_attacks': 92,
        'avg_cards': 3.5,
        'pace_factor': 1.25,
        'physicality': 1.15,
        'style': 'High-pressing, counter-attacking',
        'tier': 1,
        'over_25_goals_rate': 0.58,
        'btts_rate': 0.52,
        'second_half_goals_ratio': 0.58,
        'comeback_rate': 0.32,
        'fatigue_factor': 0.80,
        'volatility': 0.80
    },
    'Ligue 1': {
        'avg_goals_per_game': 2.68,
        'home_advantage': 1.30,
        'avg_shots': 23.5,
        'avg_sot': 7.8,
        'avg_corners': 10.0,
        'avg_dangerous_attacks': 80,
        'avg_cards': 4.0,
        'pace_factor': 1.10,
        'physicality': 1.10,
        'style': 'Athletic, transition-focused',
        'tier': 1,
        'over_25_goals_rate': 0.46,
        'btts_rate': 0.44,
        'second_half_goals_ratio': 0.53,
        'comeback_rate': 0.26,
        'fatigue_factor': 0.87,
        'volatility': 0.72
    },
    
    # === SECOND DIVISION LEAGUES ===
    
    # ENGLAND
    'Championship (ENG)': {
        'avg_goals_per_game': 2.65,
        'home_advantage': 1.38,
        'avg_shots': 23.8,
        'avg_sot': 7.6,
        'avg_corners': 10.5,
        'avg_dangerous_attacks': 82,
        'avg_cards': 4.2,
        'pace_factor': 1.18,
        'physicality': 1.25,
        'style': 'Physical, high-tempo, competitive',
        'tier': 2,
        'over_25_goals_rate': 0.48,
        'btts_rate': 0.46,
        'second_half_goals_ratio': 0.56,
        'comeback_rate': 0.30,
        'fatigue_factor': 0.82,
        'volatility': 0.85
    },
    'League One (ENG)': {
        'avg_goals_per_game': 2.55,
        'home_advantage': 1.40,
        'avg_shots': 22.5,
        'avg_sot': 7.2,
        'avg_corners': 10.0,
        'avg_dangerous_attacks': 78,
        'avg_cards': 4.5,
        'pace_factor': 1.15,
        'physicality': 1.28,
        'style': 'Direct, physical, high intensity',
        'tier': 2,
        'over_25_goals_rate': 0.45,
        'btts_rate': 0.44,
        'second_half_goals_ratio': 0.54,
        'comeback_rate': 0.32,
        'fatigue_factor': 0.80,
        'volatility': 0.88
    },
    
    # SPAIN
    'La Liga 2 (ESP)': {
        'avg_goals_per_game': 2.25,
        'home_advantage': 1.32,
        'avg_shots': 20.5,
        'avg_sot': 6.5,
        'avg_corners': 8.8,
        'avg_dangerous_attacks': 68,
        'avg_cards': 4.8,
        'pace_factor': 0.92,
        'physicality': 1.08,
        'style': 'Technical, tactical, lower scoring',
        'tier': 2,
        'over_25_goals_rate': 0.35,
        'btts_rate': 0.32,
        'second_half_goals_ratio': 0.50,
        'comeback_rate': 0.26,
        'fatigue_factor': 0.90,
        'volatility': 0.78
    },
    
    # ITALY
    'Serie B (ITA)': {
        'avg_goals_per_game': 2.35,
        'home_advantage': 1.28,
        'avg_shots': 19.8,
        'avg_sot': 6.2,
        'avg_corners': 8.5,
        'avg_dangerous_attacks': 65,
        'avg_cards': 5.2,
        'pace_factor': 0.88,
        'physicality': 1.12,
        'style': 'Defensive, tactical battles',
        'tier': 2,
        'over_25_goals_rate': 0.38,
        'btts_rate': 0.35,
        'second_half_goals_ratio': 0.47,
        'comeback_rate': 0.24,
        'fatigue_factor': 0.92,
        'volatility': 0.75
    },
    
    # GERMANY
    '2. Bundesliga (GER)': {
        'avg_goals_per_game': 2.85,
        'home_advantage': 1.35,
        'avg_shots': 24.0,
        'avg_sot': 8.0,
        'avg_corners': 10.2,
        'avg_dangerous_attacks': 85,
        'avg_cards': 3.8,
        'pace_factor': 1.15,
        'physicality': 1.18,
        'style': 'Attacking, high-pressing like Bundesliga',
        'tier': 2,
        'over_25_goals_rate': 0.52,
        'btts_rate': 0.48,
        'second_half_goals_ratio': 0.57,
        'comeback_rate': 0.31,
        'fatigue_factor': 0.82,
        'volatility': 0.82
    },
    
    # FRANCE
    'Ligue 2 (FRA)': {
        'avg_goals_per_game': 2.35,
        'home_advantage': 1.33,
        'avg_shots': 21.2,
        'avg_sot': 6.8,
        'avg_corners': 9.2,
        'avg_dangerous_attacks': 72,
        'avg_cards': 4.3,
        'pace_factor': 1.02,
        'physicality': 1.15,
        'style': 'Physical, defensive organization',
        'tier': 2,
        'over_25_goals_rate': 0.38,
        'btts_rate': 0.36,
        'second_half_goals_ratio': 0.49,
        'comeback_rate': 0.27,
        'fatigue_factor': 0.88,
        'volatility': 0.80
    },
    
    # NETHERLANDS
    'Eredivisie': {
        'avg_goals_per_game': 3.15,
        'home_advantage': 1.38,
        'avg_shots': 27.0,
        'avg_sot': 9.2,
        'avg_corners': 11.8,
        'avg_dangerous_attacks': 95,
        'avg_cards': 3.2,
        'pace_factor': 1.30,
        'physicality': 0.90,
        'style': 'Attacking, open play',
        'tier': 1,
        'over_25_goals_rate': 0.62,
        'btts_rate': 0.55,
        'second_half_goals_ratio': 0.60,
        'comeback_rate': 0.35,
        'fatigue_factor': 0.78,
        'volatility': 0.85
    },
    'Eerste Divisie (NED)': {
        'avg_goals_per_game': 3.05,
        'home_advantage': 1.40,
        'avg_shots': 25.5,
        'avg_sot': 8.5,
        'avg_corners': 11.0,
        'avg_dangerous_attacks': 88,
        'avg_cards': 3.5,
        'pace_factor': 1.22,
        'physicality': 0.95,
        'style': 'Very attacking, youth development',
        'tier': 2,
        'over_25_goals_rate': 0.58,
        'btts_rate': 0.52,
        'second_half_goals_ratio': 0.58,
        'comeback_rate': 0.34,
        'fatigue_factor': 0.80,
        'volatility': 0.90
    },
    
    # PORTUGAL
    'Primeira Liga': {
        'avg_goals_per_game': 2.62,
        'home_advantage': 1.33,
        'avg_shots': 22.0,
        'avg_sot': 7.3,
        'avg_corners': 9.8,
        'avg_dangerous_attacks': 75,
        'avg_cards': 4.8,
        'pace_factor': 1.08,
        'physicality': 1.00,
        'style': 'Technical, competitive',
        'tier': 1,
        'over_25_goals_rate': 0.44,
        'btts_rate': 0.40,
        'second_half_goals_ratio': 0.51,
        'comeback_rate': 0.25,
        'fatigue_factor': 0.89,
        'volatility': 0.75
    },
    'Liga Portugal 2': {
        'avg_goals_per_game': 2.45,
        'home_advantage': 1.35,
        'avg_shots': 20.8,
        'avg_sot': 6.6,
        'avg_corners': 9.0,
        'avg_dangerous_attacks': 70,
        'avg_cards': 5.0,
        'pace_factor': 1.05,
        'physicality': 1.05,
        'style': 'Technical, developing talent',
        'tier': 2,
        'over_25_goals_rate': 0.40,
        'btts_rate': 0.38,
        'second_half_goals_ratio': 0.49,
        'comeback_rate': 0.28,
        'fatigue_factor': 0.87,
        'volatility': 0.82
    },
    
    # TURKEY
    'Super Lig': {
        'avg_goals_per_game': 2.75,
        'home_advantage': 1.42,
        'avg_shots': 23.8,
        'avg_sot': 7.6,
        'avg_corners': 10.3,
        'avg_dangerous_attacks': 80,
        'avg_cards': 4.5,
        'pace_factor': 1.08,
        'physicality': 1.12,
        'style': 'Passionate, home-focused',
        'tier': 1,
        'over_25_goals_rate': 0.48,
        'btts_rate': 0.45,
        'second_half_goals_ratio': 0.54,
        'comeback_rate': 0.29,
        'fatigue_factor': 0.85,
        'volatility': 0.82
    },
    
    # BELGIUM
    'Belgian Pro League': {
        'avg_goals_per_game': 2.88,
        'home_advantage': 1.35,
        'avg_shots': 24.0,
        'avg_sot': 8.0,
        'avg_corners': 10.5,
        'avg_dangerous_attacks': 82,
        'avg_cards': 3.8,
        'pace_factor': 1.12,
        'physicality': 1.08,
        'style': 'Balanced, physical',
        'tier': 1,
        'over_25_goals_rate': 0.52,
        'btts_rate': 0.48,
        'second_half_goals_ratio': 0.55,
        'comeback_rate': 0.30,
        'fatigue_factor': 0.84,
        'volatility': 0.80
    },
    
    # SCOTLAND
    'Scottish Premiership': {
        'avg_goals_per_game': 2.70,
        'home_advantage': 1.38,
        'avg_shots': 23.2,
        'avg_sot': 7.5,
        'avg_corners': 10.0,
        'avg_dangerous_attacks': 78,
        'avg_cards': 4.0,
        'pace_factor': 1.10,
        'physicality': 1.20,
        'style': 'Physical, competitive',
        'tier': 1,
        'over_25_goals_rate': 0.46,
        'btts_rate': 0.42,
        'second_half_goals_ratio': 0.53,
        'comeback_rate': 0.27,
        'fatigue_factor': 0.86,
        'volatility': 0.78
    },
    'Scottish Championship': {
        'avg_goals_per_game': 2.55,
        'home_advantage': 1.38,
        'avg_shots': 22.5,
        'avg_sot': 7.2,
        'avg_corners': 9.8,
        'avg_dangerous_attacks': 75,
        'avg_cards': 4.2,
        'pace_factor': 1.12,
        'physicality': 1.20,
        'style': 'Physical, direct, competitive',
        'tier': 2,
        'over_25_goals_rate': 0.44,
        'btts_rate': 0.41,
        'second_half_goals_ratio': 0.52,
        'comeback_rate': 0.29,
        'fatigue_factor': 0.84,
        'volatility': 0.85
    },
    
    # AUSTRIA
    'Austrian Bundesliga': {
        'avg_goals_per_game': 2.95,
        'home_advantage': 1.32,
        'avg_shots': 24.8,
        'avg_sot': 8.2,
        'avg_corners': 10.5,
        'avg_dangerous_attacks': 85,
        'avg_cards': 3.6,
        'pace_factor': 1.15,
        'physicality': 1.10,
        'style': 'Attacking, developing',
        'tier': 1,
        'over_25_goals_rate': 0.54,
        'btts_rate': 0.50,
        'second_half_goals_ratio': 0.56,
        'comeback_rate': 0.30,
        'fatigue_factor': 0.83,
        'volatility': 0.80
    },
    'Austrian 2. Liga': {
        'avg_goals_per_game': 2.75,
        'home_advantage': 1.32,
        'avg_shots': 23.2,
        'avg_sot': 7.8,
        'avg_corners': 10.0,
        'avg_dangerous_attacks': 78,
        'avg_cards': 4.0,
        'pace_factor': 1.08,
        'physicality': 1.10,
        'style': 'Developing, tactical',
        'tier': 2,
        'over_25_goals_rate': 0.50,
        'btts_rate': 0.46,
        'second_half_goals_ratio': 0.54,
        'comeback_rate': 0.31,
        'fatigue_factor': 0.82,
        'volatility': 0.85
    },
    
    # SWITZERLAND
    'Swiss Super League': {
        'avg_goals_per_game': 2.88,
        'home_advantage': 1.30,
        'avg_shots': 24.2,
        'avg_sot': 8.1,
        'avg_corners': 10.3,
        'avg_dangerous_attacks': 82,
        'avg_cards': 3.7,
        'pace_factor': 1.12,
        'physicality': 1.05,
        'style': 'Balanced, technical',
        'tier': 1,
        'over_25_goals_rate': 0.52,
        'btts_rate': 0.48,
        'second_half_goals_ratio': 0.55,
        'comeback_rate': 0.29,
        'fatigue_factor': 0.85,
        'volatility': 0.78
    },
    'Swiss Challenge League': {
        'avg_goals_per_game': 2.82,
        'home_advantage': 1.30,
        'avg_shots': 23.8,
        'avg_sot': 7.9,
        'avg_corners': 10.2,
        'avg_dangerous_attacks': 80,
        'avg_cards': 4.0,
        'pace_factor': 1.10,
        'physicality': 1.08,
        'style': 'Balanced, developing',
        'tier': 2,
        'over_25_goals_rate': 0.50,
        'btts_rate': 0.46,
        'second_half_goals_ratio': 0.54,
        'comeback_rate': 0.30,
        'fatigue_factor': 0.83,
        'volatility': 0.84
    }
}

# ================================
# ENHANCED PREDICTION ENGINE WITH TIER-SPECIFIC MODELS
# ================================
class Advanced45MinutePredictor:
    """
    Advanced prediction system with tier-specific models for 1st and 2nd divisions
    """
    
    def __init__(self, league='Premier League'):
        self.league = league
        self.league_profile = LEAGUE_PROFILES.get(league, LEAGUE_PROFILES['Premier League'])
        self.is_second_tier = self.league_profile['tier'] == 2
        
        # Tier-specific weights
        if self.is_second_tier:
            # Second tier: More weight to momentum and volatility
            self.weights = {
                'xg_weight': 0.30,
                'momentum_weight': 0.30,
                'situation_weight': 0.20,
                'fatigue_weight': 0.10,
                'psychological_weight': 0.10
            }
        else:
            # First tier: More weight to xG and tactical factors
            self.weights = {
                'xg_weight': 0.35,
                'momentum_weight': 0.25,
                'situation_weight': 0.20,
                'fatigue_weight': 0.10,
                'psychological_weight': 0.10
            }
    
    def calculate_advanced_momentum(self, first_half_stats):
        """
        Calculate comprehensive momentum score with tier-specific adjustments
        """
        home_momentum = 50
        away_momentum = 50
        
        # Base calculations (same as before)
        total_xg = first_half_stats['home_xg'] + first_half_stats['away_xg']
        if total_xg > 0:
            home_xg_share = first_half_stats['home_xg'] / total_xg
            home_momentum += (home_xg_share - 0.5) * 40
            away_momentum += (0.5 - home_xg_share) * 40
        
        home_shot_efficiency = first_half_stats['home_sot'] / max(1, first_half_stats['home_shots'])
        away_shot_efficiency = first_half_stats['away_sot'] / max(1, first_half_stats['away_shots'])
        home_momentum += (home_shot_efficiency - 0.3) * 25
        away_momentum += (away_shot_efficiency - 0.3) * 25
        
        total_da = first_half_stats['home_dangerous_attacks'] + first_half_stats['away_dangerous_attacks']
        if total_da > 0:
            home_da_share = first_half_stats['home_dangerous_attacks'] / total_da
            home_momentum += (home_da_share - 0.5) * 20
            away_momentum += (0.5 - home_da_share) * 20
        
        total_corners = first_half_stats['home_corners'] + first_half_stats['away_corners']
        if total_corners > 0:
            home_corner_share = first_half_stats['home_corners'] / total_corners
            home_momentum += (home_corner_share - 0.5) * 15
            away_momentum += (0.5 - home_corner_share) * 15
        
        home_goals_vs_xg = first_half_stats['home_goals'] - first_half_stats['home_xg']
        away_goals_vs_xg = first_half_stats['away_goals'] - first_half_stats['away_xg']
        home_momentum += home_goals_vs_xg * 8
        away_momentum += away_goals_vs_xg * 8
        
        # TIER-SPECIFIC ADJUSTMENTS
        pace_factor = self.league_profile['pace_factor']
        volatility = self.league_profile['volatility']
        
        # Second tier: Higher volatility and momentum swings
        if self.is_second_tier:
            home_momentum *= pace_factor * volatility * 1.1
            away_momentum *= pace_factor * volatility * 1.1
        else:
            home_momentum *= pace_factor
            away_momentum *= pace_factor
        
        return {
            'home': max(10, min(90, home_momentum)),
            'away': max(10, min(90, away_momentum)),
            'dominance_ratio': home_momentum / max(1, away_momentum),
            'volatility_index': volatility
        }
    
    def predict_second_half_goals_advanced(self, first_half_stats, momentum):
        """
        Advanced second half goal prediction with tier-specific models
        """
        minutes_played = first_half_stats.get('minutes_played', 45)
        
        # Calculate per-minute rates
        home_xg_rate = first_half_stats['home_xg'] / max(1, minutes_played)
        away_xg_rate = first_half_stats['away_xg'] / max(1, minutes_played)
        
        # League-average second half adjustment
        second_half_ratio = self.league_profile['second_half_goals_ratio']
        
        # Momentum-adjusted rates
        momentum_factor_home = momentum['home'] / 50
        momentum_factor_away = momentum['away'] / 50
        
        # Psychological factors
        score_diff = first_half_stats['home_goals'] - first_half_stats['away_goals']
        psychological_factors = self._calculate_psychological_factors(score_diff, momentum)
        
        # Fatigue modeling
        fatigue_factors = self._calculate_fatigue_factors(first_half_stats)
        
        # Tactical adjustments
        tactical_factors = self._calculate_tactical_adjustments(first_half_stats, score_diff)
        
        # COMBINE FACTORS WITH TIER-SPECIFIC MODIFICATIONS
        base_home_xg = home_xg_rate * 45 * momentum_factor_home
        base_away_xg = away_xg_rate * 45 * momentum_factor_away
        
        # Second tier: Higher volatility in goal expectations
        if self.is_second_tier:
            volatility_multiplier = 1.0 + (self.league_profile['volatility'] - 0.75) * 0.4
            base_home_xg *= volatility_multiplier
            base_away_xg *= volatility_multiplier
        
        home_second_half_xg = (base_home_xg * 
                              psychological_factors['home_attack'] * 
                              fatigue_factors['home_attack'] *
                              tactical_factors['home_attack'])
        
        away_second_half_xg = (base_away_xg * 
                              psychological_factors['away_attack'] * 
                              fatigue_factors['away_attack'] *
                              tactical_factors['away_attack'])
        
        # Apply league-specific second half goal ratio
        home_second_half_xg *= second_half_ratio
        away_second_half_xg *= second_half_ratio
        
        # Home advantage in second half (adjusted for tier)
        home_advantage_2h = self.league_profile['home_advantage'] * (0.75 if self.is_second_tier else 0.8)
        home_second_half_xg *= home_advantage_2h
        away_second_half_xg /= home_advantage_2h
        
        return self._calculate_goal_probabilities(home_second_half_xg, away_second_half_xg)
    
    def _calculate_psychological_factors(self, score_diff, momentum):
        """
        Calculate psychological impact with tier-specific adjustments
        """
        factors = {
            'home_attack': 1.0,
            'away_attack': 1.0,
            'home_defense': 1.0,
            'away_defense': 1.0
        }
        
        # Base psychological factors
        if score_diff > 0:  # Home leading
            if score_diff >= 2:
                factors['home_attack'] = 0.7
                factors['home_defense'] = 1.2
                factors['away_attack'] = 1.4
            else:
                factors['home_attack'] = 0.9
                factors['away_attack'] = 1.2
        
        elif score_diff < 0:  # Away leading
            if score_diff <= -2:
                factors['away_attack'] = 0.7
                factors['away_defense'] = 1.2
                factors['home_attack'] = 1.4
            else:
                factors['away_attack'] = 0.9
                factors['home_attack'] = 1.2
        
        # Second tier: Stronger momentum effects and comeback mentality
        if self.is_second_tier:
            comeback_bonus = 1.15  # Higher comeback likelihood in lower tiers
            if score_diff > 0 and momentum['away'] > 60:
                factors['away_attack'] *= comeback_bonus
            elif score_diff < 0 and momentum['home'] > 60:
                factors['home_attack'] *= comeback_bonus
        
        # Momentum override
        if momentum['home'] > 70:
            factors['home_attack'] = min(1.3, factors['home_attack'] * 1.2)
        if momentum['away'] > 70:
            factors['away_attack'] = min(1.3, factors['away_attack'] * 1.2)
        
        return factors
    
    def _calculate_fatigue_factors(self, first_half_stats):
        """
        Model physical fatigue with tier-specific adjustments
        """
        total_actions = (first_half_stats['home_shots'] + first_half_stats['away_shots'] +
                        first_half_stats['home_dangerous_attacks'] + first_half_stats['away_dangerous_attacks'])
        
        base_fatigue = self.league_profile['fatigue_factor']
        
        # Second tier: Generally higher fatigue impact due to conditioning
        if self.is_second_tier:
            base_fatigue *= 0.95  # 5% more fatigue impact
        
        intensity_factor = min(1.5, total_actions / 60)
        
        home_fatigue = base_fatigue * (0.9 + 0.1 * intensity_factor)
        away_fatigue = base_fatigue * (0.8 + 0.2 * intensity_factor)
        
        return {
            'home_attack': home_fatigue,
            'home_defense': home_fatigue * 0.95,
            'away_attack': away_fatigue,
            'away_defense': away_fatigue * 0.95
        }
    
    def _calculate_tactical_adjustments(self, first_half_stats, score_diff):
        """
        Predict tactical adjustments with tier-specific patterns
        """
        factors = {
            'home_attack': 1.0,
            'away_attack': 1.0
        }
        
        xg_diff = first_half_stats['home_xg'] - first_half_stats['away_xg']
        goal_diff = first_half_stats['home_goals'] - first_half_stats['away_goals']
        
        # Base tactical adjustments
        if goal_diff < 0 and xg_diff > 0.5:
            factors['home_attack'] = 1.3
        elif goal_diff > 0 and xg_diff < -0.5:
            factors['away_attack'] = 1.3
        
        if goal_diff > 0 and xg_diff < -0.8:
            factors['home_attack'] = 0.8
            factors['away_attack'] = 1.2
        
        # Second tier: More aggressive tactical changes
        if self.is_second_tier:
            aggression_factor = 1.1
            factors['home_attack'] = min(1.4, factors['home_attack'] * aggression_factor)
            factors['away_attack'] = min(1.4, factors['away_attack'] * aggression_factor)
        
        return factors
    
    def _calculate_goal_probabilities(self, home_xg, away_xg):
        """Calculate goal probabilities using Poisson distribution"""
        home_probs = [poisson.pmf(i, home_xg) for i in range(6)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(6)]
        
        max_prob = 0
        most_likely_score = "0-0"
        for i in range(6):
            for j in range(6):
                prob = home_probs[i] * away_probs[j]
                if prob > max_prob:
                    max_prob = prob
                    most_likely_score = f"{i}-{j}"
        
        home_win = sum(home_probs[i] * sum(away_probs[:i]) for i in range(1, 6))
        draw = sum(home_probs[i] * away_probs[i] for i in range(6))
        away_win = sum(away_probs[j] * sum(home_probs[:j]) for j in range(1, 6))
        
        btts = (1 - poisson.cdf(0, home_xg)) * (1 - poisson.cdf(0, away_xg))
        
        return {
            'home_xg': round(home_xg, 2),
           
