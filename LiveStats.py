# Leagues.py - FOOTBALL PREDICTOR PRO v8.0 (LIVE IN-GAME PREDICTION)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import base64

warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Predictor Pro v8.0", layout="wide")
st.markdown("""
# Football Predictor Pro v8.0
**Live In-Game Prediction • First Half Stats Analysis • Second Half Forecast**
*Complete European Coverage + Advanced Betting Markets*
""")

# ================================
# LEAGUE PROFILES - EXPANDED WITH 2ND DIVISIONS
# ================================
LEAGUE_PROFILES = {
    # TOP TIER LEAGUES
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
        'btts_rate': 0.48
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
        'btts_rate': 0.42
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
        'btts_rate': 0.38
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
        'btts_rate': 0.52
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
        'btts_rate': 0.44
    },
    
    # SECOND DIVISION LEAGUES
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
        'btts_rate': 0.46
    },
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
        'btts_rate': 0.32
    },
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
        'btts_rate': 0.35
    },
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
        'btts_rate': 0.48
    }
}

# ================================
# BETTING MARKET PREDICTOR
# ================================
class BettingMarketPredictor:
    """Predict various betting markets including goal lines, corners, cards"""
    
    def __init__(self, league_profile):
        self.league_profile = league_profile
    
    def predict_goal_lines(self, first_half_stats, second_half_pred, full_time_pred):
        """Predict Over/Under goal lines with probabilities"""
        
        current_goals = first_half_stats['home_goals'] + first_half_stats['away_goals']
        expected_ft_goals = full_time_pred['ft_home_xg'] + full_time_pred['ft_away_xg']
        expected_2h_goals = second_half_pred['second_half_xg_home'] + second_half_pred['second_half_xg_away']
        
        # Common goal lines
        goal_lines = [0.5, 1.5, 2.5, 3.5, 4.5]
        predictions = {}
        
        for line in goal_lines:
            # Full time probabilities
            over_prob_ft = self._calculate_over_probability(expected_ft_goals, line)
            under_prob_ft = 100 - over_prob_ft
            
            # Second half probabilities
            over_prob_2h = self._calculate_over_probability(expected_2h_goals, line - current_goals)
            under_prob_2h = 100 - over_prob_2h
            
            predictions[f'over_{line}_ft'] = round(over_prob_ft, 1)
            predictions[f'under_{line}_ft'] = round(under_prob_ft, 1)
            predictions[f'over_{line}_2h'] = round(over_prob_2h, 1)
            predictions[f'under_{line}_2h'] = round(under_prob_2h, 1)
            
            # Value bets (probability > implied probability + margin)
            implied_prob = 100 / (1 + np.exp(-0.3 * (line - expected_ft_goals)))
            predictions[f'value_over_{line}'] = over_prob_ft > implied_prob + 5
            predictions[f'value_under_{line}'] = under_prob_ft > implied_prob + 5
        
        return predictions
    
    def _calculate_over_probability(self, expected_goals, line):
        """Calculate probability of going over a goal line"""
        if line <= 0:
            return 100.0
        
        # Use Poisson distribution for over probability
        prob_under = poisson.cdf(line - 0.1, expected_goals)
        prob_over = 1 - prob_under
        return prob_over * 100
    
    def predict_corners_lines(self, first_half_stats, stats_pred):
        """Predict Over/Under corner lines"""
        
        current_corners = first_half_stats['home_corners'] + first_half_stats['away_corners']
        expected_2h_corners = stats_pred['home_corners_2h'] + stats_pred['away_corners_2h']
        expected_ft_corners = current_corners + expected_2h_corners
        
        # Common corner lines
        corner_lines = [6.5, 7.5, 8.5, 9.5, 10.5]
        predictions = {}
        
        for line in corner_lines:
            # Adjust for league average corners
            league_avg = self.league_profile['avg_corners']
            adjustment = expected_ft_corners / league_avg
            
            # Use adjusted Poisson for corners
            over_prob = self._calculate_over_probability(expected_ft_corners * 1.1, line)
            under_prob = 100 - over_prob
            
            predictions[f'corners_over_{line}'] = round(over_prob, 1)
            predictions[f'corners_under_{line}'] = round(under_prob, 1)
        
        return predictions
    
    def predict_both_teams_to_score(self, first_half_stats, second_half_pred):
        """Predict BTTS markets"""
        
        current_btts = first_half_stats['home_goals'] > 0 and first_half_stats['away_goals'] > 0
        
        # Probability both teams score in second half
        prob_home_scores_2h = 1 - np.exp(-second_half_pred['second_half_xg_home'])
        prob_away_scores_2h = 1 - np.exp(-second_half_pred['second_half_xg_away'])
        prob_btts_2h = prob_home_scores_2h * prob_away_scores_2h
        
        # Probability both teams score full time
        prob_home_scores_ft = 1 - np.exp(-second_half_pred['second_half_xg_home'])
        prob_away_scores_ft = 1 - np.exp(-second_half_pred['second_half_xg_away'])
        
        # If already scored, probability is higher
        if first_half_stats['home_goals'] > 0:
            prob_away_scores_ft *= 1.2  # Away team more likely to score if chasing
        if first_half_stats['away_goals'] > 0:
            prob_home_scores_ft *= 1.2  # Home team more likely to score if chasing
        
        prob_btts_ft = prob_home_scores_ft * prob_away_scores_ft
        
        return {
            'btts_2h_prob': round(prob_btts_2h * 100, 1),
            'btts_ft_prob': round(prob_btts_ft * 100, 1),
            'btts_yes': round(prob_btts_ft * 100, 1),
            'btts_no': round((1 - prob_btts_ft) * 100, 1)
        }
    
    def predict_win_to_nil(self, full_time_pred, first_half_stats):
        """Predict Win to Nil markets"""
        
        home_goals = first_half_stats['home_goals']
        away_goals = first_half_stats['away_goals']
        
        # Probability away team doesn't score in second half
        prob_away_no_goal_2h = np.exp(-full_time_pred['ft_away_xg'] + first_half_stats['away_xg'])
        prob_home_no_goal_2h = np.exp(-full_time_pred['ft_home_xg'] + first_half_stats['home_xg'])
        
        home_win_to_nil = full_time_pred['ft_home_win_prob'] / 100 * prob_away_no_goal_2h
        away_win_to_nil = full_time_pred['ft_away_win_prob'] / 100 * prob_home_no_goal_2h
        
        return {
            'home_win_to_nil': round(home_win_to_nil * 100, 1),
            'away_win_to_nil': round(away_win_to_nil * 100, 1)
        }
    
    def predict_double_chance(self, full_time_pred):
        """Predict Double Chance markets"""
        
        home_draw = full_time_pred['ft_home_win_prob'] + full_time_pred['ft_draw_prob']
        away_draw = full_time_pred['ft_away_win_prob'] + full_time_pred['ft_draw_prob']
        home_away = full_time_pred['ft_home_win_prob'] + full_time_pred['ft_away_win_prob']
        
        return {
            '1X': round(min(home_draw, 99.9), 1),
            'X2': round(min(away_draw, 99.9), 1),
            '12': round(min(home_away, 99.9), 1)
        }
    
    def predict_alternative_handicaps(self, full_time_pred, first_half_stats):
        """Predict Asian handicaps and alternative handicaps"""
        
        current_diff = first_half_stats['home_goals'] - first_half_stats['away_goals']
        expected_diff = full_time_pred['ft_home_xg'] - full_time_pred['ft_away_xg']
        
        handicaps = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]
        predictions = {}
        
        for handicap in handicaps:
            # Adjust expected difference by handicap
            adj_diff = expected_diff - handicap
            
            # Probability home covers handicap (home + handicap wins)
            if handicap % 1 == 0:  # Integer handicap
                prob_home = 1 - poisson.cdf(-adj_diff - 0.1, abs(adj_diff))
                prob_away = 1 - prob_home
                prob_push = poisson.pmf(-adj_diff, abs(adj_diff))
            else:  # Half handicap
                prob_home = 1 - poisson.cdf(-adj_diff, abs(adj_diff))
                prob_away = poisson.cdf(-adj_diff, abs(adj_diff))
                prob_push = 0
            
            predictions[f'ah_home_{handicap}'] = round(prob_home * 100, 1)
            predictions[f'ah_away_{handicap}'] = round(prob_away * 100, 1)
            if prob_push > 0:
                predictions[f'ah_push_{handicap}'] = round(prob_push * 100, 1)
        
        return predictions
    
    def predict_cards_lines(self, first_half_stats, momentum):
        """Predict Over/Under card lines"""
        
        # Base card expectation from league average
        league_avg_cards = self.league_profile['avg_cards']
        
        # Adjust based on match intensity and momentum difference
        momentum_diff = abs(momentum['home'] - momentum['away']) / 100
        intensity_factor = 1.0 + momentum_diff * 0.5
        
        # Derbies and important matches have more cards
        importance_factor = 1.2  # Assume important match
        
        expected_cards = league_avg_cards * intensity_factor * importance_factor
        
        # Common card lines
        card_lines = [2.5, 3.5, 4.5, 5.5]
        predictions = {}
        
        for line in card_lines:
            over_prob = self._calculate_over_probability(expected_cards, line)
            under_prob = 100 - over_prob
            
            predictions[f'cards_over_{line}'] = round(over_prob, 1)
            predictions[f'cards_under_{line}'] = round(under_prob, 1)
        
        return predictions

# ================================
# LIVE MATCH PREDICTOR
# ================================
class LiveMatchPredictor:
    """Predict second half outcome based on first half statistics"""
    
    def __init__(self, league='Premier League'):
        self.momentum_weight = 0.65
        self.historical_weight = 0.35
        self.league = league
        self.league_profile = LEAGUE_PROFILES.get(league, LEAGUE_PROFILES['Premier League'])
        self.betting_predictor = BettingMarketPredictor(self.league_profile)
    
    def calculate_momentum(self, first_half_stats):
        """Calculate team momentum from first half performance (league-adjusted)"""
        home_momentum = 0
        away_momentum = 0
        
        # League-specific weights
        pace_factor = self.league_profile['pace_factor']
        physicality = self.league_profile['physicality']
        
        # Tier adjustment - 2nd divisions often more volatile
        tier_factor = 1.1 if self.league_profile['tier'] == 2 else 1.0
        
        # xG momentum (most important)
        xg_diff = first_half_stats['home_xg'] - first_half_stats['away_xg']
        home_momentum += xg_diff * 2.5 * pace_factor * tier_factor
        away_momentum -= xg_diff * 2.5 * pace_factor * tier_factor
        
        # Shots on target momentum
        sot_diff = first_half_stats['home_sot'] - first_half_stats['away_sot']
        home_momentum += sot_diff * 0.8 * pace_factor * tier_factor
        away_momentum -= sot_diff * 0.8 * pace_factor * tier_factor
        
        # Dangerous attacks momentum
        da_diff = first_half_stats['home_dangerous_attacks'] - first_half_stats['away_dangerous_attacks']
        home_momentum += da_diff * 0.15 * tier_factor
        away_momentum -= da_diff * 0.15 * tier_factor
        
        # Corners momentum (more important in physical leagues)
        corner_diff = first_half_stats['home_corners'] - first_half_stats['away_corners']
        home_momentum += corner_diff * 0.3 * physicality * tier_factor
        away_momentum -= corner_diff * 0.3 * physicality * tier_factor
        
        # Normalize to 0-100 scale
        total = abs(home_momentum) + abs(away_momentum)
        if total > 0:
            home_momentum = (home_momentum / total) * 50 + 50
            away_momentum = (away_momentum / total) * 50 + 50
        else:
            home_momentum = away_momentum = 50
        
        return {
            'home': max(0, min(100, home_momentum)),
            'away': max(0, min(100, away_momentum))
        }
    
    def predict_second_half_goals(self, first_half_stats, momentum, historical_avg=None):
        """Predict second half goals using Bayesian updating (league-adjusted)"""
        
        # Use league-specific averages if historical not provided
        if historical_avg is None:
            league_avg = self.league_profile['avg_goals_per_game']
            historical_avg = {
                'home': league_avg * 0.55,  # Home teams score ~55% of total
                'away': league_avg * 0.45
            }
        
        # Base expected goals from first half xG rate
        if first_half_stats.get('minutes_played', 45) > 0:
            home_xg_rate = first_half_stats['home_xg'] / (first_half_stats.get('minutes_played', 45) / 45)
            away_xg_rate = first_half_stats['away_xg'] / (first_half_stats.get('minutes_played', 45) / 45)
        else:
            home_xg_rate = historical_avg['home']
            away_xg_rate = historical_avg['away']
        
        # Momentum adjustment
        momentum_factor_home = momentum['home'] / 50
        momentum_factor_away = momentum['away'] / 50
        
        # League-specific pace factor (second half multiplier)
        pace_multiplier = 1.0 + (self.league_profile['pace_factor'] - 1.0) * 0.15
        
        # Second half predictions
        home_second_half_xg = (home_xg_rate * momentum_factor_home * self.momentum_weight + 
                               historical_avg['home'] * self.historical_weight) * pace_multiplier
        
        away_second_half_xg = (away_xg_rate * momentum_factor_away * self.momentum_weight + 
                               historical_avg['away'] * self.historical_weight) * pace_multiplier
        
        # Home advantage adjustment (league-specific)
        home_advantage_factor = self.league_profile['home_advantage']
        home_second_half_xg *= home_advantage_factor
        away_second_half_xg /= (home_advantage_factor * 0.8)
        
        # Adjust for current scoreline (losing teams push forward)
        current_score_diff = first_half_stats.get('home_goals', 0) - first_half_stats.get('away_goals', 0)
        
        # Tier-specific adjustments - 2nd divisions more volatile
        tier_volatility = 1.15 if self.league_profile['tier'] == 2 else 1.0
        
        if current_score_diff < -1:
            home_second_half_xg *= 1.25 * tier_volatility
            away_second_half_xg *= 0.90
        elif current_score_diff > 1:
            home_second_half_xg *= 0.90
            away_second_half_xg *= 1.25 * tier_volatility
        elif abs(current_score_diff) == 1:
            # Close games in 2nd divisions often see comebacks
            if self.league_profile['tier'] == 2:
                if current_score_diff == -1:
                    home_second_half_xg *= 1.15
                else:
                    away_second_half_xg *= 1.15
        
        # Calculate probabilities using Poisson
        home_goals_probs = [poisson.pmf(i, home_second_half_xg) for i in range(6)]
        away_goals_probs = [poisson.pmf(i, away_second_half_xg) for i in range(6)]
        
        # Most likely second half score
        max_prob = 0
        most_likely_score = "0-0"
        
        for i in range(6):
            for j in range(6):
                prob = home_goals_probs[i] * away_goals_probs[j]
                if prob > max_prob:
                    max_prob = prob
                    most_likely_score = f"{i}-{j}"
        
        # Win probabilities for second half only
        home_win_prob = sum(home_goals_probs[i] * sum(away_goals_probs[:i]) 
                           for i in range(1, 6))
        draw_prob = sum(home_goals_probs[i] * away_goals_probs[i] for i in range(6))
        away_win_prob = sum(away_goals_probs[j] * sum(home_goals_probs[:j]) 
                           for j in range(1, 6))
        
        return {
            'second_half_xg_home': round(home_second_half_xg, 2),
            'second_half_xg_away': round(away_second_half_xg, 2),
            'most_likely_score': most_likely_score,
            'home_win_prob': round(home_win_prob * 100, 1),
            'draw_prob': round(draw_prob * 100, 1),
            'away_win_prob': round(away_win_prob * 100, 1),
            'confidence': round(max_prob * 100, 1)
        }
    
    def predict_full_time_result(self, first_half_stats, second_half_prediction):
        """Predict final full-time result"""
        
        current_home = first_half_stats.get('home_goals', 0)
        current_away = first_half_stats.get('away_goals', 0)
        
        # Expected additional goals
        additional_home = second_half_prediction['second_half_xg_home']
        additional_away = second_half_prediction['second_half_xg_away']
        
        # Full-time expected goals
        ft_home_xg = current_home + additional_home
        ft_away_xg = current_away + additional_away
        
        # Simulate full-time score distribution
        home_goals_probs = [poisson.pmf(i, additional_home) for i in range(6)]
        away_goals_probs = [poisson.pmf(i, additional_away) for i in range(6)]
        
        # Calculate full-time probabilities
        home_win_ft = 0
        draw_ft = 0
        away_win_ft = 0
        
        max_prob = 0
        most_likely_ft_score = f"{current_home}-{current_away}"
        
        for i in range(6):
            for j in range(6):
                prob = home_goals_probs[i] * away_goals_probs[j]
                final_home = current_home + i
                final_away = current_away + j
                
                if prob > max_prob:
                    max_prob = prob
                    most_likely_ft_score = f"{final_home}-{final_away}"
                
                if final_home > final_away:
                    home_win_ft += prob
                elif final_home == final_away:
                    draw_ft += prob
                else:
                    away_win_ft += prob
        
        return {
            'ft_expected_score': most_likely_ft_score,
            'ft_home_xg': round(ft_home_xg, 2),
            'ft_away_xg': round(ft_away_xg, 2),
            'ft_home_win_prob': round(home_win_ft * 100, 1),
            'ft_draw_prob': round(draw_ft * 100, 1),
            'ft_away_win_prob': round(away_win_ft * 100, 1),
            'ft_confidence': round(max_prob * 100, 1)
        }
    
    def predict_match_stats(self, first_half_stats, momentum):
        """Predict second half match statistics (league-adjusted)"""
        
        minutes_played = first_half_stats.get('minutes_played', 45)
        
        # Calculate rates from first half
        home_shot_rate = first_half_stats['home_shots'] / (minutes_played / 45) if minutes_played > 0 else self.league_profile['avg_shots'] * 0.5
        away_shot_rate = first_half_stats['away_shots'] / (minutes_played / 45) if minutes_played > 0 else self.league_profile['avg_shots'] * 0.45
        
        home_sot_rate = first_half_stats['home_sot'] / (minutes_played / 45) if minutes_played > 0 else self.league_profile['avg_sot'] * 0.5
        away_sot_rate = first_half_stats['away_sot'] / (minutes_played / 45) if minutes_played > 0 else self.league_profile['avg_sot'] * 0.45
        
        home_corner_rate = first_half_stats['home_corners'] / (minutes_played / 45) if minutes_played > 0 else self.league_profile['avg_corners'] * 0.5
        away_corner_rate = first_half_stats['away_corners'] / (minutes_played / 45) if minutes_played > 0 else self.league_profile['avg_corners'] * 0.45
        
        home_da_rate = first_half_stats['home_dangerous_attacks'] / (minutes_played / 45) if minutes_played > 0 else self.league_profile['avg_dangerous_attacks'] * 0.5
        away_da_rate = first_half_stats['away_dangerous_attacks'] / (minutes_played / 45) if minutes_played > 0 else self.league_profile['avg_dangerous_attacks'] * 0.45
        
        # Momentum adjustments
        momentum_factor_home = momentum['home'] / 50
        momentum_factor_away = momentum['away'] / 50
        
        # League pace and physicality factors
        pace_factor = self.league_profile['pace_factor']
        physicality = self.league_profile['physicality']
        
        # Tier adjustment - 2nd divisions often see more statistical variance
        tier_variance = 1.08 if self.league_profile['tier'] == 2 else 1.0
        
        # Second half predictions (adjusted for league style)
        return {
            'home_shots_2h': round(home_shot_rate * momentum_factor_home * pace_factor * tier_variance, 1),
            'away_shots_2h': round(away_shot_rate * momentum_factor_away * pace_factor * tier_variance, 1),
            'home_sot_2h': round(home_sot_rate * momentum_factor_home * pace_factor * tier_variance, 1),
            'away_sot_2h': round(away_sot_rate * momentum_factor_away * pace_factor * tier_variance, 1),
            'home_corners_2h': round(home_corner_rate * momentum_factor_home * physicality * tier_variance, 1),
            'away_corners_2h': round(away_corner_rate * momentum_factor_away * physicality * tier_variance, 1),
            'home_da_2h': round(home_da_rate * momentum_factor_home * pace_factor * 1.05 * tier_variance, 0),
            'away_da_2h': round(away_da_rate * momentum_factor_away * pace_factor * 1.05 * tier_variance, 0)
        }

# ================================
# MAIN APPLICATION
# ================================
def main():
    st.sidebar.header("🔴 LIVE MATCH ANALYSIS")
    
    # League selection with categorization
    st.sidebar.markdown("### ⚽ Select League")
    
    # Categorize leagues by tier
    top_tier_leagues = [k for k, v in LEAGUE_PROFILES.items() if v['tier'] == 1]
    second_tier_leagues = [k for k, v in LEAGUE_PROFILES.items() if v['tier'] == 2]
    
    league_category = st.sidebar.radio("League Category", ["Top Division", "Second Division"])
    
    if league_category == "Top Division":
        league = st.sidebar.selectbox("Competition", top_tier_leagues, index=0)
    else:
        league = st.sidebar.selectbox("Competition", second_tier_leagues, index=0)
    
    # Display league info with tier indicator
    league_info = LEAGUE_PROFILES[league]
    with st.sidebar.expander("📊 League Statistics"):
        tier_text = "1st Division" if league_info['tier'] == 1 else "2nd Division"
        st.metric("Tier", tier_text)
        st.metric("Avg Goals/Game", f"{league_info['avg_goals_per_game']:.2f}")
        st.metric("Home Advantage", f"{league_info['home_advantage']:.2f}")
        st.metric("Pace Factor", f"{league_info['pace_factor']:.2f}x")
        st.caption(f"**Style:** {league_info['style']}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Enter First Half Statistics")
    
    # Team names
    home_team = st.sidebar.text_input("🏠 Home Team", "Manchester City")
    away_team = st.sidebar.text_input("✈️ Away Team", "Liverpool")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚽ Current Score")
    
    col1, col2 = st.sidebar.columns(2)
    home_goals = col1.number_input("Home Goals", 0, 10, 1, key="hg")
    away_goals = col2.number_input("Away Goals", 0, 10, 0, key="ag")
    
    minutes_played = st.sidebar.slider("Minutes Played", 1, 45, 45)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 First Half Statistics")
    
    # xG
    st.sidebar.markdown("**Expected Goals (xG)**")
    col1, col2 = st.sidebar.columns(2)
    home_xg = col1.number_input("Home xG", 0.0, 10.0, 1.2, 0.1, key="hxg")
    away_xg = col2.number_input("Away xG", 0.0, 10.0, 0.7, 0.1, key="axg")
    
    # Shots
    st.sidebar.markdown("**Total Shots**")
    col1, col2 = st.sidebar.columns(2)
    home_shots = col1.number_input("Home Shots", 0, 30, 8, key="hs")
    away_shots = col2.number_input("Away Shots", 0, 30, 5, key="as")
    
    # Shots on Target
    st.sidebar.markdown("**Shots on Target**")
    col1, col2 = st.sidebar.columns(2)
    home_sot = col1.number_input("Home SoT", 0, 20, 4, key="hsot")
    away_sot = col2.number_input("Away SoT", 0, 20, 2, key="asot")
    
    # Corners
    st.sidebar.markdown("**Corners**")
    col1, col2 = st.sidebar.columns(2)
    home_corners = col1.number_input("Home Corners", 0, 15, 4, key="hc")
    away_corners = col2.number_input("Away Corners", 0, 15, 2, key="ac")
    
    # Dangerous Attacks
    st.sidebar.markdown("**Dangerous Attacks**")
    col1, col2 = st.sidebar.columns(2)
    home_da = col1.number_input("Home DA", 0, 100, 25, key="hda")
    away_da = col2.number_input("Away DA", 0, 100, 18, key="ada")
    
    # Compile first half stats
    first_half_stats = {
        'home_goals': home_goals,
        'away_goals': away_goals,
        'home_xg': home_xg,
        'away_xg': away_xg,
        'home_shots': home_shots,
        'away_shots': away_shots,
        'home_sot': home_sot,
        'away_sot': away_sot,
        'home_corners': home_corners,
        'away_corners': away_corners,
        'home_dangerous_attacks': home_da,
        'away_dangerous_attacks': away_da,
        'minutes_played': minutes_played
    }
    
    # Initialize predictor with selected league
    predictor = LiveMatchPredictor(league=league)
    
    # Main display
    tier_indicator = "🏆" if league_info['tier'] == 1 else "📈"
    st.markdown(f"## 🔴 LIVE: {home_team} {home_goals} - {away_goals} {away_team}")
    st.markdown(f"**{tier_indicator} {minutes_played}' - Half Time Analysis • {league}**")
    
    # Calculate predictions
    momentum = predictor.calculate_momentum(first_half_stats)
    second_half_pred = predictor.predict_second_half_goals(first_half_stats, momentum)
    full_time_pred = predictor.predict_full_time_result(first_half_stats, second_half_pred)
    stats_pred = predictor.predict_match_stats(first_half_stats, momentum)
    
    # Calculate betting market predictions
    betting_preds = predictor.betting_predictor.predict_goal_lines(first_half_stats, second_half_pred, full_time_pred)
    corners_preds = predictor.betting_predictor.predict_corners_lines(first_half_stats, stats_pred)
    btts_preds = predictor.betting_predictor.predict_both_teams_to_score(first_half_stats, second_half_pred)
    win_to_nil_preds = predictor.betting_predictor.predict_win_to_nil(full_time_pred, first_half_stats)
    double_chance_preds = predictor.betting_predictor.predict_double_chance(full_time_pred)
    handicap_preds = predictor.betting_predictor.predict_alternative_handicaps(full_time_pred, first_half_stats)
    cards_preds = predictor.betting_predictor.predict_cards_lines(first_half_stats, momentum)
    
    # Display momentum and basic predictions (existing code)
    # ... [Previous display code for momentum, second half predictions, full-time predictions]
    
    # NEW: ADVANCED BETTING MARKETS SECTION
    st.markdown("---")
    st.subheader("🎰 ADVANCED BETTING MARKETS")
    
    # Goal Lines
    st.markdown("### ⚽ Goal Lines (Over/Under)")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    goal_lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    for i, line in enumerate(goal_lines):
        with [col1, col2, col3, col4, col5][i]:
            over_prob = betting_preds[f'over_{line}_ft']
            under_prob = betting_preds[f'under_{line}_ft']
            over_value = betting_preds.get(f'value_over_{line}', False)
            under_value = betting_preds.get(f'value_under_{line}', False)
            
            st.metric(f"Over {line} Goals", f"{over_prob}%", 
                     delta="🔥 VALUE" if over_value else None)
            st.metric(f"Under {line} Goals", f"{under_prob}%",
                     delta="🔥 VALUE" if under_value else None)
    
    # Corners Lines
    st.markdown("### 📐 Corner Lines (Over/Under)")
    col1, col2, col3, col4 = st.columns(4)
    
    corner_lines = [6.5, 7.5, 8.5, 9.5]
    for i, line in enumerate(corner_lines):
        with [col1, col2, col3, col4][i]:
            over_prob = corners_preds[f'corners_over_{line}']
            under_prob = corners_preds[f'corners_under_{line}']
            
            st.metric(f"Over {line} Corners", f"{over_prob}%")
            st.metric(f"Under {line} Corners", f"{under_prob}%")
    
    # Both Teams to Score
    st.markdown("### 🎯 Both Teams to Score")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("BTTS Yes", f"{btts_preds['btts_yes']}%")
    with col2:
        st.metric("BTTS No", f"{btts_preds['btts_no']}%")
    with col3:
        st.metric("BTTS 2nd Half", f"{btts_preds['btts_2h_prob']}%")
    
    # Win to Nil
    st.markdown("### 🛡️ Win to Nil")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(f"{home_team} Win to Nil", f"{win_to_nil_preds['home_win_to_nil']}%")
    with col2:
        st.metric(f"{away_team} Win to Nil", f"{win_to_nil_preds['away_win_to_nil']}%")
    
    # Double Chance
    st.markdown("### 🔄 Double Chance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("1X (Home/Draw)", f"{double_chance_preds['1X']}%")
    with col2:
        st.metric("X2 (Draw/Away)", f"{double_chance_preds['X2']}%")
    with col3:
        st.metric("12 (Home/Away)", f"{double_chance_preds['12']}%")
    
    # Asian Handicaps
    st.markdown("### 📊 Asian Handicaps")
    handicaps_to_show = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
    cols = st.columns(len(handicaps_to_show))
    
    for i, handicap in enumerate(handicaps_to_show):
        with cols[i]:
            home_prob = handicap_preds[f'ah_home_{handicap}']
            away_prob = handicap_preds[f'ah_away_{handicap}']
            
            st.metric(f"Home {handicap:+g}", f"{home_prob}%")
            st.metric(f"Away {handicap:+g}", f"{away_prob}%")
    
    # Cards Lines
    st.markdown("### 🟨 Card Lines (Over/Under)")
    col1, col2, col3 = st.columns(3)
    
    card_lines = [3.5, 4.5, 5.5]
    for i, line in enumerate(card_lines):
        with [col1, col2, col3][i]:
            over_prob = cards_preds[f'cards_over_{line}']
            under_prob = cards_preds[f'cards_under_{line}']
            
            st.metric(f"Over {line} Cards", f"{over_prob}%")
            st.metric(f"Under {line} Cards", f"{under_prob}%")
    
    # Betting Recommendations
    st.markdown("---")
    st.subheader("💰 SMART BETTING RECOMMENDATIONS")
    
    recommendations = []
    
    # Goal line value bets
    for line in [1.5, 2.5, 3.5]:
        if betting_preds.get(f'value_over_{line}', False) and betting_preds[f'over_{line}_ft'] > 60:
            recommendations.append(f"✅ **OVER {line} GOALS** - Strong value ({betting_preds[f'over_{line}_ft']}% probability)")
        elif betting_preds.get(f'value_under_{line}', False) and betting_preds[f'under_{line}_ft'] > 60:
            recommendations.append(f"✅ **UNDER {line} GOALS** - Strong value ({betting_preds[f'under_{line}_ft']}% probability)")
    
    # BTTS recommendations
    if btts_preds['btts_yes'] > 65:
        recommendations.append(f"✅ **BOTH TEAMS TO SCORE - YES** ({btts_preds['btts_yes']}% probability)")
    elif btts_preds['btts_no'] > 65:
        recommendations.append(f"✅ **BOTH TEAMS TO SCORE - NO** ({btts_preds['btts_no']}% probability)")
    
    # Corner recommendations
    total_expected_corners = first_half_stats['home_corners'] + first_half_stats['away_corners'] + stats_pred['home_corners_2h'] + stats_pred['away_corners_2h']
    if total_expected_corners > 11:
        recommendations.append(f"✅ **OVER 9.5 CORNERS** - Expected {total_expected_corners:.1f} total corners")
    elif total_expected_corners < 7:
        recommendations.append(f"✅ **UNDER 8.5 CORNERS** - Expected {total_expected_corners:.1f} total corners")
    
    # Display recommendations
    if recommendations:
        for rec in recommendations:
            st.success(rec)
    else:
        st.info("⚠️ No strong value bets identified - market prices are efficient")
    
    # Risk Management
    st.markdown("---")
    st.subheader("📊 BETTING RISK MANAGEMENT")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_expected_goals = full_time_pred['ft_home_xg'] + full_time_pred['ft_away_xg']
        st.metric("Expected Total Goals", f"{total_expected_goals:.1f}")
    
    with col2:
        goal_volatility = np.sqrt(total_expected_goals)
        st.metric("Goal Volatility", f"±{goal_volatility:.1f} goals")
    
    with col3:
        confidence_score = full_time_pred['ft_confidence']
        risk_level = "LOW" if confidence_score > 70 else "MEDIUM" if confidence_score > 50 else "HIGH"
        st.metric("Risk Level", risk_level)
    
    st.info(f"**Bankroll Advice**: Bet 1-2% of bankroll on value bets, avoid bets with probability < 55%")

if __name__ == "__main__":
    main()
