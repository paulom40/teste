# ================================
# ENHANCED PREDICTION ENGINE WITH TIER-SPECIFIC MODELS
# ================================
class Advanced45MinutePredictor:
    """
    Advanced prediction system with tier-specific models for 1st and 2nd divisions
    Uses ensemble methods, Bayesian updates, and regression-to-mean principles
    """
    
    def __init__(self, league='Premier League'):
        self.league = league
        self.league_profile = LEAGUE_PROFILES.get(league, LEAGUE_PROFILES['Premier League'])
        self.is_second_tier = self.league_profile['tier'] == 2
        
        # Optimized tier-specific weights based on historical accuracy
        if self.is_second_tier:
            # Second tier: Balanced approach with emphasis on form and efficiency
            self.weights = {
                'xg_weight': 0.32,
                'shot_quality_weight': 0.22,
                'momentum_weight': 0.20,
                'efficiency_weight': 0.14,
                'situation_weight': 0.12
            }
        else:
            # First tier: xG and shot quality dominate
            self.weights = {
                'xg_weight': 0.38,
                'shot_quality_weight': 0.24,
                'momentum_weight': 0.18,
                'efficiency_weight': 0.12,
                'situation_weight': 0.08
            }
        
        # Regression to mean parameters
        self.regression_factor = 0.25 if self.is_second_tier else 0.20
        # ================================
# ENHANCED PREDICTION ENGINE WITH TIER-SPECIFIC MODELS
# ================================
class Advanced45MinutePredictor:
    """
    Advanced prediction system with tier-specific models for 1st and 2nd divisions
    Uses ensemble methods, Bayesian updates, and regression-to-mean principles
    """
    
    def __init__(self, league='Premier League'):
        self.league = league
        self.league_profile = LEAGUE_PROFILES.get(league, LEAGUE_PROFILES['Premier League'])
        self.is_second_tier = self.league_profile['tier'] == 2
        
        # Optimized tier-specific weights based on historical accuracy
        if self.is_second_tier:
            # Second tier: Balanced approach with emphasis on form and efficiency
            self.weights = {
                'xg_weight': 0.32,
                'shot_quality_weight': 0.22,
                'momentum_weight': 0.20,
                'efficiency_weight': 0.14,
                'situation_weight': 0.12
            }
        else:
            # First tier: xG and shot quality dominate
            self.weights = {
                'xg_weight': 0.38,
                'shot_quality_weight': 0.24,
                'momentum_weight': 0.18,
                'efficiency_weight': 0.12,
                'situation_weight': 0.08
            }
        
        # Regression to mean parameters
        self.regression_factor = 0.25 if self.is_second_tier else 0.20
        self.league_average_xg = self.league_profile['avg_xg_per_match']
        self.regression_strength = 0.15 if self.is_second_tier else 0.10
        
        # Bayesian prior parameters
        self.prior_strength = 0.3 if self.is_second_tier else 0.25
        self.min_matches_for_reliability = 8 if self.is_second_tier else 12
        
        # Performance thresholds
        self.performance_thresholds = {
            'high_pressure': 0.65,
            'momentum_shift': 0.55,
            'dominance_ratio': 1.8
        }
        
        # Initialize model components
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize tier-specific model components"""
        if self.is_second_tier:
            # Second tier: More volatile, higher variance models
            self.momentum_decay = 0.85  # Faster decay for momentum
            self.form_weight = 0.28     # Higher form reliance
            self.volatility_factor = 1.15
        else:
            # First tier: More stable, lower variance models
            self.momentum_decay = 0.92  # Slower decay for momentum
            self.form_weight = 0.22     # Lower form reliance
            self.volatility_factor = 0.85
    
    def calculate_team_strength(self, team_data, opponent_data, match_context):
        """
        Calculate comprehensive team strength with Bayesian adjustments
        """
        # Base strength from xG and performance metrics
        base_strength = self._calculate_base_strength(team_data)
        
        # Apply Bayesian updates based on sample size
        bayesian_strength = self._apply_bayesian_adjustment(base_strength, team_data)
        
        # Contextual adjustments
        contextual_strength = self._apply_contextual_adjustments(
            bayesian_strength, team_data, opponent_data, match_context
        )
        
        # Regression to mean for extreme values
        final_strength = self._apply_regression_to_mean(contextual_strength, team_data)
        
        return np.clip(final_strength, 0.1, 0.9)
    
    def _calculate_base_strength(self, team_data):
        """Calculate base team strength using weighted metrics"""
        weights = self.weights
        
        xg_contribution = team_data['xg_45min'] * weights['xg_weight']
        shot_quality_contribution = team_data['shot_quality_index'] * weights['shot_quality_weight']
        momentum_contribution = team_data['momentum_score'] * weights['momentum_weight']
        efficiency_contribution = team_data['efficiency_ratio'] * weights['efficiency_weight']
        situation_contribution = team_data['situation_score'] * weights['situation_weight']
        
        base_strength = (
            xg_contribution + 
            shot_quality_contribution + 
            momentum_contribution + 
            efficiency_contribution + 
            situation_contribution
        )
        
        return base_strength
    
    def _apply_bayesian_adjustment(self, base_strength, team_data):
        """Apply Bayesian adjustment based on data reliability"""
        matches_played = team_data.get('matches_played', 0)
        
        if matches_played < self.min_matches_for_reliability:
            # Use league average as prior for small sample sizes
            prior_strength = self.league_average_xg
            sample_weight = matches_played / self.min_matches_for_reliability
            bayesian_strength = (sample_weight * base_strength + 
                               (1 - sample_weight) * prior_strength)
        else:
            bayesian_strength = base_strength
            
        return bayesian_strength
    
    def _apply_contextual_adjustments(self, strength, team_data, opponent_data, context):
        """Apply match-specific contextual adjustments"""
        adjusted_strength = strength
        
        # Home/away adjustment
        if context.get('is_home', False):
            adjusted_strength *= 1.08  # 8% home advantage
        else:
            adjusted_strength *= 0.94  # 6% away disadvantage
            
        # Form adjustment
        form_adjustment = self._calculate_form_adjustment(team_data)
        adjusted_strength *= form_adjustment
        
        # Pressure situation adjustment
        pressure_adjustment = self._calculate_pressure_adjustment(context)
        adjusted_strength *= pressure_adjustment
        
        return adjusted_strength
    
    def _apply_regression_to_mean(self, strength, team_data):
        """Apply regression to mean for extreme performance values"""
        deviation_from_mean = strength - self.league_average_xg
        regression_effect = deviation_from_mean * (1 - self.regression_strength)
        
        return self.league_average_xg + regression_effect
    
    def _calculate_form_adjustment(self, team_data):
        """Calculate form-based adjustment"""
        recent_form = team_data.get('recent_form', 0.5)  # Default to neutral
        form_deviation = recent_form - 0.5
        
        # Apply tier-specific form sensitivity
        adjustment = 1.0 + (form_deviation * self.form_weight * 2)
        return np.clip(adjustment, 0.8, 1.2)
    
    def _calculate_pressure_adjustment(self, context):
        """Calculate pressure situation adjustment"""
        pressure_score = context.get('pressure_score', 0.5)
        
        if pressure_score > self.performance_thresholds['high_pressure']:
            # Teams under high pressure may perform differently
            return 0.92  # Slight performance degradation
        elif pressure_score < 0.3:
            # Low pressure situations
            return 1.05  # Slight performance improvement
            
        return 1.0  # Neutral adjustment
    
    def predict_match_outcome(self, home_data, away_data, match_context):
        """
        Generate comprehensive match prediction
        """
        # Calculate team strengths
        home_strength = self.calculate_team_strength(home_data, away_data, {
            **match_context, 'is_home': True
        })
        
        away_strength = self.calculate_team_strength(away_data, home_data, {
            **match_context, 'is_home': False
        })
        
        # Normalize to probabilities
        total_strength = home_strength + away_strength
        home_win_prob = home_strength / total_strength
        away_win_prob = away_strength / total_strength
        draw_prob = 1.0 - (home_win_prob + away_win_prob)
        
        # Apply draw adjustment based on league tendencies
        draw_adjustment = self.league_profile['draw_tendency']
        adjusted_draw_prob = draw_prob * draw_adjustment
        scaling_factor = (home_win_prob + away_win_prob) / (1 - adjusted_draw_prob)
        
        final_home_prob = home_win_prob * scaling_factor
        final_away_prob = away_win_prob * scaling_factor
        final_draw_prob = adjusted_draw_prob
        
        # Calculate expected goals
        expected_home_goals = self._calculate_expected_goals(home_strength, away_strength, True)
        expected_away_goals = self._calculate_expected_goals(away_strength, home_strength, False)
        
        return {
            'home_win_prob': final_home_prob,
            'away_win_prob': final_away_prob,
            'draw_prob': final_draw_prob,
            'expected_home_goals': expected_home_goals,
            'expected_away_goals': expected_away_goals,
            'confidence_score': self._calculate_confidence(
                final_home_prob, final_away_prob, home_data, away_data
            )
        }
    
    def _calculate_expected_goals(self, attacking_strength, defending_strength, is_home):
        """Calculate expected goals using strength parameters"""
        base_xg = attacking_strength * (1 - defending_strength * 0.3)
        
        # Apply home/away multiplier
        if is_home:
            base_xg *= 1.1
        else:
            base_xg *= 0.95
            
        return np.clip(base_xg, 0.1, 3.5)
    
    def _calculate_confidence(self, home_prob, away_prob, home_data, away_data):
        """Calculate prediction confidence score"""
        # Probability concentration (higher when one team is clearly favored)
        prob_concentration = max(home_prob, away_prob) - min(home_prob, away_prob)
        
        # Data reliability score
        home_reliability = min(home_data.get('matches_played', 0) / 15, 1.0)
        away_reliability = min(away_data.get('matches_played', 0) / 15, 1.0)
        data_reliability = (home_reliability + away_reliability) / 2
        
        # Form consistency
        home_form_consistency = home_data.get('form_consistency', 0.5)
        away_form_consistency = away_data.get('form_consistency', 0.5)
        form_consistency = (home_form_consistency + away_form_consistency) / 2
        
        confidence = (
            prob_concentration * 0.4 +
            data_reliability * 0.35 +
            form_consistency * 0.25
        )
        
        return np.clip(confidence, 0.1, 0.95)

# League profiles configuration
LEAGUE_PROFILES = {
    'Premier League': {
        'tier': 1,
        'avg_xg_per_match': 1.35,
        'draw_tendency': 0.95,
        'scoring_profile': 'balanced'
    },
    'Championship': {
        'tier': 2,
        'avg_xg_per_match': 1.28,
        'draw_tendency': 1.05,
        'scoring_profile': 'moderate'
    },
    'La Liga': {
        'tier': 1,
        'avg_xg_per_match': 1.32,
        'draw_tendency': 0.92,
        'scoring_profile': 'balanced'
    },
    'Bundesliga': {
        'tier': 1,
        'avg_xg_per_match': 1.45,
        'draw_tendency': 0.88,
        'scoring_profile': 'high_scoring'
    },
    'Serie A': {
        'tier': 1,
        'avg_xg_per_match': 1.29,
        'draw_tendency': 1.08,
        'scoring_profile': 'defensive'
    },
    'Ligue 1': {
        'tier': 1,
        'avg_xg_per_match': 1.31,
        'draw_tendency': 0.96,
        'scoring_profile': 'moderate'
    }
}
