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
