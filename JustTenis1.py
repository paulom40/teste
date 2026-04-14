import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy.stats import beta, norm
from scipy.special import expit
import hashlib

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v4.0 - 80% Accuracy Goal", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG - OTIMIZADA PARA ALTA PRECISÃO
# ==============================================================================
# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'lightgbm': 0.40,
    'random_forest': 0.30,
    'gradient_boosting': 0.20,
    'cbrf': 0.10
}

# Calibration agressiva para alta confiança
WINNER_SMOOTH = 0.35     # Mais conservador para evitar overconfidence
OU_SMOOTH = 0.45

# Limites de confiança mais rigorosos
MIN_CONFIDENCE_STRONG = 0.72  # Aumentado
MIN_CONFIDENCE_GOOD = 0.65     # Aumentado
MIN_CONFIDENCE_WEAK = 0.58

# CBRF parameters otimizados
CBRF_MOMENTUM_WINDOW = 4
CBRF_MOMENTUM_DECAY = 0.92
CBRF_WEIGHT_RECENT = 0.75

# Betaminic parameters refinados
BETAMINIC_MIN_SAMPLES = 15
BETAMINIC_SURFACE_ADJUST = {'Hard': 1.0, 'Clay': 1.04, 'Grass': 0.96}

# Feature selection
USE_FEATURE_IMPORTANCE = True
MIN_FEATURE_IMPORTANCE = 0.02

# ==============================================================================
# FEATURE ENGINEERING AVANÇADA
# ==============================================================================
class AdvancedFeatureEngineer:
    """Feature engineering avançada para tênis"""
    
    def __init__(self):
        self.feature_names = []
        self.feature_importance = {}
    
    def calculate_tennis_specific_features(self, player, matches, surface):
        """Features específicas do tênis"""
        if len(matches) == 0:
            return {
                'serve_hold_pct': 0.65,
                'break_point_conversion': 0.40,
                'tiebreak_win_pct': 0.50,
                'first_serve_pct': 0.62,
                'second_serve_win_pct': 0.48,
                'return_points_won': 0.40,
                'ace_rate': 0.05,
                'double_fault_rate': 0.03
            }
        
        # Simular estatísticas avançadas baseadas em resultados
        wins = matches[matches['winner'] == player]
        losses = matches[matches['loser'] == player]
        total_matches = len(wins) + len(losses)
        
        # Serve/hold rate (estimado)
        serve_hold = 0.65 + (len(wins) / total_matches) * 0.15
        
        # Break point conversion (estimado)
        break_conv = 0.35 + (len(wins) / total_matches) * 0.15
        
        # Tiebreak performance
        tiebreaks = matches[matches.get('tiebreaks', 0) > 0]
        tiebreak_wins = len(tiebreaks[tiebreaks['winner'] == player])
        tiebreak_pct = tiebreak_wins / len(tiebreaks) if len(tiebreaks) > 0 else 0.5
        
        return {
            'serve_hold_pct': np.clip(serve_hold, 0.55, 0.85),
            'break_point_conversion': np.clip(break_conv, 0.25, 0.60),
            'tiebreak_win_pct': tiebreak_pct,
            'first_serve_pct': 0.60 + np.random.normal(0, 0.05),
            'second_serve_win_pct': 0.45 + (len(wins) / total_matches) * 0.10,
            'return_points_won': 0.35 + (len(wins) / total_matches) * 0.15,
            'ace_rate': 0.04 + (1 if 'Zverev' in player or 'Isner' in player else 0) * 0.03,
            'double_fault_rate': 0.03 + (1 if 'Paire' in player else 0) * 0.02
        }
    
    def calculate_consistency_score(self, player, matches):
        """Calcula consistência do jogador (baixa variância = consistente)"""
        if len(matches) < 5:
            return 0.5
        
        games_played = matches['total_games'].values
        consistency = 1 - (np.std(games_played) / 15)
        return np.clip(consistency, 0, 1)
    
    def calculate_clutch_score(self, player, matches):
        """Performance em momentos decisivos"""
        if len(matches) < 10:
            return 0.5
        
        # Análise de sets decisivos e tiebreaks
        wins = matches[matches['winner'] == player]
        three_set_matches = matches[matches.get('best_of', 3) == 3]
        
        if len(three_set_matches) > 0:
            clutch_wins = len(three_set_matches[three_set_matches['winner'] == player])
            clutch_pct = clutch_wins / len(three_set_matches)
        else:
            clutch_pct = 0.5
        
        return np.clip(clutch_pct, 0.3, 0.8)
    
    def calculate_surface_specific_form(self, player, matches, surface, window=10):
        """Forma específica por superfície"""
        surface_matches = matches[matches['surface'] == surface].head(window)
        if len(surface_matches) == 0:
            return 0.5
        
        wins = len(surface_matches[surface_matches['winner'] == player])
        return wins / len(surface_matches)

# ==============================================================================
# CBRF MODEL MELHORADO
# ==============================================================================
class ImprovedCBRFModel:
    """CBRF com análise de momentum mais sofisticada"""
    
    def __init__(self, window=4, decay=0.92):
        self.window = window
        self.decay = decay
        self.player_history = defaultdict(lambda: {
            'results': [],
            'games_won': [],
            'sets_won': [],
            'momentum_trend': [],
            'confidence_trend': 0.5
        })
    
    def update_history(self, df):
        for _, row in df.sort_values('date').iterrows():
            winner, loser = row.get('winner'), row.get('loser')
            if pd.isna(winner) or pd.isna(loser):
                continue
            
            # Update winner
            self._add_result(winner, 1, row)
            # Update loser
            self._add_result(loser, 0, row)
    
    def _add_result(self, player, result, match_row):
        history = self.player_history[player]
        history['results'].append(result)
        
        if len(history['results']) > self.window:
            history['results'].pop(0)
        
        # Calculate weighted momentum
        weights = [self.decay ** i for i in range(len(history['results']))]
        weighted_sum = sum(r * w for r, w in zip(history['results'], weights))
        momentum = weighted_sum / sum(weights)
        
        history['momentum_trend'].append(momentum)
        if len(history['momentum_trend']) > 5:
            history['momentum_trend'].pop(0)
        
        # Calculate trend (aceleração/desaceleração)
        if len(history['momentum_trend']) >= 3:
            trend = history['momentum_trend'][-1] - history['momentum_trend'][-3]
            history['confidence_trend'] = 0.5 + trend * 0.5
    
    def get_momentum_score(self, player):
        history = self.player_history[player]
        if not history['results']:
            return 0.5
        
        weights = [self.decay ** i for i in range(len(history['results']))]
        momentum = sum(r * w for r, w in zip(history['results'], weights)) / sum(weights)
        
        # Add trend adjustment
        trend_adj = (history['confidence_trend'] - 0.5) * 0.15
        return np.clip(momentum + trend_adj, 0.1, 0.9)

# ==============================================================================
# ENSEMBLE MODEL
# ==============================================================================
class TennisEnsemble:
    """Ensemble de múltiplos modelos para maior precisão"""
    
    def __init__(self):
        self.models = {}
        self.calibrated = False
        self.feature_engineer = AdvancedFeatureEngineer()
    
    def train(self, X, y):
        # LGBM com hiperparâmetros otimizados
        lgbm = LGBMClassifier(
            n_estimators=350, max_depth=6, learning_rate=0.025,
            num_leaves=24, reg_alpha=1.5, reg_lambda=1.5,
            subsample=0.8, colsample_bytree=0.7,
            min_child_samples=20, class_weight='balanced',
            random_state=42, verbose=-1
        )
        
        # Random Forest para capturar interações não-lineares
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=15,
            min_samples_leaf=8, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.03,
            subsample=0.8, min_samples_split=20,
            random_state=42
        )
        
        # Train all models
        self.models['lightgbm'] = lgbm.fit(X, y)
        self.models['random_forest'] = rf.fit(X, y)
        self.models['gradient_boosting'] = gb.fit(X, y)
        
        # Calibration para probabilidades mais precisas
        self.models['lightgbm_calibrated'] = CalibratedClassifierCV(
            self.models['lightgbm'], method='sigmoid', cv=5
        ).fit(X, y)
        
        self.calibrated = True
    
    def predict_proba(self, X):
        probas = []
        weights = []
        
        for name, weight in ENSEMBLE_WEIGHTS.items():
            model_key = name if name in self.models else f"{name}_calibrated"
            if model_key in self.models:
                proba = self.models[model_key].predict_proba(X)[:, 1]
                probas.append(proba)
                weights.append(weight)
        
        # Weighted average
        proba_ensemble = np.average(probas, weights=weights, axis=0)
        
        # Apply Bayesian calibration
        proba_calibrated = self._bayesian_calibration(proba_ensemble)
        
        return np.column_stack([1 - proba_calibrated, proba_calibrated])
    
    def _bayesian_calibration(self, probs):
        """Calibração Bayesiana para probabilidades mais realistas"""
        # Beta prior (assumindo viés leve para underdog)
        alpha_prior, beta_prior = 5, 5
        
        calibrated = (probs * alpha_prior + 0.5 * beta_prior) / (alpha_prior + beta_prior)
        return np.clip(calibrated, 0.05, 0.95)

# ==============================================================================
# BETAMINIC MODEL MELHORADO
# ==============================================================================
class ImprovedBetaminicModel:
    """Betaminic com análise estatística mais profunda"""
    
    def __init__(self, min_samples=15):
        self.min_samples = min_samples
        self.player_profiles = defaultdict(lambda: {
            'avg_games': 22.0,
            'over_rate': 0.5,
            'under_rate': 0.5,
            'games_std': 4.0,
            'surface_games': {'Hard': 22.0, 'Clay': 22.0, 'Grass': 22.0},
            'confidence': 0.5
        })
    
    def update_stats(self, df):
        for player in set(df['winner'].dropna()) | set(df['loser'].dropna()):
            matches = df[(df['winner'] == player) | (df['loser'] == player)]
            if len(matches) < self.min_samples:
                continue
            
            # Base statistics
            games = matches['total_games'].values
            self.player_profiles[player]['avg_games'] = np.mean(games)
            self.player_profiles[player]['games_std'] = np.std(games)
            self.player_profiles[player]['over_rate'] = np.mean(games > 21.5)
            self.player_profiles[player]['under_rate'] = np.mean(games <= 21.5)
            
            # Surface-specific
            for surf in ['Hard', 'Clay', 'Grass']:
                surf_matches = matches[matches['surface'] == surf]
                if len(surf_matches) >= 5:
                    self.player_profiles[player]['surface_games'][surf] = surf_matches['total_games'].mean()
            
            # Confidence based on sample size
            self.player_profiles[player]['confidence'] = min(0.95, len(matches) / 50)
    
    def predict_ou(self, p1, p2, surface):
        prof1 = self.player_profiles[p1]
        prof2 = self.player_profiles[p2]
        
        # Weighted by confidence
        w1 = prof1['confidence']
        w2 = prof2['confidence']
        
        # Expected games (using surface-specific when available)
        exp_games_p1 = prof1['surface_games'].get(surface, prof1['avg_games'])
        exp_games_p2 = prof2['surface_games'].get(surface, prof2['avg_games'])
        
        exp_games = (exp_games_p1 * w1 + exp_games_p2 * w2) / (w1 + w2 + 1e-6)
        
        # Apply surface adjustment
        exp_games *= BETAMINIC_SURFACE_ADJUST.get(surface, 1.0)
        
        # Clamp to realistic range
        exp_games = np.clip(exp_games, 18, 35)
        
        # Calculate over probability using normal distribution assumption
        combined_std = np.sqrt(prof1['games_std']**2 + prof2['games_std']**2) / 2
        z_score = (21.5 - exp_games) / (combined_std + 1e-6)
        over_prob = 1 - norm.cdf(z_score)
        
        # Blend with historical rates
        hist_over = (prof1['over_rate'] * w1 + prof2['over_rate'] * w2) / (w1 + w2 + 1e-6)
        final_prob = 0.6 * over_prob + 0.4 * hist_over
        
        return np.clip(final_prob, 0.25, 0.75), exp_games

# ==============================================================================
# FEATURE ENGINEERING COMPLETA
# ==============================================================================
def build_complete_features(p1, p2, surface, player_stats, h2h_data, 
                            cbrf_model, betaminic_model, feat_engineer):
    """Constrói features completas para o modelo"""
    
    if p1 not in player_stats or p2 not in player_stats:
        return None
    
    s1 = player_stats[p1]
    s2 = player_stats[p2]
    surf = surface if surface in ['Hard', 'Clay', 'Grass'] else 'Hard'
    
    # Get match history for both players
    matches_p1 = s1.get('matches', pd.DataFrame())
    matches_p2 = s2.get('matches', pd.DataFrame())
    
    # 1. ELO Features (4)
    elo_ratio = s1['surface_elo'][surf] / (s2['surface_elo'][surf] + 1)
    elo_diff = (s1['surface_elo'][surf] - s2['surface_elo'][surf]) / 200
    win_rate_diff = s1['surface_win_rate'][surf] - s2['surface_win_rate'][surf]
    elo_trend = (s1.get('elo_trend', 0) - s2.get('elo_trend', 0)) * 2
    
    # 2. Form Features (4)
    recent_form_diff = s1.get('recent_20_form', 0.5) - s2.get('recent_20_form', 0.5)
    very_recent_diff = s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)
    consistency_diff = s1.get('consistency', 0.5) - s2.get('consistency', 0.5)
    clutch_diff = s1.get('clutch', 0.5) - s2.get('clutch', 0.5)
    
    # 3. CBRF Momentum Features (3)
    momentum_p1 = cbrf_model.get_momentum_score(p1)
    momentum_p2 = cbrf_model.get_momentum_score(p2)
    momentum_diff = momentum_p1 - momentum_p2
    
    # 4. Surface-specific form (2)
    surface_form_p1 = feat_engineer.calculate_surface_specific_form(p1, matches_p1, surf)
    surface_form_p2 = feat_engineer.calculate_surface_specific_form(p2, matches_p2, surf)
    surface_form_diff = surface_form_p1 - surface_form_p2
    
    # 5. Tennis-specific stats (6)
    tennis_stats1 = feat_engineer.calculate_tennis_specific_features(p1, matches_p1, surf)
    tennis_stats2 = feat_engineer.calculate_tennis_specific_features(p2, matches_p2, surf)
    
    serve_diff = tennis_stats1['serve_hold_pct'] - tennis_stats2['serve_hold_pct']
    break_diff = tennis_stats1['break_point_conversion'] - tennis_stats2['break_point_conversion']
    tiebreak_diff = tennis_stats1['tiebreak_win_pct'] - tennis_stats2['tiebreak_win_pct']
    return_diff = tennis_stats1['return_points_won'] - tennis_stats2['return_points_won']
    ace_diff = tennis_stats1['ace_rate'] - tennis_stats2['ace_rate']
    df_diff = tennis_stats2['double_fault_rate'] - tennis_stats1['double_fault_rate']
    
    # 6. H2H Features (2)
    h2h_ratio = h2h_data.get((p1, p2), {}).get(surf, 0.5)
    h2h_confidence = min(0.95, h2h_data.get((p1, p2), {}).get('matches', 0) / 10)
    
    # 7. Betaminic OU Features (2)
    ou_prob, exp_games = betaminic_model.predict_ou(p1, p2, surf)
    normalized_games = (exp_games - 21.5) / 15
    
    # 8. Advanced features (3)
    volatility_diff = s1.get('volatility', 0.05) - s2.get('volatility', 0.05)
    age_factor = 0  # Placeholder for age difference
    ranking_diff = s1.get('ranking', 100) - s2.get('ranking', 100)
    ranking_diff_norm = np.clip(ranking_diff / 100, -1, 1)
    
    # Combine all features
    features = [
        elo_ratio, elo_diff, win_rate_diff, elo_trend,
        recent_form_diff, very_recent_diff, consistency_diff, clutch_diff,
        momentum_diff, surface_form_diff,
        serve_diff, break_diff, tiebreak_diff, return_diff, ace_diff, df_diff,
        h2h_ratio, h2h_confidence,
        normalized_games, ou_prob,
        volatility_diff, ranking_diff_norm
    ]
    
    return features

# ==============================================================================
# ELO SYSTEM MELHORADO
# ==============================================================================
def calculate_advanced_elo(df, recent_matches=15):
    players = set(df['winner'].dropna()) | set(df['loser'].dropna())
    surface_elo = {p: {'Hard': 1500, 'Clay': 1500, 'Grass': 1500} for p in players}
    elo_history = {p: [] for p in players}
    
    # K-factors adaptativos
    K_BASE = 28
    K_SURPRISE_BONUS = 8
    
    for _, row in df.sort_values('date').iterrows():
        w, l = row['winner'], row['loser']
        surf = row.get('surface', 'Hard')
        if surf not in ['Hard', 'Clay', 'Grass']:
            surf = 'Hard'
        if pd.isna(w) or pd.isna(l):
            continue
        
        # Calculate expected scores
        r1 = surface_elo[w][surf]
        r2 = surface_elo[l][surf]
        exp_w = 1 / (1 + 10 ** ((r2 - r1) / 400))
        
        # Adaptive K-factor based on surprise level
        surprise = abs(exp_w - 0.5) * 2
        k = K_BASE + K_SURPRISE_BONUS * surprise
        
        # Update ELO
        surface_elo[w][surf] += k * (1 - exp_w)
        surface_elo[l][surf] += k * (0 - (1 - exp_w))
        
        # Store history
        elo_history[w].append(surface_elo[w][surf])
        elo_history[l].append(surface_elo[l][surf])
    
    return surface_elo, elo_history

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================
def train_advanced_model(df, feature_engineer):
    """Pipeline de treinamento completo"""
    
    # Prepare data
    df['surface'] = df.apply(lambda row: detect_surface_from_tournament(
        row.get('tournament'), row.get('surface')), axis=1)
    
    # Calculate ELO
    surface_elo, elo_history = calculate_advanced_elo(df)
    
    # Initialize models
    cbrf_model = ImprovedCBRFModel(window=CBRF_MOMENTUM_WINDOW, decay=CBRF_MOMENTUM_DECAY)
    betaminic_model = ImprovedBetaminicModel(min_samples=BETAMINIC_MIN_SAMPLES)
    
    # Update models with historical data
    cbrf_model.update_history(df)
    betaminic_model.update_stats(df)
    
    # Build player stats
    player_stats = {}
    h2h_data = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'matches': 0}))
    
    for player in set(df['winner'].dropna()) | set(df['loser'].dropna()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        # Calculate stats for each surface
        surface_win_rate = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            surf_matches = matches[matches['surface'] == surf]
            if len(surf_matches) > 0:
                wins = len(surf_matches[surf_matches['winner'] == player])
                surface_win_rate[surf] = wins / len(surf_matches)
            else:
                surface_win_rate[surf] = 0.5
        
        # Recent form
        recent = matches.sort_values('date', ascending=False).head(15)
        very_recent = matches.sort_values('date', ascending=False).head(5)
        
        # Advanced metrics
        consistency = feature_engineer.calculate_consistency_score(player, matches)
        clutch = feature_engineer.calculate_clutch_score(player, matches)
        
        # ELO trend
        elo_vals = elo_history.get(player, [1500])
        elo_trend = (elo_vals[-1] - elo_vals[0]) / 100 if len(elo_vals) > 1 else 0
        volatility = np.std(elo_vals[-10:]) / 100 if len(elo_vals) >= 10 else 0.05
        
        player_stats[player] = {
            'surface_elo': surface_elo[player],
            'surface_win_rate': surface_win_rate,
            'recent_20_form': len(recent[recent['winner'] == player]) / len(recent) if len(recent) > 0 else 0.5,
            'very_recent_form': len(very_recent[very_recent['winner'] == player]) / len(very_recent) if len(very_recent) > 0 else 0.5,
            'avg_games': matches['total_games'].mean() if 'total_games' in matches.columns else 22,
            'consistency': consistency,
            'clutch': clutch,
            'elo_trend': elo_trend,
            'volatility': volatility,
            'ranking': 100,  # Placeholder
            'matches': matches
        }
    
    # Build H2H data
    for _, row in df.iterrows():
        if pd.notna(row.get('winner')) and pd.notna(row.get('loser')):
            w, l = row['winner'], row['loser']
            surf = row.get('surface', 'Hard')
            h2h_data[w][l][surf] = h2h_data[w][l].get(surf, 0) + 1
            h2h_data[w][l]['matches'] = h2h_data[w][l].get('matches', 0) + 1
    
    # Build training dataset
    X_train, y_train = [], []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')):
            continue
        
        surf = row.get('surface', 'Hard')
        
        features = build_complete_features(
            row['winner'], row['loser'], surf, player_stats, h2h_data,
            cbrf_model, betaminic_model, feature_engineer
        )
        
        if features:
            X_train.append(features)
            y_train.append(1)  # Winner
            
            # Add reverse match
            features_rev = build_complete_features(
                row['loser'], row['winner'], surf, player_stats, h2h_data,
                cbrf_model, betaminic_model, feature_engineer
            )
            if features_rev:
                X_train.append(features_rev)
                y_train.append(0)
    
    # Train ensemble
    X_train = np.array(X_train)
    ensemble = TennisEnsemble()
    ensemble.train(X_train, y_train)
    
    return ensemble, player_stats, h2h_data, cbrf_model, betaminic_model, feature_engineer

# ==============================================================================
# PREDICTION WITH HIGH ACCURACY
# ==============================================================================
def predict_with_confidence(ensemble, player_stats, h2h_data, cbrf_model, 
                           betaminic_model, feature_engineer, match):
    """Predição com alta precisão e métricas de confiança"""
    
    p1, p2, surface = match['player1'], match['player2'], match['surface']
    
    features = build_complete_features(
        p1, p2, surface, player_stats, h2h_data,
        cbrf_model, betaminic_model, feature_engineer
    )
    
    if features is None:
        return None
    
    features = np.array([features])
    
    # Get ensemble probability
    proba = ensemble.predict_proba(features)[0][1]
    
    # Apply final calibration
    prob_p1 = 0.5 + (proba - 0.5) * WINNER_SMOOTH
    prob_p1 = np.clip(prob_p1, 0.10, 0.90)
    prob_p2 = 1 - prob_p1
    
    # Calculate confidence score
    confidence = abs(prob_p1 - 0.5) * 2
    confidence = np.clip(confidence, 0.4, 0.95)
    
    # Get CBRF momentum for reference
    momentum_p1 = cbrf_model.get_momentum_score(p1)
    momentum_p2 = cbrf_model.get_momentum_score(p2)
    momentum_diff = momentum_p1 - momentum_p2
    
    # OU prediction
    ou_prob, exp_games = betaminic_model.predict_ou(p1, p2, surface)
    
    winner_pred = p1 if prob_p1 > prob_p2 else p2
    
    # Recommendation based on confidence
    if confidence >= MIN_CONFIDENCE_STRONG:
        rec = f"🔥 STRONG {winner_pred}"
    elif confidence >= MIN_CONFIDENCE_GOOD:
        rec = f"✅ GOOD {winner_pred}"
    elif confidence >= MIN_CONFIDENCE_WEAK:
        rec = f"🟡 WEAK {winner_pred}"
    else:
        rec = f"⚪ AVOID {winner_pred}"
    
    return {
        'Tournament': match['tournament'],
        'Player1': p1,
        'Player2': p2,
        'Surface': surface,
        'Prob_P1': prob_p1,
        'Prob_P2': prob_p2,
        'Predicted_Winner': winner_pred,
        'Confidence': confidence,
        'Recommendation': rec,
        'Momentum_Edge': momentum_diff,
        'Expected_Games': round(exp_games, 1),
        'OU': "Over 21.5" if ou_prob > 0.5 else "Under 21.5",
        'OU_Prob': ou_prob
    }

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v4.0 - 80% Accuracy Goal")
    st.caption("Ensemble de Modelos | Análise de Momentum Avançada | Calibração Bayesiana")
    
    # Initialize feature engineer
    feat_engineer = AdvancedFeatureEngineer()
    
    uploaded_file = st.file_uploader("📁 Upload do ficheiro histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'ensemble' not in st.session_state:
        with st.spinner("🔄 Treinando ensemble de alta precisão... (pode levar alguns minutos)"):
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
                
                # Column mapping
                col_mapping = {
                    'tourney_date': 'date', 'winner_name': 'winner', 'loser_name': 'loser',
                    'tourney_name': 'tournament', 'score': 'score'
                }
                for old, new in col_mapping.items():
                    if old in df.columns:
                        df.rename(columns={old: new}, inplace=True)
                
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Calculate total games
                if 'total_games' not in df.columns and 'score' in df.columns:
                    def get_games(s):
                        import re
                        nums = [int(n) for n in re.findall(r'\d+', str(s)) if int(n) < 20]
                        return sum(nums) if nums else 22
                    df['total_games'] = df['score'].apply(get_games)
                elif 'total_games' not in df.columns:
                    df['total_games'] = 22
                
                # Train advanced model
                ensemble, player_stats, h2h_data, cbrf_model, betaminic_model, _ = train_advanced_model(
                    df, feat_engineer
                )
                
                st.session_state.ensemble = ensemble
                st.session_state.player_stats = player_stats
                st.session_state.h2h_data = h2h_data
                st.session_state.cbrf_model = cbrf_model
                st.session_state.betaminic_model = betaminic_model
                st.session_state.feat_engineer = feat_engineer
                st.session_state.models_ready = True
                
                st.success("✅ Ensemble treinado com sucesso!")
                st.info(f"📊 Dados: {len(df)} jogos | {len(player_stats)} jogadores")
                
            except Exception as e:
                st.error(f"Erro: {str(e)}")
    
    if st.session_state.get('models_ready'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 HOJE", use_container_width=True):
                st.session_state.matches = scrape_matches_sofascore(0)
        with col2:
            if st.button("📅 AMANHÃ", use_container_width=True):
                st.session_state.matches = scrape_matches_sofascore(1)
        
        # Manual input
        with st.expander("✏️ Previsão Manual"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.text_input("Jogador 1")
            with col_b:
                manual_p2 = st.text_input("Jogador 2")
            with col_c:
                manual_surface = st.selectbox("Superfície", ["Hard", "Clay", "Grass"])
            
            if st.button("🔮 Prever"):
                if manual_p1 and manual_p2:
                    match = {'tournament': 'Manual', 'player1': manual_p1, 
                            'player2': manual_p2, 'surface': manual_surface}
                    result = predict_with_confidence(
                        st.session_state.ensemble,
                        st.session_state.player_stats,
                        st.session_state.h2h_data,
                        st.session_state.cbrf_model,
                        st.session_state.betaminic_model,
                        st.session_state.feat_engineer,
                        match
                    )
                    if result:
                        st.dataframe(pd.DataFrame([result]).style.format({
                            'Prob_P1': '{:.1%}', 'Prob_P2': '{:.1%}',
                            'Confidence': '{:.1%}', 'OU_Prob': '{:.1%}'
                        }))
        
        # Show predictions
        if st.session_state.get('matches'):
            st.subheader("🎯 Previsões")
            results = []
            
            for match in st.session_state.matches:
                result = predict_with_confidence(
                    st.session_state.ensemble,
                    st.session_state.player_stats,
                    st.session_state.h2h_data,
                    st.session_state.cbrf_model,
                    st.session_state.betaminic_model,
                    st.session_state.feat_engineer,
                    match
                )
                if result:
                    results.append(result)
            
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results.style.format({
                    'Prob_P1': '{:.1%}', 'Prob_P2': '{:.1%}',
                    'Confidence': '{:.1%}', 'OU_Prob': '{:.1%}'
                }), use_container_width=True)
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button("📥 Download Excel", buffer.getvalue(),
                                 f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")

# Scraper function (mesma da versão anterior)
def scrape_matches_sofascore(days_ahead=0):
    try:
        target_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{target_date}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        
        data = r.json()
        matches = []
        for ev in data.get("events", []):
            if "WTA" in str(ev.get("tournament", {}).get("category", {}).get("name", "")).upper():
                continue
            matches.append({
                "tournament": ev["tournament"]["name"],
                "player1": ev["homeTeam"]["name"],
                "player2": ev["awayTeam"]["name"],
                "surface": detect_surface_from_tournament(ev["tournament"]["name"])
            })
        return matches
    except:
        return []

def detect_surface_from_tournament(tournament_name):
    t = str(tournament_name).lower()
    if 'clay' in t or 'monte carlo' in t or 'madrid' in t or 'rome' in t:
        return 'Clay'
    if 'grass' in t or 'wimbledon' in t or 'queens' in t:
        return 'Grass'
    return 'Hard'

if __name__ == "__main__":
    main()
