import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import norm
import re

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
WINNER_SMOOTH = 0.35
OU_SMOOTH = 0.45

# Limites de confiança mais rigorosos
MIN_CONFIDENCE_STRONG = 0.72
MIN_CONFIDENCE_GOOD = 0.65
MIN_CONFIDENCE_WEAK = 0.58

# CBRF parameters otimizados
CBRF_MOMENTUM_WINDOW = 4
CBRF_MOMENTUM_DECAY = 0.92

# Betaminic parameters refinados
BETAMINIC_MIN_SAMPLES = 15
BETAMINIC_SURFACE_ADJUST = {'Hard': 1.0, 'Clay': 1.04, 'Grass': 0.96}

# ==============================================================================
# SURFACE DETECTION - CORRIGIDA
# ==============================================================================
TOURNAMENT_SURFACE_MAP = {
    'monte carlo': 'Clay', 'madrid': 'Clay', 'rome': 'Clay', 'barcelona': 'Clay',
    'munich': 'Clay', 'estoril': 'Clay', 'geneva': 'Clay', 'oeiras': 'Clay',
    'santa cruz': 'Clay', 'tallahassee': 'Clay', 'busan': 'Hard', 'wuning': 'Hard',
    'wimbledon': 'Grass', 'queens': 'Grass', 'halle': 'Grass', 'newport': 'Grass',
    'stuttgart': 'Grass', 's-Hertogenbosch': 'Grass', 'eastbourne': 'Grass'
}

def detect_surface_from_tournament(tournament_name):
    """Versão simplificada - apenas 1 argumento"""
    if pd.isna(tournament_name):
        return 'Hard'
    
    t = str(tournament_name).lower()
    
    # Check mapped tournaments first
    for key, surf in TOURNAMENT_SURFACE_MAP.items():
        if key in t:
            return surf
    
    # Check by keywords
    if any(x in t for x in ['clay', 'terre', 'antuka', 'red clay']):
        return 'Clay'
    if any(x in t for x in ['grass', 'lawn', 'natural grass']):
        return 'Grass'
    
    # Default to Hard
    return 'Hard'

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
        total_matches = len(matches)
        
        # Serve/hold rate (estimado)
        serve_hold = 0.65 + (len(wins) / total_matches) * 0.15
        
        # Break point conversion (estimado)
        break_conv = 0.35 + (len(wins) / total_matches) * 0.15
        
        # Tiebreak performance
        tiebreaks = matches[matches.get('tiebreaks', 0) > 0] if 'tiebreaks' in matches.columns else pd.DataFrame()
        tiebreak_wins = len(tiebreaks[tiebreaks['winner'] == player]) if len(tiebreaks) > 0 else 0
        tiebreak_pct = tiebreak_wins / len(tiebreaks) if len(tiebreaks) > 0 else 0.5
        
        return {
            'serve_hold_pct': np.clip(serve_hold, 0.55, 0.85),
            'break_point_conversion': np.clip(break_conv, 0.25, 0.60),
            'tiebreak_win_pct': tiebreak_pct,
            'first_serve_pct': 0.60 + np.random.normal(0, 0.05),
            'second_serve_win_pct': 0.45 + (len(wins) / total_matches) * 0.10,
            'return_points_won': 0.35 + (len(wins) / total_matches) * 0.15,
            'ace_rate': 0.04,
            'double_fault_rate': 0.03
        }
    
    def calculate_consistency_score(self, player, matches):
        """Calcula consistência do jogador (baixa variância = consistente)"""
        if len(matches) < 5:
            return 0.5
        
        if 'total_games' in matches.columns:
            games_played = matches['total_games'].values
            consistency = 1 - (np.std(games_played) / 15)
            return np.clip(consistency, 0, 1)
        return 0.5
    
    def calculate_clutch_score(self, player, matches):
        """Performance em momentos decisivos"""
        if len(matches) < 10:
            return 0.5
        
        # Análise de sets decisivos
        if 'best_of' in matches.columns:
            three_set_matches = matches[matches.get('best_of', 3) == 3]
            if len(three_set_matches) > 0:
                clutch_wins = len(three_set_matches[three_set_matches['winner'] == player])
                clutch_pct = clutch_wins / len(three_set_matches)
                return np.clip(clutch_pct, 0.3, 0.8)
        
        return 0.5
    
    def calculate_surface_specific_form(self, player, matches, surface, window=10):
        """Forma específica por superfície"""
        if len(matches) == 0:
            return 0.5
        
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
            'momentum_trend': [],
            'confidence_trend': 0.5
        })
    
    def update_history(self, df):
        for _, row in df.sort_values('date').iterrows():
            winner, loser = row.get('winner'), row.get('loser')
            if pd.isna(winner) or pd.isna(loser):
                continue
            
            # Update winner
            self._add_result(winner, 1)
            # Update loser
            self._add_result(loser, 0)
    
    def _add_result(self, player, result):
        history = self.player_history[player]
        history['results'].append(result)
        
        if len(history['results']) > self.window:
            history['results'].pop(0)
        
        # Calculate weighted momentum
        weights = [self.decay ** i for i in range(len(history['results']))]
        weighted_sum = sum(r * w for r, w in zip(history['results'], weights))
        momentum = weighted_sum / sum(weights) if sum(weights) > 0 else 0.5
        
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
            if 'total_games' in matches.columns:
                games = matches['total_games'].values
                self.player_profiles[player]['avg_games'] = np.mean(games)
                self.player_profiles[player]['games_std'] = np.std(games)
                self.player_profiles[player]['over_rate'] = np.mean(games > 21.5)
                self.player_profiles[player]['under_rate'] = np.mean(games <= 21.5)
            
            # Surface-specific
            for surf in ['Hard', 'Clay', 'Grass']:
                surf_matches = matches[matches['surface'] == surf]
                if len(surf_matches) >= 5 and 'total_games' in surf_matches.columns:
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
    h2h_ratio = 0.5
    h2h_confidence = 0.5
    if (p1, p2) in h2h_data:
        surf_data = h2h_data[(p1, p2)].get(surf, 0)
        total = surf_data + h2h_data.get((p2, p1), {}).get(surf, 0) + 1
        h2h_ratio = (surf_data + 0.5) / total
        h2h_confidence = min(0.95, h2h_data[(p1, p2)].get('matches', 0) / 10)
    
    # 7. Betaminic OU Features (2)
    ou_prob, exp_games = betaminic_model.predict_ou(p1, p2, surf)
    normalized_games = (exp_games - 21.5) / 15
    
    # 8. Advanced features (2)
    volatility_diff = s1.get('volatility', 0.05) - s2.get('volatility', 0.05)
    
    # Combine all features
    features = [
        elo_ratio, elo_diff, win_rate_diff, elo_trend,
        recent_form_diff, very_recent_diff, consistency_diff, clutch_diff,
        momentum_diff, surface_form_diff,
        serve_diff, break_diff, tiebreak_diff, return_diff, ace_diff, df_diff,
        h2h_ratio, h2h_confidence,
        normalized_games, ou_prob,
        volatility_diff
    ]
    
    return features

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================
def train_advanced_model(df, feature_engineer):
    """Pipeline de treinamento completo"""
    
    # Prepare data
    df['surface'] = df.apply(lambda row: detect_surface_from_tournament(row.get('tournament')), axis=1)
    
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
    h2h_data = defaultdict(lambda: defaultdict(lambda: 0))
    
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
            'matches': matches
        }
    
    # Build H2H data
    for _, row in df.iterrows():
        if pd.notna(row.get('winner')) and pd.notna(row.get('loser')):
            w, l = row['winner'], row['loser']
            surf = row.get('surface', 'Hard')
            h2h_data[(w, l)][surf] = h2h_data[(w, l)].get(surf, 0) + 1
            h2h_data[(w, l)]['matches'] = h2h_data[(w, l)].get('matches', 0) + 1
    
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
    
    if len(X_train) == 0:
        raise ValueError("No valid training data found")
    
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
        'Momentum_Edge': round(momentum_diff, 3),
        'Expected_Games': round(exp_games, 1),
        'OU': "Over 21.5" if ou_prob > 0.5 else "Under 21.5",
        'OU_Prob': ou_prob
    }

# ==============================================================================
# SCRAPER
# ==============================================================================
def scrape_matches_sofascore(days_ahead=0):
    """Scrape matches from Sofascore API"""
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
            try:
                # Skip WTA
                category = ev.get("tournament", {}).get("category", {}).get("name", "")
                if "WTA" in str(category).upper():
                    continue
                
                tournament_name = ev["tournament"]["name"]
                surface = detect_surface_from_tournament(tournament_name)
                
                matches.append({
                    "tournament": tournament_name,
                    "player1": ev["homeTeam"]["name"],
                    "player2": ev["awayTeam"]["name"],
                    "surface": surface
                })
            except Exception:
                continue
        return matches
    except Exception:
        return []

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
                        nums = [int(n) for n in re.findall(r'\d+', str(s)) if int(n) < 20]
                        return sum(nums) if nums else 22
                    df['total_games'] = df['score'].apply(get_games)
                elif 'total_games' not in df.columns:
                    df['total_games'] = 22
                
                # Limit data for performance
                if len(df) > 10000:
                    df = df.sort_values('date', ascending=False).head(10000)
                    st.info(f"Limitando para os últimos 10,000 jogos para performance")
                
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
                st.error(f"Erro no treinamento: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.get('models_ready'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 HOJE", use_container_width=True):
                with st.spinner("Buscando jogos de hoje..."):
                    st.session_state.matches = scrape_matches_sofascore(0)
        with col2:
            if st.button("📅 AMANHÃ", use_container_width=True):
                with st.spinner("Buscando jogos de amanhã..."):
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
            
            if st.button("🔮 Prever Jogo Manual") and manual_p1 and manual_p2:
                match = {
                    'tournament': 'Manual Entry', 
                    'player1': manual_p1, 
                    'player2': manual_p2, 
                    'surface': manual_surface
                }
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
                    }), use_container_width=True)
                else:
                    st.warning("Jogador não encontrado no histórico. Certifique-se que o nome está correto.")
        
        # Show predictions
        if st.session_state.get('matches'):
            st.subheader("🎯 Previsões do Dia")
            
            results = []
            progress_bar = st.progress(0)
            for i, match in enumerate(st.session_state.matches):
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
                progress_bar.progress((i + 1) / len(st.session_state.matches))
            
            progress_bar.empty()
            
            if results:
                df_results = pd.DataFrame(results)
                
                # Color coding
                def color_recommendation(val):
                    if 'STRONG' in str(val):
                        return 'background-color: #2e7d32; color: white'
                    elif 'GOOD' in str(val):
                        return 'background-color: #4caf50; color: white'
                    elif 'WEAK' in str(val):
                        return 'background-color: #ff9800; color: black'
                    return 'background-color: #9e9e9e; color: white'
                
                styled = df_results.style.format({
                    'Prob_P1': '{:.1%}', 'Prob_P2': '{:.1%}',
                    'Confidence': '{:.1%}', 'OU_Prob': '{:.1%}'
                }).map(color_recommendation, subset=['Recommendation'])
                
                st.dataframe(styled, use_container_width=True, hide_index=True, height=700)
                
                # Summary statistics
                st.subheader("📊 Resumo das Previsões")
                col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
                with col_s1:
                    st.metric("Total Jogos", len(results))
                with col_s2:
                    strong_count = sum(1 for r in results if 'STRONG' in r['Recommendation'])
                    st.metric("STRONG Picks", strong_count)
                with col_s3:
                    good_count = sum(1 for r in results if 'GOOD' in r['Recommendation'])
                    st.metric("GOOD Picks", good_count)
                with col_s4:
                    avg_conf = df_results['Confidence'].mean()
                    st.metric("Confiança Média", f"{avg_conf:.1%}")
                with col_s5:
                    over_count = sum(1 for r in results if r['OU'] == 'Over 21.5')
                    st.metric("Over 21.5", f"{over_count}/{len(results)}")
                
                # Download button
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button(
                    "📥 Baixar Excel com Previsões",
                    buffer.getvalue(),
                    f"previsoes_atp_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )
    
    elif not uploaded_file:
        st.info("📂 Faça upload do ficheiro Excel/CSV com dados históricos para começar")
        st.markdown("""
        ### Formato esperado do ficheiro:
        - Colunas necessárias: `winner_name`, `loser_name`, `tourney_name`, `tourney_date`, `score`
        - Ou colunas em português: `vencedor`, `perdedor`, `torneio`, `data`, `placar`
        - O modelo vai tentar mapear automaticamente
        """)

if __name__ == "__main__":
    main()
