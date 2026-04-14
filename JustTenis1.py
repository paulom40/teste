import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier
import re

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v4.3 - Advanced Name Matching", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG
# ==============================================================================
WINNER_SMOOTH = 0.55
MIN_CONFIDENCE_STRONG = 0.68
MIN_CONFIDENCE_GOOD = 0.60
MIN_CONFIDENCE_WEAK = 0.52

# ==============================================================================
# ADVANCED NAME MATCHING SYSTEM
# ==============================================================================
class AdvancedNameMatcher:
    """Sistema avançado de matching de nomes de jogadores"""
    
    def __init__(self):
        self.player_database = {}  # canonical_name -> stats
        self.name_index = {}  # search_key -> canonical_name
        self.last_name_index = defaultdict(list)  # last_name -> [canonical_names]
        
    def build_database(self, player_names):
        """Build search index from player names"""
        for name in player_names:
            self.player_database[name] = name
            self._index_name(name)
    
    def _index_name(self, name):
        """Index a player name with multiple variations"""
        name_lower = name.lower()
        
        # Store original
        self.name_index[name_lower] = name
        
        # Store by last name only
        parts = name.split()
        if len(parts) >= 1:
            last_name = parts[-1].lower()
            self.last_name_index[last_name].append(name)
        
        # Store by first name initial + last name
        if len(parts) >= 2:
            first_initial = parts[0][0].lower()
            last_name = parts[-1].lower()
            key = f"{first_initial}. {last_name}"
            self.name_index[key] = name
            key2 = f"{first_initial}{last_name}"
            self.name_index[key2] = name
        
        # Store by last name with common prefixes removed
        prefixes = ['van', 'de', 'den', 'der', 'dos', 'das', 'le', 'la']
        for prefix in prefixes:
            if name_lower.startswith(prefix + ' '):
                without_prefix = ' '.join(name_lower.split()[1:])
                self.name_index[without_prefix] = name
                # Also index last name from without_prefix
                if len(without_prefix.split()) >= 1:
                    self.last_name_index[without_prefix.split()[-1]].append(name)
    
    def find_player(self, search_name, threshold=0.8):
        """Find matching player using multiple strategies"""
        if not search_name or pd.isna(search_name):
            return None
        
        search_str = str(search_name).strip().lower()
        
        # Strategy 1: Direct match
        if search_str in self.name_index:
            return self.name_index[search_str]
        
        # Strategy 2: Match by full name (case insensitive)
        for canonical in self.player_database.keys():
            if canonical.lower() == search_str:
                return canonical
        
        # Strategy 3: Match by last name only (if unique)
        parts = search_str.split()
        last_name = parts[-1] if parts else search_str
        
        if last_name in self.last_name_index:
            matches = self.last_name_index[last_name]
            if len(matches) == 1:
                return matches[0]
        
        # Strategy 4: Fuzzy matching on last name
        from difflib import get_close_matches
        all_last_names = list(self.last_name_index.keys())
        close_matches = get_close_matches(last_name, all_last_names, n=1, cutoff=threshold)
        
        if close_matches:
            matches = self.last_name_index[close_matches[0]]
            if len(matches) == 1:
                return matches[0]
        
        # Strategy 5: Try to match with reversed name (Struff J. -> Jan-Lennard Struff)
        if len(parts) >= 2 and len(parts[0]) <= 3 and '.' in parts[0]:
            # Format: "Struff J." -> search for "J. Struff"
            last = parts[0]
            first_initial = parts[1].replace('.', '')
            reversed_name = f"{first_initial}. {last}"
            if reversed_name.lower() in self.name_index:
                return self.name_index[reversed_name.lower()]
        
        # Strategy 6: Remove accents and special characters
        import unicodedata
        search_normalized = unicodedata.normalize('NFKD', search_str).encode('ASCII', 'ignore').decode('ASCII')
        for canonical in self.player_database.keys():
            canon_normalized = unicodedata.normalize('NFKD', canonical.lower()).encode('ASCII', 'ignore').decode('ASCII')
            if canon_normalized == search_normalized:
                return canonical
        
        return None

# ==============================================================================
# COMMON PLAYER NAME MAPPINGS (FALLBACK)
# ==============================================================================
COMMON_MAPPINGS = {
    # Struff
    'Struff J': 'Jan-Lennard Struff',
    'J. Struff': 'Jan-Lennard Struff',
    'Jan Struff': 'Jan-Lennard Struff',
    'Struff': 'Jan-Lennard Struff',
    
    # Cerundolo
    'Cerundolo F': 'Francisco Cerundolo',
    'F. Cerundolo': 'Francisco Cerundolo',
    'Cerundolo': 'Francisco Cerundolo',
    
    # Nagal
    'Nagal S': 'Sumit Nagal',
    'S. Nagal': 'Sumit Nagal',
    'Nagal': 'Sumit Nagal',
    
    # Shelton
    'Shelton B': 'Ben Shelton',
    'B. Shelton': 'Ben Shelton',
    'Shelton': 'Ben Shelton',
    
    # Zverev
    'Zverev A': 'Alexander Zverev',
    'A. Zverev': 'Alexander Zverev',
    'Zverev': 'Alexander Zverev',
    
    # Fonseca
    'Fonseca J': 'Joao Fonseca',
    'J. Fonseca': 'Joao Fonseca',
    'Fonseca': 'Joao Fonseca',
    
    # Tabilo
    'Tabilo A': 'Alejandro Tabilo',
    'A. Tabilo': 'Alejandro Tabilo',
    'Tabilo': 'Alejandro Tabilo',
    
    # Shapovalov
    'Shapovalov D': 'Denis Shapovalov',
    'D. Shapovalov': 'Denis Shapovalov',
    'Shapovalov': 'Denis Shapovalov',
    
    # Griekspoor
    'Griekspoor T': 'Tallon Griekspoor',
    'T. Griekspoor': 'Tallon Griekspoor',
    'Griekspoor': 'Tallon Griekspoor',
}

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
def detect_surface(tournament_name):
    """Detect surface from tournament name"""
    if pd.isna(tournament_name):
        return 'Hard'
    
    t = str(tournament_name).lower()
    
    clay_keywords = ['clay', 'monte carlo', 'madrid', 'rome', 'barcelona', 'munich', 
                     'estoril', 'geneva', 'hamburg', 'bastad', 'gstaad', 'roland garros',
                     'french open', 'rio', 'buenos aires']
    grass_keywords = ['grass', 'wimbledon', 'queens', 'halle', 'newport', 'stuttgart']
    
    if any(k in t for k in clay_keywords):
        return 'Clay'
    if any(k in t for k in grass_keywords):
        return 'Grass'
    
    return 'Hard'

# ==============================================================================
# DATA PROCESSING
# ==============================================================================
def process_historical_data(df):
    """Process historical data"""
    
    # Clean column names
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    
    # Find columns
    winner_col = None
    loser_col = None
    tournament_col = None
    date_col = None
    score_col = None
    
    for col in df.columns:
        if 'winner' in col or 'vencedor' in col:
            winner_col = col
        elif 'loser' in col or 'perdedor' in col:
            loser_col = col
        elif 'tourney' in col or 'torneio' in col or 'tournament' in col:
            tournament_col = col
        elif 'date' in col or 'data' in col:
            date_col = col
        elif 'score' in col or 'placar' in col:
            score_col = col
    
    if not winner_col or not loser_col:
        raise ValueError("Colunas de vencedor/perdedor não encontradas")
    
    # Rename
    df = df.rename(columns={
        winner_col: 'winner',
        loser_col: 'loser',
        tournament_col: 'tournament' if tournament_col else 'tournament',
        date_col: 'date' if date_col else 'date',
        score_col: 'score' if score_col else 'score'
    })
    
    # Convert date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.Timestamp.now()
    
    # Calculate total games
    if 'total_games' not in df.columns and 'score' in df.columns:
        def extract_games(score):
            if pd.isna(score):
                return 22
            numbers = re.findall(r'\d+', str(score))
            games = [int(n) for n in numbers if int(n) < 20]
            return sum(games) if games else 22
        df['total_games'] = df['score'].apply(extract_games)
    elif 'total_games' not in df.columns:
        df['total_games'] = 22
    
    # Detect surface
    if 'tournament' in df.columns:
        df['surface'] = df['tournament'].apply(detect_surface)
    else:
        df['surface'] = 'Hard'
    
    # Clean names - remove common suffixes
    def clean_name(name):
        if pd.isna(name):
            return name
        name = str(name).strip()
        # Remove ranking numbers like " (1)" or " [1]"
        name = re.sub(r'\s*[\(\[]\d+[\)\]]', '', name)
        # Remove country codes
        name = re.sub(r'\s*\([A-Z]{2,3}\)', '', name)
        return name
    
    df['winner'] = df['winner'].apply(clean_name)
    df['loser'] = df['loser'].apply(clean_name)
    
    return df

# ==============================================================================
# PLAYER STATISTICS
# ==============================================================================
def calculate_player_stats(df):
    """Calculate player statistics"""
    
    all_players = set(df['winner'].dropna()) | set(df['loser'].dropna())
    stats = {}
    
    for player in all_players:
        player_matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(player_matches) == 0:
            continue
        
        # Basic stats
        wins = len(player_matches[player_matches['winner'] == player])
        total = len(player_matches)
        win_rate = wins / total if total > 0 else 0.5
        
        # Surface stats
        surface_stats = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            surf_matches = player_matches[player_matches['surface'] == surf]
            if len(surf_matches) > 0:
                surf_wins = len(surf_matches[surf_matches['winner'] == player])
                surface_stats[surf] = surf_wins / len(surf_matches)
            else:
                surface_stats[surf] = 0.5
        
        # Recent form (last 10)
        recent = player_matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        # Very recent (last 3)
        very_recent = player_matches.sort_values('date', ascending=False).head(3)
        very_recent_wins = len(very_recent[very_recent['winner'] == player])
        very_recent_form = very_recent_wins / len(very_recent) if len(very_recent) > 0 else 0.5
        
        # Average games
        avg_games = player_matches['total_games'].mean() if 'total_games' in player_matches.columns else 22
        
        stats[player] = {
            'name': player,
            'matches': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': win_rate,
            'hard_rate': surface_stats['Hard'],
            'clay_rate': surface_stats['Clay'],
            'grass_rate': surface_stats['Grass'],
            'recent_form': recent_form,
            'very_recent_form': very_recent_form,
            'avg_games': avg_games
        }
    
    return stats

# ==============================================================================
# H2H DATA
# ==============================================================================
def calculate_h2h(df):
    """Calculate head-to-head statistics"""
    h2h = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')):
            continue
        
        w, l = row['winner'], row['loser']
        h2h[(w, l)]['wins'] += 1
        h2h[(w, l)]['total'] += 1
    
    return h2h

# ==============================================================================
# ELO RATING
# ==============================================================================
def calculate_elo(df, k=32):
    """Calculate ELO ratings"""
    players = set(df['winner'].dropna()) | set(df['loser'].dropna())
    elo = {p: 1500 for p in players}
    
    for _, row in df.sort_values('date').iterrows():
        w, l = row['winner'], row['loser']
        if pd.isna(w) or pd.isna(l):
            continue
        
        exp_w = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
        elo[w] += k * (1 - exp_w)
        elo[l] += k * (0 - (1 - exp_w))
    
    return elo

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def build_features(p1, p2, surface, player_stats, h2h, elo):
    """Build features for prediction"""
    
    s1 = player_stats.get(p1)
    s2 = player_stats.get(p2)
    
    if not s1 or not s2:
        return None
    
    # Surface win rates
    if surface == 'Clay':
        surf_rate1 = s1['clay_rate']
        surf_rate2 = s2['clay_rate']
    elif surface == 'Grass':
        surf_rate1 = s1['grass_rate']
        surf_rate2 = s2['grass_rate']
    else:
        surf_rate1 = s1['hard_rate']
        surf_rate2 = s2['hard_rate']
    
    # ELO difference
    elo1 = elo.get(p1, 1500)
    elo2 = elo.get(p2, 1500)
    elo_diff = (elo1 - elo2) / 400
    
    # Form differences
    form_diff = s1['recent_form'] - s2['recent_form']
    very_recent_diff = s1['very_recent_form'] - s2['very_recent_form']
    
    # Win rate differences
    win_rate_diff = s1['win_rate'] - s2['win_rate']
    surf_diff = surf_rate1 - surf_rate2
    
    # H2H advantage
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / h2h[(p1, p2)]['total']
    elif (p2, p1) in h2h:
        h2h_adv = 1 - (h2h[(p2, p1)]['wins'] / h2h[(p2, p1)]['total'])
    
    # Games
    games_avg = (s1['avg_games'] + s2['avg_games']) / 2
    games_norm = (games_avg - 21.5) / 8
    
    # Experience
    exp_diff = (s1['matches'] - s2['matches']) / 200
    
    # Momentum
    momentum = (s1['very_recent_form'] - s2['very_recent_form']) * 0.6 + form_diff * 0.4
    
    features = [
        elo_diff, form_diff, very_recent_diff,
        win_rate_diff, surf_diff, h2h_adv,
        games_norm, exp_diff, momentum
    ]
    
    return features

# ==============================================================================
# TRAIN MODEL
# ==============================================================================
def train_model(df, player_stats, h2h, elo):
    """Train prediction model"""
    
    X, y = [], []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')):
            continue
        
        winner = row['winner']
        loser = row['loser']
        surface = row.get('surface', 'Hard')
        
        features = build_features(winner, loser, surface, player_stats, h2h, elo)
        if features:
            X.append(features)
            y.append(1)
        
        features_rev = build_features(loser, winner, surface, player_stats, h2h, elo)
        if features_rev:
            X.append(features_rev)
            y.append(0)
    
    if len(X) == 0:
        raise ValueError("No training data")
    
    X = np.array(X)
    
    model = LGBMClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.035,
        num_leaves=16, reg_alpha=0.8, reg_lambda=0.8,
        random_state=42, verbose=-1
    )
    
    model.fit(X, y)
    return model

# ==============================================================================
# PREDICT
# ==============================================================================
def predict_match(model, p1, p2, surface, player_stats, h2h, elo, name_matcher):
    """Predict a single match"""
    
    # Try to find players
    p1_match = name_matcher.find_player(p1)
    p2_match = name_matcher.find_player(p2)
    
    # Try common mappings as fallback
    if not p1_match and p1 in COMMON_MAPPINGS:
        p1_match = COMMON_MAPPINGS[p1]
    if not p2_match and p2 in COMMON_MAPPINGS:
        p2_match = COMMON_MAPPINGS[p2]
    
    if not p1_match:
        return None, f"Jogador não encontrado: {p1}"
    if not p2_match:
        return None, f"Jogador não encontrado: {p2}"
    
    features = build_features(p1_match, p2_match, surface, player_stats, h2h, elo)
    
    if features is None:
        return None, f"Estatísticas não disponíveis"
    
    features = np.array([features])
    
    # Get probability
    prob = model.predict_proba(features)[0][1]
    
    # Apply smoothing
    prob_p1 = 0.5 + (prob - 0.5) * WINNER_SMOOTH
    prob_p1 = np.clip(prob_p1, 0.15, 0.85)
    prob_p2 = 1 - prob_p1
    
    # Confidence
    confidence = abs(prob_p1 - 0.5) * 2
    
    winner = p1_match if prob_p1 > 0.5 else p2_match
    
    # Recommendation
    if confidence >= MIN_CONFIDENCE_STRONG:
        rec = f"🔥 STRONG {winner}"
    elif confidence >= MIN_CONFIDENCE_GOOD:
        rec = f"✅ GOOD {winner}"
    elif confidence >= MIN_CONFIDENCE_WEAK:
        rec = f"🟡 WEAK {winner}"
    else:
        rec = f"⚪ AVOID {winner}"
    
    # Get stats for display
    s1 = player_stats.get(p1_match, {})
    s2 = player_stats.get(p2_match, {})
    
    momentum_edge = (s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)) * 100
    
    expected_games = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    expected_games = np.clip(expected_games, 18, 35)
    
    ou_prob = 0.5 + (expected_games - 21.5) / 20
    ou_prob = np.clip(ou_prob, 0.35, 0.65)
    
    result = {
        'Player1': p1,
        'Player2': p2,
        'Matched_As': f"{p1_match} vs {p2_match}",
        'Surface': surface,
        'Prob_P1': prob_p1,
        'Prob_P2': prob_p2,
        'Predicted_Winner': winner,
        'Confidence': confidence,
        'Recommendation': rec,
        'Momentum_Edge': round(momentum_edge, 1),
        'Expected_Games': round(expected_games, 1),
        'OU': "Over 21.5" if ou_prob > 0.5 else "Under 21.5",
        'OU_Prob': ou_prob,
        'P1_Stats': f"{s1.get('matches',0)}j {s1.get('win_rate',0):.0%}",
        'P2_Stats': f"{s2.get('matches',0)}j {s2.get('win_rate',0):.0%}"
    }
    
    return result, None

# ==============================================================================
# SCRAPER
# ==============================================================================
def scrape_matches():
    """Scrape matches"""
    try:
        target_date = datetime.now().strftime("%Y-%m-%d")
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{target_date}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        
        if r.status_code != 200:
            return []
        
        data = r.json()
        matches = []
        
        for ev in data.get("events", []):
            try:
                category = ev.get("tournament", {}).get("category", {}).get("name", "")
                if "WTA" in str(category).upper():
                    continue
                
                tournament = ev["tournament"]["name"]
                surface = detect_surface(tournament)
                
                matches.append({
                    "tournament": tournament,
                    "player1": ev["homeTeam"]["name"],
                    "player2": ev["awayTeam"]["name"],
                    "surface": surface
                })
            except:
                continue
        
        return matches
    except:
        return []

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v4.3 - Advanced Name Matching")
    st.caption("Sistema avançado de matching | Suporte a múltiplos formatos de nome")
    
    uploaded_file = st.file_uploader("📁 Upload do ficheiro histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("🔄 Processando dados..."):
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Process
                df = process_historical_data(df)
                
                st.info(f"📊 {len(df)} jogos | {len(set(df['winner']) | set(df['loser']))} jogadores")
                
                # Calculate stats
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                elo = calculate_elo(df)
                
                # Build name matcher
                name_matcher = AdvancedNameMatcher()
                name_matcher.build_database(list(player_stats.keys()))
                
                # Train model
                model = train_model(df, player_stats, h2h, elo)
                
                # Store
                st.session_state.model = model
                st.session_state.player_stats = player_stats
                st.session_state.h2h = h2h
                st.session_state.elo = elo
                st.session_state.name_matcher = name_matcher
                st.session_state.models_ready = True
                
                st.success(f"✅ Modelo treinado!")
                
                # Show sample
                with st.expander("📋 Jogadores no histórico (primeiros 30)"):
                    players_list = sorted(list(player_stats.keys()))[:30]
                    for p in players_list:
                        stats = player_stats[p]
                        st.write(f"• {p}: {stats['matches']} jogos, {stats['win_rate']:.0%} vitórias")
                
            except Exception as e:
                st.error(f"Erro: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.get('models_ready'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 HOJE", use_container_width=True):
                with st.spinner("Buscando..."):
                    st.session_state.matches = scrape_matches()
        with col2:
            if st.button("🔄 TESTAR MATCHING", use_container_width=True):
                st.rerun()
        
        # Manual test
        with st.expander("🔍 Testar Matching de Nomes", expanded=True):
            test_name = st.text_input("Digite um nome para testar:", placeholder="Ex: Struff ou Cerundolo")
            if test_name:
                found = st.session_state.name_matcher.find_player(test_name)
                if found:
                    stats = st.session_state.player_stats.get(found, {})
                    st.success(f"✅ Encontrado: {found} ({stats.get('matches', 0)} jogos)")
                else:
                    st.error(f"❌ Não encontrado: {test_name}")
        
        # Manual input
        with st.expander("✏️ Previsão Manual", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.text_input("Jogador 1", placeholder="Ex: Struff")
            with col_b:
                manual_p2 = st.text_input("Jogador 2", placeholder="Ex: Cerundolo")
            with col_c:
                manual_surface = st.selectbox("Superfície", ["Clay", "Hard", "Grass"])
            
            if st.button("🔮 PREVER", type="primary") and manual_p1 and manual_p2:
                result, error = predict_match(
                    st.session_state.model,
                    manual_p1, manual_p2, manual_surface,
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.elo,
                    st.session_state.name_matcher
                )
                if result:
                    df_result = pd.DataFrame([result])
                    display_cols = ['Player1', 'Player2', 'Matched_As', 'Surface', 
                                   'Prob_P1', 'Prob_P2', 'Predicted_Winner', 'Confidence',
                                   'Recommendation', 'Expected_Games', 'OU']
                    styled = df_result[display_cols].style.format({
                        'Prob_P1': '{:.1%}', 'Prob_P2': '{:.1%}', 'Confidence': '{:.1%}'
                    })
                    st.dataframe(styled, use_container_width=True)
                else:
                    st.error(f"Erro: {error}")
        
        # Show predictions
        if st.session_state.get('matches'):
            st.subheader("🎯 Previsões")
            
            results = []
            errors = []
            
            for match in st.session_state.matches:
                result, error = predict_match(
                    st.session_state.model,
                    match['player1'], match['player2'], match['surface'],
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.elo,
                    st.session_state.name_matcher
                )
                if result:
                    result['Tournament'] = match['tournament']
                    results.append(result)
                elif error:
                    errors.append(error)
            
            # Show errors
            if errors:
                with st.expander(f"⚠️ {len(errors)} jogadores não encontrados"):
                    for err in set(errors):
                        st.write(f"• {err}")
            
            if results:
                df_results = pd.DataFrame(results)
                
                cols = ['Tournament', 'Player1', 'Player2', 'Matched_As', 'Surface',
                       'Prob_P1', 'Prob_P2', 'Predicted_Winner', 'Confidence', 
                       'Recommendation', 'Expected_Games', 'OU']
                
                df_display = df_results[[c for c in cols if c in df_results.columns]]
                
                styled = df_display.style.format({
                    'Prob_P1': '{:.1%}', 'Prob_P2': '{:.1%}', 'Confidence': '{:.1%}'
                })
                
                st.dataframe(styled, use_container_width=True, hide_index=True)
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button("📥 Download Excel", buffer.getvalue(),
                                 f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
                
                # Summary
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    strong = sum(1 for r in results if 'STRONG' in r['Recommendation'])
                    st.metric("STRONG Picks", strong)
                with col_s2:
                    avg_conf = df_results['Confidence'].mean()
                    st.metric("Confiança Média", f"{avg_conf:.1%}")
                with col_s3:
                    st.metric("Total Jogos", len(results))

if __name__ == "__main__":
    main()
