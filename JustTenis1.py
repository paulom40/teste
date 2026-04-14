import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier
from difflib import get_close_matches
import re

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v4.2 - Name Matching Fix", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG
# ==============================================================================
WINNER_SMOOTH = 0.55
OU_SMOOTH = 0.50

MIN_CONFIDENCE_STRONG = 0.68
MIN_CONFIDENCE_GOOD = 0.60
MIN_CONFIDENCE_WEAK = 0.52

# ==============================================================================
# NAME MATCHING SYSTEM
# ==============================================================================
class PlayerNameMatcher:
    """Sistema para matching de nomes de jogadores"""
    
    def __init__(self):
        self.name_mapping = {}
        self.name_variations = defaultdict(set)
        
    def build_mapping(self, player_names):
        """Build mapping from variations to canonical names"""
        for name in player_names:
            # Store original
            self.name_mapping[name] = name
            
            # Generate variations
            variations = self._generate_variations(name)
            for var in variations:
                self.name_variations[var].add(name)
    
    def _generate_variations(self, name):
        """Generate common name variations"""
        variations = set()
        variations.add(name.lower())
        variations.add(name.upper())
        
        # Handle first name initial + last name (J. Struff -> Jan-Lennard Struff)
        parts = name.split()
        if len(parts) >= 2:
            # First initial + last name
            first_initial = parts[0][0] + "."
            variations.add(f"{first_initial} {parts[-1]}")
            variations.add(f"{first_initial}{parts[-1]}")
            
            # Just last name
            variations.add(parts[-1])
            
            # Full first name (if we have mapping)
            full_first = self._get_full_first_name(parts[0])
            if full_first:
                variations.add(f"{full_first} {parts[-1]}")
        
        return variations
    
    def _get_full_first_name(self, short_name):
        """Map short first names to full names"""
        first_names = {
            'Jan-Lennard': 'Jan-Lennard', 'Jan': 'Jan-Lennard',
            'Francisco': 'Francisco', 'Fran': 'Francisco',
            'Alejandro': 'Alejandro', 'Alex': 'Alejandro',
            'Alexander': 'Alexander', 'Alex': 'Alexander',
            'Benjamin': 'Ben', 'Ben': 'Benjamin',
            'Denis': 'Denis', 'Denys': 'Denis',
            'Tallon': 'Tallon', 'Tal': 'Tallon',
            'Zhizhen': 'Zhizhen', 'Zhen': 'Zhizhen',
            'Luciano': 'Luciano', 'Lucho': 'Luciano',
            'Zizou': 'Zizou', 'Zizo': 'Zizou',
            'Marko': 'Marko', 'Marco': 'Marko',
            'Joao': 'Joao', 'João': 'Joao',
            'Arthur': 'Arthur', 'Art': 'Arthur',
            'Flavio': 'Flavio', 'Flávio': 'Flavio',
            'Brandon': 'Brandon', 'Brand': 'Brandon',
            'Cameron': 'Cameron', 'Cam': 'Cameron',
            'Stan': 'Stan', 'Stanislas': 'Stan',
            'Adrian': 'Adrian', 'Adri': 'Adrian',
            'Jaume': 'Jaume', 'Jau': 'Jaume'
        }
        return first_names.get(short_name, short_name)
    
    def find_player(self, name, threshold=0.7):
        """Find matching player name"""
        if not name:
            return None
        
        name_lower = name.lower().strip()
        
        # Direct match
        if name_lower in self.name_mapping:
            return self.name_mapping[name_lower]
        
        # Check variations
        for var in self._generate_variations(name):
            if var.lower() in self.name_variations:
                matches = self.name_variations[var.lower()]
                if matches:
                    return list(matches)[0]
        
        # Fuzzy matching
        all_names = list(self.name_mapping.keys())
        matches = get_close_matches(name_lower, [n.lower() for n in all_names], n=1, cutoff=threshold)
        if matches:
            for original in all_names:
                if original.lower() == matches[0]:
                    return original
        
        return None

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
def detect_surface(tournament_name):
    """Detect surface from tournament name"""
    if pd.isna(tournament_name):
        return 'Hard'
    
    t = str(tournament_name).lower()
    
    clay_keywords = ['clay', 'monte carlo', 'madrid', 'rome', 'barcelona', 'munich', 
                     'estoril', 'geneva', 'hamburg', 'bastad', 'gstaad', 'umag', 'kitzbuhel',
                     'roland garros', 'french open', 'rio', 'buenos aires', 'santiago']
    grass_keywords = ['grass', 'wimbledon', 'queens', 'halle', 'newport', 'stuttgart', 
                      's-Hertogenbosch', 'eastbourne', 'mallorca']
    
    if any(k in t for k in clay_keywords):
        return 'Clay'
    if any(k in t for k in grass_keywords):
        return 'Grass'
    
    return 'Hard'

# ==============================================================================
# DATA PROCESSING
# ==============================================================================
def process_historical_data(df):
    """Process historical data and build player database"""
    
    # Clean column names
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    
    # Find correct column names
    winner_col = None
    loser_col = None
    tournament_col = None
    date_col = None
    score_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'winner' in col_lower or 'vencedor' in col_lower:
            winner_col = col
        elif 'loser' in col_lower or 'perdedor' in col_lower:
            loser_col = col
        elif 'tourney' in col_lower or 'torneio' in col_lower or 'tournament' in col_lower:
            tournament_col = col
        elif 'date' in col_lower or 'data' in col_lower:
            date_col = col
        elif 'score' in col_lower or 'placar' in col_lower:
            score_col = col
    
    if not winner_col or not loser_col:
        raise ValueError("Não foi possível encontrar colunas de vencedor/perdedor")
    
    # Rename to standard names
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
    
    # Clean player names
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    return df

# ==============================================================================
# PLAYER STATISTICS
# ==============================================================================
def calculate_player_stats(df):
    """Calculate comprehensive player statistics"""
    
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
        
        # Recent form (last 10 matches)
        recent = player_matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        # Very recent form (last 3 matches)
        very_recent = player_matches.sort_values('date', ascending=False).head(3)
        very_recent_wins = len(very_recent[very_recent['winner'] == player])
        very_recent_form = very_recent_wins / len(very_recent) if len(very_recent) > 0 else 0.5
        
        # Average games
        avg_games = player_matches['total_games'].mean() if 'total_games' in player_matches.columns else 22
        
        stats[player] = {
            'name': player,
            'matches': total,
            'win_rate': win_rate,
            'hard_rate': surface_stats['Hard'],
            'clay_rate': surface_stats['Clay'],
            'grass_rate': surface_stats['Grass'],
            'recent_form': recent_form,
            'very_recent_form': very_recent_form,
            'avg_games': avg_games,
            'wins': wins,
            'losses': total - wins
        }
    
    return stats

# ==============================================================================
# H2H DATA
# ==============================================================================
def calculate_h2h(df):
    """Calculate head-to-head statistics"""
    h2h = defaultdict(lambda: {'wins': 0, 'total': 0, 'surface_wins': defaultdict(int)})
    
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')):
            continue
        
        w, l = row['winner'], row['loser']
        surface = row.get('surface', 'Hard')
        
        h2h[(w, l)]['wins'] += 1
        h2h[(w, l)]['total'] += 1
        h2h[(w, l)]['surface_wins'][surface] += 1
    
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
    
    # Get player stats
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
    
    # Form difference
    form_diff = s1['recent_form'] - s2['recent_form']
    very_recent_diff = s1['very_recent_form'] - s2['very_recent_form']
    
    # Overall win rate difference
    win_rate_diff = s1['win_rate'] - s2['win_rate']
    
    # Surface win rate difference
    surf_diff = surf_rate1 - surf_rate2
    
    # H2H advantage
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / h2h[(p1, p2)]['total']
    elif (p2, p1) in h2h:
        h2h_adv = 1 - (h2h[(p2, p1)]['wins'] / h2h[(p2, p1)]['total'])
    
    # Games average
    games_avg = (s1['avg_games'] + s2['avg_games']) / 2
    games_norm = (games_avg - 21.5) / 8
    
    # Experience difference
    exp_diff = (s1['matches'] - s2['matches']) / 200
    
    # Momentum (recent form weighted)
    momentum = (s1['very_recent_form'] - s2['very_recent_form']) * 0.6 + (form_diff) * 0.4
    
    features = [
        elo_diff,
        form_diff,
        very_recent_diff,
        win_rate_diff,
        surf_diff,
        h2h_adv,
        games_norm,
        exp_diff,
        momentum
    ]
    
    return features

# ==============================================================================
# TRAIN MODEL
# ==============================================================================
def train_model(df, player_stats, h2h, elo):
    """Train the prediction model"""
    
    X, y = [], []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')):
            continue
        
        winner = row['winner']
        loser = row['loser']
        surface = row.get('surface', 'Hard')
        
        # Winner features
        features = build_features(winner, loser, surface, player_stats, h2h, elo)
        if features:
            X.append(features)
            y.append(1)
        
        # Loser features (reverse)
        features_rev = build_features(loser, winner, surface, player_stats, h2h, elo)
        if features_rev:
            X.append(features_rev)
            y.append(0)
    
    if len(X) == 0:
        raise ValueError("No training data available")
    
    X = np.array(X)
    
    model = LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.035,
        num_leaves=16,
        reg_alpha=0.8,
        reg_lambda=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    
    return model

# ==============================================================================
# PREDICT
# ==============================================================================
def predict_match(model, p1, p2, surface, player_stats, h2h, elo, name_matcher):
    """Predict a single match"""
    
    # Find matching players in database
    p1_match = name_matcher.find_player(p1)
    p2_match = name_matcher.find_player(p2)
    
    if not p1_match:
        st.warning(f"Jogador não encontrado: {p1}")
        return None
    if not p2_match:
        st.warning(f"Jogador não encontrado: {p2}")
        return None
    
    features = build_features(p1_match, p2_match, surface, player_stats, h2h, elo)
    
    if features is None:
        return None
    
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
    
    # Expected games (simple estimate)
    expected_games = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    expected_games = np.clip(expected_games, 18, 35)
    
    # OU prediction (simplified)
    ou_prob = 0.5 + (expected_games - 21.5) / 20
    ou_prob = np.clip(ou_prob, 0.35, 0.65)
    
    return {
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
        'OU_Prob': ou_prob
    }

# ==============================================================================
# SCRAPER
# ==============================================================================
def scrape_matches():
    """Scrape matches from Sofascore API"""
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
    st.title("🎾 ATP Predictor v4.2 - Name Matching")
    st.caption("Sistema de matching de nomes | Previsões baseadas em histórico")
    
    uploaded_file = st.file_uploader("📁 Upload do ficheiro histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("🔄 Processando dados e treinando modelo..."):
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Process data
                df = process_historical_data(df)
                
                st.info(f"📊 {len(df)} jogos carregados")
                st.info(f"👥 {len(set(df['winner']) | set(df['loser']))} jogadores únicos")
                
                # Calculate statistics
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                elo = calculate_elo(df)
                
                # Build name matcher
                name_matcher = PlayerNameMatcher()
                name_matcher.build_mapping(list(player_stats.keys()))
                
                # Train model
                model = train_model(df, player_stats, h2h, elo)
                
                # Store in session
                st.session_state.model = model
                st.session_state.player_stats = player_stats
                st.session_state.h2h = h2h
                st.session_state.elo = elo
                st.session_state.name_matcher = name_matcher
                st.session_state.models_ready = True
                
                st.success(f"✅ Modelo treinado com {len(player_stats)} jogadores!")
                
                # Show sample players
                with st.expander("📋 Jogadores no histórico"):
                    players_list = sorted(list(player_stats.keys()))[:20]
                    st.write(", ".join(players_list))
                    if len(player_stats) > 20:
                        st.write(f"... e mais {len(player_stats) - 20} jogadores")
                
            except Exception as e:
                st.error(f"Erro: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.get('models_ready'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 HOJE", use_container_width=True):
                with st.spinner("Buscando jogos..."):
                    st.session_state.matches = scrape_matches()
        with col2:
            if st.button("🔄 ATUALIZAR", use_container_width=True):
                st.rerun()
        
        # Manual input
        with st.expander("✏️ Previsão Manual", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.text_input("Jogador 1", placeholder="Ex: Carlos Alcaraz")
            with col_b:
                manual_p2 = st.text_input("Jogador 2", placeholder="Ex: Novak Djokovic")
            with col_c:
                manual_surface = st.selectbox("Superfície", ["Hard", "Clay", "Grass"])
            
            if st.button("🔮 PREVER", type="primary") and manual_p1 and manual_p2:
                result = predict_match(
                    st.session_state.model,
                    manual_p1, manual_p2, manual_surface,
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.elo,
                    st.session_state.name_matcher
                )
                if result:
                    df_result = pd.DataFrame([result])
                    styled = df_result.style.format({
                        'Prob_P1': '{:.1%}',
                        'Prob_P2': '{:.1%}',
                        'Confidence': '{:.1%}',
                        'OU_Prob': '{:.1%}'
                    })
                    st.dataframe(styled, use_container_width=True)
                else:
                    st.error("Não foi possível fazer a previsão. Verifique os nomes dos jogadores.")
        
        # Show predictions for scraped matches
        if st.session_state.get('matches'):
            st.subheader("🎯 Previsões para Hoje")
            
            results = []
            for match in st.session_state.matches:
                result = predict_match(
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
            
            if results:
                df_results = pd.DataFrame(results)
                
                # Select columns
                cols = ['Tournament', 'Player1', 'Player2', 'Surface', 'Matched_As',
                       'Prob_P1', 'Prob_P2', 'Predicted_Winner', 'Confidence', 
                       'Recommendation', 'Momentum_Edge', 'Expected_Games', 'OU']
                
                df_display = df_results[[c for c in cols if c in df_results.columns]]
                
                styled = df_display.style.format({
                    'Prob_P1': '{:.1%}',
                    'Prob_P2': '{:.1%}',
                    'Confidence': '{:.1%}'
                })
                
                st.dataframe(styled, use_container_width=True, hide_index=True)
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button(
                    "📥 Download Excel",
                    buffer.getvalue(),
                    f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )
                
                # Summary
                st.subheader("📊 Resumo")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    strong = sum(1 for r in results if 'STRONG' in r['Recommendation'])
                    st.metric("STRONG", strong)
                with col_s2:
                    good = sum(1 for r in results if 'GOOD' in r['Recommendation'])
                    st.metric("GOOD", good)
                with col_s3:
                    avg_conf = df_results['Confidence'].mean()
                    st.metric("Confiança Média", f"{avg_conf:.1%}")
                with col_s4:
                    st.metric("Total", len(results))
                
                # Show unmatched players
                unmatched = [r for r in results if 'AVOID' in r['Recommendation'] and r['Confidence'] < 0.5]
                if unmatched:
                    st.warning(f"⚠️ {len(unmatched)} jogos com baixa confiança (jogadores podem não estar no histórico)")
        
        # Debug info
        if st.checkbox("Mostrar debug info"):
            st.subheader("Debug Information")
            st.write(f"Players in database: {len(st.session_state.player_stats)}")
            st.write("Sample players:", list(st.session_state.player_stats.keys())[:10])

if __name__ == "__main__":
    main()
