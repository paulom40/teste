import warnings
from collections import defaultdict
from datetime import datetime
import io
import numpy as np
import pandas as pd
import streamlit as st
import re
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

st.set_page_config(page_title="ATP Predictor v8.0 - Manual Input", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG
# ==============================================================================
WINNER_SMOOTH = 0.55
MIN_CONFIDENCE_STRONG = 0.68
MIN_CONFIDENCE_GOOD = 0.60

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
def detect_surface(tournament_name):
    if pd.isna(tournament_name):
        return 'Hard'
    t = str(tournament_name).lower()
    clay = ['clay', 'madrid', 'rome', 'barcelona', 'munich', 'roland garros', 'bordeaux', 'aix', 'cagliari']
    grass = ['grass', 'wimbledon', 'queens', 'halle', 'stuttgart', 's-Hertogenbosch']
    if any(k in t for k in clay):
        return 'Clay'
    if any(k in t for k in grass):
        return 'Grass'
    return 'Hard'

# ==============================================================================
# PROCESS DATA - CORRIGIDO PARA SEU FORMATO
# ==============================================================================
def process_historical_data(df):
    """Process historical data and extract player names from the file"""
    
    # Mostrar colunas para debug
    st.write("### Colunas encontradas no arquivo:")
    st.write(list(df.columns))
    
    # Mapeamento das colunas do seu arquivo
    column_mapping = {}
    
    # Mapear coluna de vencedor
    for col in df.columns:
        col_lower = col.lower()
        if 'winner_name' in col_lower or 'winner' in col_lower:
            column_mapping[col] = 'winner'
        elif 'loser_name' in col_lower or 'loser' in col_lower:
            column_mapping[col] = 'loser'
        elif 'tourney_name' in col_lower or 'tournament' in col_lower or 'tourney' in col_lower:
            column_mapping[col] = 'tournament'
        elif 'tourney_date' in col_lower or 'date' in col_lower:
            column_mapping[col] = 'date'
        elif 't games' in col_lower or 'total_games' in col_lower or 'games' in col_lower:
            column_mapping[col] = 'total_games'
        elif 'surface' in col_lower:
            column_mapping[col] = 'surface'
        elif 'score' in col_lower:
            column_mapping[col] = 'score'
    
    # Renomear colunas
    df = df.rename(columns=column_mapping)
    
    st.write("### Colunas apos mapeamento:")
    st.write(list(df.columns))
    
    # Verificar se temos as colunas necessarias
    if 'winner' not in df.columns:
        st.error("Coluna 'winner' nao encontrada. Procure por 'winner_name' no arquivo.")
        return None, None
    if 'loser' not in df.columns:
        st.error("Coluna 'loser' nao encontrada. Procure por 'loser_name' no arquivo.")
        return None, None
    
    # Converter data
    if 'date' in df.columns:
        # O formato parece ser YYYYMMDD
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    else:
        df['date'] = pd.Timestamp.now()
    
    # Calcular total de games se nao existir
    if 'total_games' not in df.columns:
        # Tentar calcular a partir do placar
        if 'score' in df.columns:
            def extract_games(score):
                if pd.isna(score):
                    return 22
                # Extrair números do placar (ex: "7-6(4) 5-7 6-1" -> 7+6+5+7+6+1 = 32)
                numbers = re.findall(r'\d+', str(score))
                games = [int(n) for n in numbers if int(n) < 20]
                return sum(games) if games else 22
            df['total_games'] = df['score'].apply(extract_games)
        else:
            df['total_games'] = 22
    
    # Detectar superficie
    if 'surface' in df.columns:
        df['surface'] = df['surface'].apply(lambda x: 'Clay' if x == 'Clay' else 'Hard' if x == 'Hard' else 'Grass')
    elif 'tournament' in df.columns:
        df['surface'] = df['tournament'].apply(detect_surface)
    else:
        df['surface'] = 'Hard'
    
    # Limpar nomes
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    # Remover linhas invalidas
    df = df[df['winner'].notna() & df['loser'].notna()]
    df = df[df['winner'] != 'nan']
    df = df[df['loser'] != 'nan']
    df = df[df['winner'] != '']
    df = df[df['loser'] != '']
    df = df[df['winner'] != df['loser']]
    
    # Extrair jogadores unicos
    all_players = sorted(list(set(df['winner'].unique()) | set(df['loser'].unique())))
    
    st.success(f"Processado: {len(df)} jogos | {len(all_players)} jogadores")
    
    # Mostrar amostra dos jogadores
    with st.expander(f"Jogadores no historico ({len(all_players)} total)"):
        for i, p in enumerate(all_players[:50]):
            st.write(f"{i+1}. {p}")
        if len(all_players) > 50:
            st.write(f"... e mais {len(all_players) - 50} jogadores")
    
    return df, all_players

# ==============================================================================
# PLAYER STATISTICS
# ==============================================================================
def calculate_player_stats(df, all_players):
    """Calculate player statistics"""
    
    stats = {}
    
    for player in all_players:
        player_matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(player_matches) == 0:
            stats[player] = {
                'matches': 0, 'wins': 0, 'win_rate': 0.5,
                'recent_form': 0.5, 'very_recent_form': 0.5, 'avg_games': 22
            }
            continue
        
        wins = len(player_matches[player_matches['winner'] == player])
        total = len(player_matches)
        win_rate = wins / total if total > 0 else 0.5
        
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
            'matches': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': win_rate,
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
        w, l = row['winner'], row['loser']
        h2h[(w, l)]['wins'] += 1
        h2h[(w, l)]['total'] += 1
    
    return h2h

# ==============================================================================
# ELO RATING
# ==============================================================================
def calculate_elo(df, all_players, k=32):
    """Calculate ELO ratings"""
    elo = {p: 1500 for p in all_players}
    
    for _, row in df.sort_values('date').iterrows():
        w, l = row['winner'], row['loser']
        if w in elo and l in elo:
            exp_w = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
            elo[w] += k * (1 - exp_w)
            elo[l] += k * (0 - (1 - exp_w))
    
    return elo

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def build_features(p1, p2, surface, player_stats, h2h, elo):
    """Build features for prediction"""
    
    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})
    
    if s1.get('matches', 0) == 0 or s2.get('matches', 0) == 0:
        return None
    
    # ELO difference
    elo_diff = (elo.get(p1, 1500) - elo.get(p2, 1500)) / 400
    
    # Form differences
    form_diff = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
    very_recent_diff = s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)
    
    # Win rate differences
    win_rate_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
    
    # H2H advantage
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / h2h[(p1, p2)]['total']
    elif (p2, p1) in h2h:
        h2h_adv = 1 - (h2h[(p2, p1)]['wins'] / h2h[(p2, p1)]['total'])
    
    # Games average
    games_avg = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    games_norm = (games_avg - 21.5) / 8
    
    # Experience difference
    exp_diff = (s1.get('matches', 0) - s2.get('matches', 0)) / 200
    
    # Momentum
    momentum = very_recent_diff * 0.6 + form_diff * 0.4
    
    features = [
        elo_diff, form_diff, very_recent_diff, win_rate_diff,
        h2h_adv, games_norm, exp_diff, momentum
    ]
    
    return features

# ==============================================================================
# TRAIN MODEL
# ==============================================================================
def train_model(df, player_stats, h2h, elo):
    """Train prediction model"""
    
    X, y = [], []
    
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        surface = row.get('surface', 'Hard')
        
        features = build_features(w, l, surface, player_stats, h2h, elo)
        if features:
            X.append(features)
            y.append(1)
        
        features_rev = build_features(l, w, surface, player_stats, h2h, elo)
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
# NAME MATCHER
# ==============================================================================
class SimpleNameMatcher:
    def __init__(self, historical_names):
        self.historical_names = list(historical_names)
        self.historical_set = set(historical_names)
        self.last_name_map = defaultdict(list)
        
        for name in self.historical_names:
            parts = name.split()
            if parts:
                last_name = parts[-1].lower()
                self.last_name_map[last_name].append(name)
    
    def find_match(self, search_name):
        if not search_name:
            return None
        
        search_str = str(search_name).strip()
        search_lower = search_str.lower()
        
        # Match exato
        if search_str in self.historical_set:
            return search_str
        
        # Case insensitive
        for name in self.historical_names:
            if name.lower() == search_lower:
                return name
        
        # Match por sobrenome
        if search_lower in self.last_name_map:
            return self.last_name_map[search_lower][0]
        
        # Match parcial
        for name in self.historical_names:
            if search_lower in name.lower():
                return name
        
        return None

# ==============================================================================
# PREDICT MATCH
# ==============================================================================
def predict_match(model, p1, p2, surface, player_stats, h2h, elo, name_matcher):
    p1_match = name_matcher.find_match(p1)
    p2_match = name_matcher.find_match(p2)
    
    if not p1_match or not p2_match:
        return None, (p1, p2)
    
    features = build_features(p1_match, p2_match, surface, player_stats, h2h, elo)
    if not features:
        return None, (p1, p2)
    
    prob = model.predict_proba(np.array([features]))[0][1]
    prob_p1 = np.clip(0.5 + (prob - 0.5) * WINNER_SMOOTH, 0.15, 0.85)
    confidence = abs(prob_p1 - 0.5) * 2
    winner = p1_match if prob_p1 > 0.5 else p2_match
    
    if confidence >= MIN_CONFIDENCE_STRONG:
        rec = f"STRONG {winner}"
    elif confidence >= MIN_CONFIDENCE_GOOD:
        rec = f"GOOD {winner}"
    else:
        rec = f"AVOID {winner}"
    
    s1 = player_stats.get(p1_match, {})
    s2 = player_stats.get(p2_match, {})
    exp_games = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    
    return {
        'Busca_P1': p1,
        'Busca_P2': p2,
        'Encontrado': f"{p1_match} vs {p2_match}",
        'Superficie': surface,
        'Prob_P1': f"{prob_p1:.1%}",
        'Prob_P2': f"{1-prob_p1:.1%}",
        'Vencedor': winner,
        'Confianca': f"{confidence:.1%}",
        'Recomendacao': rec,
        'Games_Esperados': round(exp_games, 1)
    }, None

# ==============================================================================
# PARSE DE TEXTO
# ==============================================================================
def parse_colab_text(text):
    matches = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove prefixos
        line = re.sub(r'^(ATP|CHALLENGER|WTA)\s+', '', line, flags=re.IGNORECASE)
        
        # Procura por vs, VS, x
        vs_match = re.search(r'([A-Za-z\-\.\s]+?)\s+(?:vs|VS|x)\s+([A-Za-z\-\.\s]+?)(?:\s*$|\s*->)', line)
        
        if vs_match:
            p1 = vs_match.group(1).strip()
            p2 = vs_match.group(2).strip()
            
            p1 = re.sub(r'\s+', ' ', p1).strip()
            p2 = re.sub(r'\s+', ' ', p2).strip()
            
            surface = 'Clay'
            if 'hard' in line.lower():
                surface = 'Hard'
            elif 'grass' in line.lower():
                surface = 'Grass'
            
            if p1 and p2 and p1 != p2:
                matches.append({'player1': p1, 'player2': p2, 'surface': surface})
    
    return matches

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("ATP Predictor v8.0 - Previsao por Lista")
    st.caption("Upload do historico | Cole os jogos | Previsoes automaticas")
    
    uploaded_file = st.file_uploader("Upload do ficheiro historico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("Processando seu arquivo..."):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df, all_players = process_historical_data(df)
                
                if df is None or len(df) == 0:
                    return
                
                player_stats = calculate_player_stats(df, all_players)
                h2h = calculate_h2h(df)
                elo = calculate_elo(df, all_players)
                model = train_model(df, player_stats, h2h, elo)
                name_matcher = SimpleNameMatcher(all_players)
                
                st.session_state.model = model
                st.session_state.player_stats = player_stats
                st.session_state.h2h = h2h
                st.session_state.elo = elo
                st.session_state.name_matcher = name_matcher
                st.session_state.models_ready = True
                
                st.success("Modelo treinado com sucesso!")
                
            except Exception as e:
                st.error(f"Erro: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.get('models_ready'):
        st.subheader("Cole sua lista de jogos")
        
        matches_text = st.text_area(
            "Jogos (formato: Jogador1 vs Jogador2):",
            height=200,
            placeholder="Mitchell Krueger vs Tung-Lin Wu\nTrevor Svajda vs Liam Draxl\nYuta Shimizu vs Antoine Escoffier"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            surface_override = st.selectbox("Superficie", ["Clay", "Hard", "Grass"], index=0)
        
        if matches_text and st.button("FAZER PREVISOES", type="primary"):
            parsed_matches = parse_colab_text(matches_text)
            
            if parsed_matches:
                results = []
                not_found = []
                
                for match in parsed_matches:
                    surface = surface_override
                    result, missing = predict_match(
                        st.session_state.model, 
                        match['player1'], match['player2'], surface,
                        st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                        st.session_state.name_matcher
                    )
                    if result:
                        results.append(result)
                    elif missing:
                        not_found.append(missing)
                
                if not_found:
                    st.warning(f"{len(not_found)} jogos ignorados")
                
                if results:
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                    
                    buffer = io.BytesIO()
                    df_results.to_excel(buffer, index=False)
                    st.download_button("Download Excel", buffer.getvalue(), f"previsoes.xlsx")
            else:
                st.warning("Nenhum jogo detectado")

if __name__ == "__main__":
    main()
