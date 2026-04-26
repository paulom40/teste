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

st.set_page_config(page_title="ATP Predictor v6.2 - Batch Predictions", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG
# ==============================================================================
WINNER_SMOOTH = 0.35
MIN_CONFIDENCE_STRONG = 0.65
MIN_CONFIDENCE_GOOD = 0.56
MIN_CONFIDENCE_WEAK = 0.50

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
def detect_surface(tournament_name):
    if pd.isna(tournament_name):
        return 'Hard'
    t = str(tournament_name).lower()
    clay = ['clay', 'monte carlo', 'madrid', 'rome', 'barcelona', 'munich', 'roland garros']
    grass = ['grass', 'wimbledon', 'queens', 'halle']
    if any(k in t for k in clay):
        return 'Clay'
    if any(k in t for k in grass):
        return 'Grass'
    return 'Hard'

# ==============================================================================
# PROCESS DATA
# ==============================================================================
def process_historical_data(df):
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    
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
        return None, None
    
    df = df.rename(columns={
        winner_col: 'winner',
        loser_col: 'loser',
        tournament_col: 'tournament' if tournament_col else 'tournament',
        date_col: 'date' if date_col else 'date',
        score_col: 'score' if score_col else 'score'
    })
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.Timestamp.now()
    
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
    
    if 'tournament' in df.columns:
        df['surface'] = df['tournament'].apply(detect_surface)
    else:
        df['surface'] = 'Hard'
    
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    df = df[df['winner'].notna() & df['loser'].notna()]
    df = df[df['winner'] != 'nan']
    df = df[df['loser'] != 'nan']
    df = df[df['winner'] != '']
    df = df[df['loser'] != '']
    
    all_players = list(set(df['winner'].unique()) | set(df['loser'].unique()))
    
    return df, all_players

# ==============================================================================
# PLAYER STATISTICS
# ==============================================================================
def calculate_player_stats(df, all_players):
    stats = {}
    
    for player in all_players:
        player_matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(player_matches) == 0:
            stats[player] = {
                'matches': 0, 'wins': 0, 'win_rate': 0.5,
                'recent_form': 0.5, 'very_recent_form': 0.5, 'avg_games': 22,
                'hard_rate': 0.5, 'clay_rate': 0.5, 'grass_rate': 0.5
            }
            continue
        
        wins = len(player_matches[player_matches['winner'] == player])
        total = len(player_matches)
        win_rate = wins / total if total > 0 else 0.5
        
        hard_matches = player_matches[player_matches['surface'] == 'Hard']
        clay_matches = player_matches[player_matches['surface'] == 'Clay']
        grass_matches = player_matches[player_matches['surface'] == 'Grass']
        
        hard_wins = len(hard_matches[hard_matches['winner'] == player])
        clay_wins = len(clay_matches[clay_matches['winner'] == player])
        grass_wins = len(grass_matches[grass_matches['winner'] == player])
        
        hard_rate = hard_wins / len(hard_matches) if len(hard_matches) > 0 else 0.5
        clay_rate = clay_wins / len(clay_matches) if len(clay_matches) > 0 else 0.5
        grass_rate = grass_wins / len(grass_matches) if len(grass_matches) > 0 else 0.5
        
        recent = player_matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        very_recent = player_matches.sort_values('date', ascending=False).head(5)
        very_recent_wins = len(very_recent[very_recent['winner'] == player])
        very_recent_form = very_recent_wins / len(very_recent) if len(very_recent) > 0 else 0.5
        
        avg_games = player_matches['total_games'].mean() if 'total_games' in player_matches.columns else 22
        
        stats[player] = {
            'matches': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': win_rate,
            'hard_rate': hard_rate,
            'clay_rate': clay_rate,
            'grass_rate': grass_rate,
            'recent_form': recent_form,
            'very_recent_form': very_recent_form,
            'avg_games': avg_games
        }
    
    return stats

# ==============================================================================
# H2H
# ==============================================================================
def calculate_h2h(df):
    h2h = defaultdict(lambda: {'wins': 0, 'total': 0})
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        h2h[(w, l)]['wins'] += 1
        h2h[(w, l)]['total'] += 1
    return h2h

# ==============================================================================
# ELO
# ==============================================================================
def calculate_elo(df, all_players, k=32):
    elo = {p: 1500 for p in all_players}
    
    for _, row in df.sort_values('date').iterrows():
        w, l = row['winner'], row['loser']
        if w in elo and l in elo:
            exp_w = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
            elo[w] += k * (1 - exp_w)
            elo[l] += k * (0 - (1 - exp_w))
    
    return elo

# ==============================================================================
# FEATURES
# ==============================================================================
def build_features(p1, p2, surface, player_stats, h2h, elo):
    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})
    
    if s1.get('matches', 0) == 0 or s2.get('matches', 0) == 0:
        return None
    
    if surface == 'Clay':
        surf_rate1 = s1.get('clay_rate', 0.5)
        surf_rate2 = s2.get('clay_rate', 0.5)
    elif surface == 'Grass':
        surf_rate1 = s1.get('grass_rate', 0.5)
        surf_rate2 = s2.get('grass_rate', 0.5)
    else:
        surf_rate1 = s1.get('hard_rate', 0.5)
        surf_rate2 = s2.get('hard_rate', 0.5)
    
    elo1 = elo.get(p1, 1500)
    elo2 = elo.get(p2, 1500)
    elo_diff = (elo1 - elo2) / 400
    elo_diff = np.clip(elo_diff, -0.5, 0.5)
    
    form_diff = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
    very_recent_diff = s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)
    win_rate_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
    surf_diff = surf_rate1 - surf_rate2
    
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / h2h[(p1, p2)]['total']
    elif (p2, p1) in h2h:
        h2h_adv = 1 - (h2h[(p2, p1)]['wins'] / h2h[(p2, p1)]['total'])
    
    h2h_centered = h2h_adv - 0.5
    
    games_avg = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    games_norm = (games_avg - 21.5) / 8
    games_norm = np.clip(games_norm, -0.3, 0.3)
    
    exp1 = min(s1.get('matches', 0) / 200, 1)
    exp2 = min(s2.get('matches', 0) / 200, 1)
    exp_diff = exp1 - exp2
    
    momentum = very_recent_diff * 0.6 + form_diff * 0.4
    momentum = np.clip(momentum, -0.5, 0.5)
    
    features = [
        elo_diff, form_diff, very_recent_diff, win_rate_diff,
        surf_diff, h2h_centered, games_norm, exp_diff, momentum
    ]
    
    return features

# ==============================================================================
# TRAIN MODEL
# ==============================================================================
def train_model(df, player_stats, h2h, elo):
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
        n_estimators=200,
        max_depth=4,
        learning_rate=0.025,
        num_leaves=12,
        reg_alpha=1.5,
        reg_lambda=1.5,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    return model

# ==============================================================================
# NAME MATCHER
# ==============================================================================
class SimpleNameMatcher:
    def __init__(self, historical_names):
        self.historical_names = historical_names
        self.name_map = {}
        
        for name in historical_names:
            name_lower = name.lower()
            self.name_map[name_lower] = name
            
            parts = name.split()
            if parts:
                last_name = parts[-1].lower()
                if last_name not in self.name_map:
                    self.name_map[last_name] = name
    
    def find_match(self, search_name):
        if not search_name:
            return None
        
        search_str = str(search_name).strip()
        search_lower = search_str.lower()
        
        if search_str in self.historical_names:
            return search_str
        
        for name in self.historical_names:
            if name.lower() == search_lower:
                return name
        
        if search_lower in self.name_map:
            return self.name_map[search_lower]
        
        for name in self.historical_names:
            if search_lower in name.lower() or name.lower() in search_lower:
                return name
        
        return None

# ==============================================================================
# PREDICT
# ==============================================================================
def predict_match(model, p1, p2, surface, player_stats, h2h, elo, name_matcher, tournament=""):
    p1_match = name_matcher.find_match(p1)
    p2_match = name_matcher.find_match(p2)
    
    if not p1_match:
        return None, f"'{p1}' nao encontrado"
    if not p2_match:
        return None, f"'{p2}' nao encontrado"
    
    features = build_features(p1_match, p2_match, surface, player_stats, h2h, elo)
    if not features:
        return None, f"Estatisticas insuficientes"
    
    raw_prob = model.predict_proba(np.array([features]))[0][1]
    calibrated_prob = 0.5 + (raw_prob - 0.5) * WINNER_SMOOTH
    prob_p1 = np.clip(calibrated_prob, 0.35, 0.65)
    prob_p2 = 1 - prob_p1
    
    confidence = abs(prob_p1 - 0.5) * 2
    winner = p1_match if prob_p1 > 0.5 else p2_match
    
    if confidence >= MIN_CONFIDENCE_STRONG:
        rec = f"STRONG {winner}"
    elif confidence >= MIN_CONFIDENCE_GOOD:
        rec = f"GOOD {winner}"
    elif confidence >= MIN_CONFIDENCE_WEAK:
        rec = f"WEAK {winner}"
    else:
        rec = f"AVOID {winner}"
    
    s1 = player_stats.get(p1_match, {})
    s2 = player_stats.get(p2_match, {})
    
    form_diff = s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)
    momentum_edge = form_diff * 100
    exp_games = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    exp_games = np.clip(exp_games, 18, 30)
    
    result = {
        'Torneio': tournament,
        'Jogador1': p1,
        'Jogador2': p2,
        'Match_Historico': f"{p1_match} vs {p2_match}",
        'Superficie': surface,
        'Prob_P1': f"{prob_p1:.1%}",
        'Prob_P2': f"{prob_p2:.1%}",
        'Vencedor': winner,
        'Confianca': f"{confidence:.1%}",
        'Recomendacao': rec,
        'Games_Esperados': round(exp_games, 1)
    }
    
    return result, None

def parse_match_text(text, default_surface="Clay"):
    """Parse texto de matches no formato Jogador1 vs Jogador2"""
    matches = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Procurar padrao "Jogador1 vs Jogador2"
        vs_match = re.search(r'(.+?)\s+vs\s+(.+)', line, re.IGNORECASE)
        if vs_match:
            p1 = vs_match.group(1).strip()
            p2 = vs_match.group(2).strip()
            matches.append({
                'player1': p1,
                'player2': p2,
                'surface': default_surface
            })
        else:
            # Tentar separar por espacos ou tabs
            parts = re.split(r'\s{2,}|\t+', line)
            if len(parts) >= 2:
                matches.append({
                    'player1': parts[0].strip(),
                    'player2': parts[1].strip(),
                    'surface': default_surface
                })
    
    return matches

# ==============================================================================
# SCRAPER
# ==============================================================================
def scrape_matches():
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
            category = ev.get("tournament", {}).get("category", {}).get("name", "")
            if "WTA" in str(category).upper():
                continue
            
            matches.append({
                "tournament": ev["tournament"]["name"],
                "player1": ev["homeTeam"]["name"],
                "player2": ev["awayTeam"]["name"],
                "surface": detect_surface(ev["tournament"]["name"])
            })
        return matches
    except Exception:
        return []

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    st.title("ATP Predictor v6.2 - Batch Predictions")
    st.caption("Previsoes em lote | Probabilidades realistas 35%-65%")
    
    # Inicializar session_state
    if 'models_ready' not in st.session_state:
        st.session_state.models_ready = False
    if 'matches' not in st.session_state:
        st.session_state.matches = []
    
    uploaded_file = st.file_uploader("Upload do ficheiro historico", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("Processando..."):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df, all_players = process_historical_data(df)
                
                if df is None or len(df) == 0:
                    st.error("Nao foi possivel processar o arquivo")
                    return
                
                st.success(f"Carregados {len(df)} jogos e {len(all_players)} jogadores")
                
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
                st.session_state.all_players = all_players
                st.session_state.models_ready = True
                
                st.success("Modelo treinado com sucesso!")
                
            except Exception as e:
                st.error(f"Erro: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Interface after training
    if st.session_state.get('models_ready') and st.session_state.get('model'):
        # Abas para diferentes modos de entrada
        tab1, tab2, tab3 = st.tabs(["Jogos Sofascore", "Previsao Manual", "Inserir Lista"])
        
        # Tab 1: Sofascore
        with tab1:
            if st.button("Buscar jogos de hoje", use_container_width=True):
                with st.spinner("Buscando..."):
                    st.session_state.matches = scrape_matches()
                    st.rerun()
            
            if st.session_state.get('matches'):
                st.subheader("Previsoes")
                results = []
                errors = []
                
                for match in st.session_state.matches:
                    result, error = predict_match(
                        st.session_state.model, match['player1'], match['player2'], match['surface'],
                        st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                        st.session_state.name_matcher, match['tournament']
                    )
                    if result:
                        results.append(result)
                    elif error:
                        errors.append(error)
                
                if errors:
                    with st.expander(f"{len(errors)} jogadores nao encontrados"):
                        for e in set(errors):
                            st.write(e)
                
                if results:
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                    
                    # Download
                    buffer = io.BytesIO()
                    df_results.to_excel(buffer, index=False)
                    st.download_button("Download Excel", buffer.getvalue(),
                                     f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
        
        # Tab 2: Previsao manual individual
        with tab2:
            st.subheader("Previsao Individual")
            
            all_players = st.session_state.get('all_players', [])
            if all_players:
                players_with_stats = [p for p in all_players 
                                      if st.session_state.player_stats.get(p, {}).get('matches', 0) > 0]
                players_sorted = sorted(players_with_stats) if players_with_stats else sorted(all_players)
            else:
                players_sorted = []
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.selectbox("Jogador 1", [""] + players_sorted[:200])
            with col_b:
                manual_p2 = st.selectbox("Jogador 2", [""] + players_sorted[:200])
            with col_c:
                manual_surface = st.selectbox("Superficie", ["Clay", "Hard", "Grass"])
            
            if st.button("Prever", type="primary") and manual_p1 and manual_p2:
                if manual_p1 == manual_p2:
                    st.error("Selecione dois jogadores diferentes")
                else:
                    result, error = predict_match(
                        st.session_state.model, manual_p1, manual_p2, manual_surface,
                        st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                        st.session_state.name_matcher, "Manual"
                    )
                    if result:
                        st.dataframe(pd.DataFrame([result]), use_container_width=True)
                    else:
                        st.error(error)
        
        # Tab 3: Inserir lista de jogos
        with tab3:
            st.subheader("Inserir Lista de Jogos")
            st.markdown("Formato: Lehecka vs Michelsen (um por linha)")
            
            default_surface = st.selectbox("Superficie padrao", ["Clay", "Hard", "Grass"], key="batch_surface")
            
            matches_text = st.text_area(
                "Cole aqui os jogos:",
                height=200,
                placeholder="Lehecka vs Michelsen\nGriekspoor vs Musetti"
            )
            
            if st.button("Prever Lista", type="primary", use_container_width=True):
                if matches_text.strip():
                    matches_list = parse_match_text(matches_text, default_surface)
                    
                    if matches_list:
                        st.info(f"{len(matches_list)} jogos para prever")
                        
                        results = []
                        errors = []
                        
                        progress_bar = st.progress(0)
                        for i, match in enumerate(matches_list):
                            result, error = predict_match(
                                st.session_state.model, match['player1'], match['player2'], match['surface'],
                                st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                                st.session_state.name_matcher, "Batch"
                            )
                            if result:
                                results.append(result)
                            elif error:
                                errors.append(f"{match['player1']} vs {match['player2']}: {error}")
                            progress_bar.progress((i + 1) / len(matches_list))
                        progress_bar.empty()
                        
                        if errors:
                            with st.expander(f"{len(errors)} jogadores nao encontrados"):
                                for e in errors[:20]:
                                    st.write(e)
                        
                        if results:
                            st.subheader("Resultados")
                            df_results = pd.DataFrame(results)
                            st.dataframe(df_results, use_container_width=True, hide_index=True)
                            
                            # Summary
                            st.subheader("Resumo")
                            strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                            good = sum(1 for r in results if 'GOOD' in r['Recomendacao'])
                            weak = sum(1 for r in results if 'WEAK' in r['Recomendacao'])
                            
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("STRONG", strong)
                            c2.metric("GOOD", good)
                            c3.metric("WEAK", weak)
                            c4.metric("Total", len(results))
                            
                            # Download
                            buffer = io.BytesIO()
                            df_results.to_excel(buffer, index=False)
                            st.download_button("Download Excel", buffer.getvalue(),
                                             f"previsoes_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                             use_container_width=True)
                    else:
                        st.warning("Nenhum jogo encontrado")
                else:
                    st.warning("Cole a lista de jogos")
    
    elif not uploaded_file:
        st.info("Upload do ficheiro Excel/CSV com dados historicos")
        st.markdown("""
        O arquivo deve conter as colunas:
        - winner ou vencedor
        - loser ou perdedor
        """)

if __name__ == "__main__":
    main()
