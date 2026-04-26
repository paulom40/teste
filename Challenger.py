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
    clay = ['clay', 'madrid', 'rome', 'barcelona', 'munich', 'roland garros', 'atp masters 1000 madrid']
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
    """Process historical data and extract player names from the file"""
    
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
        st.error(f"Colunas nao encontradas. Colunas disponiveis: {list(df.columns)}")
        return None, None
    
    # Rename columns
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
    
    # Clean names
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    # Remove invalid rows
    df = df[df['winner'].notna() & df['loser'].notna()]
    df = df[df['winner'] != 'nan']
    df = df[df['loser'] != 'nan']
    df = df[df['winner'] != '']
    df = df[df['loser'] != '']
    df = df[df['winner'] != df['loser']]
    
    # Extract unique players
    all_players = list(set(df['winner'].unique()) | set(df['loser'].unique()))
    
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
        
        # Direct match
        if search_str in self.historical_names:
            return search_str
        
        # Remove common prefixes
        search_clean = re.sub(r'^(van|de|den|der|dos|das|le|la)\s+', '', search_lower)
        
        # Case insensitive
        for name in self.historical_names:
            if name.lower() == search_lower:
                return name
            if name.lower() == search_clean:
                return name
        
        # Last name match
        if search_lower in self.name_map:
            return self.name_map[search_lower]
        if search_clean in self.name_map:
            return self.name_map[search_clean]
        
        # Partial match
        for name in self.historical_names:
            if search_lower in name.lower() or name.lower() in search_lower:
                return name
            if search_clean in name.lower() or name.lower() in search_clean:
                return name
        
        return None

# ==============================================================================
# PREDICT MATCH
# ==============================================================================
def predict_match(model, p1, p2, surface, player_stats, h2h, elo, name_matcher):
    p1_match = name_matcher.find_match(p1)
    p2_match = name_matcher.find_match(p2)
    
    if not p1_match:
        return None, f"X '{p1}' nao encontrado no historico"
    if not p2_match:
        return None, f"X '{p2}' nao encontrado no historico"
    
    features = build_features(p1_match, p2_match, surface, player_stats, h2h, elo)
    if not features:
        return None, f"Estatisticas insuficientes para {p1_match} ou {p2_match}"
    
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
    momentum_edge = (s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)) * 100
    exp_games = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    
    has_data = s1.get('matches', 0) > 0 and s2.get('matches', 0) > 0
    data_quality = "OK" if has_data else "SEM DADOS"
    
    return {
        'Torneio': 'ATP',
        'Jogador1': p1,
        'Jogador2': p2,
        'Match_Historico': f"{p1_match} vs {p2_match}",
        'Superficie': surface,
        'Prob_P1': f"{prob_p1:.1%}",
        'Prob_P2': f"{1-prob_p1:.1%}",
        'Vencedor_Previsto': winner,
        'Confianca': f"{confidence:.1%}",
        'Recomendacao': rec,
        'Momentum': f"{momentum_edge:+.0f}",
        'Games_Esperados': round(exp_games, 1),
        'Dados': f"{data_quality} ({s1.get('matches',0)}/{s2.get('matches',0)})"
    }, None

# ==============================================================================
# PARSE DE TEXTO MELHORADO
# ==============================================================================
def parse_colab_text(text):
    """Parse o texto colado do usuario - suporta varios formatos"""
    matches = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove o prefixo "ATP" ou "CHALLENGER" se existir
        line = re.sub(r'^(ATP|CHALLENGER|ITF)\s+', '', line, flags=re.IGNORECASE)
        
        # Procura por "vs" ou "VS" ou "x"
        vs_match = re.search(r'(.+?)\s+(?:vs|VS|x)\s+(.+)', line)
        
        if vs_match:
            p1 = vs_match.group(1).strip()
            p2 = vs_match.group(2).strip()
            
            # Limpar nomes - remover múltiplos espaços e tabs
            p1 = re.sub(r'\s+', ' ', p1)
            p2 = re.sub(r'\s+', ' ', p2)
            
            # Remover caracteres especiais
            p1 = re.sub(r'[^\w\s\.\-]', '', p1)
            p2 = re.sub(r'[^\w\s\.\-]', '', p2)
            
            # Extrair favorito se tiver -> ou %
            arrow_match = re.search(r'->\s*([A-Za-z\s]+?)(?:\s*(\d+)%?)?', line)
            if arrow_match:
                favorite = arrow_match.group(1).strip()
                percentage = int(arrow_match.group(2)) / 100 if arrow_match.group(2) else None
            else:
                favorite = None
                percentage = None
            
            # Detectar superficie
            surface = 'Clay'
            if 'hard' in line.lower():
                surface = 'Hard'
            elif 'grass' in line.lower():
                surface = 'Grass'
            
            if p1 and p2 and p1 != p2:
                matches.append({
                    'player1': p1,
                    'player2': p2,
                    'surface': surface,
                    'favorite': favorite,
                    'expected_prob': percentage,
                    'tournament': 'ATP'
                })
    
    return matches

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("ATP Predictor v8.0 - Previsao por Lista")
    st.caption("Cole sua lista de jogos e o modelo fara as previsoes baseado no seu historico")
    
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
                    st.error("Nao foi possivel processar o arquivo.")
                    return
                
                st.success(f"{len(df)} jogos | {len(all_players)} jogadores")
                
                with st.expander("Amostra dos jogadores (primeiros 50)"):
                    for i, p in enumerate(sorted(all_players)[:50]):
                        st.write(f"{i+1}. {p}")
                
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
                
                st.success("Modelo treinado!")
                
            except Exception as e:
                st.error(f"Erro: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.get('models_ready'):
        st.subheader("COLE SUA LISTA DE JOGOS")
        
        st.markdown("""
        **Formatos aceitos:**
        - Lehecka vs Michelsen
        - Griekspoor vs Musetti
        - ATP Lehecka vs Michelsen
        - Lehecka x Michelsen -> Lehecka 61%
        """)
        
        matches_text = st.text_area(
            "Cole aqui os jogos:",
            height=400,
            placeholder="Lehecka vs Michelsen\nGriekspoor vs Musetti\nPrizmic vs Etcheverry"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            surface_override = st.selectbox("Superficie (padrao)", ["Clay", "Hard", "Grass"], index=0)
        with col2:
            if st.button("LIMPAR", use_container_width=True):
                st.rerun()
        
        if matches_text:
            parsed_matches = parse_colab_text(matches_text)
            
            if parsed_matches:
                st.success(f"{len(parsed_matches)} jogos detectados!")
                
                for match in parsed_matches:
                    if match['surface'] == 'Clay' and surface_override != 'Clay':
                        match['surface'] = surface_override
                
                if st.button("FAZER PREVISOES", type="primary", use_container_width=True):
                    results = []
                    errors = []
                    
                    progress_bar = st.progress(0)
                    for i, match in enumerate(parsed_matches):
                        result, error = predict_match(
                            st.session_state.model, 
                            match['player1'], match['player2'], match['surface'],
                            st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                            st.session_state.name_matcher
                        )
                        if result:
                            if match.get('expected_prob'):
                                result['Prob_Esperada'] = f"{match['expected_prob']:.0%}"
                                result['Favorito_Lista'] = match.get('favorite', '')
                            results.append(result)
                        elif error:
                            errors.append(error)
                        progress_bar.progress((i + 1) / len(parsed_matches))
                    progress_bar.empty()
                    
                    if errors:
                        with st.expander(f"{len(errors)} jogadores nao encontrados"):
                            for e in set(errors):
                                st.write(e)
                    
                    if results:
                        st.subheader("RESULTADOS DAS PREVISOES")
                        
                        df_results = pd.DataFrame(results)
                        
                        cols = ['Torneio', 'Jogador1', 'Jogador2', 'Match_Historico', 'Superficie',
                               'Prob_P1', 'Prob_P2', 'Vencedor_Previsto', 'Confianca', 'Recomendacao',
                               'Momentum', 'Games_Esperados', 'Dados']
                        
                        if 'Prob_Esperada' in df_results.columns:
                            cols.insert(6, 'Prob_Esperada')
                            cols.insert(7, 'Favorito_Lista')
                        
                        df_results = df_results[[c for c in cols if c in df_results.columns]]
                        
                        def color_recommendation(val):
                            if 'STRONG' in str(val):
                                return 'background-color: #2e7d32; color: white'
                            elif 'GOOD' in str(val):
                                return 'background-color: #4caf50; color: white'
                            elif 'AVOID' in str(val):
                                return 'background-color: #9e9e9e; color: white'
                            return ''
                        
                        styled = df_results.style.format({
                            'Prob_P1': '{:.1%}',
                            'Prob_P2': '{:.1%}',
                            'Confianca': '{:.1%}'
                        }).map(color_recommendation, subset=['Recomendacao'])
                        
                        st.dataframe(styled, use_container_width=True, hide_index=True, height=600)
                        
                        # Resumo
                        st.subheader("Resumo")
                        strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                        good = sum(1 for r in results if 'GOOD' in r['Recomendacao'])
                        avoid = sum(1 for r in results if 'AVOID' in r['Recomendacao'])
                        
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        with col_s1:
                            st.metric("STRONG", strong)
                        with col_s2:
                            st.metric("GOOD", good)
                        with col_s3:
                            st.metric("AVOID", avoid)
                        with col_s4:
                            conf_values = [float(r['Confianca'].replace('%', '')) for r in results]
                            avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
                            st.metric("Confianca Media", f"{avg_conf:.0f}%")
                        
                        # Download
                        buffer = io.BytesIO()
                        df_results.to_excel(buffer, index=False)
                        st.download_button(
                            "Download Excel com Previsoes",
                            buffer.getvalue(),
                            f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            use_container_width=True
                        )
            else:
                st.warning("Nenhum jogo detectado. Use o formato: Jogador1 vs Jogador2")
        
        # Previsao manual
        with st.expander("PREVISAO MANUAL INDIVIDUAL"):
            players_with_stats = [p for p in st.session_state.all_players 
                                  if st.session_state.player_stats.get(p, {}).get('matches', 0) > 0]
            players_sorted = sorted(players_with_stats)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.selectbox("Jogador 1", [""] + players_sorted, key="man_p1")
            with col_b:
                manual_p2 = st.selectbox("Jogador 2", [""] + players_sorted, key="man_p2")
            with col_c:
                manual_surface = st.selectbox("Superficie", ["Clay", "Hard", "Grass"], key="man_surf")
            
            if st.button("PREVER JOGO", key="man_btn") and manual_p1 and manual_p2:
                if manual_p1 == manual_p2:
                    st.error("Selecione dois jogadores diferentes!")
                else:
                    result, error = predict_match(
                        st.session_state.model, manual_p1, manual_p2, manual_surface,
                        st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                        st.session_state.name_matcher
                    )
                    if result:
                        st.dataframe(pd.DataFrame([result]), use_container_width=True)
                    else:
                        st.error(error)
    
    elif not uploaded_file:
        st.info("Faca upload do seu ficheiro Excel/CSV com dados historicos")

if __name__ == "__main__":
    main()
