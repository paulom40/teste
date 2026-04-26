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
    clay = ['clay', 'madrid', 'rome', 'barcelona', 'munich', 'roland garros']
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
    
    for col in df.columns:
        if 'winner' in col or 'vencedor' in col:
            winner_col = col
        elif 'loser' in col or 'perdedor' in col:
            loser_col = col
    
    if not winner_col or not loser_col:
        st.error(f"Colunas nao encontradas. Colunas disponiveis: {list(df.columns)}")
        return None, None
    
    # Rename columns
    df = df.rename(columns={
        winner_col: 'winner',
        loser_col: 'loser'
    })
    
    # Convert date
    if 'date' not in df.columns:
        df['date'] = pd.Timestamp.now()
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Calculate total games
    if 'total_games' not in df.columns:
        df['total_games'] = 22
    
    # Detect surface
    if 'tournament' in df.columns:
        df['surface'] = df['tournament'].apply(detect_surface)
    else:
        df['surface'] = 'Hard'
    
    # Clean names - remover caracteres especiais
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
    all_players = sorted(list(set(df['winner'].unique()) | set(df['loser'].unique())))
    
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
# SMART NAME MATCHER - COM MAPEAMENTO MANUAL
# ==============================================================================
class SmartNameMatcher:
    def __init__(self, historical_names):
        self.historical_names = historical_names
        self.historical_set = set(historical_names)
        self.last_name_map = defaultdict(list)
        self.full_name_map = {}
        
        # Criar índice de sobrenomes e mapeamento completo
        for name in historical_names:
            name_lower = name.lower()
            self.full_name_map[name_lower] = name
            
            parts = name.split()
            if parts:
                last_name = parts[-1].lower()
                self.last_name_map[last_name].append(name)
        
        # Mapeamento manual de nomes comuns (sobrenome -> nome completo)
        self.manual_mapping = {
            # ATP Top Players
            'alcaraz': 'Carlos Alcaraz',
            'sinner': 'Jannik Sinner',
            'djokovic': 'Novak Djokovic',
            'nadal': 'Rafael Nadal',
            'medvedev': 'Daniil Medvedev',
            'zverev': 'Alexander Zverev',
            'tsitsipas': 'Stefanos Tsitsipas',
            'rune': 'Holger Rune',
            'ruud': 'Casper Ruud',
            'fritz': 'Taylor Fritz',
            'tiafoe': 'Frances Tiafoe',
            'paul': 'Tommy Paul',
            'auger-aliassime': 'Felix Auger-Aliassime',
            'khachanov': 'Karen Khachanov',
            'rublev': 'Andrey Rublev',
            'de minaur': 'Alex de Minaur',
            'hurkacz': 'Hubert Hurkacz',
            'shelton': 'Ben Shelton',
            'musetti': 'Lorenzo Musetti',
            'cerundolo': 'Francisco Cerundolo',
            'etcheverry': 'Tomas Martin Etcheverry',
            'jarry': 'Nicolas Jarry',
            'baez': 'Sebastian Baez',
            'fils': 'Arthur Fils',
            'fonseca': 'Joao Fonseca',
            'lehecka': 'Jiri Lehecka',
            'griekspoor': 'Tallon Griekspoor',
            'norrie': 'Cameron Norrie',
            'sonego': 'Lorenzo Sonego',
            'davidovich': 'Alejandro Davidovich Fokina',
            
            # Adicione mais conforme necessário
            'struff': 'Jan-Lennard Struff',
            'hanfmann': 'Yannick Hanfmann',
            'altmaier': 'Daniel Altmaier',
            'koepfer': 'Dominik Koepfer',
            'mae': 'Mae Malige',
        }
    
    def find_match(self, search_name):
        if not search_name:
            return None
        
        search_str = str(search_name).strip()
        search_lower = search_str.lower()
        
        # 1. Match exato
        if search_str in self.historical_set:
            return search_str
        
        # 2. Case insensitive
        for name in self.historical_names:
            if name.lower() == search_lower:
                return name
        
        # 3. Mapeamento manual
        if search_lower in self.manual_mapping:
            manual_match = self.manual_mapping[search_lower]
            if manual_match in self.historical_set:
                return manual_match
        
        # 4. Match por sobrenome
        if search_lower in self.last_name_map:
            matches = self.last_name_map[search_lower]
            if len(matches) == 1:
                return matches[0]
            
            # Se houver múltiplos, escolher o mais provável (mais jogos)
            return matches[0]
        
        # 5. Match parcial (busca contém nome histórico)
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
        'Jogador1': p1,
        'Jogador2': p2,
        'Match_Historico': f"{p1_match} vs {p2_match}",
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
    """Parse o texto colado do usuario"""
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
            
            # Limpar nomes
            p1 = re.sub(r'\s+', ' ', p1)
            p2 = re.sub(r'\s+', ' ', p2)
            p1 = p1.strip()
            p2 = p2.strip()
            
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
                    'surface': surface
                })
    
    return matches

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("ATP Predictor v8.0 - Previsao por Lista")
    st.caption("Normalizacao automatica de nomes | Mapeamento manual incluso")
    
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
                
                # Mostrar TODOS os jogadores para referencia
                with st.expander("JOGADORES NO HISTORICO (use estes nomes exatos)"):
                    for i, p in enumerate(sorted(all_players)):
                        st.write(f"{i+1}. {p}")
                
                player_stats = calculate_player_stats(df, all_players)
                h2h = calculate_h2h(df)
                elo = calculate_elo(df, all_players)
                model = train_model(df, player_stats, h2h, elo)
                name_matcher = SmartNameMatcher(all_players)
                
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
    
    if st.session_state.get('models_ready'):
        st.subheader("COLE SUA LISTA DE JOGOS")
        
        st.markdown(f"""
        **Dica:** Use os nomes EXATOS da lista acima.
        
        Exemplos que funcionam:
        - `Jiri Lehecka vs Alex Michelsen`
        - `Lehecka vs Michelsen` (se o sobrenome for único)
        """)
        
        matches_text = st.text_area(
            "Cole aqui os jogos:",
            height=300,
            placeholder="Lehecka vs Michelsen\nGriekspoor vs Musetti\nPrizmic vs Etcheverry"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            surface_override = st.selectbox("Superficie", ["Clay", "Hard", "Grass"], index=0)
        with col2:
            if st.button("LIMPAR", use_container_width=True):
                st.rerun()
        
        if matches_text:
            parsed_matches = parse_colab_text(matches_text)
            
            if parsed_matches:
                st.info(f"{len(parsed_matches)} jogos detectados. Processando...")
                
                for match in parsed_matches:
                    if surface_override != 'Clay':
                        match['surface'] = surface_override
                
                if st.button("FAZER PREVISOES", type="primary", use_container_width=True):
                    results = []
                    not_found = []
                    
                    progress_bar = st.progress(0)
                    for i, match in enumerate(parsed_matches):
                        result, missing = predict_match(
                            st.session_state.model, 
                            match['player1'], match['player2'], match['surface'],
                            st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                            st.session_state.name_matcher
                        )
                        if result:
                            results.append(result)
                        elif missing:
                            not_found.append(missing)
                        progress_bar.progress((i + 1) / len(parsed_matches))
                    progress_bar.empty()
                    
                    if not_found:
                        st.warning(f"{len(not_found)} jogos nao processados")
                        with st.expander("Clique para ver os jogadores nao encontrados"):
                            for p1, p2 in not_found:
                                st.write(f"  - {p1} vs {p2}")
                    
                    if results:
                        st.subheader(f"RESULTADOS ({len(results)} jogos)")
                        
                        df_results = pd.DataFrame(results)
                        
                        styled = df_results.style.format({
                            'Prob_P1': '{:.1%}',
                            'Prob_P2': '{:.1%}',
                            'Confianca': '{:.1%}'
                        })
                        
                        st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
                        
                        # Resumo
                        strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                        good = sum(1 for r in results if 'GOOD' in r['Recomendacao'])
                        
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("STRONG", strong)
                        with col_s2:
                            st.metric("GOOD", good)
                        with col_s3:
                            st.metric("Total", len(results))
                        
                        # Download
                        buffer = io.BytesIO()
                        df_results.to_excel(buffer, index=False)
                        st.download_button(
                            "Download Excel",
                            buffer.getvalue(),
                            f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                        )
                    else:
                        st.error("Nenhum jogo encontrado. Verifique os nomes na lista acima.")
            else:
                st.warning("Nenhum jogo detectado")
    
    elif not uploaded_file:
        st.info("Faca upload do seu ficheiro Excel/CSV com dados historicos")

if __name__ == "__main__":
    main()
