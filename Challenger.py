import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import re

warnings.filterwarnings('ignore')

st.set_page_config(page_title="ATP Predictor v7.4 - Dashboard", page_icon="🎾", layout="wide")

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
    
    for col in df.columns:
        if 'winner' in col or 'vencedor' in col:
            winner_col = col
        elif 'loser' in col or 'perdedor' in col:
            loser_col = col
    
    if not winner_col or not loser_col:
        return None, None
    
    df = df.rename(columns={winner_col: 'winner', loser_col: 'loser'})
    
    if 'date' not in df.columns:
        df['date'] = pd.Timestamp.now()
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    if 'total_games' not in df.columns:
        df['total_games'] = 22
    
    if 'surface' not in df.columns:
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
                'recent_form': 0.5, 'very_recent_form': 0.5, 'avg_games': 22
            }
            continue
        
        wins = len(player_matches[player_matches['winner'] == player])
        total = len(player_matches)
        win_rate = wins / total if total > 0 else 0.5
        
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
            'win_rate': win_rate,
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

def calculate_elo_by_surface(df, all_players, surface, k=32):
    elo = {p: 1500 for p in all_players}
    df_surface = df[df['surface'] == surface].sort_values('date')
    
    for _, row in df_surface.iterrows():
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
    
    elo1 = elo.get(p1, 1500)
    elo2 = elo.get(p2, 1500)
    elo_diff = (elo1 - elo2) / 400
    elo_diff = np.clip(elo_diff, -0.5, 0.5)
    
    form_diff = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
    very_recent_diff = s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)
    win_rate_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
    
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
        h2h_centered, games_norm, exp_diff, momentum
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
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        num_leaves=12,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    return model

# ==============================================================================
# SIMPLE NAME MATCHER
# ==============================================================================
def find_player(name, historical_names):
    if not name:
        return None
    
    name_str = str(name).strip()
    name_lower = name_str.lower()
    
    if name_str in historical_names:
        return name_str
    
    for player in historical_names:
        if player.lower() == name_lower:
            return player
    
    parts = name_lower.split()
    if parts:
        last_name = parts[-1]
        for player in historical_names:
            if player.lower().endswith(last_name):
                return player
    
    for player in historical_names:
        if name_lower in player.lower():
            return player
    
    return None

# ==============================================================================
# PREDICT
# ==============================================================================
def predict_match(model, p1, p2, surface, player_stats, h2h, elo, historical_names, tournament=""):
    p1_match = find_player(p1, historical_names)
    p2_match = find_player(p2, historical_names)
    
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

# ==============================================================================
# PARSE MATCH TEXT
# ==============================================================================
def parse_match_text(text, default_surface="Clay"):
    matches = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
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
            parts = line.split()
            if len(parts) >= 2:
                matches.append({
                    'player1': parts[0].strip(),
                    'player2': parts[1].strip(),
                    'surface': default_surface
                })
    
    return matches

# ==============================================================================
# SCRAPER v7.3 — HOJE + AMANHÃ + FALLBACK + LOGS
# ==============================================================================
def scrape_matches():
    target_date = datetime.now().strftime("%Y-%m-%d")
    tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    dates_to_check = [target_date, tomorrow_date]

    endpoints = [
        "https://api.sofascore.com/api/v1/sport/tennis/events/{date}",
        "https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{date}",
        "https://api.sofascore.com/api/v1/sport/tennis/{date}/events"
    ]

    headers = {"User-Agent": "Mozilla/5.0"}

    all_matches = []
    logs = []

    for d in dates_to_check:
        logs.append(f"📅 Procurando jogos para {d}")

        for ep in endpoints:
            url = ep.format(date=d)
            logs.append(f"🔎 Testando endpoint: {url}")

            try:
                r = requests.get(url, headers=headers, timeout=10)

                if r.status_code != 200:
                    logs.append(f"❌ HTTP {r.status_code}")
                    continue

                data = r.json()
                events = data.get("events", [])

                if not events:
                    logs.append("⚠️ Endpoint OK mas sem eventos")
                    continue

                logs.append(f"✅ Encontrados {len(events)} eventos")

                for ev in events:
                    category = ev.get("tournament", {}).get("category", {}).get("name", "")
                    if "WTA" in str(category).upper():
                        continue

                    all_matches.append({
                        "tournament": ev["tournament"]["name"],
                        "player1": ev["homeTeam"]["name"],
                        "player2": ev["awayTeam"]["name"],
                        "surface": detect_surface(ev["tournament"]["name"])
                    })

                break

            except Exception as e:
                logs.append(f"💥 Erro: {e}")

    return all_matches, logs
def main():
    st.title("ATP Predictor v7.4 - Dashboard")
    st.caption("Sistema simples de matching | Previsoes em lote + Dashboard Estatístico")
    
    if 'models_ready' not in st.session_state:
        st.session_state.models_ready = False
    if 'matches' not in st.session_state:
        st.session_state.matches = []
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'run_predictions' not in st.session_state:
        st.session_state.run_predictions = False
    
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

                with st.expander("Jogadores no historico (amostra)"):
                    for i, p in enumerate(sorted(all_players)[:30]):
                        st.write(f"{i+1}. {p}")
                    if len(all_players) > 30:
                        st.write(f"... e mais {len(all_players) - 30} jogadores")

                player_stats = calculate_player_stats(df, all_players)
                h2h = calculate_h2h(df)

                elo_global = calculate_elo(df, all_players)
                elo_clay = calculate_elo_by_surface(df, all_players, "Clay")
                elo_hard = calculate_elo_by_surface(df, all_players, "Hard")
                elo_grass = calculate_elo_by_surface(df, all_players, "Grass")

                model = train_model(df, player_stats, h2h, elo_global)

                st.session_state.model = model
                st.session_state.player_stats = player_stats
                st.session_state.h2h = h2h
                st.session_state.elo = elo_global
                st.session_state.elo_clay = elo_clay
                st.session_state.elo_hard = elo_hard
                st.session_state.elo_grass = elo_grass
                st.session_state.all_players = all_players
                st.session_state.models_ready = True

                st.success("Modelo treinado com sucesso!")

            except Exception as e:
                st.error(f"Erro: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.get('models_ready') and st.session_state.get('model'):
        tab1, tab2, tab3, tab4 = st.tabs([
            "Jogos Sofascore",
            "Previsao Manual",
            "Inserir Lista",
            "Dashboard Estatístico"
        ])
        # ========================= TAB 1 — Sofascore =========================
        with tab1:
            if st.button("Buscar jogos de hoje", use_container_width=True):
                with st.spinner("Buscando..."):
                    matches, logs = scrape_matches()
                    st.session_state.matches = matches
                    st.session_state.logs = logs
                    st.session_state.run_predictions = False

            if st.session_state.get('logs'):
                with st.expander("Ver detalhes da busca"):
                    for line in st.session_state.logs:
                        st.write(line)

            if st.session_state.get('matches'):
                st.write(f"{len(st.session_state.matches)} jogos encontrados")

                if st.button("🔮 Prever agora", type="primary", use_container_width=True):
                    st.session_state.run_predictions = True

            if st.session_state.get('run_predictions', False) and st.session_state.get('matches'):
                st.subheader("Previsões")
                results = []
                errors = []

                for match in st.session_state.matches:
                    result, error = predict_match(
                        st.session_state.model,
                        match['player1'],
                        match['player2'],
                        match['surface'],
                        st.session_state.player_stats,
                        st.session_state.h2h,
                        st.session_state.elo,
                        st.session_state.all_players,
                        match['tournament']
                    )
                    if result:
                        results.append(result)
                    elif error:
                        errors.append(error)

                if errors:
                    with st.expander(f"{len(errors)} jogadores nao encontrados"):
                        for e in errors:
                            st.write(e)

                if results:
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)

                    buffer = io.BytesIO()
                    df_results.to_excel(buffer, index=False)
                    st.download_button(
                        "Download Excel",
                        buffer.getvalue(),
                        f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                    )

        # ====================== TAB 2 — Previsão Manual ======================
        with tab2:
            st.subheader("Previsao Individual")

            all_players = st.session_state.get('all_players', [])
            players_sorted = sorted(all_players) if all_players else []

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.selectbox("Jogador 1", [""] + players_sorted)
            with col_b:
                manual_p2 = st.selectbox("Jogador 2", [""] + players_sorted)
            with col_c:
                manual_surface = st.selectbox("Superficie", ["Clay", "Hard", "Grass"])

            if st.button("Prever", type="primary"):
                if not manual_p1 or not manual_p2:
                    st.error("Selecione os dois jogadores")
                elif manual_p1 == manual_p2:
                    st.error("Selecione dois jogadores diferentes")
                else:
                    result, error = predict_match(
                        st.session_state.model,
                        manual_p1,
                        manual_p2,
                        manual_surface,
                        st.session_state.player_stats,
                        st.session_state.h2h,
                        st.session_state.elo,
                        st.session_state.all_players,
                        "Manual"
                    )
                    if result:
                        st.dataframe(pd.DataFrame([result]), use_container_width=True, hide_index=True)
                    else:
                        st.error(error)
        # ====================== TAB 3 — Lista de Jogos =======================
        with tab3:
            st.subheader("Inserir Lista de Jogos")

            st.info("Formatos aceitos: 'Lehecka vs Michelsen' ou 'Jiri Lehecka vs Alex Michelsen'")

            default_surface = st.selectbox("Superficie padrao", ["Clay", "Hard", "Grass"], key="batch_surface")

            matches_text = st.text_area(
                "Cole aqui os jogos (um por linha):",
                height=300,
                placeholder="Lehecka vs Michelsen\nGriekspoor vs Musetti\nPrizmic vs Etcheverry"
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
                                st.session_state.model,
                                match['player1'],
                                match['player2'],
                                match['surface'],
                                st.session_state.player_stats,
                                st.session_state.h2h,
                                st.session_state.elo,
                                st.session_state.all_players,
                                "Batch"
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

                            strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                            good = sum(1 for r in results if 'GOOD' in r['Recomendacao'])
                            weak = sum(1 for r in results if 'WEAK' in r['Recomendacao'])

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("STRONG", strong)
                            c2.metric("GOOD", good)
                            c3.metric("WEAK", weak)
                            c4.metric("Total", len(results))

                            buffer = io.BytesIO()
                            df_results.to_excel(buffer, index=False)
                            st.download_button(
                                "Download Excel",
                                buffer.getvalue(),
                                f"previsoes_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                use_container_width=True
                            )
                    else:
                        st.warning("Nenhum jogo encontrado")
                else:
                    st.warning("Cole a lista de jogos")

        # ====================== TAB 4 — Dashboard Estatístico =======================
        with tab4:
            st.header("📊 Dashboard Estatístico — Jogadores ATP")

            stats = st.session_state.player_stats
            elo = st.session_state.elo
            h2h = st.session_state.h2h
            players = st.session_state.all_players

            # ---------------------- TABELA GERAL ----------------------
            st.subheader("📌 Estatísticas Gerais")

            df_stats = pd.DataFrame([
                {
                    "Jogador": p,
                    "ELO": round(elo.get(p, 1500), 1),
                    "Win Rate": round(stats[p]["win_rate"] * 100, 1),
                    "Forma (10 jogos)": round(stats[p]["recent_form"] * 100, 1),
                    "Forma (5 jogos)": round(stats[p]["very_recent_form"] * 100, 1),
                    "Média Games": round(stats[p]["avg_games"], 1),
                    "Jogos": stats[p]["matches"]
                }
                for p in players
            ])

            df_stats = df_stats.sort_values("ELO", ascending=False)
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

            # ---------------------- RANKING ELO GLOBAL ----------------------
            st.subheader("🏆 Ranking ELO Global")

            df_top_elo = df_stats.head(20).sort_values("ELO")
            fig_elo = df_top_elo.plot(
                x="Jogador", y="ELO", kind="barh", figsize=(10, 6), title="Top 20 ELO"
            ).get_figure()
            st.pyplot(fig_elo)

            # ---------------------- RANKING ELO POR SUPERFÍCIE ----------------------
            st.subheader("🏟️ Ranking ELO por Superfície")

            surface_option = st.selectbox(
                "Escolha a superfície",
                ["Clay", "Hard", "Grass"],
                key="elo_surface_select"
            )

            elo_map = {
                "Clay": st.session_state.elo_clay,
                "Hard": st.session_state.elo_hard,
                "Grass": st.session_state.elo_grass
            }

            elo_surface = elo_map[surface_option]

            df_elo_surface = pd.DataFrame([
                {"Jogador": p, "ELO": round(elo_surface.get(p, 1500), 1)}
                for p in players
            ]).sort_values("ELO", ascending=False).head(30)

            st.dataframe(df_elo_surface, use_container_width=True, hide_index=True)

            fig_elo_surface = df_elo_surface.sort_values("ELO").plot(
                x="Jogador", y="ELO", kind="barh",
                figsize=(10, 6),
                title=f"Top 30 ELO — {surface_option}"
            ).get_figure()
            st.pyplot(fig_elo_surface)

            # ---------------------- H2H SIMPLES ----------------------
            st.subheader("⚔️ Head-to-Head (H2H) Rápido")

            col1, col2 = st.columns(2)
            with col1:
                h2h_p1 = st.selectbox("Jogador 1 (H2H)", [""] + sorted(players), key="h2h_p1_simple")
            with col2:
                h2h_p2 = st.selectbox("Jogador 2 (H2H)", [""] + sorted(players), key="h2h_p2_simple")

            if h2h_p1 and h2h_p2 and h2h_p1 != h2h_p2:
                w1 = h2h.get((h2h_p1, h2h_p2), {"wins": 0, "total": 0})
                w2 = h2h.get((h2h_p2, h2h_p1), {"wins": 0, "total": 0})
                total = w1["total"] + w2["total"]

                st.write(f"**{h2h_p1} vitórias:** {w1['wins']}")
                st.write(f"**{h2h_p2} vitórias:** {w2['wins']}")
                st.write(f"**Total de jogos:** {total}")

                if total > 0:
                    df_h2h_simple = pd.DataFrame({
                        "Jogador": [h2h_p1, h2h_p2],
                        "Vitórias": [w1["wins"], w2["wins"]]
                    })
                    fig_h2h_simple = df_h2h_simple.plot(
                        x="Jogador", y="Vitórias", kind="bar",
                        title=f"H2H — {h2h_p1} vs {h2h_p2}", figsize=(6, 4)
                    ).get_figure()
                    st.pyplot(fig_h2h_simple)

            # ---------------------- COMPARAÇÃO 1v1 COMPLETA ----------------------
            st.subheader("⚔️ Comparação 1v1 — Forma, H2H e Média de Games")

            colc1, colc2 = st.columns(2)
            with colc1:
                p1 = st.selectbox("Jogador 1", [""] + sorted(players), key="comp_p1")
            with colc2:
                p2 = st.selectbox("Jogador 2", [""] + sorted(players), key="comp_p2")

            if p1 and p2 and p1 != p2:
                s1 = stats[p1]
                s2 = stats[p2]

                # FORMA RECENTE
                st.markdown("### 🔥 Forma Recente (10 e 5 jogos)")
                df_form = pd.DataFrame({
                    "Jogador": [p1, p2],
                    "Forma (10 jogos)": [s1["recent_form"] * 100, s2["recent_form"] * 100],
                    "Forma (5 jogos)": [s1["very_recent_form"] * 100, s2["very_recent_form"] * 100]
                })
                st.dataframe(df_form, hide_index=True, use_container_width=True)

                fig_form = df_form.set_index("Jogador").plot(
                    kind="bar",
                    figsize=(6, 4),
                    title="Comparação de Forma Recente"
                ).get_figure()
                st.pyplot(fig_form)

                # H2H DETALHADO
                st.markdown("### ⚔️ Head-to-Head (H2H) Detalhado")
                w1 = h2h.get((p1, p2), {"wins": 0, "total": 0})
                w2 = h2h.get((p2, p1), {"wins": 0, "total": 0})
                total = w1["total"] + w2["total"]

                st.write(f"**{p1} vitórias:** {w1['wins']}")
                st.write(f"**{p2} vitórias:** {w2['wins']}")
                st.write(f"**Total de jogos:** {total}")

                if total > 0:
                    df_h2h = pd.DataFrame({
                        "Jogador": [p1, p2],
                        "Vitórias": [w1["wins"], w2["wins"]]
                    })
                    fig_h2h = df_h2h.plot(
                        x="Jogador", y="Vitórias", kind="bar",
                        title=f"H2H — {p1} vs {p2}", figsize=(6, 4)
                    ).get_figure()
                    st.pyplot(fig_h2h)

                # MÉDIA DE GAMES
                st.markdown("### 🎾 Média de Games por Jogo")
                df_games = pd.DataFrame({
                    "Jogador": [p1, p2],
                    "Média Games": [s1["avg_games"], s2["avg_games"]]
                })
                st.dataframe(df_games, hide_index=True, use_container_width=True)

                fig_games = df_games.set_index("Jogador").plot(
                    kind="bar",
                    figsize=(6, 4),
                    title="Média de Games"
                ).get_figure()
                st.pyplot(fig_games)

                # OVER/UNDER 21.5
                st.markdown("### 📈 Probabilidade Over/Under 21.5 (Histórico)")

                avg_games = (s1["avg_games"] + s2["avg_games"]) / 2
                prob_over = min(max((avg_games - 21.5) / 10 + 0.5, 0.05), 0.95)
                prob_under = 1 - prob_over

                df_ou = pd.DataFrame({
                    "Linha": ["Over 21.5", "Under 21.5"],
                    "Probabilidade": [prob_over * 100, prob_under * 100]
                })
                st.dataframe(df_ou, hide_index=True, use_container_width=True)

                fig_ou = df_ou.plot(
                    x="Linha", y="Probabilidade", kind="bar",
                    title="Probabilidade Over/Under 21.5",
                    figsize=(6, 4)
                ).get_figure()
                st.pyplot(fig_ou)

    elif not uploaded_file:
        st.info("Upload do ficheiro Excel/CSV com dados historicos")
        st.markdown("""
        O arquivo deve conter as colunas:
        - `winner` ou `vencedor` - nome do vencedor
        - `loser` ou `perdedor` - nome do perdedor
        - opcional: `date`, `total_games`, `surface`
        """)

if __name__ == "__main__":
    main()
