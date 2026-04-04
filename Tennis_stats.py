import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
from io import BytesIO

st.set_page_config(page_title="Tênis Predictor", page_icon="🎾", layout="wide")
st.title("🎾 Partidas Hoje + Predictor Stats")

tab1, tab2, tab3 = st.tabs(["📅 Partidas Hoje", "🔍 Previsão Personalizada", "📈 Modeling Strategy"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])

# ====================== CARREGAR STATS ======================
@st.cache_data
def load_stats(file):
    if not file:
        return pd.DataFrame()
    try:
        df = pd.read_excel(file)
        def norm(name):
            if not isinstance(name, str): return ""
            n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            return ''.join(filter(str.isalnum, n.lower().strip()))
        df['winner_clean'] = df['winner_name'].apply(norm)
        df['loser_clean'] = df['loser_name'].apply(norm)
        st.sidebar.success(f"✅ {len(df)} jogos carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar stats: {e}")
        return pd.DataFrame()

df_stats = load_stats(uploaded_file)

# ====================== FUNÇÕES AUXILIARES ======================
def norm(name):
    if not isinstance(name, str): return ""
    n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return ''.join(filter(str.isalnum, n.lower().strip()))

def find_best_player_stats(player_name, df):
    if df.empty or not player_name: 
        return pd.Series(dtype='object')
    clean_name = norm(player_name)
    best_match = None
    best_score = 0.0
    for _, row in df.iterrows():
        for col in ['winner_clean', 'loser_clean']:
            clean_db = row.get(col, "")
            if not clean_db: continue
            similarity = SequenceMatcher(None, clean_name, clean_db).ratio()
            if clean_name in clean_db or clean_db in clean_name:
                similarity = max(similarity, 0.95)
            score = similarity * 100
            if score > best_score:
                best_score = score
                best_match = row
    return best_match if best_score >= 60 else pd.Series(dtype='object')

def predict_from_stats(p1_stats, p2_stats, superficie="Hard"):
    def safe(v):
        try: return float(v) if pd.notna(v) else 0.0
        except: return 0.0

    def serve_win(stats):
        svpt = safe(stats.get('w_svpt', 0))
        if svpt == 0: return 0.65
        return (safe(stats.get('w_1stWon', 0)) + safe(stats.get('w_2ndWon', 0))) / svpt

    serve1 = serve_win(p1_stats)
    serve2 = serve_win(p2_stats)
    return1 = 1 - serve2
    return2 = 1 - serve1

    p1_point_win = (serve1 + return1) / 2
    p2_point_win = (serve2 + return2) / 2

    surface_factor = {'Clay': 1.08, 'Hard': 1.0, 'Grass': 0.93, 'Indoor': 1.02}.get(superficie, 1.0)

    diff = (p1_point_win - p2_point_win) * 100
    prob_p1 = 1 / (1 + 10 ** (-diff / 38))

    hold1 = serve1 ** 1.85
    hold2 = serve2 ** 1.85
    break_prob = (1 - hold1 + 1 - hold2) / 2
    games_per_set = 9.6 + 4.2 * break_prob
    total_esperado = round(games_per_set * 2.15 * surface_factor, 2)

    prob_over = max(0.38, min(0.78, 0.5 + (total_esperado - 21.5) * 0.085))

    return {
        "Prob_J1_%": round(prob_p1 * 100, 1),
        "Total_Esperado": total_esperado,
        "Prob_Over_21.5_%": round(prob_over, 1),
        "Prob_Under_21.5_%": round(100 - prob_over, 1),
        "Serve_J1_%": round(serve1 * 100, 1),
        "BP_Saved_J1_%": round(safe(p1_stats.get('w_bpSaved',0)) / max(safe(p1_stats.get('w_bpFaced',1)), 1) * 100, 1),
    }

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'kigali', 'santiago', 'punto cana', 'heilbronn', 'perugia']):
        return 'Clay'
    if any(k in t for k in ['grass', 'birmingham']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'cherbourg']):
        return 'Indoor'
    return 'Hard'

# ====================== SCRAPING LEVE (BeautifulSoup + requests) ======================
def get_flashscore_matches():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        }
        response = requests.get("https://www.flashscore.pt/tenis/", headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        matches = []

        # Encontrar os jogos agendados
        events = soup.find_all("div", class_="event__match")
        for event in events[:70]:
            try:
                tournament = event.find("div", class_="event__tournament")
                tournament_name = tournament.get_text(strip=True) if tournament else "Desconhecido"

                j1 = event.find("div", class_="event__participant--home")
                jogador1 = j1.get_text(strip=True) if j1 else "?"

                j2 = event.find("div", class_="event__participant--away")
                jogador2 = j2.get_text(strip=True) if j2 else "?"

                time_el = event.find("div", class_="event__time")
                horario = time_el.get_text(strip=True) if time_el else "?"

                if horario not in ["AO VIVO", "Terminado", "Cancelado", ""]:
                    superficie = detect_surface(tournament_name)
                    matches.append({
                        'torneio': tournament_name,
                        'jogador_1': jogador1,
                        'jogador_2': jogador2,
                        'horario': horario,
                        'superficie': superficie
                    })
            except:
                continue

        return pd.DataFrame(matches)
    except Exception as e:
        st.error(f"Erro no scraping: {e}")
        return pd.DataFrame()

# ====================== ABA 1 - PARTIDAS HOJE ======================
with tab1:
    st.header("Partidas de Hoje + Previsão Automática")

    if st.button("🔄 Buscar Partidas do Flashscore + Calcular", type="primary"):
        if df_stats.empty:
            st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx")
        else:
            with st.spinner("Buscando partidas..."):
                df_flash = get_flashscore_matches()

                if df_flash.empty:
                    st.warning("Não foi possível obter partidas. Tenta novamente.")
                else:
                    results = []
                    for _, row in df_flash.iterrows():
                        p1 = find_best_player_stats(row['jogador_1'], df_stats)
                        p2 = find_best_player_stats(row['jogador_2'], df_stats)

                        if not p1.empty and not p2.empty:
                            pred = predict_from_stats(p1, p2, row['superficie'])
                            results.append([pred["Prob_J1_%"], pred["Total_Esperado"], 
                                          pred["Prob_Over_21.5_%"], pred["Serve_J1_%"], pred["BP_Saved_J1_%"]])
                        else:
                            results.append([None, None, None, None, None])

                    df_flash[['Prob_J1_%', 'Total_Esperado', 'Prob_Over_21.5_%', 'Serve_J1_%', 'BP_Saved_J1_%']] = pd.DataFrame(results)

                    st.success(f"✅ {len(df_flash)} partidas analisadas")
                    st.dataframe(df_flash, use_container_width=True, hide_index=True)

                    csv = df_flash.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Exportar CSV", csv, f"previsoes_hoje_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

# ====================== ABA 2 - PREVISÃO PERSONALIZADA ======================
with tab2:
    st.header("🔍 Previsão Personalizada")

    if df_stats.empty:
        st.info("Carregue o ficheiro Challenger1.xlsx")
    else:
        player_list = pd.concat([df_stats['winner_name'], df_stats['loser_name']]).drop_duplicates().sort_values().tolist()

        col1, col2 = st.columns(2)
        with col1:
            jogador_a = st.selectbox("Jogador A", options=player_list, key="ja")
        with col2:
            jogador_b = st.selectbox("Jogador B", options=player_list, key="jb")

        superficie = st.selectbox("Superfície", ["Hard", "Clay", "Grass", "Indoor"])

        if st.button("Calcular Previsão", type="primary"):
            if jogador_a == jogador_b:
                st.error("Escolha jogadores diferentes!")
            else:
                p1 = find_best_player_stats(jogador_a, df_stats)
                p2 = find_best_player_stats(jogador_b, df_stats)

                if p1.empty or p2.empty:
                    st.error("Não foi possível encontrar stats suficientes.")
                else:
                    result = predict_from_stats(p1, p2, superficie)
                    st.success("Previsão Calculada!")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(f"{jogador_a} vence", f"{result['Prob_J1_%']}%")
                    with c2:
                        st.metric(f"{jogador_b} vence", f"{result['Prob_Under_21.5_%'] if 'Prob_Under_21.5_%' in result else 100 - result['Prob_J1_%']}%")
                    st.metric("Total Esperado", f"{result['Total_Esperado']} jogos")
                    st.metric("Over 21.5", f"{result['Prob_Over_21.5_%']}%")

# ====================== ABA 3 - MODELING STRATEGY ======================
with tab3:
    st.header("📈 Recommended Modeling Strategy")
    st.markdown("""
    ### Estratégia Recomendada

    1. **Feature Engineering**
       - Rank Difference, Points Difference
       - Average Total Games (últimos jogos)
       - Serve % e Return % por superfície

    2. **Modelo Híbrido**
       - Vitória → XGBoost / Logistic Regression
       - Total Jogos → Markov Chain Simulation

    3. **Validação**
       - 10-fold Cross-Validation

    **Melhor prática atual:**
    Machine Learning para vencedor + **Markov Chains** para Total de Jogos.
    """)

st.caption("Versão leve com BeautifulSoup + requests")
