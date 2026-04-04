import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
from io import BytesIO
import unicodedata
import subprocess
from difflib import SequenceMatcher

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Partidas Hoje + Predictor Stats")

tab1, tab2, tab3 = st.tabs(["📅 Partidas Hoje", "🔍 Previsão Personalizada", "📈 Recommended Modeling Strategy"])

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
        st.sidebar.error(f"Erro ao carregar: {e}")
        return pd.DataFrame()

df_stats = load_stats(uploaded_file)

# ====================== FUNÇÕES AUXILIARES ======================
def norm(name):
    if not isinstance(name, str): return ""
    n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return ''.join(filter(str.isalnum, n.lower().strip()))

def find_best_player_stats(player_name, df):
    if df.empty or not player_name: return pd.Series(dtype='object')
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

# ====================== SCRAPING FLASHSCORE ======================
async def get_flashscore_matches():
    matches = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
        )
        page = await browser.new_page()
        try:
            await page.goto("https://www.flashscore.pt/tenis/", timeout=90000)
            await page.wait_for_timeout(15000)

            try:
                tab = await page.query_selector("text=Agendados")
                if tab:
                    await tab.click()
                    await page.wait_for_timeout(10000)
            except: pass

            elements = await page.query_selector_all(".event__match")
            for el in elements[:70]:
                try:
                    tour = await el.query_selector(".event__tournament")
                    tournament = (await tour.inner_text()).strip() if tour else "Desconhecido"
                    p1 = await el.query_selector(".event__participant--home")
                    j1 = (await p1.inner_text()).strip() if p1 else "?"
                    p2 = await el.query_selector(".event__participant--away")
                    j2 = (await p2.inner_text()).strip() if p2 else "?"
                    time_el = await el.query_selector(".event__time")
                    horario = (await time_el.inner_text()).strip() if time_el else "?"

                    if horario not in ["AO VIVO", "Terminado", "Cancelado", ""]:
                        superficie = detect_surface(tournament)
                        matches.append({
                            'torneio': tournament,
                            'jogador_1': j1,
                            'jogador_2': j2,
                            'horario': horario,
                            'superficie': superficie
                        })
                except: continue
        finally:
            await browser.close()
    return pd.DataFrame(matches)

# ====================== ABA 1 - PARTIDAS HOJE ======================
with tab1:
    st.header("Partidas de Hoje + Previsão Automática")

    if st.button("🔄 Buscar Partidas do Flashscore + Calcular Previsões", type="primary"):
        if df_stats.empty:
            st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx")
        else:
            with st.spinner("Buscando partidas e calculando..."):
                df_flash = asyncio.run(get_flashscore_matches())

                if df_flash.empty:
                    st.warning("Nenhuma partida encontrada.")
                else:
                    results = []
                    for _, row in df_flash.iterrows():
                        p1 = find_best_player_stats(row['jogador_1'], df_stats)
                        p2 = find_best_player_stats(row['jogador_2'], df_stats)

                        if not p1.empty and not p2.empty:
                            pred = predict_from_stats(p1, p2, row['superficie'])
                            results.append([
                                pred["Prob_J1_%"], 
                                pred["Total_Esperado"], 
                                pred["Prob_Over_21.5_%"],
                                pred["Serve_J1_%"],
                                pred["BP_Saved_J1_%"]
                            ])
                        else:
                            results.append([None, None, None, None, None])

                    df_flash[['Prob_J1_%', 'Total_Esperado', 'Prob_Over_21.5_%', 'Serve_J1_%', 'BP_Saved_J1_%']] = pd.DataFrame(results)

                    st.success(f"✅ {len(df_flash)} partidas analisadas")
                    st.dataframe(df_flash, use_container_width=True, hide_index=True)

                    # Download
                    csv = df_flash.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Exportar CSV", csv, f"previsoes_hoje_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

# ====================== ABA 2 - PREVISÃO PERSONALIZADA ======================
with tab2:
    st.header("🔍 Previsão Personalizada")

    if not player_list := pd.concat([df_raw['winner_name'], df_raw['loser_name']]).drop_duplicates().sort_values().tolist() if not df_raw.empty else []:
        st.info("Carregue o ficheiro para ativar esta aba")
    else:
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
                result = predict_match(jogador_a, jogador_b, superficie)  # função simplificada
                if result:
                    st.success("Previsão Calculada!")
                    c1, c2 = st.columns(2)
                    with c1: st.metric(f"{jogador_a} vence", f"{result['Prob_J1_%']}%")
                    with c2: st.metric(f"{jogador_b} vence", f"{result['Prob_B_Vitória_%'] if 'Prob_B_Vitória_%' in result else 100-result['Prob_J1_%']}%")
                    st.metric("Total Esperado", f"{result['Total_Esperado']} jogos")
                    st.metric("Over 21.5", f"{result['Prob_Over_21.5_%']}%")

# ====================== ABA 3 - MODELING STRATEGY ======================
with tab3:
    st.header("📈 Recommended Modeling Strategy")
    st.markdown("""
    ### Estratégia Recomendada para Modelar Tênis

    **Para melhores resultados em Vitória + Total de Jogos (Over/Under 21.5):**

    1. **Feature Engineering**
       - Rank Difference, Points Difference
       - Average Total Games (últimos 5-10 jogos)
       - Serve % e Return % por superfície
       - Break Points Saved / Faced

    2. **Hybrid Modeling (Melhor Abordagem)**
       - **Win Probability**: XGBoost ou Logistic Regression
       - **Total Games**: Markov Chain Simulation ou Poisson Distribution
       - **Combinação Final**: Ensemble dos dois modelos

    3. **Validação**
       - 10-fold Cross-Validation (time-based)
       - Testar por superfície e nível de torneio

    **Conclusão da comunidade:**
    > Machine Learning é excelente para prever o vencedor, mas **Markov Chains** e simulações são atualmente as mais precisas para o mercado de Total de Jogos.
    """)

st.caption("Web scraping Flashscore + Predictor baseado em stats reais")
