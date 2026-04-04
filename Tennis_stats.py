import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
from io import BytesIO
import unicodedata
import subprocess
from difflib import SequenceMatcher

st.set_page_config(page_title="Tênis Hoje - Predictor Stats", page_icon="🎾", layout="wide")
st.title("🎾 Partidas de Tênis Hoje + Predictor por Stats Reais")

tab1, tab2 = st.tabs(["📅 Partidas Hoje", "🔮 Teste Manual"])

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
            # Normalização mais agressiva
            name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            name = ''.join(filter(str.isalnum, name.lower().strip()))
            return name
        
        df['winner_clean'] = df['winner_name'].apply(norm)
        df['loser_clean'] = df['loser_name'].apply(norm)
        
        st.sidebar.success(f"✅ {len(df)} jogos com stats carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar ficheiro: {e}")
        return pd.DataFrame()

df_stats = load_stats(uploaded_file)

# ====================== FUNÇÃO DE MATCHING MELHORADA ======================
def find_best_player_stats(player_name: str, df_stats):
    if df_stats.empty or not player_name:
        return pd.Series(dtype='object')
    
    clean_name = unicodedata.normalize('NFKD', player_name).encode('ascii', 'ignore').decode('utf-8')
    clean_name = ''.join(filter(str.isalnum, clean_name.lower().strip()))
    
    if len(clean_name) < 4:
        return pd.Series(dtype='object')

    best_match = None
    best_score = 0.0

    for _, row in df_stats.iterrows():
        # Testa tanto winner quanto loser
        for col in ['winner_clean', 'loser_clean']:
            clean_db = row.get(col, "")
            if not clean_db:
                continue
            
            # Fuzzy matching com SequenceMatcher
            similarity = SequenceMatcher(None, clean_name, clean_db).ratio()
            
            # Bonus se for match exato no início ou fim
            if clean_name in clean_db or clean_db in clean_name:
                similarity = max(similarity, 0.95)
            
            score = similarity * 100
            
            if score > best_score:
                best_score = score
                best_match = row

    # Limiar mais inteligente
    if best_score < 62 or best_match is None:   # aumentado de 55 para 62
        return pd.Series(dtype='object')
    
    return best_match

# ====================== PREDICTOR ======================
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
    prob_p1_win = 1 / (1 + 10 ** (-diff / 38))

    hold1 = serve1 ** 1.85
    hold2 = serve2 ** 1.85
    break_prob = (1 - hold1 + 1 - hold2) / 2
    games_per_set = 9.6 + 4.2 * break_prob
    total_esperado = round(games_per_set * 2.15 * surface_factor, 2)

    prob_over = max(0.38, min(0.78, 0.5 + (total_esperado - 21.5) * 0.085))

    return {
        "prob_p1_win": round(prob_p1_win * 100, 1),
        "prob_p2_win": round(100 - prob_p1_win * 100, 1),
        "total_esperado": total_esperado,
        "prob_over_21_5": round(prob_over * 100, 1),
        "serve1_pct": round(serve1 * 100, 1),
        "return1_pct": round(return1 * 100, 1),
        "bp_saved1": round(safe(p1_stats.get('w_bpSaved',0)) / max(safe(p1_stats.get('w_bpFaced',1)), 1) * 100, 1),
    }

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
            await page.wait_for_timeout(12000)
            
            try:
                tab = await page.query_selector("text=Agendados")
                if tab:
                    await tab.click()
                    await page.wait_for_timeout(8000)
            except: pass

            elements = await page.query_selector_all(".event__match")
            for el in elements[:80]:
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

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'kigali', 'santiago', 'punto cana', 'heilbronn', 'perugia', 'prostejov']):
        return 'Clay'
    if any(k in t for k in ['grass', 'birmingham']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'cherbourg']):
        return 'Indoor'
    return 'Hard'

# ====================== ABA 1 - PARTIDAS HOJE ======================
with tab1:
    st.header("Partidas de Hoje + Predictor Automático")

    if st.button("🔄 Buscar Partidas + Calcular Predictor", type="primary"):
        if df_stats.empty:
            st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx na barra lateral.")
        else:
            with st.spinner("Buscando partidas e calculando stats..."):
                df_flash = asyncio.run(get_flashscore_matches())

                if df_flash.empty:
                    st.warning("Nenhuma partida encontrada.")
                else:
                    results = []
                    for _, row in df_flash.iterrows():
                        p1_stats = find_best_player_stats(row['jogador_1'], df_stats)
                        p2_stats = find_best_player_stats(row['jogador_2'], df_stats)

                        if not p1_stats.empty and not p2_stats.empty:
                            pred = predict_from_stats(p1_stats, p2_stats, row['superficie'])
                            results.append([
                                pred['prob_p1_win'], 
                                pred['total_esperado'], 
                                pred['prob_over_21_5'],
                                pred['serve1_pct'],
                                pred['bp_saved1']
                            ])
                        else:
                            results.append([None, None, None, None, None])

                    df_flash[['Prob_J1_%', 'Total_Esperado', 'Prob_Over_21.5_%', 'Serve_J1_%', 'BP_Saved_J1_%']] = pd.DataFrame(results)

                    st.success(f"✅ {len(df_flash)} partidas analisadas")
                    
                    st.dataframe(
                        df_flash,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "torneio": "🏆 Torneio",
                            "jogador_1": "🎾 Jogador 1",
                            "jogador_2": "🎾 Jogador 2",
                            "horario": "⏰ Horário",
                            "superficie": "🏟️ Superfície",
                            "Prob_J1_%": st.column_config.NumberColumn("Prob J1 (%)", format="%.1f"),
                            "Total_Esperado": st.column_config.NumberColumn("Total Esperado", format="%.2f"),
                            "Prob_Over_21.5_%": st.column_config.NumberColumn("Over 21.5 (%)", format="%.1f"),
                            "Serve_J1_%": st.column_config.NumberColumn("Serve J1 (%)", format="%.1f"),
                            "BP_Saved_J1_%": st.column_config.NumberColumn("BP Saved J1 (%)", format="%.1f"),
                        }
                    )

                    # Download
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("📥 CSV", df_flash.to_csv(index=False).encode('utf-8'),
                                          f"tenis_predictor_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

# ====================== ABA 2 - TESTE MANUAL ======================
with tab2:
    st.header("🔮 Teste Manual do Predictor")
    nome1 = st.text_input("Nome Jogador 1", "Mitchell Krueger")
    nome2 = st.text_input("Nome Jogador 2", "Tung-Lin Wu")
    superficie = st.selectbox("Superfície", ["Hard", "Clay", "Grass", "Indoor"], index=0)

    if st.button("Calcular Previsão"):
        if df_stats.empty:
            st.error("Carregue o ficheiro primeiro!")
        else:
            p1_stats = find_best_player_stats(nome1, df_stats)
            p2_stats = find_best_player_stats(nome2, df_stats)

            if p1_stats.empty or p2_stats.empty:
                st.error("Um dos jogadores não foi encontrado com boa correspondência.")
            else:
                pred = predict_from_stats(p1_stats, p2_stats, superficie)
                st.success("Previsão Calculada!")
                st.metric(f"{nome1} vence", f"{pred['prob_p1_win']}%")
                st.metric("Total Esperado", f"{pred['total_esperado']} jogos")
                st.metric("Over 21.5", f"{pred['prob_over_21_5']}%")
                st.write(f"Serve {nome1}: **{pred['serve1_pct']}%** | BP Saved: **{pred['bp_saved1']}%**")

st.caption("Matching de nomes melhorado com fuzzy + busca winner/loser • Superfície considerada")
