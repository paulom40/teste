import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
from io import BytesIO
import unicodedata
import subprocess
from difflib import SequenceMatcher  # Para fuzzy matching

# ====================== CONFIGURAÇÃO ======================
st.set_page_config(page_title="Tênis Hoje - WELO + Total + AI", page_icon="🎾", layout="wide")
st.title("🎾 Partidas de Tênis Hoje + WELO Melhorado + TennisPredictions.ai")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

# ====================== INSTALAÇÃO PLAYWRIGHT ======================
def install_playwright_browser():
    try:
        result = subprocess.run(
            ["playwright", "install", "chromium"],
            timeout=240,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            st.sidebar.success("✅ Chromium instalado com sucesso")
            return True
        else:
            st.sidebar.warning("⚠️ Playwright instalou com avisos")
            return False
    except Exception as e:
        st.sidebar.error(f"Erro na instalação: {e}")
        return False

if 'browser_installed' not in st.session_state:
    install_playwright_browser()
    st.session_state.browser_installed = True

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger.xlsm")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger.xlsm", type=["xlsm", "xlsx"])

@st.cache_data
def load_welo_data(file):
    try:
        xls = pd.ExcelFile(file)
        df = pd.read_excel(xls, sheet_name="Jogadores>20")
        
        def normalize_name(name):
            if not isinstance(name, str): return ""
            name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            name = name.lower().strip()
            name = ''.join(filter(str.isalnum, name))
            return name
        
        df['Jogador_clean'] = df['Jogador'].apply(normalize_name)
        st.sidebar.success(f"✅ {len(df)} jogadores carregados do Excel")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar ficheiro: {e}")
        return pd.DataFrame()

df_welo = pd.DataFrame()
if uploaded_file:
    df_welo = load_welo_data(uploaded_file)

# ====================== FUNÇÃO WELO MELHORADA ======================
def get_welo(jogador_nome: str, superficie: str, df_welo) -> tuple[float, str]:
    """Retorna (welo_final, fonte) com blend 60% superfície + 40% geral"""
    if df_welo.empty or not jogador_nome:
        return 1484.0, "Default"

    clean_flash = unicodedata.normalize('NFKD', str(jogador_nome)).encode('ascii', 'ignore').decode('utf-8')
    clean_flash = ''.join(filter(str.isalnum, clean_flash.lower().strip()))

    if len(clean_flash) < 4:
        return 1484.0, "Default"

    best_match = None
    best_score = 0.0

    for _, row in df_welo.iterrows():
        clean_excel = row.get('Jogador_clean', "")
        if not clean_excel:
            continue
        
        # Fuzzy matching melhorado
        similarity = SequenceMatcher(None, clean_flash, clean_excel).ratio()
        score = similarity * 100
        
        if score > best_score:
            best_score = score
            best_match = row

    if best_score < 55 or best_match is None:   # limiar mais flexível
        return 1484.0, "Default"

    # ELO da superfície específica
    surface_map = {'clay': 'ELO Clay', 'hard': 'ELO Hard', 'grass': 'ELO Grass', 'indoor': 'ELO Indoor'}
    col_surface = surface_map.get(superficie.lower())

    elo_surface = None
    if col_surface and col_surface in best_match.index:
        val = best_match[col_surface]
        if pd.notna(val) and str(val).strip():
            elo_surface = float(val)

    # ELO geral (média)
    elo_cols = ['ELO Hard', 'ELO Clay', 'ELO Grass', 'ELO Indoor']
    values = [float(best_match[c]) for c in elo_cols if c in best_match.index and pd.notna(best_match[c])]
    elo_geral = round(sum(values) / len(values), 1) if values else 1484.0

    if elo_surface is not None:
        welo_final = round(0.60 * elo_surface + 0.40 * elo_geral, 1)
        fonte = f"{superficie.capitalize()} (Blend)"
    else:
        welo_final = elo_geral
        fonte = "Blended (Geral)"

    return welo_final, fonte

# ====================== OUTRAS FUNÇÕES ======================
def calcular_linha_total(welo1: float, welo2: float, superficie: str) -> tuple:
    dif = abs(welo1 - welo2)
    base_jogos = {'Clay': 22.8, 'Hard': 22.4, 'Grass': 21.9, 'Indoor': 22.6}.get(superficie, 22.5)
    ajuste_dif = -0.035 * dif
    total_esperado = max(18.5, min(27.0, base_jogos + ajuste_dif))
    prob_mais_21_5 = max(0.35, min(0.78, 0.5 + (total_esperado - 22.0) * 0.08))
    return round(total_esperado, 2), round(prob_mais_21_5 * 100, 1)

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'kigali', 'santiago', 'punto cana', 'bucharest', 'houston', 'marrakech', 'rio', 'barcelona']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon', 'halle', 'queens', 'eastbourne']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'stockholm', 'basel']):
        return 'Indoor'
    return 'Hard'

def normalize_match_name(j1, j2):
    def clean(name):
        name = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('utf-8')
        return ''.join(filter(str.isalnum, name.lower().strip()))
    return f"{clean(j1)} vs {clean(j2)}"

# ====================== SCRAPING ======================
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

async def get_tennispredictions_data():
    predictions = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
        )
        page = await browser.new_page()
        try:
            await page.goto("https://tennispredictions.ai/", timeout=90000)
            await page.wait_for_timeout(10000)

            rows = await page.query_selector_all("table tr")
            for row in rows[1:]:
                try:
                    cells = await row.query_selector_all("td")
                    if len(cells) < 3: 
                        continue

                    matchup = (await cells[1].inner_text()).strip() if len(cells) > 1 else ""
                    prediction_text = (await cells[2].inner_text()).strip() if len(cells) > 2 else ""

                    if " " in prediction_text:
                        winner_str, prob_str = prediction_text.split(maxsplit=1)
                        pred_vencedor = winner_str.strip()
                        prob_ai = float(prob_str.replace("%", "").strip())
                    else:
                        pred_vencedor = prediction_text
                        prob_ai = None

                    predictions.append({
                        'matchup': matchup,
                        'pred_vencedor': pred_vencedor,
                        'prob_ai': prob_ai
                    })
                except:
                    continue
        finally:
            await browser.close()
    return pd.DataFrame(predictions)

# ====================== PROCESSAMENTO PRINCIPAL ======================
def process_matches(df_flash, df_pred, df_welo):
    if df_flash.empty:
        return pd.DataFrame()

    df = df_flash.copy()

    # Merge com previsões AI
    if not df_pred.empty:
        df['match_key'] = df.apply(lambda row: normalize_match_name(row['jogador_1'], row['jogador_2']), axis=1)
        df_pred['match_key'] = df_pred['matchup'].apply(
            lambda x: normalize_match_name(*[p.strip() for p in str(x).split(" VS ")]) if " VS " in str(x) else str(x).lower()
        )
        df = pd.merge(df, df_pred[['match_key', 'pred_vencedor', 'prob_ai']], on='match_key', how='left')
        df.rename(columns={'pred_vencedor': 'Pred_AI', 'prob_ai': 'Prob_AI_%'}, inplace=True)
        df.drop(columns=['match_key'], inplace=True, errors='ignore')

    # Calcula WELO melhorado
    df[['WELO_J1', 'Fonte_WELO_J1']] = df.apply(
        lambda row: pd.Series(get_welo(row['jogador_1'], row['superficie'], df_welo)), axis=1
    )
    df[['WELO_J2', 'Fonte_WELO_J2']] = df.apply(
        lambda row: pd.Series(get_welo(row['jogador_2'], row['superficie'], df_welo)), axis=1
    )
    df['Dif_WELO'] = abs(df['WELO_J1'] - df['WELO_J2'])

    # Linha Total
    resultados = df.apply(lambda row: calcular_linha_total(row['WELO_J1'], row['WELO_J2'], row['superficie']), axis=1)
    df['Total_Esperado'] = [r[0] for r in resultados]
    df['Prob_Mais_21.5'] = [r[1] for r in resultados]

    # Probabilidade implícita WELO
    df['Prob_WELO_%'] = df.apply(
        lambda row: round(100 / (1 + 10**((row['WELO_J2'] - row['WELO_J1']) / 400)), 1) 
        if pd.notna(row.get('WELO_J1')) and pd.notna(row.get('WELO_J2')) else None, axis=1
    )

    # Probabilidade Combinada (melhor modelo)
    df['Prob_Combined_%'] = df.apply(
        lambda row: round(0.62 * row.get('Prob_WELO_%', 50) + 0.38 * row.get('Prob_AI_%', 50), 1)
        if pd.notna(row.get('Prob_AI_%')) else row.get('Prob_WELO_%'), axis=1
    )

    # Diferença e Value
    df['Dif_Prob'] = df.apply(
        lambda row: round(row.get('Prob_AI_%', 0) - row.get('Prob_WELO_%', 0), 1) 
        if pd.notna(row.get('Prob_AI_%')) else None, axis=1
    )
    
    df['Value'] = df.apply(
        lambda row: "🔥 HIGH VALUE" if abs(row.get('Dif_Prob', 0)) >= 12 else 
                   "✅ Bom Value" if abs(row.get('Dif_Prob', 0)) >= 7 else "", axis=1
    )

    return df

# ====================== EXECUÇÃO ======================
if st.button("🔄 Buscar Partidas + WELO Melhorado + AI", type="primary"):
    if df_welo.empty:
        st.warning("⚠️ Carregue primeiro o ficheiro Challenger.xlsm na barra lateral.")
    else:
        with st.spinner("Buscando Flashscore + TennisPredictions.ai + calculando modelo..."):
            try:
                df_flash = asyncio.run(get_flashscore_matches())
                df_pred = asyncio.run(get_tennispredictions_data())

                df = process_matches(df_flash, df_pred, df_welo)

                if df.empty:
                    st.warning("Nenhuma partida encontrada.")
                else:
                    st.success(f"✅ {len(df)} partidas analisadas | {df['Prob_AI_%'].notna().sum()} com previsão AI")

                    # Tabela com formatação condicional
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "torneio": "🏆 Torneio",
                            "jogador_1": "🎾 Jogador 1",
                            "jogador_2": "🎾 Jogador 2",
                            "horario": "⏰ Horário",
                            "superficie": "🏟️ Superfície",
                            "WELO_J1": st.column_config.NumberColumn("WELO J1", format="%.1f"),
                            "WELO_J2": st.column_config.NumberColumn("WELO J2", format="%.1f"),
                            "Dif_WELO": st.column_config.NumberColumn("Dif WELO", format="%.1f"),
                            "Fonte_WELO_J1": "Fonte J1",
                            "Fonte_WELO_J2": "Fonte J2",
                            "Total_Esperado": st.column_config.NumberColumn("Total Esperado", format="%.2f"),
                            "Prob_Mais_21.5": st.column_config.NumberColumn("Prob >21.5 (%)", format="%.1f"),
                            "Pred_AI": "🤖 Pred AI",
                            "Prob_AI_%": st.column_config.NumberColumn("Prob AI (%)", format="%.0f"),
                            "Prob_Combined_%": st.column_config.NumberColumn("Prob Combinada (%)", format="%.1f"),
                            "Dif_Prob": st.column_config.NumberColumn("Dif Prob (AI-WELO)", format="%.1f"),
                            "Value": "Value Bet",
                        }
                    )

                    # Downloads
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("📥 CSV", df.to_csv(index=False).encode('utf-8'),
                                          f"tenis_hoje_melhorado_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
                    with col2:
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                        output.seek(0)
                        st.download_button("📊 Excel", output,
                                          f"tenis_hoje_melhorado_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"Erro durante o processamento: {e}")

else:
    st.info("Carregue o ficheiro Challenger.xlsm na sidebar e clique no botão.")

st.caption("Modelo Melhorado: WELO Blend (60/40) • Probabilidade Combinada • Detecção de Value Bets")
