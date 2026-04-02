import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
from io import BytesIO
import unicodedata
import subprocess
import os

# ====================== INSTALAÇÃO FORÇADA DO BROWSER ======================
def install_playwright_browser():
    try:
        st.info("🔧 Instalando Chromium do Playwright... (pode demorar 15-40 segundos na primeira vez)")
        
        # Comando mais completo e com dependências do sistema
        result = subprocess.run(
            ["playwright", "install", "chromium", "--with-deps"],
            capture_output=True,
            text=True,
            timeout=180,   # 3 minutos de timeout
            check=False
        )
        
        if result.returncode == 0:
            st.success("✅ Browser Chromium instalado com sucesso!")
            return True
        else:
            st.warning(f"Aviso durante instalação: {result.stderr[-300:]}")  # mostra só o final
            # Tenta uma segunda vez de forma mais simples
            subprocess.run(["playwright", "install", "chromium"], timeout=120, check=False)
            return True
    except subprocess.TimeoutExpired:
        st.error("Timeout durante instalação do browser. Tente novamente.")
        return False
    except Exception as e:
        st.error(f"Erro na instalação: {e}")
        return False

# Executa a instalação assim que a app carrega
if 'playwright_installed' not in st.session_state:
    install_playwright_browser()
    st.session_state.playwright_installed = True

# ====================== CONFIGURAÇÃO ======================
st.set_page_config(page_title="Tênis Hoje - WELO", page_icon="🎾", layout="wide")

st.title("🎾 Partidas de Tênis Hoje + WELO por Superfície")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

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
            if not isinstance(name, str):
                return ""
            name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            name = name.lower().strip()
            name = ''.join(filter(str.isalnum, name))
            return name
        
        df['Jogador_clean'] = df['Jogador'].apply(normalize_name)
        st.sidebar.success(f"✅ {len(df)} jogadores carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar ficheiro: {e}")
        return pd.DataFrame()

df_welo = pd.DataFrame()
if uploaded_file:
    df_welo = load_welo_data(uploaded_file)

# ====================== FUNÇÃO WELO ======================
def get_welo(jogador_nome: str, superficie: str, df_welo) -> float:
    if df_welo.empty or not jogador_nome:
        return 1484.0
    
    clean_name = unicodedata.normalize('NFKD', str(jogador_nome)).encode('ascii', 'ignore').decode('utf-8')
    clean_name = ''.join(filter(str.isalnum, clean_name.lower().strip()))
    
    if not clean_name:
        return 1484.0
    
    mask = (
        df_welo['Jogador_clean'].str.contains(clean_name, na=False) |
        df_welo['Jogador_clean'].apply(lambda x: clean_name in str(x) if pd.notna(x) else False)
    )
    
    match = df_welo[mask]
    if match.empty:
        return 1484.0
    
    surface_map = {'clay': 'ELO Clay', 'hard': 'ELO Hard', 'grass': 'ELO Grass', 'indoor': 'ELO Indoor'}
    col = surface_map.get(superficie.lower())
    
    if col and col in match.columns:
        val = match[col].iloc[0]
        if pd.notna(val) and str(val).strip() != '':
            return round(float(val), 1)
    
    elo_cols = ['ELO Hard', 'ELO Clay', 'ELO Grass', 'ELO Indoor']
    values = [float(match[c].iloc[0]) for c in elo_cols if c in match.columns and pd.notna(match[c].iloc[0])]
    return round(sum(values) / len(values), 1) if values else 1484.0

# ====================== DETEÇÃO DE SUPERFÍCIE ======================
def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'kigali', 'santiago', 'punto cana', 'bucharest', 'houston', 'marrakech', 'rio', 'barcelona']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon', 'halle', 'queens', 'eastbourne']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'stockholm', 'basel']):
        return 'Indoor'
    return 'Hard'

# ====================== BUSCAR PARTIDAS ======================
async def get_flashscore_matches():
    matches = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--single-process']
        )
        page = await browser.new_page()
        
        try:
            await page.goto("https://www.flashscore.pt/tenis/", timeout=60000)
            await page.wait_for_timeout(9000)   # mais tempo para carregar
            
            try:
                tab = await page.query_selector("text=Agendados")
                if tab:
                    await tab.click()
                    await page.wait_for_timeout(6000)
            except:
                pass
            
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
                except:
                    continue
        finally:
            await browser.close()
    
    return pd.DataFrame(matches)

# ====================== BOTÃO ======================
if st.button("🔄 Buscar Partidas e Calcular WELO", type="primary"):
    if df_welo.empty:
        st.warning("⚠️ Carregue primeiro o ficheiro Challenger.xlsm na barra lateral.")
    else:
        with st.spinner("Acedendo ao FlashScore e calculando WELO..."):
            try:
                df = asyncio.run(get_flashscore_matches())
                
                if not df.empty:
                    df['WELO_J1'] = df.apply(lambda row: get_welo(row['jogador_1'], row['superficie'], df_welo), axis=1)
                    df['WELO_J2'] = df.apply(lambda row: get_welo(row['jogador_2'], row['superficie'], df_welo), axis=1)
                    
                    st.success(f"✅ {len(df)} partidas encontradas!")
                    
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
                        }
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("📥 CSV", df.to_csv(index=False).encode('utf-8'), 
                                          f"tenis_hoje_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
                    with col2:
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                        output.seek(0)
                        st.download_button("📊 Excel", output, 
                                          f"tenis_hoje_welo_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.warning("Nenhuma partida agendada encontrada.")
            except Exception as e:
                st.error(f"Erro ao aceder ao FlashScore: {str(e)}")
                st.info("Tente clicar no botão novamente em 10-20 segundos.")
else:
    st.info("Carregue o ficheiro na sidebar e clique no botão.")

st.caption("Instalação automática do browser | Matching melhorado de nomes")
