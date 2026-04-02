import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
from io import BytesIO
import unicodedata

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
            # Remove acentos e caracteres especiais
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

# ====================== FUNÇÃO WELO CORRIGIDA ======================
def get_welo(jogador_nome: str, superficie: str, df_welo) -> float:
    if df_welo.empty or not jogador_nome:
        return 1484.0
    
    # Normaliza o nome do FlashScore
    clean_name = unicodedata.normalize('NFKD', str(jogador_nome)).encode('ascii', 'ignore').decode('utf-8')
    clean_name = ''.join(filter(str.isalnum, clean_name.lower().strip()))
    
    if not clean_name:
        return 1484.0
    
    # Matching melhorado (bidirecional)
    mask = (
        df_welo['Jogador_clean'].str.contains(clean_name, na=False) |
        clean_name.str.contains(df_welo['Jogador_clean'], na=False)  # corrigido aqui
    )
    
    match = df_welo[mask]
    
    if match.empty:
        return 1484.0
    
    # Prioridade: ELO específico da superfície
    surface_map = {'clay': 'ELO Clay', 'hard': 'ELO Hard', 'grass': 'ELO Grass', 'indoor': 'ELO Indoor'}
    col = surface_map.get(superficie.lower())
    
    if col and col in match.columns:
        val = match[col].iloc[0]
        if pd.notna(val) and str(val).strip() != '':
            return round(float(val), 1)
    
    # Fallback: média dos ELOs
    elo_cols = ['ELO Hard', 'ELO Clay', 'ELO Grass', 'ELO Indoor']
    values = []
    for c in elo_cols:
        if c in match.columns:
            v = match[c].iloc[0]
            if pd.notna(v) and str(v).strip() != '':
                values.append(float(v))
    
    return round(sum(values) / len(values), 1) if values else 1484.0

# ====================== DETEÇÃO DE SUPERFÍCIE ======================
def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    clay_kw = ['clay', 'saibro', 'kigali', 'santiago', 'punto cana', 'bucharest', 'houston', 'marrakech', 'rio', 'barcelona']
    grass_kw = ['grass', 'wimbledon', 'halle', 'queens', 'eastbourne']
    indoor_kw = ['indoor', 'stockholm', 'basel', 'vienna']
    
    if any(k in t for k in clay_kw):
        return 'Clay'
    elif any(k in t for k in grass_kw):
        return 'Grass'
    elif any(k in t for k in indoor_kw):
        return 'Indoor'
    return 'Hard'

# ====================== BUSCAR PARTIDAS ======================
async def get_flashscore_matches():
    matches = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = await browser.new_page()
        try:
            await page.goto("https://www.flashscore.pt/tenis/", timeout=30000)
            await page.wait_for_timeout(7000)
            
            try:
                tab = await page.query_selector("text=Agendados")
                if tab:
                    await tab.click()
                    await page.wait_for_timeout(4000)
            except:
                pass
            
            elements = await page.query_selector_all(".event__match")
            for el in elements[:60]:
                try:
                    tour_el = await el.query_selector(".event__tournament")
                    tournament = (await tour_el.inner_text()).strip() if tour_el else "Desconhecido"
                    
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

# ====================== EXECUÇÃO ======================
if st.button("🔄 Buscar Partidas e Calcular WELO", type="primary"):
    if df_welo.empty:
        st.warning("⚠️ Por favor, carregue primeiro o ficheiro Challenger.xlsm na barra lateral.")
    else:
        with st.spinner("A buscar partidas no FlashScore e calcular WELO..."):
            df = asyncio.run(get_flashscore_matches())
            
            if not df.empty:
                # Aplicar WELO
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
                
                # Downloads
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Baixar CSV",
                        df.to_csv(index=False).encode('utf-8'),
                        f"tenis_hoje_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv"
                    )
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name="Partidas")
                    output.seek(0)
                    st.download_button(
                        "📊 Baixar Excel",
                        output,
                        f"tenis_hoje_welo_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("Nenhuma partida agendada encontrada no momento.")
else:
    st.info("👈 Carregue o ficheiro Challenger.xlsm na barra lateral e clique no botão.")

st.caption("Matching de nomes melhorado | WELO por superfície")
