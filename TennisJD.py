import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
from io import BytesIO

st.set_page_config(page_title="Tênis Hoje - WELO", page_icon="🎾", layout="wide")

st.title("🎾 Partidas de Tênis Hoje + WELO por Superfície")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Dados WELO")
    uploaded_file = st.file_uploader(
        "Carregue o ficheiro Challenger.xlsm",
        type=["xlsm", "xlsx"],
        help="Deve conter a aba 'Jogadores>20'"
    )

@st.cache_data
def load_welo_data(file):
    try:
        xls = pd.ExcelFile(file)
        df = pd.read_excel(xls, sheet_name="Jogadores>20")
        
        # Coluna para matching robusto
        df['Jogador_clean'] = (
            df['Jogador'].astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'[^a-z0-9áéíóúãõçâêîôûàèìòùäëïöü]', '', regex=True)
            .str.replace(r'\s+', '', regex=True)
        )
        
        st.sidebar.success(f"✅ {len(df)} jogadores carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar ficheiro: {e}")
        return pd.DataFrame()

df_welo = pd.DataFrame()
if uploaded_file:
    df_welo = load_welo_data(uploaded_file)

# ====================== FUNÇÃO WELO POR SUPERFÍCIE ======================
def get_welo(jogador_nome, superficie, df_welo):
    if df_welo.empty or not jogador_nome:
        return 1484.0
    
    # Limpeza leve para matching
    clean = ''.join(filter(str.isalnum, str(jogador_nome).lower().strip()))
    
    match = df_welo[df_welo['Jogador_clean'] == clean]
    if match.empty:
        return 1484.0
    
    # Mapeamento superfície → coluna
    surface_map = {
        'clay': 'ELO Clay',
        'hard': 'ELO Hard',
        'grass': 'ELO Grass',
        'indoor': 'ELO Indoor'
    }
    
    col = surface_map.get(superficie.lower())
    
    if col and col in match.columns:
        val = match[col].iloc[0]
        if pd.notna(val) and str(val).strip() != '':
            return float(val)
    
    # Fallback: média dos ELOs disponíveis
    elo_cols = ['ELO Hard', 'ELO Clay', 'ELO Grass', 'ELO Indoor']
    values = []
    for c in elo_cols:
        if c in match.columns:
            v = match[c].iloc[0]
            if pd.notna(v) and str(v).strip() != '':
                values.append(float(v))
    
    return round(sum(values) / len(values), 1) if values else 1484.0

# ====================== DETEÇÃO MELHORADA DE SUPERFÍCIE ======================
def detect_surface(tournament_name: str) -> str:
    name = str(tournament_name).lower()
    
    clay_keywords = ['clay', 'saibro', 'roland', 'paris', 'madrid', 'rome', 'monte carlo', 'barcelona', 
                     'bucharest', 'houston', 'marrakech', 'santiago', 'kigali', 'punto cana', 'rio']
    grass_keywords = ['grass', 'wimbledon', 'halle', 'queens', 'eastbourne', 'mallorca', 'stuttgart']
    indoor_keywords = ['indoor', 'stockholm', 'basel', 'vienna', 'paris masters']
    
    if any(k in name for k in clay_keywords):
        return 'Clay'
    elif any(k in name for k in grass_keywords):
        return 'Grass'
    elif any(k in name for k in indoor_keywords):
        return 'Indoor'
    else:
        return 'Hard'  # default mais comum

# ====================== BUSCAR PARTIDAS ======================
async def get_flashscore_matches():
    matches = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = await browser.new_page()
        
        try:
            await page.goto("https://www.flashscore.pt/tenis/", timeout=30000)
            await page.wait_for_timeout(7000)
            
            # Tentar aba "Agendados"
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
                    tour = await el.query_selector(".event__tournament")
                    tournament = (await tour.inner_text()).strip() if tour else "Desconhecido"
                    
                    p1 = await el.query_selector(".event__participant--home")
                    jogador1 = (await p1.inner_text()).strip() if p1 else "?"
                    
                    p2 = await el.query_selector(".event__participant--away")
                    jogador2 = (await p2.inner_text()).strip() if p2 else "?"
                    
                    time_el = await el.query_selector(".event__time")
                    horario = (await time_el.inner_text()).strip() if time_el else "?"
                    
                    if horario not in ["AO VIVO", "Terminado", "Cancelado", ""]:
                        superficie = detect_surface(tournament)
                        
                        matches.append({
                            'torneio': tournament,
                            'jogador_1': jogador1,
                            'jogador_2': jogador2,
                            'horario': horario,
                            'superficie': superficie,
                            'status': 'Agendado'
                        })
                except:
                    continue
        except Exception as e:
            st.error(f"Erro FlashScore: {e}")
        finally:
            await browser.close()
    
    return pd.DataFrame(matches)

# ====================== BOTÃO PRINCIPAL ======================
if st.button("🔄 Buscar Partidas e Calcular WELO", type="primary"):
    if df_welo.empty and uploaded_file is None:
        st.warning("⚠️ Carregue primeiro o ficheiro Challenger.xlsm na barra lateral.")
    else:
        with st.spinner("A buscar partidas no FlashScore e calcular WELO..."):
            df = asyncio.run(get_flashscore_matches())
            
            if not df.empty:
                # Calcular WELO usando a superfície detetada
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
                    st.download_button("📥 CSV", df.to_csv(index=False).encode('utf-8'), 
                                     f"tenis_hoje_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
                
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name="Partidas")
                    output.seek(0)
                    st.download_button("📊 Excel", output, 
                                     f"tenis_hoje_welo_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("Nenhuma partida agendada encontrada neste momento.")

else:
    st.info("Carregue o ficheiro na sidebar e clique no botão para atualizar.")

st.caption("FlashScore • WELO por superfície • Nomes mantidos originais")
