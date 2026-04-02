import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
from io import BytesIO
import unicodedata
import time

# Instalar selenium e webdriver se necessário
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    st.error("Selenium não está instalado. Por favor, instale com: pip install selenium webdriver-manager")
    st.stop()

# ====================== CONFIGURAÇÃO ======================
st.set_page_config(page_title="Tênis Hoje - WELO (Selenium)", page_icon="🎾", layout="wide")

st.title("🎾 Partidas de Tênis Hoje + WELO por Jogador/Superfície")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')} - Versão Selenium")

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
        st.sidebar.success(f"✅ {len(df)} jogadores carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar ficheiro: {e}")
        return pd.DataFrame()

df_welo = pd.DataFrame()
if uploaded_file:
    df_welo = load_welo_data(uploaded_file)

# ====================== WELO POR JOGADOR E SUPERFÍCIE ======================
def get_player_welo(jogador_nome: str, superficie: str, df_welo) -> float:
    if df_welo.empty or not jogador_nome:
        return 1484.0
    
    clean_flash = unicodedata.normalize('NFKD', str(jogador_nome)).encode('ascii', 'ignore').decode('utf-8')
    clean_flash = ''.join(filter(str.isalnum, clean_flash.lower().strip()))
    
    if len(clean_flash) < 5:
        return 1484.0
    
    best_match = None
    best_score = 0
    
    for _, row in df_welo.iterrows():
        clean_excel = row.get('Jogador_clean', '')
        if not clean_excel: continue
            
        score = 0
        if clean_flash in clean_excel or clean_excel in clean_flash:
            score = 100
        elif len(clean_flash) > 6 and len(clean_excel) > 6:
            common = len(set(clean_flash) & set(clean_excel))
            if common > len(clean_flash) * 0.65:
                score = 75
        
        if score > best_score:
            best_score = score
            best_match = row
    
    if best_score < 60 or best_match is None:
        return 1484.0
    
    surface_map = {'clay': 'ELO Clay', 'hard': 'ELO Hard', 'grass': 'ELO Grass', 'indoor': 'ELO Indoor'}
    col = surface_map.get(superficie.lower())
    
    if col and col in best_match.index:
        val = best_match[col]
        if pd.notna(val) and str(val).strip() != '':
            return round(float(val), 1)
    
    elo_cols = ['ELO Hard', 'ELO Clay', 'ELO Grass', 'ELO Indoor']
    values = [float(best_match[c]) for c in elo_cols if c in best_match.index and pd.notna(best_match[c])]
    return round(sum(values) / len(values), 1) if values else 1484.0

# ====================== LINHA TOTAL ======================
def calcular_linha_total(welo1: float, welo2: float, superficie: str):
    dif = abs(welo1 - welo2)
    base = {'Clay': 23.1, 'Hard': 22.5, 'Grass': 22.0, 'Indoor': 22.7}.get(superficie, 22.6)
    ajuste = -0.042 * dif
    total_esperado = round(base + ajuste, 2)
    total_esperado = max(19.0, min(27.0, total_esperado))
    prob = max(40, min(78, 53 + (total_esperado - 22.0) * 7))
    return total_esperado, round(prob, 1)

# ====================== DETECÇÃO SUPERFÍCIE ======================
def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    clay_kw = ['clay', 'saibro', 'kigali', 'santiago', 'punto cana', 'bucharest', 'houston', 'marrakech', 'rio', 'barcelona', 'murcia', 'girona', 'oeiras']
    grass_kw = ['grass', 'wimbledon', 'halle', 'queens', 'eastbourne']
    indoor_kw = ['indoor', 'stockholm', 'basel']
    
    if any(k in t for k in clay_kw):
        return 'Clay'
    elif any(k in t for k in grass_kw):
        return 'Grass'
    elif any(k in t for k in indoor_kw):
        return 'Indoor'
    return 'Hard'

# ====================== SCRAPING COM SELENIUM ======================
def get_flashscore_matches_selenium():
    matches = []
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    try:
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.flashscore.pt/tenis/")
        time.sleep(8)  # Aguarda carregamento inicial

        # Tenta clicar na aba "Agendados"
        try:
            scheduled_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Agendados')]"))
            )
            scheduled_tab.click()
            time.sleep(5)
        except:
            pass

        # Pega todas as partidas
        games = driver.find_elements(By.CSS_SELECTOR, "div.event__match")
        
        for game in games[:80]:
            try:
                tournament = game.find_element(By.CSS_SELECTOR, "div.event__tournament").text.strip()
                j1 = game.find_element(By.CSS_SELECTOR, "div.event__participant--home").text.strip()
                j2 = game.find_element(By.CSS_SELECTOR, "div.event__participant--away").text.strip()
                horario = game.find_element(By.CSS_SELECTOR, "div.event__time").text.strip()

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
                
        driver.quit()
        return pd.DataFrame(matches)
    
    except Exception as e:
        st.error(f"Erro no Selenium: {e}")
        if 'driver' in locals():
            driver.quit()
        return pd.DataFrame()

# ====================== EXECUÇÃO ======================
if st.button("🔄 Buscar Partidas com Selenium + Calcular WELO", type="primary"):
    if df_welo.empty:
        st.warning("⚠️ Carregue primeiro o ficheiro Challenger.xlsm na barra lateral.")
    else:
        with st.spinner("Abrindo navegador e buscando partidas no FlashScore... (pode demorar 15-25 segundos)"):
            df = get_flashscore_matches_selenium()
            
            if not df.empty:
                df['WELO_J1'] = df.apply(lambda row: get_player_welo(row['jogador_1'], row['superficie'], df_welo), axis=1)
                df['WELO_J2'] = df.apply(lambda row: get_player_welo(row['jogador_2'], row['superficie'], df_welo), axis=1)
                df['Dif_WELO'] = abs(df['WELO_J1'] - df['WELO_J2'])
                
                resultados = df.apply(lambda row: calcular_linha_total(row['WELO_J1'], row['WELO_J2'], row['superficie']), axis=1)
                df['Total_Esperado'] = [r[0] for r in resultados]
                df['Prob_Mais_21.5'] = [r[1] for r in resultados]
                
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
                        "Dif_WELO": st.column_config.NumberColumn("Dif WELO", format="%.1f"),
                        "Total_Esperado": st.column_config.NumberColumn("Total Esperado", format="%.2f"),
                        "Prob_Mais_21.5": st.column_config.NumberColumn("Prob >21.5 (%)", format="%.1f"),
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
                st.warning("Não foi possível extrair partidas. Tente novamente ou use o modo manual.")
else:
    st.info("Clique no botão acima para buscar as partidas usando Selenium.")

st.caption("Versão com Selenium | Mais estável que Playwright no Streamlit Cloud")
