import streamlit as st
import pandas as pd
from datetime import datetime
import time
import unicodedata
from difflib import SequenceMatcher
from io import BytesIO
import math

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Partidas Hoje + Predictor Stats")

tab1, tab2, tab3 = st.tabs(["📅 Partidas Hoje", "🔍 Previsão Personalizada", "📈 Modeling Strategy"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])
   
    if st.button("🗑️ Limpar Cache"):
        st.cache_data.clear()
        st.success("Cache limpo!")

# (Mantém as funções load_stats, calculate_elo_by_surface, norm, find_best_player_stats, 
# get_player_elo, predict_from_stats exatamente como no teu código anterior)

# ====================== SELENIUM + SOFASCORE ======================
@st.cache_data(ttl=600, show_spinner=False)
def get_todays_matches_sofascore():
    """Scraping com Selenium no Sofascore - mais confiável para ténis"""
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    url = f"https://www.sofascore.com/tennis/{today_str}"
    
    options = Options()
    options.add_argument("--headless")           # Sem interface gráfica
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    try:
        with st.spinner("🌐 Abrindo Sofascore com Selenium (pode demorar 10-20s)..."):
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.get(url)
            time.sleep(8)  # Espera carregar o JavaScript

            matches = []
            
            # Seletores atualizados para Sofascore Tennis (2026)
            match_elements = driver.find_elements(By.CSS_SELECTOR, 
                "div[class*='EventCell'], div[class*='match'], div[class*='fixture']")
            
            if not match_elements:
                match_elements = driver.find_elements(By.CSS_SELECTOR, 
                    "[data-testid='event-cell'], .Match, .Event")
            
            for elem in match_elements[:40]:   # Limite para não pegar lixo
                try:
                    # Horário
                    time_elem = elem.find_element(By.CSS_SELECTOR, 
                        "div[class*='time'], span[class*='time'], [data-testid='event-time']")
                    horario = time_elem.text.strip() if time_elem else "11:00"
                    
                    # Jogadores
                    players = elem.find_elements(By.CSS_SELECTOR, 
                        "div[class*='participant'], span[class*='name'], .Player")
                    if len(players) >= 2:
                        j1 = players[0].text.strip()
                        j2 = players[1].text.strip()
                    else:
                        continue
                    
                    # Torneio
                    tour_elem = elem.find_element(By.CSS_SELECTOR, 
                        "div[class*='tournament'], span[class*='league'], .Competition")
                    torneio = tour_elem.text.strip() if tour_elem else "ATP/WTA Tour"
                    
                    # Normalizar torneios conhecidos
                    if "Monte Carlo" in torneio or "Monte-Carlo" in torneio:
                        torneio = "ATP Monte-Carlo Masters"
                    elif "Marrakech" in torneio:
                        torneio = "ATP Marrakech"
                    elif "Houston" in torneio:
                        torneio = "ATP Houston"
                    elif "Charleston" in torneio:
                        torneio = "WTA Charleston"
                    
                    superficie = "Clay"
                    
                    if j1 and j2 and len(j1) > 3 and len(j2) > 3:
                        matches.append({
                            'torneio': torneio,
                            'jogador_1': j1,
                            'jogador_2': j2,
                            'horario': horario,
                            'superficie': superficie
                        })
                except:
                    continue
            
            driver.quit()
            
            if len(matches) >= 5:
                st.success(f"✅ {len(matches)} partidas carregadas do Sofascore!")
                return pd.DataFrame(matches)
                
    except Exception as e:
        st.error(f"Erro no Selenium: {str(e)[:120]}")
        if 'driver' in locals():
            driver.quit()
    
    # ====================== FALLBACK ======================
    st.warning("Sofascore não retornou dados suficientes. Usando fallback.")
    return get_fallback_matches()

def get_fallback_matches():
    data = [
        {'torneio': 'ATP Houston - Final', 'jogador_1': 'Tommy Paul', 'jogador_2': 'Roman Burruchaga', 'horario': '22:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Marrakech - Final', 'jogador_1': 'Marco Trungelliti', 'jogador_2': 'Rafael Jodar', 'horario': '15:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Bucharest - Final', 'jogador_1': 'Mariano Navone', 'jogador_2': 'Botic van de Zandschulp', 'horario': '14:00', 'superficie': 'Clay'},
        {'torneio': 'WTA Charleston - Final', 'jogador_1': 'Jessica Pegula', 'jogador_2': 'Emma Navarro', 'horario': '18:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Vit Kopriva', 'jogador_2': 'Matteo Arnaldi', 'horario': '11:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Alexander Shevchenko', 'jogador_2': 'Andrea Pellegrino', 'horario': '11:30', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Hugo Gaston', 'jogador_2': 'Titouan Droguet', 'horario': '13:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Richard Gasquet', 'jogador_2': 'Valentin Royer', 'horario': '13:30', 'superficie': 'Clay'},
    ]
    return pd.DataFrame(data)

# ====================== Na ABA 1 ======================
with tab1:
    st.header(f"📅 Partidas de Tênis - {datetime.now().strftime('%d/%m/%Y')}")
   
    if df_stats.empty:
        st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx")
    else:
        if st.button("🔄 Buscar Partidas de Hoje (Sofascore)", type="primary", use_container_width=True):
            matches_df = get_todays_matches_sofascore()
            st.session_state.cached_matches = matches_df

        # Resto do código de cálculo de previsões igual ao anterior...
