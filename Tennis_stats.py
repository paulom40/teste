import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import unicodedata
from difflib import SequenceMatcher
import time
from io import BytesIO
import math

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Tennis Predictor Pro - API RapidAPI")

# ====================== CONFIGURAÇÃO API ======================
RAPIDAPI_KEY = "bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a44"
RAPIDAPI_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])
   
    st.markdown("---")
    st.markdown("### ⚙️ Configurações API")
   
    # Mostrar status da API
    if st.button("🔌 Testar Conexão API"):
        with st.spinner("Testando..."):
            try:
                headers = {
                    "x-rapidapi-host": RAPIDAPI_HOST,
                    "x-rapidapi-key": RAPIDAPI_KEY
                }
                test_url = f"https://{RAPIDAPI_HOST}/tennis/v2/atp/rankings"
                response = requests.get(test_url, headers=headers, timeout=5)
               
                if response.status_code == 200:
                    st.success("✅ API Conectada!")
                else:
                    st.error(f"❌ Erro: {response.status_code} - {response.text[:200]}")
            except Exception as e:
                st.error(f"❌ Erro de conexão: {str(e)[:150]}")
   
    st.markdown("---")
    if st.button("🗑️ Limpar Cache"):
        st.cache_data.clear()
        st.success("Cache limpo!")

# ====================== FUNÇÕES API ======================
@st.cache_data(ttl=1800)  # Cache 30 min
def get_matches_from_rapidapi(date_str=None):
    """Busca partidas via RapidAPI Tennis"""
   
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
   
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
   
    matches = []
   
    # Endpoints disponíveis
    endpoints = [
        f"https://{RAPIDAPI_HOST}/tennis/v2/atp/matches/schedule/{date_str}",
        f"https://{RAPIDAPI_HOST}/tennis/v2/wta/matches/schedule/{date_str}",
        f"https://{RAPIDAPI_HOST}/tennis/v2/atp/matches/live",
        f"https://{RAPIDAPI_HOST}/tennis/v2/wta/matches/live",
    ]
   
    for url in endpoints:
        try:
            response = requests.get(url, headers=headers, timeout=10)
           
            if response.status_code == 200:
                data = response.json()
               
                events = []
                if isinstance(data, dict):
                    events = data.get('matches') or data.get('events') or data.get('data') or []
                elif isinstance(data, list):
                    events = data
               
                for event in events:
                    try:
                        if not isinstance(event, dict):
                            continue
                        
                        # Extrair jogadores
                        home = event.get('homeTeam') or event.get('home') or {}
                        away = event.get('awayTeam') or event.get('away') or {}
                        
                        player1 = home.get('name') or home.get('shortName') if isinstance(home, dict) else str(home)
                        player2 = away.get('name') or away.get('shortName') if isinstance(away, dict) else str(away)
                        
                        if not player1 or not player2:
                            continue
                        
                        # Torneio
                        tournament = event.get('tournament') or {}
                        torneio = tournament.get('name') if isinstance(tournament, dict) else str(tournament)
                        
                        # Horário
                        start_time = event.get('startTimestamp') or event.get('time')
                        if start_time:
                            try:
                                if isinstance(start_time, (int, float)):
                                    horario = datetime.fromtimestamp(start_time).strftime('%H:%M')
                                else:
                                    horario = str(start_time)
                            except:
                                horario = 'TBD'
                        else:
                            horario = 'TBD'
                        
                        # Superfície
                        surface = event.get('groundType') or event.get('surface') or ''
                        if surface:
                            surface_lower = str(surface).lower()
                            if 'clay' in surface_lower:
                                superficie = 'Clay'
                            elif 'grass' in surface_lower:
                                superficie = 'Grass'
                            elif 'hard' in surface_lower:
                                superficie = 'Hard'
                            else:
                                superficie = detect_surface(torneio)
                        else:
                            superficie = detect_surface(torneio)
                        
                        matches.append({
                            'torneio': torneio or 'Torneio Desconhecido',
                            'jogador_1': player1,
                            'jogador_2': player2,
                            'horario': horario,
                            'superficie': superficie
                        })
                    except:
                        continue
        except:
            continue
   
    if matches:
        df = pd.DataFrame(matches)
        df = df.drop_duplicates(subset=['jogador_1', 'jogador_2'])
        return df
   
    return pd.DataFrame()

def detect_surface(tournament: str) -> str:
    """Detecta superfície pelo nome do torneio"""
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'terre', 'barletta', 'marrakech', 'monte-carlo', 'bucarest', 
                           'houston', 'barcelona', 'madrid', 'rome', 'roland garros', 'french']):
        return 'Clay'
    if any(k in t for k in ['grass', 'relva', 'wimbledon', 'halle', 'queens']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'coberta', 'paris masters', 'vienna', 'basel']):
        return 'Indoor'
    return 'Hard'

# ====================== CARREGAR STATS ======================
@st.cache_data(ttl=3600)
def load_stats(file):
    if not file:
        return pd.DataFrame()
    try:
        df = pd.read_excel(file)
       
        def norm(name):
            if not isinstance(name, str):
                return ""
            n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            return ''.join(filter(str.isalnum, n.lower().strip()))
       
        df['winner_clean'] = df['winner_name'].apply(norm)
        df['loser_clean'] = df['loser_name'].apply(norm)
       
        if 'surface' not in df.columns:
            df['surface'] = 'Hard'
       
        df = calculate_elo_by_surface(df)
       
        st.sidebar.success(f"✅ {len(df)} jogos carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar ficheiro: {e}")
        return pd.DataFrame()

# (Mantive as funções calculate_elo_by_surface, norm, find_best_player_stats, predict_from_stats, 
#  get_player_elo e to_excel iguais às que tinhas - não alterei para não quebrar a lógica)

df_stats = load_stats(uploaded_file)

# ====================== RESTO DO CÓDIGO ======================
# ... (o resto do teu código permanece igual: funções auxiliares, interface principal, etc.)

# Apenas colei aqui o final para completar:

# ====================== INTERFACE PRINCIPAL ======================
st.markdown(f"## 📅 Partidas de Tênis - {datetime.now().strftime('%d/%m/%Y')}")

if df_stats.empty:
    st.error("⚠️ **Carregue o ficheiro Challenger1.xlsx na barra lateral!**")
else:
    # (Aqui continua todo o código que tinhas a partir da seleção de data até o final)
    # Copia e cola o resto exatamente como estava no teu código original a partir desta linha:
    # "    # Seleção de data" até o final.
