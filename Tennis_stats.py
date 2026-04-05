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
    
    # Adicionar seleção de tipo de torneio
    tour_type = st.selectbox("Tipo de Torneio", ["atp", "wta"])
   
    if st.button("🔌 Testar Conexão API"):
        with st.spinner("Testando..."):
            try:
                headers = {
                    "x-rapidapi-host": RAPIDAPI_HOST,
                    "x-rapidapi-key": RAPIDAPI_KEY
                }
                
                # Tentar endpoint de ranking que deve existir
                # Baseado na documentação, tente um destes:
                test_url = f"https://{RAPIDAPI_HOST}/getRanking"
                
                # Parâmetros para a requisição
                params = {
                    "tour": tour_type,
                    "limit": "10"
                }
                
                response = requests.get(test_url, headers=headers, params=params, timeout=10)
               
                if response.status_code == 200:
                    st.success("✅ API Conectada com sucesso!")
                    data = response.json()
                    st.json(data)  # Mostra a estrutura da resposta
                else:
                    st.error(f"❌ Erro: {response.status_code}")
                    st.code(f"Resposta: {response.text[:300]}")
                    
                    # Tentar endpoint alternativo
                    st.info("Tentando endpoint alternativo...")
                    test_url2 = f"https://{RAPIDAPI_HOST}/getAllFixtures"
                    response2 = requests.get(test_url2, headers=headers, timeout=10)
                    if response2.status_code == 200:
                        st.success("✅ Endpoint /getAllFixtures funciona!")
                        st.json(response2.json())
                    else:
                        st.error(f"❌ Também falhou: {response2.status_code}")
                        
            except Exception as e:
                st.error(f"❌ Erro de conexão: {str(e)[:150]}")
   
    st.markdown("---")
    if st.button("🗑️ Limpar Cache"):
        st.cache_data.clear()
        st.success("Cache limpo!")

# ====================== FUNÇÕES API ======================
@st.cache_data(ttl=1800)
def get_matches_from_rapidapi(date_str=None, tour="atp"):
    """Busca partidas via RapidAPI Tennis usando endpoints corretos"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
   
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
   
    matches = []
    
    # Tenta diferentes endpoints que podem existir
    endpoints_to_try = [
        # Primeiro tenta getAllFixtures
        {"url": f"https://{RAPIDAPI_HOST}/getAllFixtures", "method": "get"},
        # Depois tenta com parâmetros
        {"url": f"https://{RAPIDAPI_HOST}/getAllFixtures", "method": "get", "params": {"tour": tour}},
        # Tenta getTournamentResults
        {"url": f"https://{RAPIDAPI_HOST}/getTournamentResults", "method": "get"},
        # Tenta getLiveMatches
        {"url": f"https://{RAPIDAPI_HOST}/getLiveMatches", "method": "get"},
    ]
    
    for endpoint in endpoints_to_try:
        try:
            if endpoint["method"] == "get":
                params = endpoint.get("params", {})
                response = requests.get(endpoint["url"], headers=headers, params=params, timeout=10)
            else:
                response = requests.post(endpoint["url"], headers=headers, timeout=10)
           
            if response.status_code == 200:
                data = response.json()
                
                # Extrair partidas da resposta
                fixtures = []
                if isinstance(data, dict):
                    # Verificar diferentes estruturas possíveis
                    for key in ['results', 'fixtures', 'matches', 'data', 'items']:
                        if key in data and isinstance(data[key], list):
                            fixtures = data[key]
                            break
                    # Se não encontrou com as chaves, pega todos os valores que são listas
                    if not fixtures:
                        for value in data.values():
                            if isinstance(value, list) and len(value) > 0:
                                fixtures = value
                                break
                elif isinstance(data, list):
                    fixtures = data
                
                for fixture in fixtures:
                    try:
                        if not isinstance(fixture, dict):
                            continue
                        
                        # Extrair jogadores
                        player1 = None
                        player2 = None
                        
                        # Tentar diferentes formatos
                        if 'player1' in fixture and 'player2' in fixture:
                            p1 = fixture['player1']
                            p2 = fixture['player2']
                            player1 = p1.get('name') if isinstance(p1, dict) else str(p1)
                            player2 = p2.get('name') if isinstance(p2, dict) else str(p2)
                        elif 'home_team' in fixture and 'away_team' in fixture:
                            player1 = fixture['home_team'].get('name') if isinstance(fixture['home_team'], dict) else str(fixture['home_team'])
                            player2 = fixture['away_team'].get('name') if isinstance(fixture['away_team'], dict) else str(fixture['away_team'])
                        elif 'players' in fixture and isinstance(fixture['players'], list) and len(fixture['players']) >= 2:
                            player1 = fixture['players'][0].get('name') if isinstance(fixture['players'][0], dict) else str(fixture['players'][0])
                            player2 = fixture['players'][1].get('name') if isinstance(fixture['players'][1], dict) else str(fixture['players'][1])
                        
                        if not player1 or not player2:
                            continue
                        
                        # Torneio
                        tournament = fixture.get('tournament') or fixture.get('competition') or {}
                        torneio = tournament.get('name') if isinstance(tournament, dict) else str(tournament) if tournament else "Torneio"
                        
                        # Horário
                        start_time = fixture.get('start_time') or fixture.get('date') or fixture.get('datetime')
                        horario = 'TBD'
                        if start_time:
                            try:
                                if isinstance(start_time, str):
                                    if 'T' in start_time:
                                        horario = start_time.split('T')[1][:5]
                                    else:
                                        horario = start_time[:5] if len(start_time) >= 5 else start_time
                                elif isinstance(start_time, (int, float)):
                                    horario = datetime.fromtimestamp(start_time).strftime('%H:%M')
                            except:
                                pass
                        
                        # Superfície
                        superficie = detect_surface(torneio)
                        if 'surface' in fixture:
                            surface_str = str(fixture['surface']).lower()
                            if 'clay' in surface_str:
                                superficie = 'Clay'
                            elif 'grass' in surface_str:
                                superficie = 'Grass'
                            elif 'indoor' in surface_str:
                                superficie = 'Indoor'
                        
                        matches.append({
                            'torneio': torneio,
                            'jogador_1': player1,
                            'jogador_2': player2,
                            'horario': horario,
                            'superficie': superficie
                        })
                    except Exception as e:
                        continue
                
                # Se encontrou partidas, retorna
                if matches:
                    break
                    
        except Exception as e:
            continue
    
    if matches:
        df = pd.DataFrame(matches)
        df = df.drop_duplicates(subset=['jogador_1', 'jogador_2'])
        return df
   
    return pd.DataFrame()

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'terre', 'barletta', 'marrakech', 'monte-carlo', 'bucarest',
                           'houston', 'barcelona', 'madrid', 'rome', 'roland garros', 'french', 'bastad']):
        return 'Clay'
    if any(k in t for k in ['grass', 'relva', 'wimbledon', 'halle', 'queens', 'newport']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'coberta', 'paris masters', 'vienna', 'basel']):
        return 'Indoor'
    return 'Hard'

# Resto do código permanece igual...
# (as funções load_stats, calculate_elo_by_surface, norm, find_best_player_stats, 
#  predict_from_stats, get_player_elo, to_excel continuam iguais)

# ====================== INTERFACE PRINCIPAL ======================
st.markdown(f"## 📅 Partidas de Tênis - {datetime.now().strftime('%d/%m/%Y')}")

if df_stats.empty:
    st.error("⚠️ **Carregue o ficheiro Challenger1.xlsx na barra lateral!**")
else:
    # Seleção de data
    col_date1, col_date2, col_date3 = st.columns([2, 1, 1])
   
    with col_date1:
        date_selected = st.date_input(
            "📅 Selecione a data:",
            value=datetime.now(),
            min_value=datetime.now() - timedelta(days=7),
            max_value=datetime.now() + timedelta(days=14)
        )
   
    with col_date2:
        if st.button("🔄 BUSCAR PARTIDAS", type="primary", use_container_width=True):
            with st.spinner("🌐 Buscando via RapidAPI..."):
                date_str = date_selected.strftime("%Y-%m-%d")
                matches_df = get_matches_from_rapidapi(date_str, tour_type)
               
                if not matches_df.empty:
                    st.session_state.matches = matches_df
                    st.success(f"✅ {len(matches_df)} partidas encontradas!")
                    st.rerun()
                else:
                    st.warning("⚠️ Nenhuma partida encontrada. Tente outra data ou adicione manualmente.")
                    st.info("💡 Dica: O botão 'Testar Conexão API' mostra a estrutura exata da resposta")
   
    with col_date3:
        if st.button("➕ Adicionar", use_container_width=True):
            st.session_state.show_form = True
    
    # Resto do código continua igual...
