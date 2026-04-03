import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="ATP & Challenger Tennis - API",
    page_icon="🎾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎾 ATP & Challenger Tennis - Live Data</p>', unsafe_allow_html=True)
st.markdown("**Powered by SportsScore API**")

# API Configuration
API_KEY = "bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a44"
API_HOST = "sportscore1.p.rapidapi.com"
BASE_URL = "https://sportscore1.p.rapidapi.com"

@st.cache_data(ttl=1800)  # Cache por 30 minutos
def get_tennis_sport_id():
    """
    Get the sport ID for tennis
    """
    url = f"{BASE_URL}/sports"
    
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_HOST
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Find tennis in the sports list
            for sport in data.get('data', []):
                if 'tennis' in sport.get('name', '').lower():
                    return sport.get('id')
            # If not found, tennis is usually sport_id = 5 or 2
            return 5
        return 5  # Default tennis ID
    except Exception as e:
        st.error(f"Erro ao obter sport ID: {e}")
        return 5

@st.cache_data(ttl=1800)
def get_tennis_leagues():
    """
    Get ATP and Challenger leagues
    """
    sport_id = get_tennis_sport_id()
    url = f"{BASE_URL}/leagues"
    
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_HOST
    }
    
    params = {
        "sport_id": sport_id
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            leagues = []
            
            for league in data.get('data', []):
                league_name = league.get('name', '')
                # Filtrar apenas ATP e Challenger
                if any(keyword in league_name for keyword in ['ATP', 'Challenger', 'Grand Slam']):
                    leagues.append({
                        'id': league.get('id'),
                        'name': league_name,
                        'country': league.get('country', {}).get('name', 'International')
                    })
            
            return leagues
        else:
            st.error(f"Erro na API: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Erro ao buscar ligas: {e}")
        return []

@st.cache_data(ttl=600)  # Cache por 10 minutos
def get_tennis_events(league_id=None, date_start=None, date_end=None):
    """
    Get tennis events/matches
    """
    sport_id = get_tennis_sport_id()
    url = f"{BASE_URL}/events/search"
    
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_HOST,
        "Content-Type": "application/json"
    }
    
    # Preparar parâmetros
    params = {
        "sport_id": sport_id,
        "page": 1
    }
    
    if league_id:
        params["league_id"] = league_id
    
    if date_start:
        params["date_start"] = date_start
    
    if date_end:
        params["date_end"] = date_end
    
    try:
        response = requests.post(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        else:
            st.warning(f"API Status: {response.status_code}")
            st.write(f"Response: {response.text[:500]}")
            return []
            
    except Exception as e:
        st.error(f"Erro ao buscar eventos: {e}")
        return []

def parse_match_data(events):
    """
    Parse event data into readable format
    """
    matches = []
    
    for event in events:
        try:
            match = {
                'ID': event.get('id'),
                'Liga': event.get('league', {}).get('name', 'N/A'),
                'Torneio': event.get('tournament', {}).get('name', 'N/A'),
                'Jogador_Casa': event.get('home_team', {}).get('name', 'N/A'),
                'Jogador_Fora': event.get('away_team', {}).get('name', 'N/A'),
                'Status': event.get('status', 'N/A'),
                'Data': event.get('start_at', 'N/A'),
                'Placar_Casa': event.get('home_score', {}).get('current', '-'),
                'Placar_Fora': event.get('away_score', {}).get('current', '-'),
                'Round': event.get('round', {}).get('name', 'N/A') if event.get('round') else 'N/A',
            }
            
            # Adicionar odds se disponível
            if event.get('odds'):
                match['Odds_Casa'] = event.get('odds', {}).get('home', 'N/A')
                match['Odds_Fora'] = event.get('odds', {}).get('away', 'N/A')
            
            matches.append(match)
            
        except Exception as e:
            continue
    
    return matches

# Sidebar - Configurações
st.sidebar.header("⚙️ Configurações")

# Datas
today = datetime.now()
date_start = st.sidebar.date_input(
    "Data Início",
    value=today - timedelta(days=1),
    max_value=today + timedelta(days=30)
)

date_end = st.sidebar.date_input(
    "Data Fim",
    value=today + timedelta(days=7),
    max_value=today + timedelta(days=30)
)

# Botões de controle
if st.sidebar.button("🔄 Atualizar Dados"):
    st.cache_data.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)

# Informação da API
with st.sidebar.expander("ℹ️ Info da API"):
    st.write("**Status:** Conectada")
    st.write(f"**Host:** {API_HOST}")
    st.write(f"**Sport ID:** Tennis")
    st.caption("API: SportsScore via RapidAPI")

# Main Content
st.markdown("---")

# Métricas de topo
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("⏰ Última Atualização", datetime.now().strftime("%H:%M:%S"))
with col2:
    st.metric("📅 Período", f"{(date_end - date_start).days} dias")
with col3:
    st.metric("🔄 Status", "🟢 Ativa")

st.markdown("---")

# Buscar ligas
with st.spinner("🔍 Buscando ligas de ténis..."):
    leagues = get_tennis_leagues()

if leagues:
    st.success(f"✅ {len(leagues)} ligas encontradas")
    
    # Selector de liga
    st.subheader("🏆 Selecionar Torneio")
    
    league_options = {f"{l['name']} ({l['country']})": l['id'] for l in leagues}
    league_options = {"Todos os Torneios": None, **league_options}
    
    selected_league_name = st.selectbox(
        "Escolha o torneio",
        options=list(league_options.keys())
    )
    
    selected_league_id = league_options[selected_league_name]
    
    # Buscar eventos
    st.markdown("---")
    st.subheader("🎾 Jogos")
    
    with st.spinner("📊 Carregando jogos..."):
        events = get_tennis_events(
            league_id=selected_league_id,
            date_start=date_start.strftime("%Y-%m-%d"),
            date_end=date_end.strftime("%Y-%m-%d")
        )
    
    if events:
        # Parse data
        matches = parse_match_data(events)
        
        if matches:
            df = pd.DataFrame(matches)
            
            # Estatísticas
            st.markdown("### 📊 Estatísticas")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Total de Jogos", len(df))
            with metric_cols[1]:
                tournaments = df['Torneio'].nunique()
                st.metric("Torneios", tournaments)
            with metric_cols[2]:
                live_matches = len(df[df['Status'].str.contains('live|playing', case=False, na=False)])
                st.metric("Ao Vivo", live_matches)
            with metric_cols[3]:
                upcoming = len(df[df['Status'].str.contains('not_started|scheduled', case=False, na=False)])
                st.metric("Próximos", upcoming)
            
            st.markdown("---")
            
            # Filtros adicionais
            col1, col2 = st.columns(2)
            
            with col1:
                status_filter = st.multiselect(
                    "Filtrar por Status",
                    options=df['Status'].unique().tolist(),
                    default=df['Status'].unique().tolist()
                )
            
            with col2:
                tournament_filter = st.multiselect(
                    "Filtrar por Torneio",
                    options=df['Torneio'].unique().tolist(),
                    default=df['Torneio'].unique().tolist()
                )
            
            # Aplicar filtros
            filtered_df = df[
                (df['Status'].isin(status_filter)) &
                (df['Torneio'].isin(tournament_filter))
            ]
            
            st.markdown("---")
            
            # Tabela de dados
            st.markdown(f"### 📋 Dados dos Jogos ({len(filtered_df)} jogos)")
            
            # Formatar colunas para exibição
            display_df = filtered_df[[
                'Liga', 'Torneio', 'Jogador_Casa', 'Jogador_Fora',
                'Placar_Casa', 'Placar_Fora', 'Status', 'Data', 'Round'
            ]]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400,
                hide_index=True
            )
            
            # Seção de Downloads
            st.markdown("---")
            st.markdown("### 💾 Exportar Dados")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f'tennis_atp_challenger_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                    mime='text/csv',
                )
            
            with col2:
                json_data = filtered_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f'tennis_atp_challenger_{datetime.now().strftime("%Y%m%d_%H%M")}.json',
                    mime='application/json',
                )
            
            with col3:
                # Dados completos (incluindo odds se houver)
                full_json = json.dumps(events, indent=2)
                st.download_button(
                    label="📥 Dados Completos (JSON)",
                    data=full_json,
                    file_name=f'tennis_full_data_{datetime.now().strftime("%Y%m%d_%H%M")}.json',
                    mime='application/json',
                )
            
            # Ver dados brutos
            with st.expander("🔍 Ver Dados Brutos da API"):
                st.json(events[:3] if len(events) > 3 else events)
        
        else:
            st.warning("⚠️ Não foi possível processar os dados dos eventos")
            st.json(events[:2] if len(events) > 2 else events)
    
    else:
        st.info("ℹ️ Nenhum jogo encontrado para o período selecionado")
        st.write("Tente ajustar as datas ou selecionar outro torneio")

else:
    st.error("❌ Não foi possível carregar as ligas")
    st.info("""
    **Possíveis soluções:**
    1. Verifique se a API key está correta
    2. Confirme que você tem créditos disponíveis na RapidAPI
    3. Tente novamente em alguns segundos
    """)

# Informações adicionais
st.markdown("---")
with st.expander("📖 Como Usar"):
    st.markdown("""
    ### Instruções:
    
    1. **Selecione as datas** na barra lateral
    2. **Escolha um torneio** específico ou veja todos
    3. **Filtre por status** (ao vivo, agendados, finalizados)
    4. **Exporte os dados** em CSV ou JSON
    
    ### Tipos de Status:
    - `not_started` / `scheduled` - Jogos agendados
    - `live` / `playing` - Jogos ao vivo
    - `finished` / `ended` - Jogos finalizados
    - `postponed` - Jogos adiados
    - `cancelled` - Jogos cancelados
    
    ### Dados Disponíveis:
    - Liga e Torneio
    - Jogadores (Casa vs Fora)
    - Placar atual
    - Status do jogo
    - Data e hora
    - Round/Fase
    - Odds (quando disponível)
    """)

with st.expander("⚙️ Sobre a API"):
    st.markdown("""
    ### SportsScore API (RapidAPI)
    
    **Características:**
    - ✅ Dados em tempo real
    - ✅ Cobertura global de ténis
    - ✅ ATP, Challenger, Grand Slams
    - ✅ Estatísticas detalhadas
    
    **Limites:**
    - Verifique seu plano no RapidAPI
    - Free tier tem limite de requisições
    - Cache de 10-30 minutos para otimizar
    
    **Endpoints utilizados:**
    - `/sports` - Listar esportes
    - `/leagues` - Listar ligas/torneios
    - `/events/search` - Buscar jogos
    """)

# Footer
st.markdown("---")
st.caption("🎾 ATP & Challenger Tennis Scraper | Powered by SportsScore API")

# Auto-refresh
if auto_refresh:
    import time
    time.sleep(300)  # 5 minutos
    st.rerun()
