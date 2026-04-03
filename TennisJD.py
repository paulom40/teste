import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Tennis API Explorer", page_icon="🎾", layout="wide")

st.title("🎾 SportsScore API Explorer")
st.markdown("**Explorar endpoints e encontrar dados de ténis**")

# API Config
API_KEY = "bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a44"
API_HOST = "sportscore1.p.rapidapi.com"
BASE_URL = "https://sportscore1.p.rapidapi.com"

headers = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": API_HOST,
    "Content-Type": "application/json"
}

# Sidebar
st.sidebar.header("🔧 API Explorer")

# Available endpoints to try
endpoints = {
    "Sports List": "/sports",
    "Leagues": "/leagues",
    "Events Search": "/events/search",
    "Live Events": "/events/live",
    "Fixtures": "/fixtures",
    "Today Events": "/events/today",
    "Competitions": "/competitions",
}

selected_endpoint = st.sidebar.selectbox(
    "Selecione Endpoint",
    list(endpoints.keys())
)

endpoint_url = endpoints[selected_endpoint]

# Parameters
st.sidebar.subheader("⚙️ Parâmetros")

sport_id = st.sidebar.number_input("Sport ID", min_value=1, max_value=100, value=5)
league_id = st.sidebar.number_input("League ID (opcional)", min_value=0, value=0)
page = st.sidebar.number_input("Página", min_value=1, value=1)

today = datetime.now()
date_start = st.sidebar.date_input("Data Início", value=today - timedelta(days=1))
date_end = st.sidebar.date_input("Data Fim", value=today + timedelta(days=7))

# Build params
params = {"page": page}

if "sport" in selected_endpoint.lower() or selected_endpoint == "Leagues":
    if sport_id > 0:
        params["sport_id"] = sport_id

if "events" in selected_endpoint.lower() or "fixtures" in selected_endpoint.lower():
    params["sport_id"] = sport_id
    params["date_start"] = date_start.strftime("%Y-%m-%d")
    params["date_end"] = date_end.strftime("%Y-%m-%d")
    
    if league_id > 0:
        params["league_id"] = league_id

# Main content
st.markdown("---")

# Endpoint info
st.subheader(f"📡 Endpoint: {endpoint_url}")
st.code(f"GET {BASE_URL}{endpoint_url}")

# Show parameters
if params:
    st.markdown("**Parâmetros:**")
    st.json(params)

# Execute request
if st.button("🚀 Executar Requisição", type="primary"):
    
    with st.spinner("Fazendo requisição..."):
        try:
            # Choose method based on endpoint
            if "search" in endpoint_url:
                response = requests.post(
                    f"{BASE_URL}{endpoint_url}",
                    headers=headers,
                    params=params,
                    timeout=15
                )
            else:
                response = requests.get(
                    f"{BASE_URL}{endpoint_url}",
                    headers=headers,
                    params=params,
                    timeout=15
                )
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Resposta da API")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status Code", response.status_code)
            with col2:
                st.metric("Tempo", f"{response.elapsed.total_seconds():.2f}s")
            with col3:
                status_emoji = "✅" if response.status_code == 200 else "❌"
                st.metric("Status", status_emoji)
            
            # Response content
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Try to extract data array
                    if isinstance(data, dict) and 'data' in data:
                        items = data['data']
                        st.success(f"✅ {len(items)} itens encontrados")
                        
                        # Show first item
                        if items:
                            st.markdown("### 🔍 Primeiro Item:")
                            st.json(items[0])
                            
                            # Try to create dataframe
                            try:
                                df = pd.DataFrame(items)
                                st.markdown(f"### 📊 Tabela de Dados ({len(df)} linhas)")
                                st.dataframe(df, use_container_width=True, height=300)
                                
                                # Download
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "📥 Download CSV",
                                    csv,
                                    f"api_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    "text/csv"
                                )
                            except:
                                pass
                        
                        # Show all data
                        with st.expander("Ver JSON Completo"):
                            st.json(data)
                    else:
                        st.json(data)
                        
                except Exception as e:
                    st.error(f"Erro ao processar JSON: {e}")
                    st.text("Resposta bruta:")
                    st.code(response.text[:2000])
            else:
                st.error(f"Erro {response.status_code}")
                st.text("Resposta:")
                st.code(response.text[:2000])
                
        except Exception as e:
            st.error(f"❌ Erro na requisição: {e}")

# Quick tests section
st.markdown("---")
st.subheader("⚡ Testes Rápidos")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏃 Testar /sports"):
        try:
            r = requests.get(f"{BASE_URL}/sports", headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                st.success(f"✅ {len(data.get('data', []))} desportos")
                with st.expander("Ver"):
                    st.json(data.get('data', [])[:5])
            else:
                st.error(f"❌ {r.status_code}")
        except Exception as e:
            st.error(str(e))

with col2:
    if st.button("🏃 Testar /leagues"):
        try:
            r = requests.get(
                f"{BASE_URL}/leagues",
                headers=headers,
                params={"sport_id": sport_id},
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                st.success(f"✅ {len(data.get('data', []))} ligas")
                with st.expander("Ver"):
                    st.json(data.get('data', [])[:5])
            else:
                st.error(f"❌ {r.status_code}")
        except Exception as e:
            st.error(str(e))

with col3:
    if st.button("🏃 Testar /events/search"):
        try:
            r = requests.post(
                f"{BASE_URL}/events/search",
                headers=headers,
                params={
                    "sport_id": sport_id,
                    "date_start": date_start.strftime("%Y-%m-%d"),
                    "date_end": date_end.strftime("%Y-%m-%d"),
                    "page": 1
                },
                timeout=15
            )
            if r.status_code == 200:
                data = r.json()
                st.success(f"✅ {len(data.get('data', []))} eventos")
                with st.expander("Ver"):
                    st.json(data.get('data', [])[:3])
            else:
                st.error(f"❌ {r.status_code}")
        except Exception as e:
            st.error(str(e))

# Documentation
st.markdown("---")
with st.expander("📚 Documentação da API"):
    st.markdown("""
    ### Endpoints Comuns:
    
    - **GET /sports** - Lista todos os desportos
    - **GET /leagues?sport_id=X** - Ligas de um desporto
    - **POST /events/search** - Buscar eventos/jogos
    - **GET /events/live** - Eventos ao vivo
    - **GET /fixtures** - Próximos jogos
    
    ### Sport IDs Comuns:
    - **1** - Football/Soccer
    - **2** - Basketball  
    - **5** - Tennis (comum)
    - **12** - Tennis (alternativo)
    
    ### Parâmetros Úteis:
    - `sport_id` - ID do desporto
    - `league_id` - ID da liga/torneio
    - `date_start` - Data início (YYYY-MM-DD)
    - `date_end` - Data fim (YYYY-MM-DD)
    - `page` - Número da página
    - `status` - Status do evento (live, finished, etc)
    
    ### Response Format:
    ```json
    {
        "data": [...],
        "meta": {
            "total": 100,
            "page": 1,
            "per_page": 50
        }
    }
    ```
    """)

with st.expander("🔍 Encontrar Tennis"):
    st.markdown("""
    ### Como Encontrar Ténis:
    
    **Método 1: Via /sports**
    1. Execute `/sports`
    2. Procure por "tennis" no nome
    3. Anote o `id`
    
    **Método 2: Testar IDs comuns**
    - Sport ID: 5, 2, 1, 12
    - Execute `/events/search` com cada um
    - Veja qual retorna jogos de ténis
    
    **Método 3: Via /leagues**
    1. Teste diferentes sport_ids
    2. Procure ligas como "ATP", "WTA", "Grand Slam"
    3. Use o league_id nos eventos
    """)

st.markdown("---")
st.caption(f"🔑 API Key: {API_KEY[:15]}... | Host: {API_HOST}")
