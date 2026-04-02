import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
from io import BytesIO
import unicodedata
import time

# ====================== CONFIGURAÇÃO ======================
st.set_page_config(page_title="Tênis Hoje - WELO + Total", page_icon="🎾", layout="wide")

st.title("🎾 Partidas de Tênis Hoje + WELO + Linha Total")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

# ====================== CONFIGURAÇÃO DA API ======================
RAPIDAPI_KEY = "bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a4"
RAPIDAPI_HOST = "sportscore1.p.rapidapi.com"

# Cache para evitar muitas requisições
@st.cache_data(ttl=300)  # Cache de 5 minutos
def get_tennis_matches():
    """
    Busca partidas de ténis do dia atual via SportScore API
    """
    matches = []
    
    try:
        # Data de hoje e próximos 7 dias
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Endpoint para buscar eventos de ténis (sport_id=2)
        url = "https://sportscore1.p.rapidapi.com/api/v1/events"
        
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": RAPIDAPI_HOST,
            "Content-Type": "application/json"
        }
        
        # Parâmetros para filtrar ténis e partidas futuras
        params = {
            "sport_id": 2,  # 2 = Tennis
            "page": 1,
            "status": "not_started"  # Partidas não iniciadas
        }
        
        st.info("🔄 Buscando partidas de ténis da API...")
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Processar os dados
            if 'data' in data and isinstance(data['data'], list):
                for event in data['data']:
                    try:
                        # Extrair informações da partida
                        tournament = event.get('league', {}).get('name', 'Torneio Desconhecido')
                        home_team = event.get('home_team', {}).get('name', '')
                        away_team = event.get('away_team', {}).get('name', '')
                        
                        # Data e hora
                        start_time = event.get('starting_at', '')
                        horario = ''
                        if start_time:
                            try:
                                dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                                horario = dt.strftime('%H:%M')
                            except:
                                horario = start_time
                        
                        # Verificar se tem ambos os jogadores
                        if home_team and away_team:
                            # Detectar superfície
                            superficie = detect_surface(tournament)
                            
                            matches.append({
                                'torneio': tournament,
                                'jogador_1': home_team,
                                'jogador_2': away_team,
                                'horario': horario,
                                'superficie': superficie,
                                'status': event.get('status', 'Agendado')
                            })
                    except Exception as e:
                        continue
                
                st.success(f"✅ Encontradas {len(matches)} partidas de ténis")
            else:
                st.warning("Nenhuma partida encontrada para hoje")
        else:
            st.error(f"Erro na API: {response.status_code}")
            st.json(response.text if response.text else "Sem detalhes")
            
    except requests.exceptions.Timeout:
        st.error("⏰ Timeout na requisição. A API demorou muito para responder.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erro na requisição: {str(e)}")
    except Exception as e:
        st.error(f"❌ Erro inesperado: {str(e)}")
    
    return pd.DataFrame(matches)

# ====================== FUNÇÃO PARA BUSCAR MAIS PARTIDAS ======================
def get_more_matches():
    """
    Busca partidas de ténis de datas específicas
    """
    matches = []
    
    try:
        # Buscar partidas dos próximos 7 dias
        for i in range(7):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            
            url = f"https://sportscore1.p.rapidapi.com/api/v1/events/date/{date}"
            
            headers = {
                "X-RapidAPI-Key": RAPIDAPI_KEY,
                "X-RapidAPI-Host": RAPIDAPI_HOST
            }
            
            params = {
                "sport_id": 2  # Tennis
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and isinstance(data['data'], list):
                    for event in data['data']:
                        try:
                            tournament = event.get('league', {}).get('name', 'Torneio Desconhecido')
                            home_team = event.get('home_team', {}).get('name', '')
                            away_team = event.get('away_team', {}).get('name', '')
                            
                            if home_team and away_team:
                                superficie = detect_surface(tournament)
                                
                                matches.append({
                                    'torneio': tournament,
                                    'jogador_1': home_team,
                                    'jogador_2': away_team,
                                    'horario': date,
                                    'superficie': superficie,
                                    'status': event.get('status', 'Agendado')
                                })
                        except:
                            continue
            
            time.sleep(0.5)  # Evitar rate limiting
            
    except Exception as e:
        st.error(f"Erro ao buscar mais partidas: {str(e)}")
    
    return pd.DataFrame(matches)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger.xlsm")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger.xlsm", type=["xlsm", "xlsx"])
    
    st.markdown("---")
    st.header("⚙️ Opções")
    days_option = st.radio(
        "Período das partidas:",
        ["Hoje apenas", "Próximos 7 dias"],
        help="Selecione 'Hoje apenas' para partidas do dia atual"
    )

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

# ====================== FUNÇÃO WELO ======================
def get_welo(jogador_nome: str, superficie: str, df_welo) -> float:
    if df_welo.empty or not jogador_nome:
        return 1484.0
    
    clean_flash = unicodedata.normalize('NFKD', str(jogador_nome)).encode('ascii', 'ignore').decode('utf-8')
    clean_flash = ''.join(filter(str.isalnum, clean_flash.lower().strip()))
    
    if len(clean_flash) < 5:
        return 1484.0
    
    best_match = None
    best_score = 0
    
    for _, row in df_welo.iterrows():
        clean_excel = row['Jogador_clean']
        if not clean_excel: continue
            
        score = 0
        if clean_flash in clean_excel or clean_excel in clean_flash:
            score = 100
        elif len(clean_flash) > 6 and len(clean_excel) > 6:
            common = len(set(clean_flash) & set(clean_excel))
            if common > len(clean_flash) * 0.65:
                score = 70
        
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

# ====================== CÁLCULO DA LINHA TOTAL ======================
def calcular_linha_total(welo1: float, welo2: float, superficie: str) -> tuple:
    """
    Retorna (Total_Esperado, Prob_Mais_21.5)
    """
    dif = abs(welo1 - welo2)
    
    # Base média de jogos por superfície
    base_jogos = {
        'Clay': 22.8,
        'Hard': 22.4,
        'Grass': 21.9,
        'Indoor': 22.6
    }.get(superficie, 22.5)
    
    # Ajuste pela diferença de nível
    ajuste_dif = -0.035 * dif   # quanto maior a diferença, menos jogos esperados
    
    total_esperado = base_jogos + ajuste_dif
    total_esperado = max(18.5, min(27.0, total_esperado))  # limite realista
    
    # Probabilidade de Mais de 21.5
    prob_mais_21_5 = max(0.35, min(0.78, 0.5 + (total_esperado - 22.0) * 0.08))
    
    return round(total_esperado, 2), round(prob_mais_21_5 * 100, 1)

# ====================== FUNÇÃO DE SUPORTE ======================
def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'kigali', 'santiago', 'punto cana', 'bucharest', 'houston', 'marrakech', 'rio', 'barcelona', 'monte carlo']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon', 'halle', 'queens', 'eastbourne']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'stockholm', 'basel', 'vienna']):
        return 'Indoor'
    return 'Hard'

# ====================== EXECUÇÃO PRINCIPAL ======================
if st.button("🔄 Buscar Partidas + Calcular WELO + Linha Total", type="primary"):
    if df_welo.empty:
        st.warning("⚠️ Carregue primeiro o ficheiro Challenger.xlsm na barra lateral.")
    else:
        with st.spinner("Buscando partidas da API e calculando WELO..."):
            if days_option == "Hoje apenas":
                df = get_tennis_matches()
            else:
                df = get_more_matches()
            
            if not df.empty:
                # Calcular WELO para cada jogador
                df['WELO_J1'] = df.apply(lambda row: get_welo(row['jogador_1'], row['superficie'], df_welo), axis=1)
                df['WELO_J2'] = df.apply(lambda row: get_welo(row['jogador_2'], row['superficie'], df_welo), axis=1)
                df['Dif_WELO'] = abs(df['WELO_J1'] - df['WELO_J2'])
                
                # Calcular linha total
                resultados = df.apply(lambda row: calcular_linha_total(row['WELO_J1'], row['WELO_J2'], row['superficie']), axis=1)
                df['Total_Esperado'] = [r[0] for r in resultados]
                df['Prob_Mais_21.5'] = [r[1] for r in resultados]
                
                st.success(f"✅ {len(df)} partidas analisadas!")
                
                # Mostrar tabela
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
                        "status": "📊 Status",
                        "WELO_J1": st.column_config.NumberColumn("WELO J1", format="%.1f"),
                        "WELO_J2": st.column_config.NumberColumn("WELO J2", format="%.1f"),
                        "Dif_WELO": st.column_config.NumberColumn("Dif WELO", format="%.1f"),
                        "Total_Esperado": st.column_config.NumberColumn("Total Esperado", format="%.2f"),
                        "Prob_Mais_21.5": st.column_config.NumberColumn("Prob >21.5 (%)", format="%.1f"),
                    }
                )
                
                # Botões de download
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📥 CSV", df.to_csv(index=False).encode('utf-8'), 
                                      f"tenis_hoje_total_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    output.seek(0)
                    st.download_button("📊 Excel", output, 
                                      f"tenis_hoje_total_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("⚠️ Nenhuma partida encontrada. Verifique a sua conexão ou tente novamente mais tarde.")
else:
    st.info("📌 Carregue o ficheiro Challenger.xlsm na sidebar e clique no botão para buscar partidas reais de ténis da API.")

st.caption("🎾 Dados via SportScore API • WELO por superfície • Estimativa de Total de Jogos • Probabilidade > 21.5")
