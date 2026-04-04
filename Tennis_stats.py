import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
from io import BytesIO
import time
import json

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Partidas Hoje + Predictor Stats")

tab1, tab2, tab3 = st.tabs(["📅 Partidas Hoje", "🔍 Previsão Personalizada", "📈 Modeling Strategy"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])
    
    # Opção para usar partidas manuais se o scraping falhar
    use_manual = st.checkbox("✏️ Usar entrada manual de partidas", value=False)

# ====================== CARREGAR STATS ======================
@st.cache_data
def load_stats(file):
    if not file:
        return pd.DataFrame()
    try:
        df = pd.read_excel(file)
        def norm(name):
            if not isinstance(name, str): return ""
            n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            return ''.join(filter(str.isalnum, n.lower().strip()))
        df['winner_clean'] = df['winner_name'].apply(norm)
        df['loser_clean'] = df['loser_name'].apply(norm)
        st.sidebar.success(f"✅ {len(df)} jogos carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar stats: {e}")
        return pd.DataFrame()

df_stats = load_stats(uploaded_file)

# ====================== FUNÇÕES AUXILIARES ======================
def norm(name):
    if not isinstance(name, str): return ""
    n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return ''.join(filter(str.isalnum, n.lower().strip()))

def find_best_player_stats(player_name, df):
    if df.empty or not player_name: 
        return pd.Series(dtype='object')
    clean_name = norm(player_name)
    best_match = None
    best_score = 0.0
    for _, row in df.iterrows():
        for col in ['winner_clean', 'loser_clean']:
            clean_db = row.get(col, "")
            if not clean_db: continue
            similarity = SequenceMatcher(None, clean_name, clean_db).ratio()
            if clean_name in clean_db or clean_db in clean_name:
                similarity = max(similarity, 0.95)
            score = similarity * 100
            if score > best_score:
                best_score = score
                best_match = row
    return best_match if best_score >= 60 else pd.Series(dtype='object')

def predict_from_stats(p1_stats, p2_stats, superficie="Hard"):
    def safe(v):
        try: return float(v) if pd.notna(v) else 0.0
        except: return 0.0

    def serve_win(stats):
        svpt = safe(stats.get('w_svpt', 0))
        if svpt == 0: return 0.65
        return (safe(stats.get('w_1stWon', 0)) + safe(stats.get('w_2ndWon', 0))) / svpt

    serve1 = serve_win(p1_stats)
    serve2 = serve_win(p2_stats)
    return1 = 1 - serve2
    return2 = 1 - serve1

    p1_point_win = (serve1 + return1) / 2
    p2_point_win = (serve2 + return2) / 2

    surface_factor = {'Clay': 1.08, 'Hard': 1.0, 'Grass': 0.93, 'Indoor': 1.02}.get(superficie, 1.0)

    diff = (p1_point_win - p2_point_win) * 100
    prob_p1 = 1 / (1 + 10 ** (-diff / 38))

    hold1 = serve1 ** 1.85
    hold2 = serve2 ** 1.85
    break_prob = (1 - hold1 + 1 - hold2) / 2
    games_per_set = 9.6 + 4.2 * break_prob
    total_esperado = round(games_per_set * 2.15 * surface_factor, 2)

    prob_over = max(0.38, min(0.78, 0.5 + (total_esperado - 21.5) * 0.085))

    return {
        "Prob_J1_%": round(prob_p1 * 100, 1),
        "Total_Esperado": total_esperado,
        "Prob_Over_21.5_%": round(prob_over, 1),
        "Prob_Under_21.5_%": round(100 - prob_over, 1),
        "Serve_J1_%": round(serve1 * 100, 1),
        "BP_Saved_J1_%": round(safe(p1_stats.get('w_bpSaved',0)) / max(safe(p1_stats.get('w_bpFaced',1)), 1) * 100, 1),
    }

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'kigali', 'santiago', 'punto cana', 'heilbronn', 'perugia']):
        return 'Clay'
    if any(k in t for k in ['grass', 'birmingham']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'cherbourg']):
        return 'Indoor'
    return 'Hard'

# ====================== NOVA VERSÃO: API SOFASCORE ======================
def get_sofascore_api_matches():
    """Usa a API não oficial do Sofascore para obter partidas"""
    try:
        # Endpoint da API do Sofascore para tênis (partidas ao vivo + agendadas)
        url = "https://api.sofascore.com/api/v1/sport/tennis/events/live-and-upcoming"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "pt-PT,pt;q=0.9"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            
            # Procurar eventos
            events = data.get('events', [])
            
            for event in events[:50]:  # Limitar a 50 partidas
                try:
                    # Extrair informações
                    tournament = event.get('tournament', {}).get('name', 'Desconhecido')
                    home_team = event.get('homeTeam', {}).get('name', '')
                    away_team = event.get('awayTeam', {}).get('name', '')
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Horário (timestamp)
                    start_timestamp = event.get('startTimestamp', 0)
                    if start_timestamp:
                        horario = datetime.fromtimestamp(start_timestamp).strftime('%H:%M')
                    else:
                        horario = '?'
                    
                    # Verificar se é hoje (opcional)
                    event_date = datetime.fromtimestamp(start_timestamp).date() if start_timestamp else None
                    today = datetime.now().date()
                    
                    # Superfície
                    superficie = detect_surface(tournament)
                    
                    matches.append({
                        'torneio': tournament,
                        'jogador_1': home_team,
                        'jogador_2': away_team,
                        'horario': horario,
                        'superficie': superficie,
                        'data': event_date
                    })
                    
                except Exception as e:
                    continue
            
            return pd.DataFrame(matches)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Erro na API: {str(e)[:100]}")
        return pd.DataFrame()

# ====================== ENTRADA MANUAL DE PARTIDAS ======================
def manual_matches_input():
    """Interface para inserir partidas manualmente"""
    st.subheader("📝 Inserir Partidas Manualmente")
    
    num_matches = st.number_input("Número de partidas", min_value=1, max_value=20, value=3)
    
    matches = []
    for i in range(int(num_matches)):
        st.markdown(f"**Partida {i+1}**")
        col1, col2, col3, col4 = st.columns([2,2,1,1])
        
        with col1:
            j1 = st.text_input(f"Jogador A {i+1}", key=f"j1_{i}")
        with col2:
            j2 = st.text_input(f"Jogador B {i+1}", key=f"j2_{i}")
        with col3:
            torneio = st.text_input(f"Torneio {i+1}", "Challenger", key=f"tor_{i}")
        with col4:
            superficie = st.selectbox(f"Superfície {i+1}", ["Hard", "Clay", "Grass", "Indoor"], key=f"sup_{i}")
        
        if j1 and j2:
            matches.append({
                'torneio': torneio,
                'jogador_1': j1,
                'jogador_2': j2,
                'horario': 'Manual',
                'superficie': superficie
            })
    
    return pd.DataFrame(matches) if matches else pd.DataFrame()

# ====================== ABA 1 - PARTIDAS HOJE ======================
with tab1:
    st.header("Partidas de Hoje + Previsão Automática")
    
    # Botão para buscar partidas
    col1, col2 = st.columns([1,3])
    with col1:
        fetch_matches = st.button("🔄 Buscar Partidas (API)", type="primary", use_container_width=True)
    
    df_flash = pd.DataFrame()
    
    if fetch_matches:
        if df_stats.empty:
            st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx na barra lateral.")
        else:
            with st.spinner("A obter partidas da API do Sofascore..."):
                df_flash = get_sofascore_api_matches()
                
                if df_flash.empty:
                    st.warning("⚠️ Não foi possível obter partidas automaticamente.")
                    st.info("💡 Podes usar a opção 'Entrada Manual' no sidebar ou a aba 'Previsão Personalizada'.")
                    
                    # Opção de exemplo
                    st.markdown("---")
                    st.subheader("📋 Exemplo de partidas para teste")
                    
                    example_matches = pd.DataFrame({
                        'torneio': ['Challenger Santiago', 'Challenger Kigali', 'Challenger Perugia'],
                        'jogador_1': ['Joao Sousa', 'Carlos Taberner', 'Federico Coria'],
                        'jogador_2': ['Gastao Elias', 'Pedro Sousa', 'Thiago Monteiro'],
                        'horario': ['15:00', '13:00', '11:00'],
                        'superficie': ['Clay', 'Clay', 'Clay']
                    })
                    
                    if st.button("📊 Usar exemplo para teste"):
                        df_flash = example_matches
                        st.rerun()
                else:
                    st.success(f"✅ {len(df_flash)} partidas encontradas!")
    
    # Se temos partidas (ou por API ou por exemplo)
    if not df_flash.empty:
        with st.spinner("Calculando previsões..."):
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in df_flash.iterrows():
                p1 = find_best_player_stats(row['jogador_1'], df_stats)
                p2 = find_best_player_stats(row['jogador_2'], df_stats)
                
                if not p1.empty and not p2.empty:
                    pred = predict_from_stats(p1, p2, row['superficie'])
                    results.append([
                        pred["Prob_J1_%"], 
                        pred["Total_Esperado"], 
                        pred["Prob_Over_21.5_%"],
                        pred["Serve_J1_%"],
                        pred["BP_Saved_J1_%"]
                    ])
                else:
                    results.append([None, None, None, None, None])
                    if p1.empty:
                        st.warning(f"⚠️ Sem stats para: {row['jogador_1']}")
                    if p2.empty:
                        st.warning(f"⚠️ Sem stats para: {row['jogador_2']}")
                
                progress_bar.progress((idx + 1) / len(df_flash))
                time.sleep(0.1)
            
            df_flash[['Prob_J1_%', 'Total_Esperado', 'Prob_Over_21.5_%', 'Serve_J1_%', 'BP_Saved_J1_%']] = pd.DataFrame(results)
            
            # Formatar para exibição
            display_df = df_flash.copy()
            display_df['Prob_J1_%'] = display_df['Prob_J1_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
            display_df['Prob_Over_21.5_%'] = display_df['Prob_Over_21.5_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Download
            csv = df_flash.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Exportar CSV", 
                csv, 
                f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                "text/csv"
            )
    
    # Entrada manual alternativa
    if use_manual or df_flash.empty:
        st.markdown("---")
        manual_df = manual_matches_input()
        
        if st.button("📊 Calcular Previsões para Partidas Manuais", type="secondary"):
            if not manual_df.empty:
                with st.spinner("Calculando..."):
                    results = []
                    for _, row in manual_df.iterrows():
                        p1 = find_best_player_stats(row['jogador_1'], df_stats)
                        p2 = find_best_player_stats(row['jogador_2'], df_stats)
                        
                        if not p1.empty and not p2.empty:
                            pred = predict_from_stats(p1, p2, row['superficie'])
                            results.append([pred["Prob_J1_%"], pred["Total_Esperado"], pred["Prob_Over_21.5_%"]])
                        else:
                            results.append([None, None, None])
                    
                    manual_df[['Prob_J1_%', 'Total_Esperado', 'Prob_Over_21.5_%']] = pd.DataFrame(results)
                    st.dataframe(manual_df, use_container_width=True, hide_index=True)

# ====================== ABA 2 - PREVISÃO PERSONALIZADA ======================
with tab2:
    st.header("🔍 Previsão Personalizada")

    if df_stats.empty:
        st.info("Carregue o ficheiro Challenger1.xlsx")
    else:
        player_list = pd.concat([df_stats['winner_name'], df_stats['loser_name']]).drop_duplicates().sort_values().tolist()

        col1, col2 = st.columns(2)
        with col1:
            jogador_a = st.selectbox("Jogador A", options=player_list, key="ja")
        with col2:
            jogador_b = st.selectbox("Jogador B", options=player_list, key="jb")

        superficie = st.selectbox("Superfície", ["Hard", "Clay", "Grass", "Indoor"])

        if st.button("Calcular Previsão", type="primary"):
            if jogador_a == jogador_b:
                st.error("Escolha jogadores diferentes!")
            else:
                p1 = find_best_player_stats(jogador_a, df_stats)
                p2 = find_best_player_stats(jogador_b, df_stats)

                if p1.empty or p2.empty:
                    st.error("Não foi possível encontrar stats para um dos jogadores.")
                else:
                    result = predict_from_stats(p1, p2, superficie)
                    st.success("Previsão Calculada!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{jogador_a} vence", f"{result['Prob_J1_%']}%")
                        st.metric(f"{jogador_b} vence", f"{100 - result['Prob_J1_%']}%")
                    with col2:
                        st.metric("Total Esperado", f"{result['Total_Esperado']} jogos")
                        st.metric("Over 21.5", f"{result['Prob_Over_21.5_%']}%")
                    with col3:
                        st.metric("Under 21.5", f"{result['Prob_Under_21.5_%']}%")
                        st.metric("Serve % Jogador A", f"{result['Serve_J1_%']}%")

# ====================== ABA 3 - MODELING STRATEGY ======================
with tab3:
    st.header("📈 Recommended Modeling Strategy")
    st.markdown("""
    ### Estratégia Recomendada

    1. **Feature Engineering**
       - Rank Difference
       - Average Total Games
       - Serve % e Return % por superfície

    2. **Modelo Híbrido**
       - Vitória → XGBoost / Logistic Regression
       - Total Jogos → Markov Chain Simulation

    **Melhor prática atual:**
    Machine Learning para vencedor + **Markov Chains** para Total de Jogos.
    
    ### Limitações conhecidas
    - O scraping direto pode falhar devido a bloqueios
    - Usar a API do Sofascore (não oficial) é mais estável
    - Para produção, considerar usar dados de APIs pagas como Tennis Data API ou Tennis Abstract
    """)

st.caption("Versão com API Sofascore + Entrada Manual • Baseado no teu ficheiro Challenger1.xlsx")
