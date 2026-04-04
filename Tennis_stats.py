import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
import time

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Partidas Hoje + Predictor Stats")

tab1, tab2, tab3 = st.tabs(["📅 Partidas Hoje", "🔍 Previsão Personalizada", "📈 Modeling Strategy"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])
    
    st.markdown("---")
    st.caption("Dados de partidas obtidos via API do Sofascore")

# ====================== CARREGAR STATS ======================
@st.cache_data
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
        st.sidebar.success(f"✅ {len(df)} jogos carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro: {e}")
        return pd.DataFrame()

df_stats = load_stats(uploaded_file)

# ====================== FUNÇÕES AUXILIARES ======================
def norm(name):
    if not isinstance(name, str): 
        return ""
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
            if not clean_db: 
                continue
            similarity = SequenceMatcher(None, clean_name, clean_db).ratio()
            if clean_name in clean_db or clean_db in clean_name:
                similarity = max(similarity, 0.95)
            if similarity > best_score:
                best_score = similarity
                best_match = row
    return best_match if best_score >= 0.6 else pd.Series(dtype='object')

def predict_from_stats(p1_stats, p2_stats, superficie="Hard"):
    def safe(v):
        try: 
            return float(v) if pd.notna(v) else 0.0
        except: 
            return 0.0

    def serve_win(stats):
        svpt = safe(stats.get('w_svpt', 0))
        if svpt == 0: 
            return 0.65
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
        "Prob_Over_21.5_%": round(prob_over * 100, 1),
        "Prob_Under_21.5_%": round((1 - prob_over) * 100, 1),
        "Serve_J1_%": round(serve1 * 100, 1),
        "BP_Saved_J1_%": round(safe(p1_stats.get('w_bpSaved',0)) / max(safe(p1_stats.get('w_bpFaced',1)), 1) * 100, 1),
    }

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'barletta', 'marrakech', 'monte-carlo', 'bucarest', 'houston']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon']):
        return 'Grass'
    if any(k in t for k in ['indoor']):
        return 'Indoor'
    return 'Hard'

# ====================== API SOFASCORE ======================
def get_matches_from_sofascore():
    """Obtém partidas de tênis de hoje via API do Sofascore"""
    try:
        url = "https://api.sofascore.com/api/v1/sport/tennis/events/live-and-upcoming"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            
            events = data.get('events', [])
            
            for event in events:
                try:
                    tournament = event.get('tournament', {}).get('name', 'Torneio')
                    home_team = event.get('homeTeam', {}).get('name', '')
                    away_team = event.get('awayTeam', {}).get('name', '')
                    
                    if not home_team or not away_team:
                        continue
                    
                    start_timestamp = event.get('startTimestamp', 0)
                    if start_timestamp:
                        horario = datetime.fromtimestamp(start_timestamp).strftime('%H:%M')
                    else:
                        horario = '?'
                    
                    status = event.get('status', {}).get('description', 'Agendado')
                    
                    if status not in ['Ended', 'Canceled']:
                        superficie = detect_surface(tournament)
                        matches.append({
                            'torneio': tournament,
                            'jogador_1': home_team,
                            'jogador_2': away_team,
                            'horario': horario,
                            'status': status,
                            'superficie': superficie
                        })
                except Exception as e:
                    continue
            
            if matches:
                return pd.DataFrame(matches[:30])
        
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Erro na API: {str(e)[:100]}")
        return pd.DataFrame()

# ====================== PARTIDAS FALLBACK ======================
def get_fallback_matches():
    """Partidas de hoje (4 abril 2026)"""
    return pd.DataFrame([
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Vit Kopriva', 'jogador_2': 'Matteo Arnaldi', 'horario': '11:00', 'superficie': 'Clay', 'status': 'Qualifying'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Alexander Shevchenko', 'jogador_2': 'Andrea Pellegrino', 'horario': '11:00', 'superficie': 'Clay', 'status': 'Qualifying'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Francesco Maestrelli', 'jogador_2': 'Alexander Blockx', 'horario': '11:00', 'superficie': 'Clay', 'status': 'Qualifying'},
        {'torneio': 'ATP 250 Marrakech', 'jogador_1': 'Luciano Darderi', 'jogador_2': 'Marco Trungelliti', 'horario': '14:00', 'superficie': 'Clay', 'status': 'Semifinal'},
        {'torneio': 'ATP 250 Marrakech', 'jogador_1': 'Rafael Jodar', 'jogador_2': 'Camilo Ugo Carabelli', 'horario': '16:30', 'superficie': 'Clay', 'status': 'Semifinal'},
        {'torneio': 'ATP 250 Bucharest', 'jogador_1': 'Mariano Navone', 'jogador_2': 'Botic Van De Zandschulp', 'horario': '14:00', 'superficie': 'Clay', 'status': 'Semifinal'},
        {'torneio': 'ATP 250 Bucharest', 'jogador_1': 'Fabian Marozsan', 'jogador_2': 'Daniel Merida', 'horario': '16:00', 'superficie': 'Clay', 'status': 'Semifinal'},
        {'torneio': 'ATP 250 Houston', 'jogador_1': 'Tommy Paul', 'jogador_2': 'Frances Tiafoe', 'horario': '22:00', 'superficie': 'Clay', 'status': 'Semifinal'},
        {'torneio': 'ATP 250 Houston', 'jogador_1': 'Thiago Tirante', 'jogador_2': 'Roman Burruchaga', 'horario': '03:00+1', 'superficie': 'Clay', 'status': 'Semifinal'},
        {'torneio': 'Challenger Barletta', 'jogador_1': 'Michele Ribecai', 'jogador_2': 'Mili Poljicak', 'horario': '10:00', 'superficie': 'Clay', 'status': 'Quarterfinal'},
        {'torneio': 'Challenger Barletta', 'jogador_1': 'Enrico Dalla Valle', 'jogador_2': 'Lukas Neumayer', 'horario': '10:00', 'superficie': 'Clay', 'status': 'Quarterfinal'},
        {'torneio': 'Challenger Sao Leopoldo', 'jogador_1': 'Paulo Andre Saraiva', 'jogador_2': 'Facundo Diaz Acosta', 'horario': '14:00', 'superficie': 'Clay', 'status': 'Round of 16'},
    ])

# ====================== ABA 1 - PARTIDAS HOJE ======================
with tab1:
    st.header("📅 Partidas de Tênis - 4 de Abril 2026")
    
    if df_stats.empty:
        st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx na barra lateral.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            use_api = st.button("🔄 Buscar Partidas (API Sofascore)", type="primary", use_container_width=True)
        with col2:
            use_fallback = st.button("📋 Usar Partidas de Hoje (Fallback)", use_container_width=True)
        
        matches_df = pd.DataFrame()
        
        if use_api:
            with st.spinner("Buscando partidas da API do Sofascore..."):
                matches_df = get_matches_from_sofascore()
                if matches_df.empty:
                    st.warning("⚠️ API não retornou dados. Usando fallback...")
                    matches_df = get_fallback_matches()
                else:
                    st.success(f"✅ {len(matches_df)} partidas encontradas via API!")
        
        if use_fallback:
            matches_df = get_fallback_matches()
            st.success(f"✅ {len(matches_df)} partidas carregadas (dados de 04/04/2026)")
        
        if not matches_df.empty:
            with st.spinner("Calculando previsões..."):
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in matches_df.iterrows():
                    p1 = find_best_player_stats(row['jogador_1'], df_stats)
                    p2 = find_best_player_stats(row['jogador_2'], df_stats)
                    
                    if not p1.empty and not p2.empty:
                        pred = predict_from_stats(p1, p2, row['superficie'])
                        results.append([pred["Prob_J1_%"], pred["Total_Esperado"], pred["Prob_Over_21.5_%"], pred["Serve_J1_%"]])
                    else:
                        results.append([None, None, None, None])
                        if p1.empty and row['jogador_1']:
                            st.warning(f"⚠️ Sem stats: {row['jogador_1']}")
                        if p2.empty and row['jogador_2']:
                            st.warning(f"⚠️ Sem stats: {row['jogador_2']}")
                    
                    progress_bar.progress((idx + 1) / len(matches_df))
                    time.sleep(0.05)
                
                matches_df[['Prob_J1_%', 'Total_Esperado', 'Prob_Over_21.5_%', 'Serve_J1_%']] = pd.DataFrame(results)
                
                # Formatar exibição
                display_df = matches_df.copy()
                display_df['Prob_J1_%'] = display_df['Prob_J1_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                display_df['Prob_Over_21.5_%'] = display_df['Prob_Over_21.5_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Exportar
                csv = matches_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Exportar CSV", csv, f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
        else:
            st.info("👆 Clique em um dos botões acima para carregar as partidas de hoje")

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
                    st.error("Stats não encontrados para um dos jogadores.")
                else:
                    result = predict_from_stats(p1, p2, superficie)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{jogador_a} vence", f"{result['Prob_J1_%']}%")
                        st.metric(f"{jogador_b} vence", f"{100 - result['Prob_J1_%']}%")
                    with col2:
                        st.metric("Total Esperado", f"{result['Total_Esperado']} jogos")
                        st.metric("Over 21.5", f"{result['Prob_Over_21.5_%']}%")
                    with col3:
                        st.metric("Under 21.5", f"{result['Prob_Under_21.5_%']}%")
                        st.metric("Serve %", f"{result['Serve_J1_%']}%")

# ====================== ABA 3 - MODELING STRATEGY ======================
with tab3:
    st.header("📈 Sobre o Modelo")
    st.markdown("""
    ### 🎯 Como as Previsões são Calculadas
    
    **Fórmula de Probabilidade (Baseada em Elo):**
    
    P = 1 / (1 + 10^(-diff/38))
    
    **Fatores Considerados:**
    - Percentual de pontos de saque
    - Percentual de pontos de retorno  
    - Fator superfície (Clay +8%, Grass -7%)
    
    ### 📊 Partidas de Hoje (04/04/2026)
    
    - ATP Monte-Carlo Masters - Qualifying (Terra Batida)
    - ATP 250 Marrakech - Semifinais
    - ATP 250 Bucharest - Semifinais
    - ATP 250 Houston - Semifinais
    - Challenger Barletta - Quartas
    - Challenger Sao Leopoldo - Oitavas
    
    ### 🔗 Fontes de Dados
    
    - API do Sofascore (tempo real)
    - Dados históricos do seu Excel
    """)

st.caption("🎾 Tênis Predictor Pro • Dados de 04/04/2026 • API Sofascore")
