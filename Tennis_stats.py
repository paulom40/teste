import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
import time
import random
import json

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Partidas Hoje + Predictor Stats")

tab1, tab2, tab3 = st.tabs(["📅 Partidas Hoje", "🔍 Previsão Personalizada", "📈 Modeling Strategy"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])
    
    st.markdown("---")
    st.header("⚙️ Configurações")
    
    use_manual = st.checkbox("✏️ Usar entrada manual de partidas", value=False)
    use_stealth = st.checkbox("🕵️ Modo Stealth (headers rotativos)", value=True)

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

def get_random_headers():
    """Headers rotativos sem dependências externas"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1'
    ]
    
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'pt-PT,pt;q=0.9,en;q=0.8,en-US;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    }

def random_delay():
    """Delay aleatório para evitar detecção"""
    time.sleep(random.uniform(1.5, 3.5))

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
        "Prob_Over_21.5_%": round(prob_over * 100, 1),
        "Prob_Under_21.5_%": round((1 - prob_over) * 100, 1),
        "Serve_J1_%": round(serve1 * 100, 1),
        "BP_Saved_J1_%": round(safe(p1_stats.get('w_bpSaved',0)) / max(safe(p1_stats.get('w_bpFaced',1)), 1) * 100, 1),
    }

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'kigali', 'santiago', 'heilbronn', 'perugia', 'barletta', 'rome']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon', 'birmingham']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'cherbourg', 'bergerac']):
        return 'Indoor'
    return 'Hard'

# ====================== SCRAPING MELHORADO ======================
def get_matches_from_soft tennis():
    """Tenta obter partidas de múltiplas fontes"""
    
    # Lista de URLs para tentar
    urls = [
        "https://www.sofascore.com/tennis",
        "https://www.flashscore.com/tennis/",
        "https://www.atptour.com/en/scores/current-results",
        "https://www.tennis.com.au/tournaments/"
    ]
    
    all_matches = []
    
    for url in urls:
        try:
            if use_stealth:
                headers = get_random_headers()
            else:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            response = requests.get(url, headers=headers, timeout=15)
            random_delay()
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Procurar por padrões de nomes de jogadores
                text = soup.get_text()
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                # Procurar por padrões de partidas (ex: "Player Name vs Player Name")
                matches_found = []
                for i, line in enumerate(lines):
                    if ' vs ' in line.lower() or ' v ' in line.lower():
                        parts = line.split(' vs ') if ' vs ' in line.lower() else line.split(' v ')
                        if len(parts) == 2:
                            p1 = parts[0].strip()
                            p2 = parts[1].strip()
                            if len(p1) > 2 and len(p2) > 2 and not any(x in p1.lower() for x in ['http', 'www', 'click']):
                                matches_found.append((p1, p2))
                
                # Também procurar por nomes próximos no texto
                for i in range(len(lines) - 2):
                    line1 = lines[i]
                    line2 = lines[i+1]
                    if len(line1.split()) in [2,3] and len(line2.split()) in [2,3]:
                        if len(line1) > 3 and len(line2) > 3 and not any(x in line1.lower() for x in ['final', 'quarter', 'semi']):
                            matches_found.append((line1, line2))
                
                # Adicionar matches encontrados
                for p1, p2 in matches_found[:20]:  # Limitar a 20 partidas
                    all_matches.append({
                        'torneio': f"Torneio {len(all_matches)+1}",
                        'jogador_1': p1,
                        'jogador_2': p2,
                        'horario': datetime.now().strftime('%H:%M'),
                        'superficie': 'Hard'
                    })
                
                if all_matches:
                    st.info(f"✅ Encontradas {len(all_matches)} partidas em {url}")
                    break
                    
        except Exception as e:
            continue
    
    # Remover duplicatas
    if all_matches:
        df = pd.DataFrame(all_matches)
        df = df.drop_duplicates(subset=['jogador_1', 'jogador_2'])
        return df
    
    return pd.DataFrame()

# ====================== FALLBACK: PARTIDAS PRÉ-DEFINIDAS ======================
def get_predefined_matches():
    """Partidas pré-definidas para quando o scraping falha"""
    return pd.DataFrame({
        'torneio': [
            'Challenger Santiago', 
            'Challenger Kigali', 
            'Challenger Perugia',
            'ATP Buenos Aires',
            'Challenger Phoenix'
        ],
        'jogador_1': [
            'Joao Sousa',
            'Carlos Taberner',
            'Federico Coria',
            'Sebastian Baez',
            'Alejandro Tabilo'
        ],
        'jogador_2': [
            'Gastao Elias',
            'Pedro Sousa',
            'Thiago Monteiro',
            'Cameron Norrie',
            'Roman Safiullin'
        ],
        'horario': [
            '15:00', '13:00', '11:00', '17:00', '19:00'
        ],
        'superficie': [
            'Clay', 'Clay', 'Clay', 'Clay', 'Hard'
        ]
    })

# ====================== ENTRADA MANUAL ======================
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
    
    # Botões
    col1, col2, col3 = st.columns(3)
    with col1:
        fetch_matches = st.button("🔄 Buscar Partidas (Web)", type="primary", use_container_width=True)
    with col2:
        use_example = st.button("📋 Usar Partidas Exemplo", use_container_width=True)
    with col3:
        clear = st.button("🗑️ Limpar", use_container_width=True)
    
    # Inicializar session state para matches
    if 'current_matches' not in st.session_state:
        st.session_state.current_matches = pd.DataFrame()
    
    if fetch_matches:
        if df_stats.empty:
            st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx na barra lateral.")
        else:
            with st.spinner("Buscando partidas da web..."):
                df_matches = get_matches_from_soft tennis()
                
                if df_matches.empty:
                    st.warning("⚠️ Não foi possível obter partidas automaticamente.")
                    st.info("💡 Use uma das opções abaixo:\n- Clique em 'Usar Partidas Exemplo'\n- Ative 'Entrada Manual' no sidebar\n- Insira partidas manualmente abaixo")
                    
                    # Mostrar opção de exemplo
                    if st.button("📊 Carregar exemplo agora"):
                        st.session_state.current_matches = get_predefined_matches()
                        st.rerun()
                else:
                    st.session_state.current_matches = df_matches
                    st.success(f"✅ {len(df_matches)} partidas encontradas!")
                    st.rerun()
    
    if use_example:
        st.session_state.current_matches = get_predefined_matches()
        st.success("✅ Exemplo carregado!")
        st.rerun()
    
    if clear:
        st.session_state.current_matches = pd.DataFrame()
        st.rerun()
    
    # Processar e mostrar partidas
    if not st.session_state.current_matches.empty:
        df_matches = st.session_state.current_matches.copy()
        
        with st.spinner("Calculando previsões..."):
            results = []
            progress_bar = st.progress(0)
            
            # Detectar superfície se necessário
            if 'superficie' not in df_matches.columns:
                df_matches['superficie'] = df_matches['torneio'].apply(detect_surface)
            
            for idx, row in df_matches.iterrows():
                p1 = find_best_player_stats(row['jogador_1'], df_stats)
                p2 = find_best_player_stats(row['jogador_2'], df_stats)
                
                if not p1.empty and not p2.empty:
                    pred = predict_from_stats(p1, p2, row.get('superficie', 'Hard'))
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
                
                progress_bar.progress((idx + 1) / len(df_matches))
                time.sleep(0.05)
            
            df_matches[['Prob_J1_%', 'Total_Esperado', 'Prob_Over_21.5_%', 'Serve_J1_%', 'BP_Saved_J1_%']] = pd.DataFrame(results)
            
            # Formatar para exibição
            display_df = df_matches.copy()
            display_df['Prob_J1_%'] = display_df['Prob_J1_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
            display_df['Prob_Over_21.5_%'] = display_df['Prob_Over_21.5_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
            display_df['Serve_J1_%'] = display_df['Serve_J1_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Download
            csv = df_matches.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Exportar CSV", 
                csv, 
                f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                "text/csv",
                use_container_width=True
            )
    
    # Modo manual (quando ativado no sidebar)
    if use_manual or st.session_state.current_matches.empty:
        st.markdown("---")
        manual_df = manual_matches_input()
        
        if st.button("📊 Calcular Previsões Manuais", type="secondary", use_container_width=True):
            if not manual_df.empty and not df_stats.empty:
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
                    
                    # Download manual
                    csv_manual = manual_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Exportar CSV Manual", 
                        csv_manual, 
                        f"previsoes_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                        "text/csv"
                    )

# ====================== ABA 2 - PREVISÃO PERSONALIZADA ======================
with tab2:
    st.header("🔍 Previsão Personalizada")

    if df_stats.empty:
        st.info("Carregue o ficheiro Challenger1.xlsx")
    else:
        # Lista de jogadores
        all_players = pd.concat([df_stats['winner_name'], df_stats['loser_name']]).drop_duplicates().sort_values()
        
        # Busca rápida
        search_a = st.text_input("🔎 Buscar Jogador A", placeholder="Digite o nome...")
        search_b = st.text_input("🔎 Buscar Jogador B", placeholder="Digite o nome...")
        
        # Filtrar lista
        if search_a:
            player_list_a = [p for p in all_players if search_a.lower() in p.lower()]
        else:
            player_list_a = all_players.tolist()[:100]
            
        if search_b:
            player_list_b = [p for p in all_players if search_b.lower() in p.lower()]
        else:
            player_list_b = all_players.tolist()[:100]
        
        col1, col2 = st.columns(2)
        with col1:
            jogador_a = st.selectbox("Jogador A", options=player_list_a, key="ja")
        with col2:
            jogador_b = st.selectbox("Jogador B", options=player_list_b, key="jb")

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
                        st.metric(f"🏆 {jogador_a} vence", f"{result['Prob_J1_%']}%")
                        st.metric(f"🏆 {jogador_b} vence", f"{100 - result['Prob_J1_%']}%")
                    with col2:
                        st.metric("📊 Total Esperado", f"{result['Total_Esperado']} jogos")
                        st.metric("📈 Over 21.5", f"{result['Prob_Over_21.5_%']}%")
                    with col3:
                        st.metric("📉 Under 21.5", f"{result['Prob_Under_21.5_%']}%")
                        st.metric("🎾 Serve %", f"{result['Serve_J1_%']}%")

# ====================== ABA 3 - MODELING STRATEGY ======================
with tab3:
    st.header("📈 Recommended Modeling Strategy")
    
    st.markdown("""
    ### 🎯 Estratégia Atual
    
    **Modelo de Probabilidades Baseado em:**
    - Percentual de pontos de saque
    - Percentual de pontos de retorno
    - Fator de correção por superfície
    - Elo rating implícito
    
    ### 📊 Features Utilizadas
    
    1. **Serve Performance**
       - w_1stWon (Primeiros saques ganhos)
       - w_2ndWon (Segundos saques ganhos)
       - w_svpt (Total pontos de saque)
    
    2. **Return Performance**
       - w_1stWon (derivado do adversário)
       - Break points saved/converted
    
    3. **Surface Adjustment**
       - Clay: +8% fator correção
       - Grass: -7% fator correção
       - Indoor: +2% fator correção
    
    ### 🔄 Próximos Passos
    
    1. **Adicionar ranking ATP**
    2. **Histórico de confrontos diretos**
    3. **Forma recente (últimos 5 jogos)**
    4. **Modelo de Machine Learning (XGBoost)**
    
    ### 💡 Como Melhorar as Previsões
    
    - Carregue mais dados históricos no Excel
    - Inclua estatísticas por superfície
    - Adicione colunas: `surface`, `round`, `best_of`
    """)

st.caption("🎾 Tênis Predictor Pro v2.0 • Sem dependências externas • Funciona no Streamlit Cloud")
