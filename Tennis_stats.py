import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
import time
import random
import re

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
def is_valid_name(name):
    """Verifica se o nome é válido (não contém caracteres estranhos)"""
    if not isinstance(name, str) or len(name) < 2:
        return False
    
    # Verificar se tem caracteres imprimíveis válidos
    try:
        # Contar caracteres alfanuméricos válidos
        valid_chars = sum(1 for c in name if c.isalnum() or c.isspace() or c in '.-')
        ratio = valid_chars / len(name) if len(name) > 0 else 0
        
        # Nome válido se pelo menos 70% dos caracteres são normais
        if ratio < 0.7:
            return False
            
        # Verificar se não tem muitos caracteres especiais estranhos
        if re.search(r'[^\w\s\.\-áéíóúãõçÀÈÍÓÚÂÊÔÃÕÇ]', name):
            return False
            
        return True
    except:
        return False

def clean_player_name(name):
    """Limpa e normaliza nome do jogador"""
    if not isinstance(name, str):
        return ""
    
    # Remover caracteres não imprimíveis
    name = ''.join(char for char in name if char.isprintable())
    
    # Remover caracteres estranhos comuns em scraping corrompido
    name = re.sub(r'[^\w\s\.\-áéíóúãõçÀÈÍÓÚÂÊÔÃÕÇ]', '', name)
    
    # Limitar a 50 caracteres
    name = name[:50]
    
    return name.strip()

def norm(name):
    if not isinstance(name, str): return ""
    n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return ''.join(filter(str.isalnum, n.lower().strip()))

def get_random_headers():
    """Headers rotativos"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
    ]
    
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'pt-PT,pt;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }

def find_best_player_stats(player_name, df):
    if df.empty or not player_name or not is_valid_name(player_name): 
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
            
            if similarity > best_score:
                best_score = similarity
                best_match = row
    
    return best_match if best_score >= 0.6 else pd.Series(dtype='object')

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
    if any(k in t for k in ['clay', 'saibro', 'kigali', 'santiago', 'heilbronn', 'perugia']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon', 'birmingham']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'cherbourg']):
        return 'Indoor'
    return 'Hard'

# ====================== PARTIDAS PRÉ-DEFINIDAS (RECOMENDADO) ======================
def get_predefined_matches():
    """Partidas pré-definidas com nomes válidos"""
    return pd.DataFrame({
        'torneio': [
            'Challenger Santiago', 
            'Challenger Kigali', 
            'Challenger Perugia',
            'ATP Buenos Aires',
            'Challenger Phoenix',
            'Challenger Rome',
            'ATP Santiago',
            'Challenger Barletta'
        ],
        'jogador_1': [
            'Joao Sousa',
            'Carlos Taberner',
            'Federico Coria',
            'Sebastian Baez',
            'Alejandro Tabilo',
            'Thiago Monteiro',
            'Camilo Ugo Carabelli',
            'Francisco Cerundolo'
        ],
        'jogador_2': [
            'Gastao Elias',
            'Pedro Sousa',
            'Thiago Monteiro',
            'Cameron Norrie',
            'Roman Safiullin',
            'Juan Manuel Cerundolo',
            'Tomas Martin Etcheverry',
            'Facundo Diaz Acosta'
        ],
        'horario': [
            '15:00', '13:00', '11:00', '17:00', '19:00', '14:00', '16:00', '12:00'
        ],
        'superficie': [
            'Clay', 'Clay', 'Clay', 'Clay', 'Hard', 'Clay', 'Clay', 'Clay'
        ]
    })

# ====================== ENTRADA MANUAL ======================
def manual_matches_input():
    """Interface para inserir partidas manualmente"""
    st.subheader("📝 Inserir Partidas Manualmente")
    
    st.info("💡 Dica: Use nomes exatamente como aparecem no seu ficheiro Excel")
    
    num_matches = st.number_input("Número de partidas", min_value=1, max_value=20, value=3)
    
    matches = []
    for i in range(int(num_matches)):
        st.markdown(f"**Partida {i+1}**")
        col1, col2, col3, col4 = st.columns([2,2,1,1])
        
        with col1:
            j1 = st.text_input(f"Jogador A {i+1}", key=f"j1_{i}", placeholder="Ex: Joao Sousa")
        with col2:
            j2 = st.text_input(f"Jogador B {i+1}", key=f"j2_{i}", placeholder="Ex: Gastao Elias")
        with col3:
            torneio = st.text_input(f"Torneio {i+1}", "Challenger", key=f"tor_{i}")
        with col4:
            superficie = st.selectbox(f"Superfície {i+1}", ["Hard", "Clay", "Grass", "Indoor"], key=f"sup_{i}")
        
        if j1 and j2 and is_valid_name(j1) and is_valid_name(j2):
            matches.append({
                'torneio': torneio,
                'jogador_1': clean_player_name(j1),
                'jogador_2': clean_player_name(j2),
                'horario': 'Manual',
                'superficie': superficie
            })
        elif j1 or j2:
            st.warning(f"⚠️ Partida {i+1}: Nomes inválidos ou vazios")
    
    return pd.DataFrame(matches) if matches else pd.DataFrame()

# ====================== SCRAPING SIMPLES (OPCIONAL) ======================
def get_matches_simple():
    """Versão simples de scraping - pode ser desativada se der problemas"""
    try:
        url = "https://www.atptour.com/en/scores/current-results"
        headers = get_random_headers()
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            matches = []
            
            # Procurar por texto que parece nome de jogador
            text = soup.get_text()
            # Padrão de nome: duas palavras com iniciais maiúsculas
            name_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            names = re.findall(name_pattern, text)
            
            # Pegar pares de nomes próximos
            for i in range(0, len(names)-1, 2):
                if i+1 < len(names):
                    p1 = names[i]
                    p2 = names[i+1]
                    if len(p1) > 3 and len(p2) > 3 and p1 != p2:
                        if is_valid_name(p1) and is_valid_name(p2):
                            matches.append({
                                'torneio': 'ATP Tour',
                                'jogador_1': p1,
                                'jogador_2': p2,
                                'horario': datetime.now().strftime('%H:%M'),
                                'superficie': 'Hard'
                            })
            
            if matches:
                return pd.DataFrame(matches[:10])
        
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# ====================== ABA 1 - PARTIDAS HOJE ======================
with tab1:
    st.header("Partidas de Hoje + Previsão Automática")
    
    st.info("🎾 **Recomendação:** Use as 'Partidas Exemplo' ou 'Entrada Manual' para melhores resultados")
    
    # Botões
    col1, col2, col3 = st.columns(3)
    with col1:
        use_example = st.button("📋 Usar Partidas Exemplo", type="primary", use_container_width=True)
    with col2:
        fetch_matches = st.button("🔄 Tentar Scraping Web", use_container_width=True)
    with col3:
        clear = st.button("🗑️ Limpar", use_container_width=True)
    
    # Inicializar session state
    if 'current_matches' not in st.session_state:
        st.session_state.current_matches = pd.DataFrame()
    
    if use_example:
        st.session_state.current_matches = get_predefined_matches()
        st.success("✅ Partidas exemplo carregadas!")
        st.rerun()
    
    if fetch_matches:
        if df_stats.empty:
            st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx na barra lateral.")
        else:
            with st.spinner("Tentando obter partidas da web..."):
                df_matches = get_matches_simple()
                
                if df_matches.empty:
                    st.warning("⚠️ Não foi possível obter partidas da web.")
                    st.info("💡 Use 'Partidas Exemplo' ou ative 'Entrada Manual' no sidebar")
                else:
                    # Filtrar nomes válidos
                    df_matches = df_matches[
                        df_matches['jogador_1'].apply(is_valid_name) & 
                        df_matches['jogador_2'].apply(is_valid_name)
                    ]
                    
                    if not df_matches.empty:
                        st.session_state.current_matches = df_matches
                        st.success(f"✅ {len(df_matches)} partidas encontradas!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Nomes encontrados são inválidos")
    
    if clear:
        st.session_state.current_matches = pd.DataFrame()
        st.rerun()
    
    # Processar e mostrar partidas
    if not st.session_state.current_matches.empty:
        df_matches = st.session_state.current_matches.copy()
        
        # Limpar nomes
        df_matches['jogador_1'] = df_matches['jogador_1'].apply(clean_player_name)
        df_matches['jogador_2'] = df_matches['jogador_2'].apply(clean_player_name)
        
        # Remover linhas com nomes vazios
        df_matches = df_matches[df_matches['jogador_1'].str.len() > 2]
        df_matches = df_matches[df_matches['jogador_2'].str.len() > 2]
        
        if df_matches.empty:
            st.warning("⚠️ Nenhuma partida válida após limpeza")
        else:
            with st.spinner("Calculando previsões..."):
                results = []
                progress_bar = st.progress(0)
                
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
                        # Mostrar apenas se o nome parece válido
                        if is_valid_name(row['jogador_1']) and len(row['jogador_1']) > 2:
                            st.warning(f"⚠️ Sem stats para: {row['jogador_1']}")
                        if is_valid_name(row['jogador_2']) and len(row['jogador_2']) > 2:
                            st.warning(f"⚠️ Sem stats para: {row['jogador_2']}")
                    
                    progress_bar.progress((idx + 1) / len(df_matches))
                    time.sleep(0.05)
                
                df_matches[['Prob_J1_%', 'Total_Esperado', 'Prob_Over_21.5_%', 'Serve_J1_%', 'BP_Saved_J1_%']] = pd.DataFrame(results)
                
                # Formatar para exibição
                display_df = df_matches.copy()
                display_df['Prob_J1_%'] = display_df['Prob_J1_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                display_df['Prob_Over_21.5_%'] = display_df['Prob_Over_21.5_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                
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
    
    # Modo manual
    if use_manual:
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
                            if not p1.empty:
                                st.warning(f"⚠️ Sem stats para: {row['jogador_1']}")
                            if not p2.empty:
                                st.warning(f"⚠️ Sem stats para: {row['jogador_2']}")
                    
                    manual_df[['Prob_J1_%', 'Total_Esperado', 'Prob_Over_21.5_%']] = pd.DataFrame(results)
                    st.dataframe(manual_df, use_container_width=True, hide_index=True)

# ====================== ABA 2 - PREVISÃO PERSONALIZADA ======================
with tab2:
    st.header("🔍 Previsão Personalizada")

    if df_stats.empty:
        st.info("📁 Carregue o ficheiro Challenger1.xlsx na barra lateral")
    else:
        # Lista de jogadores do Excel
        all_players = pd.concat([df_stats['winner_name'], df_stats['loser_name']]).drop_duplicates().sort_values().tolist()
        
        st.success(f"✅ {len(all_players)} jogadores disponíveis no seu ficheiro")
        
        col1, col2 = st.columns(2)
        with col1:
            jogador_a = st.selectbox("Jogador A", options=all_players, key="ja")
        with col2:
            jogador_b = st.selectbox("Jogador B", options=all_players, key="jb")

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
    ### 🎯 Modelo de Probabilidades
    
    **Baseado em:**
    - Percentual de pontos de saque
    - Percentual de pontos de retorno  
    - Fator de correção por superfície
    
    ### 💡 Recomendações
    
    1. **Use a aba 'Previsão Personalizada'** para melhores resultados
    2. **Carregue mais dados** no seu Excel para melhor precisão
    3. **Use 'Partidas Exemplo'** para testar o sistema
    
    ### 📊 Próximas Melhorias
    
    - Adicionar ranking ATP
    - Histórico de confrontos diretos
    - Forma recente (últimos 5 jogos)
    """)

st.caption("🎾 Tênis Predictor Pro v2.1 • Validação de nomes • Recomendado usar Partidas Exemplo")
