import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
from io import BytesIO
import time
import random
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from fake_useragent import UserAgent
import cloudscraper

# Configuração da página
st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Partidas Hoje + Predictor Stats")

tab1, tab2, tab3 = st.tabs(["📅 Partidas Hoje", "🔍 Previsão Personalizada", "📈 Modeling Strategy"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])
    
    st.markdown("---")
    st.header("⚙️ Configurações de Scraping")
    
    scraping_method = st.selectbox(
        "Método de scraping",
        ["Auto (Recomendado)", "API Sofascore", "Selenium (Mais Robusto)", "CloudScraper", "Manual"]
    )
    
    use_proxy = st.checkbox("Usar rotação de User-Agent", value=True)
    delay_range = st.slider("Delay entre requests (segundos)", 1, 10, (2, 5))

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
        
        # Calcular estatísticas adicionais
        df['total_games'] = df.get('w_svpt', 0) + df.get('l_svpt', 0)
        df['avg_games_per_match'] = df.groupby('winner_clean')['total_games'].transform('mean')
        
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

def get_random_user_agent():
    """Retorna um User-Agent aleatório"""
    ua = UserAgent()
    return ua.random

def get_random_headers():
    """Headers com rotação"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
    ]
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'pt-PT,pt;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }

def random_delay():
    """Delay aleatório para evitar detecção"""
    delay = random.uniform(st.session_state.get('delay_min', 2), st.session_state.get('delay_max', 5))
    time.sleep(delay)

# ====================== SELENIUM SCRAPER (MAIS ROBUSTO) ======================
def setup_selenium_driver():
    """Configura o driver do Selenium com opções anti-detecção"""
    chrome_options = Options()
    
    # Opções para evitar detecção
    chrome_options.add_argument('--headless')  # Modo headless para servidores
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Random User Agent
    if use_proxy:
        chrome_options.add_argument(f'user-agent={get_random_user_agent()}')
    
    # Desabilitar imagens para carregamento mais rápido
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except WebDriverException as e:
        st.error(f"Erro ao iniciar Selenium: {e}")
        return None

def get_matches_selenium():
    """Obtém partidas usando Selenium (mais robusto contra JavaScript)"""
    driver = setup_selenium_driver()
    if not driver:
        return pd.DataFrame()
    
    matches = []
    
    try:
        # URLs para tentar (múltiplas fontes)
        urls = [
            "https://www.flashscore.com/tennis/",
            "https://www.sofascore.com/tennis",
            "https://www.atptour.com/en/scores/results"
        ]
        
        for url in urls:
            st.info(f"Tentando: {url}")
            driver.get(url)
            random_delay()
            
            # Aguardar carregamento da página
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Scroll para carregar conteúdo dinâmico
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            random_delay()
            
            # Extrair HTML
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Padrões comuns para encontrar partidas
            selectors = [
                'div[class*="event"]',
                'div[class*="match"]',
                'div[class*="game"]',
                'div[data-testid*="match"]',
                'div[class*="tennis"]'
            ]
            
            for selector in selectors:
                events = soup.select(selector)
                if events:
                    for event in events[:30]:
                        try:
                            # Tentar extrair nomes dos jogadores
                            text = event.get_text()
                            lines = [line.strip() for line in text.split('\n') if line.strip()]
                            
                            # Procurar por nomes que parecem jogadores (2-3 palavras)
                            players = [line for line in lines if len(line.split()) in [2, 3] and len(line) > 3]
                            
                            if len(players) >= 2:
                                j1, j2 = players[0], players[1]
                                
                                # Extrair torneio
                                tournament = "Tênis"
                                for line in lines:
                                    if any(word in line.lower() for word in ['challenger', 'atp', 'wta', 'open', 'cup']):
                                        tournament = line
                                        break
                                
                                matches.append({
                                    'torneio': tournament,
                                    'jogador_1': j1,
                                    'jogador_2': j2,
                                    'horario': datetime.now().strftime('%H:%M'),
                                    'superficie': 'Hard'  # Será detectado depois
                                })
                        except:
                            continue
                    
                    if matches:
                        break
            
            if matches:
                st.success(f"✅ Encontradas {len(matches)} partidas com Selenium")
                break
                
    except Exception as e:
        st.error(f"Erro no Selenium: {str(e)[:200]}")
    finally:
        driver.quit()
    
    # Remover duplicatas
    if matches:
        df = pd.DataFrame(matches)
        df = df.drop_duplicates(subset=['jogador_1', 'jogador_2'])
        return df
    
    return pd.DataFrame()

# ====================== CLOUDSCRAPER (BYPASS CLOUDFLARE) ======================
def get_matches_cloudscraper():
    """Usa cloudscraper para bypassar Cloudflare"""
    try:
        scraper = cloudscraper.create_scraper()
        
        urls = [
            "https://www.flashscore.com/tennis/",
            "https://www.sofascore.com/tennis"
        ]
        
        for url in urls:
            headers = get_random_headers()
            response = scraper.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                matches = []
                
                # Procurar por padrões de partidas
                match_elements = soup.find_all(['div', 'a'], class_=lambda x: x and ('match' in str(x).lower() or 'event' in str(x).lower()))
                
                for elem in match_elements[:50]:
                    text = elem.get_text(strip=True)
                    if 'vs' in text.lower() or ' - ' in text:
                        parts = text.split('vs') if 'vs' in text.lower() else text.split('-')
                        if len(parts) >= 2:
                            j1 = parts[0].strip()
                            j2 = parts[1].strip().split()[0] if parts[1] else ''
                            
                            if j1 and j2 and len(j1) > 2 and len(j2) > 2:
                                matches.append({
                                    'torneio': 'Torneio',
                                    'jogador_1': j1,
                                    'jogador_2': j2,
                                    'horario': 'Hoje',
                                    'superficie': detect_surface('')
                                })
                
                if matches:
                    return pd.DataFrame(matches)
                    
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Erro no CloudScraper: {e}")
        return pd.DataFrame()

# ====================== API SOFASCORE (MÉTODO ORIGINAL) ======================
def get_matches_sofascore_api():
    """Usa a API do Sofascore"""
    try:
        url = "https://api.sofascore.com/api/v1/sport/tennis/events/live-and-upcoming"
        headers = get_random_headers()
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            
            for event in data.get('events', [])[:50]:
                try:
                    tournament = event.get('tournament', {}).get('name', 'Desconhecido')
                    home = event.get('homeTeam', {}).get('name', '')
                    away = event.get('awayTeam', {}).get('name', '')
                    
                    if home and away:
                        matches.append({
                            'torneio': tournament,
                            'jogador_1': home,
                            'jogador_2': away,
                            'horario': datetime.fromtimestamp(event.get('startTimestamp', 0)).strftime('%H:%M'),
                            'superficie': detect_surface(tournament)
                        })
                except:
                    continue
            
            return pd.DataFrame(matches)
        
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Erro na API: {e}")
        return pd.DataFrame()

# ====================== FUNÇÃO PRINCIPAL DE SCRAPING ======================
def get_matches_robust():
    """Tenta múltiplos métodos até conseguir dados"""
    
    # Dicionário de métodos disponíveis
    methods = {
        "API Sofascore": get_matches_sofascore_api,
        "CloudScraper": get_matches_cloudscraper,
        "Selenium": get_matches_selenium
    }
    
    # Selecionar método baseado na escolha do usuário
    if scraping_method == "Auto (Recomendado)":
        # Tentar em ordem de eficiência
        for method_name, method_func in methods.items():
            st.info(f"🔄 Tentando {method_name}...")
            df = method_func()
            if not df.empty:
                st.success(f"✅ Sucesso com {method_name}!")
                return df
            random_delay()
    elif scraping_method in methods:
        df = methods[scraping_method]()
        if not df.empty:
            return df
    elif scraping_method == "Manual":
        return pd.DataFrame()  # Será tratado pelo modo manual
    
    return pd.DataFrame()

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'kigali', 'santiago', 'heilbronn', 'perugia', 'barletta', 'rome']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon', 'birmingham', 'newport', 's-Hertogenbosch']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'cherbourg', 'bergerac', 'pau', 'quimper']):
        return 'Indoor'
    return 'Hard'

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

# ====================== INTERFACE MANUAL ======================
def manual_matches_input():
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
    
    # Salvar configurações na sessão
    st.session_state['delay_min'] = delay_range[0]
    st.session_state['delay_max'] = delay_range[1]
    
    col1, col2 = st.columns([1, 3])
    with col1:
        fetch_matches = st.button("🔄 Buscar Partidas", type="primary", use_container_width=True)
    
    df_matches = pd.DataFrame()
    
    if fetch_matches:
        if df_stats.empty:
            st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx na barra lateral.")
        else:
            with st.spinner("Buscando partidas com método robusto..."):
                df_matches = get_matches_robust()
                
                if df_matches.empty and scraping_method != "Manual":
                    st.warning("⚠️ Não foi possível obter partidas automaticamente.")
                    st.info("💡 Dicas:\n- Tente selecionar 'Selenium (Mais Robusto)' no sidebar\n- Ou use o modo 'Manual'\n- Verifique sua conexão com internet")
                    
                    # Exemplo para teste
                    st.markdown("---")
                    st.subheader("📋 Carregar exemplo para teste")
                    
                    if st.button("📊 Carregar exemplo de partidas"):
                        example_matches = pd.DataFrame({
                            'torneio': ['Challenger Santiago', 'Challenger Kigali', 'Challenger Perugia'],
                            'jogador_1': ['Joao Sousa', 'Carlos Taberner', 'Federico Coria'],
                            'jogador_2': ['Gastao Elias', 'Pedro Sousa', 'Thiago Monteiro'],
                            'horario': ['15:00', '13:00', '11:00'],
                            'superficie': ['Clay', 'Clay', 'Clay']
                        })
                        df_matches = example_matches
                        st.rerun()
    
    # Processar partidas encontradas
    if not df_matches.empty:
        with st.spinner("Calculando previsões..."):
            results = []
            progress_bar = st.progress(0)
            
            # Detectar superfície para cada partida
            df_matches['superficie'] = df_matches['torneio'].apply(detect_surface)
            
            for idx, row in df_matches.iterrows():
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
                
                progress_bar.progress((idx + 1) / len(df_matches))
                time.sleep(0.1)
            
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
                "text/csv"
            )
    
    # Modo manual
    if scraping_method == "Manual" or df_matches.empty:
        st.markdown("---")
        manual_df = manual_matches_input()
        
        if st.button("📊 Calcular Previsões", type="secondary"):
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

    ### Métodos de Scraping Implementados

    - **Selenium**: Mais robusto, executa JavaScript, bypassa proteções básicas
    - **CloudScraper**: Bypassa Cloudflare e proteções similares
    - **API Sofascore**: Mais rápido, mas pode ter rate limiting
    - **Rotação de Headers**: Evita detecção por User-Agent

    ### Para Produção

    Recomenda-se usar uma combinação:
    1. Tentar API primeiro (mais rápida)
    2. Fallback para Selenium (mais robusto)
    3. Cache de resultados para evitar scraping excessivo
    """)

st.caption("Versão com Selenium + CloudScraper + Rotação de Headers • Baseado no teu ficheiro Challenger1.xlsx")
