import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
from io import BytesIO
import unicodedata
import re
import json
from bs4 import BeautifulSoup
import time

# ====================== CONFIGURAÇÃO ======================
st.set_page_config(page_title="Tênis Hoje - WELO + Total", page_icon="🎾", layout="wide")

st.title("🎾 Partidas de Tênis Hoje + WELO + Linha Total")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

# ====================== CONFIGURAÇÃO DA API ======================
RAPIDAPI_KEY = "bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a44"
RAPIDAPI_HOST = "sportscore1.p.rapidapi.com"

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Ficheiro WELO")
    uploaded_file = st.file_uploader("Carregue o ficheiro Challenger.xlsm", type=["xlsm", "xlsx"])
    
    st.markdown("---")
    st.header("⚙️ Configurações")
    fonte_dados = st.selectbox(
        "Fonte de dados:",
        ["Automático (Tenta todas)", "Dados de Demonstração", "ATP Tour (Scraping)", "Tennis API (RapidAPI)"]
    )
    
    if uploaded_file:
        st.success("✅ Ficheiro carregado!")

# ====================== FUNÇÃO PARA CARREGAR WELO ======================
@st.cache_data
def load_welo_data(file):
    """Carrega e processa o ficheiro Challenger.xlsm"""
    try:
        xls = pd.ExcelFile(file)
        
        # Procurar sheet correta
        sheet_name = None
        for sheet in xls.sheet_names:
            if 'jogadores' in sheet.lower() or 'players' in sheet.lower() or '20' in sheet:
                sheet_name = sheet
                break
        
        if not sheet_name:
            sheet_name = xls.sheet_names[0]
        
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Normalizar nomes
        def normalize_name(name):
            if not isinstance(name, str):
                return ""
            name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            name = name.lower().strip()
            name = re.sub(r'[^\w\s]', '', name)
            name = ''.join(filter(str.isalnum, name))
            return name
        
        # Identificar coluna de nomes
        name_col = None
        for col in df.columns:
            if 'jogador' in col.lower() or 'player' in col.lower() or 'nome' in col.lower():
                name_col = col
                break
        
        if not name_col:
            name_col = df.columns[0]
        
        df['Jogador_clean'] = df[name_col].apply(normalize_name)
        
        # Garantir colunas ELO
        for col in ['ELO Clay', 'ELO Hard', 'ELO Grass', 'ELO Indoor']:
            if col not in df.columns:
                df[col] = 1484.0
        
        return df, name_col
        
    except Exception as e:
        st.sidebar.error(f"❌ Erro: {str(e)}")
        return pd.DataFrame(), None

# ====================== FUNÇÃO WELO ======================
def get_welo(jogador_nome: str, superficie: str, df_welo, name_col: str) -> float:
    """Calcula WELO do jogador para a superfície específica"""
    
    if df_welo.empty or not jogador_nome:
        return 1484.0
    
    clean_nome = unicodedata.normalize('NFKD', str(jogador_nome)).encode('ascii', 'ignore').decode('utf-8')
    clean_nome = re.sub(r'[^\w\s]', '', clean_nome).lower().strip()
    clean_nome = ''.join(filter(str.isalnum, clean_nome))
    
    if len(clean_nome) < 3:
        return 1484.0
    
    # Procurar jogador
    best_match = None
    best_score = 0
    
    for idx, row in df_welo.iterrows():
        clean_excel = row['Jogador_clean']
        if not clean_excel:
            continue
        
        score = 0
        if clean_nome == clean_excel:
            score = 100
        elif clean_nome in clean_excel or clean_excel in clean_nome:
            score = 90
        elif len(clean_nome) > 4 and len(clean_excel) > 4:
            if clean_nome[:4] == clean_excel[:4]:
                score = 75
            elif clean_nome[-4:] == clean_excel[-4:]:
                score = 70
            else:
                common = len(set(clean_nome) & set(clean_excel))
                max_len = max(len(clean_nome), len(clean_excel))
                if max_len > 0:
                    similarity = common / max_len
                    if similarity > 0.6:
                        score = 60 * similarity
        
        if score > best_score:
            best_score = score
            best_match = row
    
    if best_score < 60 or best_match is None:
        return 1484.0
    
    # Mapear superfície
    surface_col = {
        'clay': 'ELO Clay',
        'hard': 'ELO Hard',
        'grass': 'ELO Grass',
        'indoor': 'ELO Indoor'
    }.get(superficie.lower(), 'ELO Hard')
    
    if surface_col in best_match.index:
        val = best_match[surface_col]
        if pd.notna(val) and str(val).strip() != '' and val != 0:
            try:
                return round(float(val), 1)
            except:
                pass
    
    # Fallback: média
    elo_cols = ['ELO Hard', 'ELO Clay', 'ELO Grass', 'ELO Indoor']
    valores = []
    for col in elo_cols:
        if col in best_match.index:
            val = best_match[col]
            if pd.notna(val) and str(val).strip() != '' and val != 0:
                try:
                    valores.append(float(val))
                except:
                    pass
    
    if valores:
        return round(sum(valores) / len(valores), 1)
    
    return 1484.0

# ====================== CÁLCULO DA LINHA TOTAL ======================
def calcular_linha_total(welo1: float, welo2: float, superficie: str) -> tuple:
    dif = abs(welo1 - welo2)
    
    base_games = {
        'Clay': 22.8,
        'Hard': 22.4,
        'Grass': 21.9,
        'Indoor': 22.6
    }.get(superficie, 22.5)
    
    ajuste_dif = -0.025 * dif
    total_esperado = base_games + ajuste_dif
    total_esperado = max(18.5, min(27.0, total_esperado))
    
    if total_esperado >= 22.5:
        prob_mais = 0.55 + (total_esperado - 22.5) * 0.08
    elif total_esperado >= 21.5:
        prob_mais = 0.45 + (total_esperado - 21.5) * 0.10
    else:
        prob_mais = 0.35 + (total_esperado - 18.5) * 0.02
    
    prob_mais = max(0.25, min(0.85, prob_mais))
    
    return round(total_esperado, 2), round(prob_mais * 100, 1)

# ====================== FONTES DE DADOS ======================
def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'monte carlo', 'madrid', 'rome', 'barcelona', 'stuttgart', 'houston', 'marrakech', 'bucharest']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon', 'halle', 'queens', 'eastbourne', 's-Hertogenbosch']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'stockholm', 'basel', 'vienna', 'metz']):
        return 'Indoor'
    return 'Hard'

def get_demo_matches():
    """Fonte 1: Dados de demonstração (sempre disponível)"""
    matches = [
        {'torneio': 'ATP Masters 1000 Monte Carlo', 'jogador_1': 'Novak Djokovic', 'jogador_2': 'Carlos Alcaraz', 'horario': '14:30', 'data': datetime.now().strftime('%Y-%m-%d'), 'superficie': 'Clay'},
        {'torneio': 'ATP Masters 1000 Monte Carlo', 'jogador_1': 'Jannik Sinner', 'jogador_2': 'Daniil Medvedev', 'horario': '16:00', 'data': datetime.now().strftime('%Y-%m-%d'), 'superficie': 'Clay'},
        {'torneio': 'WTA 500 Stuttgart', 'jogador_1': 'Iga Swiatek', 'jogador_2': 'Elena Rybakina', 'horario': '12:00', 'data': datetime.now().strftime('%Y-%m-%d'), 'superficie': 'Clay'},
        {'torneio': 'ATP 250 Houston', 'jogador_1': 'Ben Shelton', 'jogador_2': 'Frances Tiafoe', 'horario': '20:00', 'data': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'), 'superficie': 'Clay'},
        {'torneio': 'ATP Challenger Oeiras', 'jogador_1': 'Nuno Borges', 'jogador_2': 'João Sousa', 'horario': '15:00', 'data': (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'), 'superficie': 'Clay'},
        {'torneio': 'ATP 500 Barcelona', 'jogador_1': 'Casper Ruud', 'jogador_2': 'Stefanos Tsitsipas', 'horario': '18:30', 'data': datetime.now().strftime('%Y-%m-%d'), 'superficie': 'Clay'},
        {'torneio': 'WTA 1000 Madrid', 'jogador_1': 'Aryna Sabalenka', 'jogador_2': 'Jessica Pegula', 'horario': '17:00', 'data': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'), 'superficie': 'Clay'},
    ]
    return pd.DataFrame(matches)

def get_atp_scraping():
    """Fonte 2: Scraping do site da ATP"""
    matches = []
    
    try:
        url = "https://www.atptour.com/en/scores/current"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Procurar por partidas
            match_cards = soup.find_all('div', class_='match-card')
            
            for card in match_cards[:15]:
                try:
                    # Extrair jogadores
                    players = card.find_all('span', class_='player-name')
                    if len(players) >= 2:
                        player1 = players[0].get_text(strip=True)
                        player2 = players[1].get_text(strip=True)
                        
                        # Extrair torneio
                        tournament_div = card.find('div', class_='tourney-title')
                        tournament = tournament_div.get_text(strip=True) if tournament_div else "ATP Tour"
                        
                        matches.append({
                            'torneio': tournament,
                            'jogador_1': player1,
                            'jogador_2': player2,
                            'horario': 'Hoje',
                            'data': datetime.now().strftime('%Y-%m-%d'),
                            'superficie': detect_surface(tournament)
                        })
                except:
                    continue
    except:
        pass
    
    return pd.DataFrame(matches)

def get_tennis_api():
    """Fonte 3: Tennis API (RapidAPI)"""
    matches = []
    
    # Tentar API alternativa de ténis
    url = "https://tennisapi1.p.rapidapi.com/api/tennis/events/live"
    
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "tennisapi1.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Processar dados conforme estrutura da API
            # (adaptar conforme necessário)
            pass
    except:
        pass
    
    return pd.DataFrame(matches)

def get_all_matches_auto():
    """Tenta todas as fontes automaticamente"""
    
    # Tentar ATP scraping primeiro
    st.info("🔍 Tentando ATP Tour scraping...")
    df = get_atp_scraping()
    
    if not df.empty:
        st.success(f"✅ Encontradas {len(df)} partidas via ATP Tour")
        return df
    
    # Fallback para dados de demonstração
    st.info("📊 Usando dados de demonstração...")
    df = get_demo_matches()
    st.success(f"✅ {len(df)} partidas de demonstração carregadas")
    
    return df

# ====================== EXPORTAR EXCEL ======================
def export_to_excel(df_analise):
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_analise.to_excel(writer, sheet_name='Análise Partidas', index=False)
        
        # Estatísticas
        stats_data = {
            'Métrica': ['Total Partidas', 'Média WELO J1', 'Média WELO J2', 'Média Total Esperado', 'Média Prob >21.5%', 'Data'],
            'Valor': [
                len(df_analise),
                f"{df_analise['WELO_J1'].mean():.1f}",
                f"{df_analise['WELO_J2'].mean():.1f}",
                f"{df_analise['Total_Esperado'].mean():.2f}",
                f"{df_analise['Prob_Mais_21.5'].mean():.1f}%",
                datetime.now().strftime('%d/%m/%Y %H:%M')
            ]
        }
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='Estatísticas', index=False)
    
    output.seek(0)
    return output

# ====================== MAIN ======================
df_welo = pd.DataFrame()
name_col = None

if uploaded_file:
    df_welo, name_col = load_welo_data(uploaded_file)

# Botão principal
if st.button("🔄 Buscar Partidas + Calcular WELO", type="primary", use_container_width=True):
    
    if df_welo.empty:
        st.error("❌ Carregue o ficheiro Challenger.xlsm primeiro!")
        st.stop()
    
    with st.spinner("Buscando partidas..."):
        
        # Escolher fonte
        if fonte_dados == "Automático (Tenta todas)":
            df_partidas = get_all_matches_auto()
        elif fonte_dados == "Dados de Demonstração":
            df_partidas = get_demo_matches()
            st.info("📊 Usando dados de demonstração")
        elif fonte_dados == "ATP Tour (Scraping)":
            df_partidas = get_atp_scraping()
            if df_partidas.empty:
                st.warning("⚠️ Nenhuma partida encontrada no ATP Tour")
                df_partidas = get_demo_matches()
        else:
            df_partidas = get_demo_matches()
        
        if not df_partidas.empty:
            # Calcular WELO
            df_partidas['WELO_J1'] = df_partidas.apply(
                lambda row: get_welo(row['jogador_1'], row['superficie'], df_welo, name_col), 
                axis=1
            )
            
            df_partidas['WELO_J2'] = df_partidas.apply(
                lambda row: get_welo(row['jogador_2'], row['superficie'], df_welo, name_col),
                axis=1
            )
            
            df_partidas['Dif_WELO'] = abs(df_partidas['WELO_J1'] - df_partidas['WELO_J2'])
            
            # Calcular linha total
            resultados = df_partidas.apply(
                lambda row: calcular_linha_total(row['WELO_J1'], row['WELO_J2'], row['superficie']),
                axis=1
            )
            
            df_partidas['Total_Esperado'] = [r[0] for r in resultados]
            df_partidas['Prob_Mais_21.5'] = [f"{r[1]}%" for r in resultados]
            
            # Estatísticas
            st.success(f"✅ {len(df_partidas)} partidas analisadas!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Partidas", len(df_partidas))
            with col2:
                st.metric("Média WELO", f"{df_partidas[['WELO_J1', 'WELO_J2']].mean().mean():.0f}")
            with col3:
                prob_mean = df_partidas['Prob_Mais_21.5'].str.replace('%', '').astype(float).mean()
                st.metric("Média Prob >21.5", f"{prob_mean:.1f}%")
            
            # Tabela
            st.dataframe(
                df_partidas,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "torneio": "🏆 Torneio",
                    "jogador_1": "🎾 Jogador 1",
                    "jogador_2": "🎾 Jogador 2",
                    "horario": "⏰ Horário",
                    "data": "📅 Data",
                    "superficie": "🏟️ Superfície",
                    "WELO_J1": st.column_config.NumberColumn("WELO J1", format="%.1f"),
                    "WELO_J2": st.column_config.NumberColumn("WELO J2", format="%.1f"),
                    "Dif_WELO": st.column_config.NumberColumn("Diferença", format="%.1f"),
                    "Total_Esperado": st.column_config.NumberColumn("Total Esperado", format="%.2f"),
                    "Prob_Mais_21.5": st.column_config.TextColumn("Prob >21.5"),
                }
            )
            
            # Oportunidades
            st.subheader("🎯 Recomendações")
            over_df = df_partidas[df_partidas['Prob_Mais_21.5'].str.replace('%', '').astype(float) > 65]
            under_df = df_partidas[df_partidas['Prob_Mais_21.5'].str.replace('%', '').astype(float) < 40]
            
            for _, row in over_df.iterrows():
                st.success(f"🔴 OVER 21.5: {row['jogador_1']} vs {row['jogador_2']} ({row['Prob_Mais_21.5']})")
            
            for _, row in under_df.iterrows():
                st.info(f"🔵 UNDER 21.5: {row['jogador_1']} vs {row['jogador_2']} ({100 - float(row['Prob_Mais_21.5'].replace('%', '')):.0f}%)")
            
            # Exportação
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📄 CSV", df_partidas.to_csv(index=False).encode('utf-8'), f"tenis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
            with col2:
                st.download_button("📊 Excel", export_to_excel(df_partidas), f"tenis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")

else:
    st.info("""
    ### 🎾 Como usar:
    
    1. **Carregue o ficheiro Challenger.xlsm**
    2. **Escolha a fonte de dados** (recomendado: Automático)
    3. **Clique em buscar** para ver as análises
    
    ### Fontes de dados disponíveis:
    - **Automático**: Tenta ATP scraping → Demonstração
    - **ATP Tour**: Scraping do site oficial
    - **Demonstração**: Dados de exemplo (sempre funciona)
    
    **Nota:** A API SportScore não retorna dados de ténis no plano gratuito. Use as alternativas acima.
    """)

st.caption("🎾 Múltiplas fontes de dados • WELO real • Análise de Totais")
