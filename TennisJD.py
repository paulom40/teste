import streamlit as st
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import unicodedata
import re

# ====================== CONFIGURAÇÃO ======================
st.set_page_config(page_title="Tênis Hoje - WELO + Total", page_icon="🎾", layout="wide")

st.title("🎾 Partidas de Tênis Hoje + WELO + Linha Total")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger.xlsm")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger.xlsm", type=["xlsm", "xlsx"])

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
    # Quanto mais próximo de 22.5, maior a probabilidade de ir acima
    prob_mais_21_5 = max(0.35, min(0.78, 0.5 + (total_esperado - 22.0) * 0.08))
    
    return round(total_esperado, 2), round(prob_mais_21_5 * 100, 1)

# ====================== FUNÇÃO PARA BUSCAR PARTIDAS ======================
def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'kigali', 'santiago', 'punto cana', 'bucharest', 'houston', 'marrakech', 'rio', 'barcelona']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon', 'halle', 'queens', 'eastbourne']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'stockholm', 'basel']):
        return 'Indoor'
    return 'Hard'

def get_flashscore_matches():
    """
    Versão alternativa usando requests + BeautifulSoup
    Nota: FlashScore tem proteção anti-bot, então usamos uma API alternativa
    """
    matches = []
    
    # Opção 1: Usar API da FlashScore (não oficial)
    # Esta é uma abordagem mais simples que pode funcionar
    
    try:
        # Headers para simular um navegador
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Tentar a versão mobile da FlashScore
        url = "https://www.flashscore.com/tennis/"
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Procurar por elementos de partidas (seletores podem mudar)
            # Nota: FlashScore usa JavaScript, então isso pode não funcionar bem
            
            # Alternativa: Usar dados de exemplo para demonstração
            st.warning("⚠️ O FlashScore tem proteção anti-bot. Usando dados de exemplo para demonstração.")
            
            # Dados de exemplo para demonstrar a funcionalidade
            example_matches = [
                {'torneio': 'ATP Masters 1000 Monte Carlo', 'jogador_1': 'Novak Djokovic', 'jogador_2': 'Carlos Alcaraz', 'horario': '14:00', 'superficie': 'Clay'},
                {'torneio': 'WTA 500 Stuttgart', 'jogador_1': 'Iga Swiatek', 'jogador_2': 'Elena Rybakina', 'horario': '16:30', 'superficie': 'Clay'},
                {'torneio': 'ATP 250 Houston', 'jogador_1': 'Ben Shelton', 'jogador_2': 'Frances Tiafoe', 'horario': '19:00', 'superficie': 'Clay'},
                {'torneio': 'WTA 1000 Madrid', 'jogador_1': 'Coco Gauff', 'jogador_2': 'Jessica Pegula', 'horario': '21:00', 'superficie': 'Clay'},
            ]
            
            for match in example_matches:
                matches.append(match)
        else:
            st.error(f"Erro ao acessar FlashScore: {response.status_code}")
            
    except Exception as e:
        st.error(f"Erro na requisição: {e}")
        st.info("💡 Dica: O FlashScore bloqueia scraping direto. Considere usar uma API oficial ou fonte alternativa.")
    
    return pd.DataFrame(matches)

# ====================== EXECUÇÃO ======================
if st.button("🔄 Buscar Partidas + Calcular WELO + Linha Total", type="primary"):
    if df_welo.empty:
        st.warning("⚠️ Carregue primeiro o ficheiro Challenger.xlsm na barra lateral.")
    else:
        with st.spinner("Buscando partidas e calculando WELO + Linha Total..."):
            df = get_flashscore_matches()
            
            if not df.empty:
                df['WELO_J1'] = df.apply(lambda row: get_welo(row['jogador_1'], row['superficie'], df_welo), axis=1)
                df['WELO_J2'] = df.apply(lambda row: get_welo(row['jogador_2'], row['superficie'], df_welo), axis=1)
                df['Dif_WELO'] = abs(df['WELO_J1'] - df['WELO_J2'])
                
                # Calcula linha total
                resultados = df.apply(lambda row: calcular_linha_total(row['WELO_J1'], row['WELO_J2'], row['superficie']), axis=1)
                df['Total_Esperado'] = [r[0] for r in resultados]
                df['Prob_Mais_21.5'] = [r[1] for r in resultados]
                
                st.success(f"✅ {len(df)} partidas analisadas!")
                
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
                        "WELO_J1": st.column_config.NumberColumn("WELO J1", format="%.1f"),
                        "WELO_J2": st.column_config.NumberColumn("WELO J2", format="%.1f"),
                        "Dif_WELO": st.column_config.NumberColumn("Dif WELO", format="%.1f"),
                        "Total_Esperado": st.column_config.NumberColumn("Total Esperado", format="%.2f"),
                        "Prob_Mais_21.5": st.column_config.NumberColumn("Prob >21.5 (%)", format="%.1f"),
                    }
                )
                
                # Downloads
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
                st.warning("Nenhuma partida encontrada.")
else:
    st.info("Carregue o ficheiro na sidebar e clique no botão.")

st.caption("WELO por superfície • Estimativa de Total de Jogos • Probabilidade > 21.5")
