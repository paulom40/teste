import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
import unicodedata

# ====================== CONFIGURAÇÃO ======================
st.set_page_config(page_title="Tênis Hoje - WELO + Total", page_icon="🎾", layout="wide")

st.title("🎾 Partidas de Tênis + WELO + Linha Total")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Ficheiros")
    
    st.subheader("1. Ficheiro WELO (Challenger.xlsm)")
    welo_file = st.file_uploader("Escolha o ficheiro Challenger.xlsm", type=["xlsm", "xlsx"], key="welo")
    
    st.subheader("2. Ficheiro de Partidas (ATP/Challenger)")
    matches_file = st.file_uploader("Escolha o ficheiro CSV de partidas", type=["csv"], key="matches")
    
    st.markdown("---")
    st.info("💡 O ficheiro CSV deve conter colunas: winner_name, loser_name, surface, tourney_name, score")

# ====================== CARREGAR DADOS WELO ======================
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
        st.sidebar.success(f"✅ {len(df)} jogadores WELO carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar WELO: {e}")
        return pd.DataFrame()

# ====================== CARREGAR PARTIDAS ======================
@st.cache_data
def load_matches_data(file):
    try:
        df = pd.read_csv(file)
        
        # Verificar colunas necessárias
        required_cols = ['winner_name', 'loser_name', 'surface']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Colunas faltando no CSV: {missing_cols}")
            st.info(f"Colunas disponíveis: {list(df.columns)}")
            return pd.DataFrame()
        
        # Filtrar apenas partidas completas (com vencedor)
        df = df[df['winner_name'].notna() & df['loser_name'].notna()]
        
        # Adicionar colunas necessárias se não existirem
        if 'tourney_name' not in df.columns:
            df['tourney_name'] = 'Torneio'
        
        if 'score' not in df.columns:
            df['score'] = ''
        
        # Calcular total de games se não existir
        if 'T Games' not in df.columns:
            df['T Games'] = df.apply(lambda x: calculate_games_from_score(x.get('score', '')), axis=1)
        
        st.sidebar.success(f"✅ {len(df)} partidas carregadas")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar partidas: {e}")
        return pd.DataFrame()

def calculate_games_from_score(score):
    """Calcula total de games a partir do placar"""
    if not score or pd.isna(score):
        return 0
    
    total = 0
    import re
    # Extrair sets (ex: 6-4, 3-6, 6-2)
    sets = re.findall(r'(\d+)-(\d+)', score)
    for set_score in sets:
        total += int(set_score[0]) + int(set_score[1])
    
    return total

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
def calcular_linha_total(welo1: float, welo2: float, superficie: str, games_reais=None) -> tuple:
    """
    Retorna (Total_Esperado, Prob_Mais_21.5, Diferenca_Real)
    """
    dif = abs(welo1 - welo2)
    
    # Base média de jogos por superfície (baseado nos dados reais)
    base_jogos = {
        'Clay': 22.8,
        'Hard': 22.4,
        'Grass': 21.9,
        'Indoor': 22.6
    }.get(superficie, 22.5)
    
    # Ajuste pela diferença de nível
    ajuste_dif = -0.035 * dif
    
    total_esperado = base_jogos + ajuste_dif
    total_esperado = max(18.5, min(27.0, total_esperado))
    
    # Probabilidade de Mais de 21.5
    prob_mais_21_5 = max(0.35, min(0.78, 0.5 + (total_esperado - 22.0) * 0.08))
    
    # Calcular diferença se tivermos games reais
    diferenca = None
    if games_reais and games_reais > 0:
        diferenca = games_reais - total_esperado
    
    return round(total_esperado, 2), round(prob_mais_21_5 * 100, 1), diferenca

# ====================== DETECTAR SUPERFÍCIE ======================
def get_surface_display(surface):
    """Converte superfície para formato amigável"""
    surface_map = {
        'Clay': '🏟️ Clay',
        'Hard': '🎾 Hard',
        'Grass': '🌿 Grass',
        'Indoor': '🏠 Indoor'
    }
    return surface_map.get(surface, surface)

# ====================== EXECUÇÃO PRINCIPAL ======================
# Carregar dados
df_welo = pd.DataFrame()
df_matches = pd.DataFrame()

if welo_file:
    df_welo = load_welo_data(welo_file)

if matches_file:
    df_matches = load_matches_data(matches_file)

# Botão de análise
if st.button("🔄 Analisar Partidas + Calcular WELO + Linha Total", type="primary"):
    if df_welo.empty:
        st.warning("⚠️ Carregue primeiro o ficheiro Challenger.xlsm na barra lateral.")
    elif df_matches.empty:
        st.warning("⚠️ Carregue primeiro o ficheiro CSV de partidas na barra lateral.")
    else:
        with st.spinner("Analisando partidas e calculando WELO..."):
            
            # Criar DataFrame para análise
            analysis_df = df_matches.copy()
            
            # Calcular WELO para cada jogador
            analysis_df['WELO_Winner'] = analysis_df.apply(
                lambda row: get_welo(row['winner_name'], row['surface'], df_welo), axis=1
            )
            analysis_df['WELO_Loser'] = analysis_df.apply(
                lambda row: get_welo(row['loser_name'], row['surface'], df_welo), axis=1
            )
            analysis_df['Dif_WELO'] = abs(analysis_df['WELO_Winner'] - analysis_df['WELO_Loser'])
            
            # Calcular linha total
            games_col = 'T Games' if 'T Games' in analysis_df.columns else None
            
            resultados = analysis_df.apply(
                lambda row: calcular_linha_total(
                    row['WELO_Winner'], 
                    row['WELO_Loser'], 
                    row['surface'],
                    row.get(games_col, 0) if games_col else None
                ), axis=1
            )
            
            analysis_df['Total_Esperado'] = [r[0] for r in resultados]
            analysis_df['Prob_Mais_21.5'] = [r[1] for r in resultados]
            if games_col:
                analysis_df['Diferenca_Real_vs_Esperado'] = [r[2] for r in resultados if r[2] is not None]
            
            # Estatísticas resumo
            st.subheader("📊 Estatísticas de Análise")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Partidas", len(analysis_df))
            with col2:
                st.metric("Média WELO Vencedor", f"{analysis_df['WELO_Winner'].mean():.0f}")
            with col3:
                st.metric("Média Total Esperado", f"{analysis_df['Total_Esperado'].mean():.2f}")
            with col4:
                if games_col:
                    st.metric("Média Total Real", f"{analysis_df[games_col].mean():.2f}")
            
            # Mostrar dados
            st.subheader("🎾 Detalhe das Partidas")
            
            # Selecionar colunas para exibir
            display_cols = ['tourney_name', 'winner_name', 'loser_name', 'surface', 'score']
            display_cols.extend(['WELO_Winner', 'WELO_Loser', 'Dif_WELO', 'Total_Esperado', 'Prob_Mais_21.5'])
            if games_col and 'Diferenca_Real_vs_Esperado' in analysis_df.columns:
                display_cols.append('Diferenca_Real_vs_Esperado')
            
            st.dataframe(
                analysis_df[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "tourney_name": "🏆 Torneio",
                    "winner_name": "🏆 Vencedor",
                    "loser_name": "📉 Perdedor",
                    "surface": "🏟️ Superfície",
                    "score": "📊 Placar",
                    "WELO_Winner": st.column_config.NumberColumn("WELO Vencedor", format="%.1f"),
                    "WELO_Loser": st.column_config.NumberColumn("WELO Perdedor", format="%.1f"),
                    "Dif_WELO": st.column_config.NumberColumn("Dif WELO", format="%.1f"),
                    "Total_Esperado": st.column_config.NumberColumn("Total Esperado", format="%.2f"),
                    "Prob_Mais_21.5": st.column_config.NumberColumn("Prob >21.5 (%)", format="%.1f"),
                    "Diferenca_Real_vs_Esperado": st.column_config.NumberColumn("Dif Real-Esperado", format="%.1f"),
                }
            )
            
            # Botões de download
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download CSV",
                    analysis_df.to_csv(index=False).encode('utf-8'),
                    f"analise_tenis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )
            with col2:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    analysis_df.to_excel(writer, index=False)
                output.seek(0)
                st.download_button(
                    "📊 Download Excel",
                    output,
                    f"analise_tenis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Gráfico de distribuição
            st.subheader("📈 Distribuição do Total de Games Esperado")
            st.bar_chart(analysis_df['Total_Esperado'].value_counts().sort_index())

else:
    st.info("""
    ### 📋 Como usar:
    
    1. **Carregue o ficheiro Challenger.xlsm** (dados WELO dos jogadores)
    2. **Carregue o ficheiro CSV de partidas** (ATP/Challenger com resultados)
    3. **Clique no botão** para analisar
    
    ### Formato esperado do CSV:
    - `winner_name`: Nome do vencedor
    - `loser_name`: Nome do perdedor  
    - `surface`: Superfície (Clay, Hard, Grass, Indoor)
    - `tourney_name`: Nome do torneio
    - `score`: Placar (opcional)
    - `T Games`: Total de games (opcional)
    """)

st.caption("🎾 Análise WELO • Estimativa de Total de Jogos • Probabilidade > 21.5")
