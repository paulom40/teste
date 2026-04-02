import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
from io import BytesIO
import unicodedata
import re
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

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
    st.header("⚙️ Opções da API")
    dias_busca = st.slider("Dias para buscar partidas", 1, 7, 3, help="Buscar partidas dos próximos X dias")
    
    if uploaded_file:
        st.success("✅ Ficheiro carregado com sucesso!")
    else:
        st.warning("⚠️ Necessário carregar o ficheiro Challenger.xlsm")

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
        
        st.sidebar.success(f"✅ Sheet '{sheet_name}' carregada com {len(df)} linhas")
        
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

# ====================== FUNÇÃO PARA BUSCAR PARTIDAS DA API (GRATUITA) ======================
def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'kigali', 'santiago', 'punto cana', 'bucharest', 'houston', 'marrakech', 'rio', 'barcelona', 'monte carlo', 'stuttgart']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon', 'halle', 'queens', 'eastbourne']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'stockholm', 'basel', 'vienna']):
        return 'Indoor'
    return 'Hard'

def get_tennis_matches_from_api(dias=3):
    """
    Busca partidas de ténis usando endpoints GRATUITOS da SportScore API
    """
    matches = []
    
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    
    params = {
        "sport_id": 2  # Tennis
    }
    
    # Buscar partidas dos próximos X dias
    for i in range(dias):
        date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        
        # Endpoint GRATUITO: eventos por data
        url = f"https://sportscore1.p.rapidapi.com/api/v1/events/date/{date}"
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and isinstance(data['data'], list):
                    for event in data['data']:
                        try:
                            # Extrair informações
                            tournament = event.get('league', {}).get('name', 'Torneio')
                            home_team = event.get('home_team', {}).get('name', '')
                            away_team = event.get('away_team', {}).get('name', '')
                            
                            # Horário
                            start_time = event.get('starting_at', '')
                            horario = ''
                            if start_time and 'T' in str(start_time):
                                horario = str(start_time).split('T')[1][:5]
                            
                            # Status da partida
                            status = event.get('status', '')
                            
                            # Apenas partidas não iniciadas
                            if home_team and away_team and status != 'finished':
                                superficie = detect_surface(tournament)
                                
                                matches.append({
                                    'torneio': tournament,
                                    'jogador_1': home_team,
                                    'jogador_2': away_team,
                                    'horario': horario,
                                    'data': date,
                                    'superficie': superficie,
                                    'status': status
                                })
                        except Exception as e:
                            continue
                            
            elif response.status_code == 401:
                st.error(f"❌ Erro 401 na API para data {date}. Chave inválida ou sem permissão.")
                return pd.DataFrame()
                
        except Exception as e:
            st.warning(f"⚠️ Erro ao buscar data {date}: {str(e)}")
    
    return pd.DataFrame(matches)

# ====================== FUNÇÃO PARA EXPORTAR EXCEL ======================
def export_to_excel(df_analise):
    """Exporta para Excel formatado"""
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Análise
        df_analise.to_excel(writer, sheet_name='Análise Partidas', index=False)
        
        # Sheet 2: Estatísticas
        stats_data = {
            'Métrica': [
                'Total de Partidas',
                'Média WELO Jogador 1',
                'Média WELO Jogador 2',
                'Média Diferença WELO',
                'Média Total Esperado',
                'Média Probabilidade >21.5%',
                'Partidas com OVER (>65%)',
                'Partidas com UNDER (<40%)',
                'Data da Análise'
            ],
            'Valor': [
                len(df_analise),
                f"{df_analise['WELO_J1'].mean():.1f}",
                f"{df_analise['WELO_J2'].mean():.1f}",
                f"{df_analise['Dif_WELO'].mean():.1f}",
                f"{df_analise['Total_Esperado'].mean():.2f}",
                f"{df_analise['Prob_Mais_21.5'].mean():.1f}%",
                len(df_analise[df_analise['Prob_Mais_21.5'] > 65]),
                len(df_analise[df_analise['Prob_Mais_21.5'] < 40]),
                datetime.now().strftime('%d/%m/%Y %H:%M')
            ]
        }
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='Estatísticas', index=False)
        
        # Sheet 3: Oportunidades OVER
        over_df = df_analise[df_analise['Prob_Mais_21.5'] > 65].copy()
        if not over_df.empty:
            over_df.to_excel(writer, sheet_name='Oportunidades OVER', index=False)
        
        # Sheet 4: Oportunidades UNDER
        under_df = df_analise[df_analise['Prob_Mais_21.5'] < 40].copy()
        if not under_df.empty:
            under_df.to_excel(writer, sheet_name='Oportunidades UNDER', index=False)
        
        # Formatação
        workbook = writer.book
        sheet1 = writer.sheets['Análise Partidas']
        
        # Cabeçalho
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        
        for cell in sheet1[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center')
        
        # Ajustar largura
        for column in sheet1.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            sheet1.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return output

# ====================== MAIN ======================
# Carregar dados WELO
df_welo = pd.DataFrame()
name_col = None

if uploaded_file:
    df_welo, name_col = load_welo_data(uploaded_file)

# Botão principal
if st.button("🔄 Buscar Partidas da API + Calcular WELO Real", type="primary", use_container_width=True):
    
    if df_welo.empty:
        st.error("❌ É necessário carregar o ficheiro Challenger.xlsm na barra lateral!")
        st.stop()
    
    with st.spinner(f"🔍 Buscando partidas da API para os próximos {dias_busca} dias..."):
        df_partidas = get_tennis_matches_from_api(dias=dias_busca)
        
        if df_partidas.empty:
            st.warning("⚠️ Nenhuma partida encontrada nos próximos dias. Verifique a chave API ou tente mais tarde.")
            st.info("💡 Dica: O plano gratuito pode ter limitações. Os endpoints estão a funcionar?")
        else:
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
            
            # Salvar no session state
            st.session_state.df_analise = df_partidas
            
            # Estatísticas
            st.success(f"✅ {len(df_partidas)} partidas encontradas nos próximos {dias_busca} dias!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Partidas", len(df_partidas))
            with col2:
                st.metric("Média WELO J1", f"{df_partidas['WELO_J1'].mean():.0f}")
            with col3:
                st.metric("Média WELO J2", f"{df_partidas['WELO_J2'].mean():.0f}")
            with col4:
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
            st.subheader("🎯 Oportunidades Destacadas")
            
            over_df = df_partidas[df_partidas['Prob_Mais_21.5'].str.replace('%', '').astype(float) > 65]
            under_df = df_partidas[df_partidas['Prob_Mais_21.5'].str.replace('%', '').astype(float) < 40]
            
            for _, row in over_df.iterrows():
                st.success(f"🔴 **{row['jogador_1']} vs {row['jogador_2']}** - {row['Prob_Mais_21.5']} OVER (Total: {row['Total_Esperado']})")
            
            for _, row in under_df.iterrows():
                st.info(f"🔵 **{row['jogador_1']} vs {row['jogador_2']}** - {100 - float(row['Prob_Mais_21.5'].replace('%', '')):.1f}% UNDER (Total: {row['Total_Esperado']})")
            
            # Exportação
            st.markdown("---")
            st.subheader("📥 Exportar Resultados")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                csv_data = df_partidas.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Exportar CSV",
                    data=csv_data,
                    file_name=f"tenis_analise_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_exp2:
                excel_file = export_to_excel(df_partidas)
                st.download_button(
                    label="📊 Exportar Excel",
                    data=excel_file,
                    file_name=f"tenis_analise_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

else:
    st.info("""
    ### 🎾 Como usar:
    
    1. **Carregue o ficheiro Challenger.xlsm** na barra lateral
    2. **Selecione quantos dias** quer buscar (1-7 dias)
    3. **Clique no botão** para buscar partidas reais da API
    4. **Exporte os resultados** para CSV ou Excel
    
    ### Endpoints GRATUITOS utilizados:
    - `/api/v1/events/date/{data}` - Eventos por data
    - Filtro automático para ténis (sport_id=2)
    - Busca partidas dos próximos dias
    
    ### Limitações do plano gratuito:
    - Número limitado de requests por mês
    - Apenas endpoints básicos
    - Sem dados históricos (apenas atuais/futuros)
    """)

st.caption("🎾 API SportScore (Plano Gratuito) • WELO real do ficheiro • Estimativa de Total de Games")
