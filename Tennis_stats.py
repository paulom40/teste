import streamlit as st
import pandas as pd
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
from io import BytesIO

st.set_page_config(page_title="Tênis Predictor - Stats", page_icon="🎾", layout="wide")
st.title("🎾 Predictor de Tênis por Stats Reais")
st.caption("Baseado no ficheiro Challenger1.xlsx")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])

# ====================== CARREGAR E PROCESSAR DADOS ======================
@st.cache_data
def load_and_process_data(file):
    if not file:
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(file)
        
        # Normalização de nomes
        def norm(name):
            if not isinstance(name, str): 
                return ""
            n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            return ''.join(filter(str.isalnum, n.lower().strip()))
        
        df['winner_clean'] = df['winner_name'].apply(norm)
        df['loser_clean'] = df['loser_name'].apply(norm)
        
        st.sidebar.success(f"✅ {len(df)} jogos carregados com stats detalhadas")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar ficheiro: {e}")
        return pd.DataFrame()

df_raw = load_and_process_data(uploaded_file)

# ====================== PREDICTOR ======================
def predict_from_row(row, superficie):
    def safe(v):
        try: return float(v) if pd.notna(v) else 0.0
        except: return 0.0

    # Stats do Winner (Jogador 1)
    p1 = {
        'w_svpt': safe(row.get('w_svpt')),
        'w_1stWon': safe(row.get('w_1stWon')),
        'w_2ndWon': safe(row.get('w_2ndWon')),
        'w_bpSaved': safe(row.get('w_bpSaved')),
        'w_bpFaced': safe(row.get('w_bpFaced'))
    }
    
    # Stats do Loser (Jogador 2)
    p2 = {
        'w_svpt': safe(row.get('l_svpt')),
        'w_1stWon': safe(row.get('l_1stWon')),
        'w_2ndWon': safe(row.get('l_2ndWon')),
        'w_bpSaved': safe(row.get('l_bpSaved')),
        'w_bpFaced': safe(row.get('l_bpFaced'))
    }

    def serve_win(stats):
        svpt = stats['w_svpt']
        if svpt == 0: return 0.65
        return (stats['w_1stWon'] + stats['w_2ndWon']) / svpt

    serve1 = serve_win(p1)
    serve2 = serve_win(p2)
    return1 = 1 - serve2
    return2 = 1 - serve1

    p1_point_win = (serve1 + return1) / 2
    p2_point_win = (serve2 + return2) / 2

    surface_factor = {'Clay': 1.08, 'Hard': 1.0, 'Grass': 0.93, 'Indoor': 1.02}.get(superficie, 1.0)

    diff = (p1_point_win - p2_point_win) * 100
    prob_p1_win = 1 / (1 + 10 ** (-diff / 38))

    hold1 = serve1 ** 1.85
    hold2 = serve2 ** 1.85
    break_prob = (1 - hold1 + 1 - hold2) / 2
    games_per_set = 9.6 + 4.2 * break_prob
    total_esperado = round(games_per_set * 2.15 * surface_factor, 2)

    prob_over = max(0.38, min(0.78, 0.5 + (total_esperado - 21.5) * 0.085))

    return {
        "Jogador_1": row['winner_name'],
        "Jogador_2": row['loser_name'],
        "Superficie": superficie,
        "Prob_J1_%": round(prob_p1_win * 100, 1),
        "Total_Esperado": total_esperado,
        "Prob_Over_21.5_%": round(prob_over * 100, 1),
        "Serve_J1_%": round(serve1 * 100, 1),
        "BP_Saved_J1_%": round(safe(p1['w_bpSaved']) / max(safe(p1['w_bpFaced']), 1) * 100, 1),
        "Score": row.get('score', ''),
        "Round": row.get('round', '')
    }

# ====================== EXECUÇÃO ======================
if df_raw.empty:
    st.info("👆 Carregue o ficheiro Challenger1.xlsx na barra lateral para começar.")
else:
    st.success(f"Ficheiro carregado com {len(df_raw)} jogos")

    # Filtro por superfície
    surfaces = df_raw['surface'].dropna().unique().tolist()
    selected_surface = st.selectbox("Filtrar por Superfície", options=["Todas"] + surfaces)

    if selected_surface != "Todas":
        df_filtered = df_raw[df_raw['surface'] == selected_surface].copy()
    else:
        df_filtered = df_raw.copy()

    # Calcular previsões
    if st.button("🚀 Calcular Predictor para todos os jogos", type="primary"):
        with st.spinner("A aplicar predictor em todos os jogos..."):
            predictions = []
            for _, row in df_filtered.iterrows():
                pred = predict_from_row(row, row.get('surface', 'Hard'))
                predictions.append(pred)
            
            result_df = pd.DataFrame(predictions)
            
            # Ordenar por probabilidade mais alta
            result_df = result_df.sort_values(by="Prob_J1_%", ascending=False)

            st.success(f"✅ Previsões calculadas para {len(result_df)} jogos")
            
            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Prob_J1_%": st.column_config.NumberColumn("Probabilidade J1 (%)", format="%.1f"),
                    "Total_Esperado": st.column_config.NumberColumn("Total Esperado", format="%.2f"),
                    "Prob_Over_21.5_%": st.column_config.NumberColumn("Over 21.5 (%)", format="%.1f"),
                    "Serve_J1_%": st.column_config.NumberColumn("Serve J1 (%)", format="%.1f"),
                    "BP_Saved_J1_%": st.column_config.NumberColumn("BP Saved J1 (%)", format="%.1f"),
                }
            )

            # Download
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download CSV",
                    result_df.to_csv(index=False).encode('utf-8'),
                    f"predictor_tenis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )

st.caption("Predictor baseado em stats reais (serve, return, break points) • Superfície considerada")
