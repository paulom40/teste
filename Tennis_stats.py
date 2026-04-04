import streamlit as st
import pandas as pd
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
from io import BytesIO

st.set_page_config(page_title="Tênis Predictor", page_icon="🎾", layout="wide")
st.title("🎾 Predictor de Tênis - Escolha Jogadores")

tab1, tab2 = st.tabs(["📊 Todos os Jogos", "🔍 Previsão Personalizada"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])

# ====================== CARREGAR DADOS ======================
@st.cache_data
def load_data(file):
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
        
        # Lista única de jogadores para seleção
        all_players = pd.concat([
            df['winner_name'], 
            df['loser_name']
        ]).drop_duplicates().sort_values().tolist()
        
        st.sidebar.success(f"✅ {len(df)} jogos | {len(all_players)} jogadores únicos")
        return df, all_players
    except Exception as e:
        st.sidebar.error(f"Erro: {e}")
        return pd.DataFrame(), []

df_raw, player_list = load_data(uploaded_file)

# ====================== PREDICTOR ======================
def predict_match(jogador_a, jogador_b, superficie="Hard"):
    if df_raw.empty:
        return None
    
    def norm(name):
        if not isinstance(name, str): return ""
        n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
        return ''.join(filter(str.isalnum, n.lower().strip()))
    
    # Encontrar stats do Jogador A e B
    p1_stats = find_best_player_stats(jogador_a, df_raw)
    p2_stats = find_best_player_stats(jogador_b, df_raw)
    
    if p1_stats.empty or p2_stats.empty:
        return None
    
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
    prob_a_win = 1 / (1 + 10 ** (-diff / 38))

    hold1 = serve1 ** 1.85
    hold2 = serve2 ** 1.85
    break_prob = (1 - hold1 + 1 - hold2) / 2
    games_per_set = 9.6 + 4.2 * break_prob
    total_esperado = round(games_per_set * 2.15 * surface_factor, 2)

    prob_over = max(0.38, min(0.78, 0.5 + (total_esperado - 21.5) * 0.085))
    prob_under = 100 - prob_over

    return {
        "Jogador_A": jogador_a,
        "Jogador_B": jogador_b,
        "Superficie": superficie,
        "Prob_A_Vitória_%": round(prob_a_win * 100, 1),
        "Prob_B_Vitória_%": round(100 - prob_a_win * 100, 1),
        "Total_Esperado": total_esperado,
        "Prob_Over_21.5_%": round(prob_over, 1),
        "Prob_Under_21.5_%": round(prob_under, 1),
        "Serve_A_%": round(serve1 * 100, 1),
        "BP_Saved_A_%": round(safe(p1_stats.get('w_bpSaved',0)) / max(safe(p1_stats.get('w_bpFaced',1)), 1) * 100, 1),
    }

def find_best_player_stats(player_name, df):
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

def norm(name):
    if not isinstance(name, str): return ""
    n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return ''.join(filter(str.isalnum, n.lower().strip()))

# ====================== ABA 1 - TODOS OS JOGOS ======================
with tab1:
    st.header("Todos os Jogos")
    if not df_raw.empty:
        if st.button("Calcular para todos os jogos"):
            # (podes manter ou remover esta parte)
            st.info("Funcionalidade disponível na aba 'Previsão Personalizada'")

# ====================== ABA 2 - PREVISÃO PERSONALIZADA ======================
with tab2:
    st.header("🔍 Previsão Personalizada")
    st.write("Selecione os dois jogadores e a superfície")

    col1, col2 = st.columns(2)
    with col1:
        jogador_a = st.selectbox("Jogador A", options=player_list, key="ja")
    with col2:
        jogador_b = st.selectbox("Jogador B", options=player_list, key="jb")

    superficie = st.selectbox("Superfície", ["Hard", "Clay", "Grass", "Indoor"], index=0)

    if st.button("🚀 Calcular Previsão", type="primary"):
        if jogador_a == jogador_b:
            st.error("Selecione dois jogadores diferentes!")
        else:
            with st.spinner("Calculando previsão..."):
                result = predict_match(jogador_a, jogador_b, superficie)
                
                if result is None:
                    st.error("Não foi possível encontrar stats suficientes para um dos jogadores.")
                else:
                    st.success("Previsão Calculada!")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(f"🏆 {result['Jogador_A']} vence", f"{result['Prob_A_Vitória_%']}%")
                    with c2:
                        st.metric(f"🏆 {result['Jogador_B']} vence", f"{result['Prob_B_Vitória_%']}%")
                    
                    st.metric("📊 Total de Jogos Esperado", f"{result['Total_Esperado']} jogos")
                    
                    over, under = st.columns(2)
                    with over:
                        st.metric("Over 21.5", f"{result['Prob_Over_21.5_%']}%")
                    with under:
                        st.metric("Under 21.5", f"{result['Prob_Under_21.5_%']}%")
                    
                    st.write(f"**Serve {result['Jogador_A']}**: {result['Serve_A_%']}%")
                    st.write(f"**BP Saved {result['Jogador_A']}**: {result['BP_Saved_A_%']}%")

                    # ====================== BOTÃO EXPORTAR PARA EXCEL ======================
                    result_df = pd.DataFrame([result])
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        result_df.to_excel(writer, index=False, sheet_name='Previsão')
                    output.seek(0)

                    st.download_button(
                        label="📥 Exportar Previsão para Excel",
                        data=output,
                        file_name=f"previsao_{jogador_a}_vs_{jogador_b}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

st.caption("Selecione Jogador A, Jogador B e Superfície → Calcule → Exporte para Excel")
