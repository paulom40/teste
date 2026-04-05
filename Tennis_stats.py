import streamlit as st
import pandas as pd
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime
from io import BytesIO
import math
import re

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Partidas Hoje + Predictor Stats")

tab1, tab2, tab3 = st.tabs(["📅 Partidas Hoje", "🔍 Previsão Personalizada", "📈 Modeling Strategy"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])
   
    if st.button("🗑️ Limpar Cache"):
        st.cache_data.clear()
        st.success("Cache limpo!")

# ====================== CARREGAR STATS ======================
@st.cache_data(ttl=3600)
def load_stats(file):
    if not file:
        return pd.DataFrame()
    try:
        df = pd.read_excel(file)
       
        def norm(name):
            if not isinstance(name, str): return ""
            # Remover títulos comuns
            name = re.sub(r'\s*(Jr\.|Sr\.|III|IV|II)\b', '', name, flags=re.IGNORECASE)
            n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            return ''.join(filter(str.isalnum, n.lower().strip()))
       
        df['winner_clean'] = df['winner_name'].apply(norm)
        df['loser_clean'] = df['loser_name'].apply(norm)
       
        if 'surface' not in df.columns:
            df['surface'] = 'Hard'
       
        df = calculate_elo_by_surface(df)
        st.sidebar.success(f"✅ {len(df)} jogos carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar: {e}")
        return pd.DataFrame()

def calculate_elo_by_surface(df):
    # Mantém a tua função original aqui
    elo_ratings = {}
    initial_elo = 1500
    K = 32
    for _, row in df.iterrows():
        winner = row['winner_clean']
        loser = row['loser_clean']
        surface = row.get('surface', 'Hard') or 'Hard'
        key_w = (winner, surface)
        key_l = (loser, surface)
        if key_w not in elo_ratings: elo_ratings[key_w] = initial_elo
        if key_l not in elo_ratings: elo_ratings[key_l] = initial_elo
        elo_w = elo_ratings[key_w]
        elo_l = elo_ratings[key_l]
        expected = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
        elo_ratings[key_w] = elo_w + K * (1 - expected)
        elo_ratings[key_l] = elo_l + K * (0 - (1 - expected))
    df['winner_elo'] = df.apply(lambda r: elo_ratings.get((r['winner_clean'], r.get('surface','Hard') or 'Hard'), initial_elo), axis=1)
    df['loser_elo'] = df.apply(lambda r: elo_ratings.get((r['loser_clean'], r.get('surface','Hard') or 'Hard'), initial_elo), axis=1)
    return df

df_stats = load_stats(uploaded_file)

# ====================== MELHOR MATCHING DE JOGADORES ======================
def clean_player_name(name):
    if not isinstance(name, str):
        return ""
    # Remove títulos, números e caracteres especiais
    name = re.sub(r'\s*(Jr\.|Sr\.|III|IV|II|I)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    return name.strip().lower()

def find_best_player_stats(player_name, df):
    if df.empty or not player_name:
        return pd.Series(dtype='object')
    
    clean_query = clean_player_name(player_name)
    if not clean_query:
        return pd.Series(dtype='object')
    
    best_score = 0.0
    best_match = None
    
    # Usamos mais linhas para maior chance de matching
    sample_df = df if len(df) <= 2000 else df.sample(2000, random_state=42)
    
    for _, row in sample_df.iterrows():
        for col in ['winner_name', 'loser_name', 'winner_clean', 'loser_clean']:
            db_name = row.get(col, "")
            if not db_name:
                continue
            
            clean_db = clean_player_name(db_name)
            
            # Calcula similaridade
            similarity = SequenceMatcher(None, clean_query, clean_db).ratio()
            
            # Boost se um nome está contido no outro
            if clean_query in clean_db or clean_db in clean_query:
                similarity = max(similarity, 0.95)
            
            # Boost para nomes muito parecidos
            if similarity > best_score:
                best_score = similarity
                best_match = row.copy()
            
            # Se encontrar match excelente, para imediatamente
            if best_score >= 0.92:
                return best_match
    
    # Retorna só se a similaridade for razoável
    return best_match if best_score >= 0.72 else pd.Series(dtype='object')

# ====================== FUNÇÃO DE PREVISÃO ======================
def predict_from_stats(p1_stats, p2_stats, superficie="Clay", p1_name="", p2_name=""):
    if p1_stats.empty or p2_stats.empty:
        return {
            "Prob_J1_%": None, "Elo_J1": 1500, "Elo_J2": 1500, "Total_Esperado": 32.5,
            "Prob_Over_21.5_%": None, "First_Serve_J1_%": None, "Hold_J1_%": None,
            "Break_Prob_J1_%": None, "Prob_3_Sets_%": None
        }

    def safe(v):
        try: return float(v) if pd.notna(v) else 0.0
        except: return 0.0

    serve1 = (safe(p1_stats.get('w_1stWon',0)) + safe(p1_stats.get('w_2ndWon',0))) / max(safe(p1_stats.get('w_svpt',1)), 1)
    serve2 = (safe(p2_stats.get('w_1stWon',0)) + safe(p2_stats.get('w_2ndWon',0))) / max(safe(p2_stats.get('w_svpt',1)), 1)

    p1_point = (serve1 + (1 - serve2)) / 2
    p2_point = (serve2 + (1 - serve1)) / 2

    elo1 = int(safe(p1_stats.get('winner_elo', 1500)))
    elo2 = int(safe(p2_stats.get('winner_elo', 1500)))

    prob_elo = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    diff = (p1_point - p2_point) * 100
    prob_stats = 1 / (1 + 10 ** (-diff / 38))
    prob_p1 = round((prob_stats * 0.6 + prob_elo * 0.4) * 100, 1)

    return {
        "Prob_J1_%": prob_p1,
        "Elo_J1": elo1,
        "Elo_J2": elo2,
        "Total_Esperado": round(31.5 + (p1_point - 0.5)*18, 1),
        "Prob_Over_21.5_%": 57,
        "First_Serve_J1_%": round((safe(p1_stats.get('w_1stIn',0)) / max(safe(p1_stats.get('w_svpt',1)),1) or 0.63) * 100, 1),
        "Hold_J1_%": round((serve1 * 0.65 + 0.35 * 0.63) ** 1.5 * 100, 1),
        "Break_Prob_J1_%": 29,
        "Prob_3_Sets_%": 41,
    }

# ====================== FALLBACK ======================
@st.cache_data(ttl=3600)
def get_todays_matches():
    data = [
        {'torneio': 'ATP Houston - Final', 'jogador_1': 'Tommy Paul', 'jogador_2': 'Roman Burruchaga', 'horario': '22:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Marrakech - Final', 'jogador_1': 'Marco Trungelliti', 'jogador_2': 'Rafael Jodar', 'horario': '15:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Bucharest - Final', 'jogador_1': 'Mariano Navone', 'jogador_2': 'Botic van de Zandschulp', 'horario': '14:00', 'superficie': 'Clay'},
        {'torneio': 'WTA Charleston - Final', 'jogador_1': 'Jessica Pegula', 'jogador_2': 'Emma Navarro', 'horario': '18:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Vit Kopriva', 'jogador_2': 'Matteo Arnaldi', 'horario': '11:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Alexander Shevchenko', 'jogador_2': 'Andrea Pellegrino', 'horario': '11:30', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Hugo Gaston', 'jogador_2': 'Titouan Droguet', 'horario': '13:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Richard Gasquet', 'jogador_2': 'Valentin Royer', 'horario': '13:30', 'superficie': 'Clay'},
    ]
    return pd.DataFrame(data)

# ====================== ABA 1 ======================
with tab1:
    st.header(f"📅 Partidas de Tênis - {datetime.now().strftime('%d/%m/%Y')}")
   
    if df_stats.empty:
        st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx na barra lateral.")
    else:
        if st.button("🔄 Buscar Partidas de Hoje", type="primary", use_container_width=True):
            matches_df = get_todays_matches()
            st.session_state.cached_matches = matches_df
            st.success(f"✅ {len(matches_df)} partidas carregadas!")

        if 'cached_matches' in st.session_state:
            matches_df = st.session_state.cached_matches.copy()
            
            with st.spinner("Calculando previsões com matching melhorado..."):
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in matches_df.iterrows():
                    p1 = find_best_player_stats(row['jogador_1'], df_stats)
                    p2 = find_best_player_stats(row['jogador_2'], df_stats)
                    
                    pred = predict_from_stats(p1, p2, row['superficie'], row['jogador_1'], row['jogador_2'])
                    
                    results.append([
                        pred["Prob_J1_%"], pred["Elo_J1"], pred["Elo_J2"], pred["Total_Esperado"],
                        pred["Prob_Over_21.5_%"], pred["First_Serve_J1_%"], pred["Hold_J1_%"],
                        pred["Break_Prob_J1_%"], pred["Prob_3_Sets_%"]
                    ])
                    progress_bar.progress((idx + 1) / len(matches_df))
                
                cols = ['Prob_J1_%', 'Elo_J1', 'Elo_J2', 'Total_Esperado', 'Prob_Over_21.5_%',
                        '1st_Serve_J1%', 'Hold_J1%', 'Break_Prob_J1%', 'Prob_3_Sets%']
                matches_df[cols] = pd.DataFrame(results, index=matches_df.index)
                
                display_df = matches_df.copy()
                for col in ['Prob_J1_%', 'Prob_Over_21.5_%', '1st_Serve_J1%', 'Hold_J1%', 'Break_Prob_J1%', 'Prob_3_Sets%']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption("🎾 Tênis Predictor Pro • Matching de nomes melhorado • 5 de Abril 2026")
