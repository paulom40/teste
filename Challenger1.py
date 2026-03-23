import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import requests
from io import BytesIO
from datetime import datetime, timedelta
import re
import warnings
import json

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CHALLENGER 3+ Sets Predictor",
    page_icon="🎾",
    layout="wide"
)

# ============================================================
# 1. NORMALIZAÇÃO E PARSING
# ============================================================

def normalize_columns(df):
    col_map = {
        "winner_name": "Winner", "loser_name": "Loser",
        "winner_rank": "WRank",  "loser_rank": "LRank",
        "winner_rank_points": "WPts", "loser_rank_points": "LPts",
        "tourney_date": "Date",  "score": "Score",
        "best_of": "BestOf",     "round": "Round", "minutes": "Minutes",
        "winner_hand": "WHand",  "loser_hand": "LHand",
        "winner_ht": "WHt",      "loser_ht": "LHt",
        "winner_age": "WAge",    "loser_age": "LAge",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "Score" in df.columns:
        df = parse_score(df)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")

    if "Surface" not in df.columns:
        for alt in ["surface", "tourney_surface"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Surface"})
                break
        else:
            df["Surface"] = "Hard"

    return df


def parse_score(df):
    def _parse(score):
        if pd.isna(score):
            return [np.nan] * 11
        sets = re.findall(r"(\d+)-(\d+)(?:\(\d+\))?", str(score))
        w = [int(s[0]) for s in sets]
        l = [int(s[1]) for s in sets]
        while len(w) < 5:
            w.append(np.nan)
        while len(l) < 5:
            l.append(np.nan)
        wsets = sum(
            1 for a, b in zip(w[:5], l[:5])
            if not (pd.isna(a) or pd.isna(b)) and a > b
        )
        return w[:5] + l[:5] + [wsets]

    parsed = df["Score"].apply(_parse)
    cols = ["W1","W2","W3","W4","W5","L1","L2","L3","L4","L5","Wsets"]
    for i, col in enumerate(cols):
        df[col] = [row[i] for row in parsed]
    return df


def add_total_games_col(df):
    total = pd.Series(0.0, index=df.index)
    for i in range(1, 6):
        wc, lc = f"W{i}", f"L{i}"
        if wc in df.columns and lc in df.columns:
            w = pd.to_numeric(df[wc], errors="coerce").fillna(-1)
            l = pd.to_numeric(df[lc], errors="coerce").fillna(-1)
            valid = (w >= 0) & (l >= 0)
            total += np.where(valid, w + l, 0)
    return total.where(total > 0, other=np.nan)


def extract_surface_from_tournament(tournament_name):
    """
    Extrai o tipo de superfície do nome do torneio
    """
    if not isinstance(tournament_name, str):
        return "Hard"
    
    tournament_lower = tournament_name.lower()
    
    if "clay" in tournament_lower:
        return "Clay"
    elif "grass" in tournament_lower or "wimbledon" in tournament_lower:
        return "Grass"
    elif "hard" in tournament_lower:
        return "Hard"
    else:
        return "Hard"


# ============================================================
# 2. CARREGAR HISTÓRICO (GitHub ou Excel)
# ============================================================

def fetch_challenger_github_data():
    try:
        url = "https://github.com/paulom40/teste/raw/main/Challenger.xlsx"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        df = pd.read_excel(BytesIO(response.content))
        df = normalize_columns(df)
        return df, "GitHub Challenger Database"
    except Exception as e:
        st.warning(f"Não foi possível obter dados do GitHub: {e}")
        return None, None


def load_custom_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df = normalize_columns(df)
        st.sidebar.success(f"Histórico carregado: {uploaded_file.name}")
        return df, uploaded_file.name
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar ficheiro: {e}")
        return None, None


# ============================================================
# 3. API — JOGOS DE HOJE E AMANHÃ (VERSÃO CORRIGIDA)
# ============================================================

def fetch_matches_from_api():
    """
    Busca os jogos da API Tennis com tratamento de erros melhorado.
    """
    
    # Configuração da API
    API_URL = "https://api.api-tennis.com/tennis/"
    API_KEY = "7e3c6125ceaf5442372a487f9948c083a8778bb9604f49d8b33efc0e005f275c"
    
    # Datas para buscar (hoje e amanhã)
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Parâmetros da API
    params = {
        "method": "get_fixtures",
        "APIkey": API_KEY,
        "date_start": today,
        "date_stop": tomorrow,
    }
    
    try:
        with st.spinner(f"A buscar jogos de {today} a {tomorrow}..."):
            response = requests.get(API_URL, params=params, timeout=15)
            
            # Verificar status HTTP
            if response.status_code != 200:
                st.error(f"API retornou status {response.status_code}")
                st.code(f"Resposta: {response.text[:500]}")
                return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])
            
            # Verificar se a resposta está vazia
            if not response.text:
                st.error("API retornou resposta vazia")
                return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])
            
            # Tentar fazer parse do JSON
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                st.error(f"Erro ao decodificar JSON: {e}")
                st.write("Resposta da API (primeiros 500 caracteres):")
                st.code(response.text[:500])
                return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])
        
        # Verificar se a API retornou sucesso
        if data.get("success") != 1:
            st.error(f"API retornou erro: {data.get('error', 'Erro desconhecido')}")
            return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])
        
        matches = data.get("result", [])
        
        if not matches:
            st.info(f"Nenhum jogo encontrado para {today} e {tomorrow}.")
            
            # Tentar buscar apenas jogos de hoje
            params_today = {
                "method": "get_fixtures",
                "APIkey": API_KEY,
                "date_start": today,
                "date_stop": today,
            }
            response_today = requests.get(API_URL, params=params_today, timeout=15)
            data_today = response_today.json()
            matches_today = data_today.get("result", [])
            
            if matches_today:
                st.info(f"Encontrados {len(matches_today)} jogos apenas para hoje.")
                matches = matches_today
            else:
                return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])
        
        # Converter para DataFrame
        df_api = pd.DataFrame(matches)
        
        # Verificar se as colunas esperadas existem
        required_cols = ["event_date", "event_first_player", "event_second_player"]
        missing_cols = [col for col in required_cols if col not in df_api.columns]
        
        if missing_cols:
            st.error(f"Colunas em falta na resposta da API: {missing_cols}")
            st.write("Colunas disponíveis:", list(df_api.columns))
            return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])
        
        # Mapear os campos da API
        df_api["Date"] = pd.to_datetime(df_api["event_date"])
        df_api["Winner"] = df_api["event_first_player"]
        df_api["Loser"] = df_api["event_second_player"]
        
        # Extrair superfície do nome do torneio
        if "tournament_name" in df_api.columns:
            df_api["Surface"] = df_api["tournament_name"].apply(extract_surface_from_tournament)
        else:
            df_api["Surface"] = "Hard"
        
        # Filtrar apenas jogos que ainda não começaram (se a coluna existir)
        if "event_status" in df_api.columns:
            df_api = df_api[df_api["event_status"] == ""]
        
        # Selecionar apenas as colunas necessárias
        result_df = df_api[["Date", "Winner", "Loser", "Surface"]].copy()
        
        # Remover duplicados e valores nulos
        result_df = result_df.drop_duplicates()
        result_df = result_df.dropna(subset=["Winner", "Loser"])
        
        if len(result_df) > 0:
            st.success(f"✅ Encontrados {len(result_df)} jogos para previsão")
        
        return result_df
        
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão com a API: {e}")
        return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])
    except Exception as e:
        st.error(f"Erro inesperado: {e}")
        return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])


def get_today_and_tomorrow_matches(df_matches):
    if df_matches.empty:
        return df_matches

    dfm = df_matches.copy()
    dfm["Date"] = pd.to_datetime(dfm["Date"]).dt.normalize()

    today = pd.Timestamp.now().normalize()
    tomorrow = today + pd.Timedelta(days=1)

    filtered = dfm[(dfm["Date"] == today) | (dfm["Date"] == tomorrow)]
    
    if len(filtered) == 0:
        st.info(f"Nenhum jogo para hoje ({today.date()}) ou amanhã ({tomorrow.date()})")
    
    return filtered


# ============================================================
# 4. MODELO PARA PROBABILIDADE DE 3+ SETS
# ============================================================

def train_three_sets_model(df_hist):
    dfm = df_hist.copy()

    if "Wsets" not in dfm.columns:
        st.error("Erro: coluna Wsets não existe.")
        return None

    # Calcular variável alvo: 3+ sets
    dfm["three_sets"] = (dfm["Wsets"] >= 2).astype(int)
    
    # Features para o modelo
    features = ["WRank", "LRank", "WPts", "LPts"]
    
    # Verificar se todas as features existem
    missing_features = [f for f in features if f not in dfm.columns]
    if missing_features:
        st.error(f"Features em falta no dataset: {missing_features}")
        return None
    
    # Remover linhas com valores nulos
    dfm = dfm.dropna(subset=features + ["three_sets"])

    if len(dfm) < 30:
        st.warning(f"Apenas {len(dfm)} jogos com dados completos. Mínimo recomendado: 30")
        return None

    X = dfm[features]
    y = dfm["three_sets"]

    # Treinar modelo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Calcular acurácia para feedback
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test) if len(X_test) > 0 else 0
    
    st.sidebar.success(f"✅ Modelo treinado com {len(dfm)} jogos")
    st.sidebar.info(f"📊 Acurácia - Treino: {train_score:.1%} | Teste: {test_score:.1%}")

    return model


def predict_three_sets_for_upcoming(upcoming_df, hist_df, model):
    df_up = upcoming_df.copy()

    # Obter últimos rankings conhecidos para cada jogador
    hist_sorted = hist_df.sort_values("Date")

    # Mapear últimos valores conhecidos
    w_rank_map = hist_sorted.groupby("Winner")["WRank"].last().to_dict()
    l_rank_map = hist_sorted.groupby("Loser")["LRank"].last().to_dict()
    w_pts_map  = hist_sorted.groupby("Winner")["WPts"].last().to_dict()
    l_pts_map  = hist_sorted.groupby("Loser")["LPts"].last().to_dict()

    # Preencher features com valores padrão quando não encontrados
    df_up["WRank"] = df_up["Winner"].map(w_rank_map).fillna(300)
    df_up["LRank"] = df_up["Loser"].map(l_rank_map).fillna(300)
    df_up["WPts"]  = df_up["Winner"].map(w_pts_map).fillna(30)
    df_up["LPts"]  = df_up["Loser"].map(l_pts_map).fillna(30)

    # Garantir que os valores são numéricos
    for col in ["WRank", "LRank", "WPts", "LPts"]:
        df_up[col] = pd.to_numeric(df_up[col], errors="coerce").fillna(300)

    features = ["WRank", "LRank", "WPts", "LPts"]
    
    # Prever probabilidades
    df_up["prob_3_sets"] = model.predict_proba(df_up[features])[:, 1]

    return df_up.sort_values("prob_3_sets", ascending=False)


# ============================================================
# 5. SIDEBAR — CARREGAR HISTÓRICO
# ============================================================

st.sidebar.header("📂 Histórico de jogos")

uploaded_file = st.sidebar.file_uploader(
    "Escolhe um ficheiro Excel (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:
    df, source_name = load_custom_excel(uploaded_file)
else:
    df, source_name = fetch_challenger_github_data()

if df is None or df.empty:
    st.error("Não foi possível carregar nenhum histórico.")
    st.stop()

st.sidebar.info(f"Fonte atual: {source_name}")
st.sidebar.write(f"Total de jogos: {len(df)}")


# ============================================================
# 6. UI PRINCIPAL — PREDIÇÃO 3+ SETS
# ============================================================

st.title("🎾 CHALLENGER — Predição de jogos com 3+ sets")
st.caption(f"Fonte de dados: {source_name} | Total de jogos históricos: {len(df)}")

# Treinar modelo
with st.spinner("A treinar modelo de probabilidade de 3+ sets..."):
    model_3sets = train_three_sets_model(df)

if model_3sets is None:
    st.error("Não foi possível treinar o modelo de 3+ sets.")
    st.stop()

st.markdown("---")
st.header("📅 Jogos de hoje e amanhã")

# Buscar jogos da API
api_matches = fetch_matches_from_api()

if api_matches.empty:
    st.warning("⚠️ Não foi possível obter jogos da API.")
    
    # Opção para carregar jogos manualmente
    st.subheader("📤 Carrega um ficheiro com jogos para prever")
    st.markdown("""
    O ficheiro Excel deve conter as colunas:
    - **Date** (data do jogo)
    - **Winner** (nome do jogador favorito)
    - **Loser** (nome do adversário)
    - **Surface** (opcional: Clay, Grass, Hard)
    """)
    
    manual_file = st.file_uploader(
        "Escolhe um ficheiro Excel (.xlsx)",
        type=["xlsx"],
        key="manual_matches"
    )
    
    if manual_file is not None:
        try:
            manual_df = pd.read_excel(manual_file)
            required_cols = ["Date", "Winner", "Loser"]
            missing = [col for col in required_cols if col not in manual_df.columns]
            
            if missing:
                st.error(f"Colunas em falta no ficheiro: {missing}")
                st.stop()
            
            manual_df["Date"] = pd.to_datetime(manual_df["Date"])
            if "Surface" not in manual_df.columns:
                manual_df["Surface"] = "Hard"
            
            upcoming = manual_df
            st.success(f"✅ Carregados {len(upcoming)} jogos para previsão")
            
        except Exception as e:
            st.error(f"Erro ao carregar ficheiro: {e}")
            st.stop()
    else:
        st.info("👆 Carrega um ficheiro Excel para fazer previsões ou verifica a ligação à API.")
        st.stop()
else:
    upcoming = get_today_and_tomorrow_matches(api_matches)
    
    if upcoming.empty:
        st.info("📭 Nenhum jogo encontrado para hoje ou amanhã.")
        
        # Mostrar todos os jogos disponíveis para debug
        if len(api_matches) > 0:
            st.write("Jogos disponíveis na API:")
            st.dataframe(api_matches)
        st.stop()

# Fazer previsões
preds = predict_three_sets_for_upcoming(upcoming, df, model_3sets)

# Mostrar resultados
st.subheader("📋 Jogos ordenados por probabilidade de 3+ sets")
st.dataframe(
    preds[["Date", "Winner", "Loser", "Surface", "prob_3_sets"]]
    .style.format({"prob_3_sets": "{:.1%}"})
)

st.subheader("🔥 TOP 5 jogos mais prováveis de irem a 3+ sets")
top_n = min(5, len(preds))
for _, row in preds.head(top_n).iterrows():
    prob_percent = row['prob_3_sets'] * 100
    st.markdown(
        f"**{row['Winner']} vs {row['Loser']}** — "
        f"**{prob_percent:.1f}%** probabilidade de 3+ sets "
        f"({row['Surface']}, {row['Date'].date()})"
    )

# Estatísticas adicionais
st.markdown("---")
st.subheader("📊 Estatísticas")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jogos no histórico", len(df))
with col2:
    st.metric("Jogos para prever", len(preds))
with col3:
    prob_media = preds["prob_3_sets"].mean()
    st.metric("Probabilidade média", f"{prob_media:.1%}")

# Mostrar distribuição de probabilidades se houver dados suficientes
if len(preds) > 1:
    st.subheader("📈 Distribuição das probabilidades")
    st.bar_chart(preds["prob_3_sets"].value_counts(bins=10).sort_index())
