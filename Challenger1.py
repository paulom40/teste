import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import requests
from io import BytesIO
from datetime import datetime
import re
import warnings

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
        while len(w) < 5: w.append(np.nan)
        while len(l) < 5: l.append(np.nan)
        wsets = sum(1 for a, b in zip(w[:5], l[:5])
                    if not (pd.isna(a) or pd.isna(b)) and a > b)
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
# 3. SIDEBAR — CARREGAR HISTÓRICO
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
# 4. MODELO PARA PROBABILIDADE DE 3+ SETS
# ============================================================

def train_three_sets_model(df_hist):
    dfm = df_hist.copy()

    if "Wsets" not in dfm.columns:
        st.error("Erro: coluna Wsets não existe.")
        return None

    dfm["Total_Games"] = add_total_games_col(dfm)
    dfm["three_sets"] = (dfm["Wsets"] >= 2).astype(int)

    features = ["WRank", "LRank", "WPts", "LPts", "Total_Games"]
    dfm = dfm.dropna(subset=features + ["three_sets"])

    if len(dfm) < 80:
        st.warning("Poucos dados para treinar o modelo.")
        return None

    X = dfm[features]
    y = dfm["three_sets"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    return model


# ============================================================
# 5. API — JOGOS DE HOJE E AMANHÃ
# ============================================================

def fetch_matches_from_api():
    """
    Ajusta esta função à tua API real.
    Tem de devolver colunas: Date, Winner, Loser, Surface
    """
    API_URL = "COLOCA_AQUI_A_TUA_URL"
    API_KEY = "COLOCA_AQUI_A_TUA_API_KEY"

    headers = {"Authorization": f"Bearer {API_KEY}"}

    try:
        r = requests.get(API_URL, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"Erro ao obter jogos da API: {e}")
        return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])

    df_api = pd.DataFrame(data)

    # Ajustar nomes conforme JSON real
    if "date" in df_api.columns:
        df_api["Date"] = pd.to_datetime(df_api["date"]).dt.normalize()

    rename_map = {}
    if "player1" in df_api.columns: rename_map["player1"] = "Winner"
    if "player2" in df_api.columns: rename_map["player2"] = "Loser"
    if "surface" in df_api.columns: rename_map["surface"] = "Surface"

    df_api = df_api.rename(columns=rename_map)

    for col in ["Winner", "Loser", "Surface"]:
        if col not in df_api.columns:
            df_api[col] = ""

    return df_api[["Date", "Winner", "Loser", "Surface"]]


def get_today_and_tomorrow_matches(df_matches):
    if df_matches.empty:
        return df_matches

    dfm = df_matches.copy()
    dfm["Date"] = pd.to_datetime(dfm["Date"]).dt.normalize()

    today = pd.Timestamp.today().normalize()
    tomorrow = today + pd.Timedelta(days=1)

    return dfm[(dfm["Date"] == today) | (dfm["Date"] == tomorrow)]


def predict_three_sets_for_upcoming(upcoming_df, hist_df, model):
    df_up = upcoming_df.copy()

    hist_sorted = hist_df.sort_values("Date")

    w_rank_map = hist_sorted.groupby("Winner")["WRank"].last()
    l_rank_map = hist_sorted.groupby("Loser")["LRank"].last()
    w_pts_map  = hist_sorted.groupby("Winner")["WPts"].last()
    l_pts_map  = hist_sorted.groupby("Loser")["LPts"].last()

    df_up["WRank"] = df_up["Winner"].map(w_rank_map).fillna(200)
    df_up["LRank"] = df_up["Loser"].map(l_rank_map).fillna(200)
    df_up["WPts"]  = df_up["Winner"].map(w_pts_map).fillna(50)
    df_up["LPts"]  = df_up["Loser"].map(l_pts_map).fillna(50)

    df_up["Total_Games"] = 22

    features = ["WRank", "LRank", "WPts", "LPts", "Total_Games"]
    df_up["prob_3_sets"] = model.predict_proba(df_up[features])[:, 1]

    return df_up.sort_values("prob_3_sets", ascending=False)
# ============================================================
# 6. UI PRINCIPAL — PREDIÇÃO 3+ SETS
# ============================================================

st.title("🎾 CHALLENGER — Predição de jogos com 3+ sets")
st.caption(f"Fonte de dados: {source_name} | Total de jogos históricos: {len(df)}")

with st.spinner("A treinar modelo de probabilidade de 3+ sets..."):
    model_3sets = train_three_sets_model(df)

if model_3sets is None:
    st.error("Não foi possível treinar o modelo de 3+ sets.")
    st.stop()

st.markdown("---")
st.header("📅 Jogos de hoje e amanhã")

with st.spinner("A obter jogos da API..."):
    api_matches = fetch_matches_from_api()

if api_matches.empty:
    st.warning("Nenhum jogo obtido da API.")
    st.stop()

upcoming = get_today_and_tomorrow_matches(api_matches)

if upcoming.empty:
    st.info("A API não devolveu jogos para hoje ou amanhã.")
    st.stop()

preds = predict_three_sets_for_upcoming(upcoming, df, model_3sets)

st.subheader("📋 Jogos ordenados por probabilidade de 3+ sets")
st.dataframe(
    preds[["Date", "Winner", "Loser", "Surface", "prob_3_sets"]]
    .style.format({"prob_3_sets": "{:.1%}"})
)

st.subheader("🔥 TOP 5 jogos mais prováveis de irem a 3+ sets")
top_n = min(5, len(preds))
for _, row in preds.head(top_n).iterrows():
    st.markdown(
        f"**{row['Winner']} vs {row['Loser']}** — "
        f"{row['prob_3_sets']:.1%} probabilidade "
        f"({row['Surface']}, {row['Date'].date()})"
    )
