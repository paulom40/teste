# ============================================================
#   🔥 PREDIÇÃO DE 3+ SETS PARA JOGOS DE HOJE E AMANHÃ
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import requests

# ============================================================
# 1. Treinar modelo para prever probabilidade de 3+ sets
# ============================================================

def train_three_sets_model(df_hist):
    df = df_hist.copy()

    if "Wsets" not in df.columns:
        st.error("Erro: coluna Wsets não existe. O parse_score não correu.")
        return None

    df["Total_Games"] = add_total_games_col(df)
    df["three_sets"] = (df["Wsets"] >= 2).astype(int)

    features = ["WRank", "LRank", "WPts", "LPts", "Total_Games"]
    df = df.dropna(subset=features + ["three_sets"])

    if len(df) < 80:
        st.warning("Poucos dados para treinar o modelo de 3 sets.")
        return None

    X = df[features]
    y = df["three_sets"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    return model


# ============================================================
# 2. Buscar jogos da API (corrigido)
# ============================================================

def fetch_matches_from_api():
    url = "https://api-tennis.com/admin"

    api_key = "7e3c6125ceaf5442372a487f9948c083a8778bb9604f49d8b33efc0e005f275c"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data)

    # Ajusta estes nomes conforme o JSON real da tua API
    df["Date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.rename(columns={
        "player1": "Winner",
        "player2": "Loser",
        "surface": "Surface"
    })

    return df


# ============================================================
# 3. Filtrar jogos de hoje e amanhã
# ============================================================

def get_today_and_tomorrow_matches(df_matches):
    if df_matches.empty:
        return df_matches

    df = df_matches.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    today = pd.Timestamp.today().normalize()
    tomorrow = today + pd.Timedelta(days=1)

    return df[(df["Date"] == today) | (df["Date"] == tomorrow)]


# ============================================================
# 4. Prever probabilidade de 3+ sets para jogos futuros
# ============================================================

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
# 5. Secção Streamlit final
# ============================================================

st.markdown("---")
st.header("🎾 Jogos de Hoje e Amanhã — Probabilidade de 3+ Sets")

df_hist = df.copy()

with st.spinner("A treinar modelo de 3+ sets..."):
    model_3sets = train_three_sets_model(df_hist)

if model_3sets is None:
    st.error("Não foi possível treinar o modelo.")
    st.stop()

api_matches = fetch_matches_from_api()

if api_matches.empty:
    st.warning("Nenhum jogo obtido da API.")
    st.stop()

upcoming = get_today_and_tomorrow_matches(api_matches)

if upcoming.empty:
    st.info("A API não devolveu jogos para hoje ou amanhã.")
    st.stop()

preds = predict_three_sets_for_upcoming(upcoming, df_hist, model_3sets)

st.subheader("📋 Jogos ordenados por probabilidade de 3+ sets")
st.dataframe(
    preds[["Date", "Winner", "Loser", "Surface", "prob_3_sets"]]
    .style.format({"prob_3_sets": "{:.1%}"})
)

st.subheader("🔥 TOP 5 jogos mais prováveis de irem a 3+ sets")
for _, row in preds.head(5).iterrows():
    st.markdown(
        f"**{row['Winner']} vs {row['Loser']}** — "
        f"{row['prob_3_sets']:.1%} probabilidade "
        f"({row['Surface']}, {row['Date'].date()})"
    )
