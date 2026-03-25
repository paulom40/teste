import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from io import BytesIO
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Tennis Predictor", layout="wide")

# ============================================================
# CONSTANTS
# ============================================================

ELO_START = 1500
K = 32

FEATURE_COLS = [
    "elo_diff",
    "elo_abs",
    "surface_enc",
    "round_enc"
]

SURFACE_ENC = {"Clay": 0, "Hard": 1, "Grass": 2}

# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    url = "https://github.com/paulom40/teste/raw/main/Challenger.xlsx"
    df = pd.read_excel(url)

    # DATE FIX (CRÍTICO)
    df["Date"] = pd.to_datetime(
        df["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    df = df.dropna(subset=["Date"])

    df["Surface"] = df.get("surface", "Hard")
    df["Winner"] = df.get("winner_name")
    df["Loser"] = df.get("loser_name")

    # SCORE FIX (CRÍTICO)
    df["score"] = df.get("score", df.get("Score"))

    return df.sort_values("Date")


# ============================================================
# SCORE PARSER
# ============================================================

def parse_games(score):
    if pd.isna(score):
        return np.nan

    sets = re.findall(r"(\d+)-(\d+)", str(score))
    if len(sets) == 0:
        return np.nan

    return sum(int(a) + int(b) for a, b in sets)


# ============================================================
# ELO SYSTEM
# ============================================================

class EloSystem:
    def __init__(self):
        self.elo = {}

    def get(self, p):
        return self.elo.get(p, ELO_START)

    def expected(self, ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update(self, w, l):
        ra, rb = self.get(w), self.get(l)
        exp = self.expected(ra, rb)

        self.elo[w] = ra + K * (1 - exp)
        self.elo[l] = rb + K * (0 - (1 - exp))

    def snapshot(self, w, l):
        ra, rb = self.get(w), self.get(l)

        return {
            "elo_w": ra,
            "elo_l": rb,
            "elo_diff": ra - rb,
            "elo_abs": abs(ra - rb),
        }


# ============================================================
# BUILD FEATURES
# ============================================================

def build(df):
    elo = EloSystem()
    rows = []

    for _, r in df.iterrows():
        w, l = r["Winner"], r["Loser"]

        if pd.isna(w) or pd.isna(l):
            continue

        snap = elo.snapshot(w, l)

        total_games = parse_games(r["score"])

        rows.append({
            **snap,
            "surface_enc": SURFACE_ENC.get(r["Surface"], 1),
            "round_enc": 1,
            "total_games": total_games
        })

        elo.update(w, l)

    if len(rows) == 0:
        st.error("❌ Nenhum jogo processado")
        st.stop()

    df_feat = pd.DataFrame(rows)

    # DEBUG
    st.write("Colunas:", df_feat.columns.tolist())

    if "total_games" not in df_feat.columns:
        st.error("❌ total_games não existe")
        st.stop()

    df_feat = df_feat[df_feat["total_games"].notna()]

    if len(df_feat) < 50:
        st.error("❌ Poucos jogos com score válido")
        st.stop()

    return elo, df_feat


# ============================================================
# TRAIN MODEL
# ============================================================

def train(df):
    X = df[FEATURE_COLS]
    y = (df["total_games"] >= 22).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    return model, acc


# ============================================================
# API MATCHES
# ============================================================

def fetch_matches():
    url = "https://api.api-tennis.com/tennis/"
    key = "YOUR_API_KEY"

    try:
        r = requests.get(url, params={
            "method": "get_fixtures",
            "APIkey": key
        }, timeout=10)

        data = r.json()

        if data.get("success") != 1:
            return pd.DataFrame()

        df = pd.DataFrame(data["result"])

        df["Winner"] = df["event_first_player"]
        df["Loser"] = df["event_second_player"]
        df["Surface"] = "Hard"

        return df[["Winner", "Loser", "Surface"]]

    except:
        return pd.DataFrame()


# ============================================================
# PREDICT
# ============================================================

def predict(matches, elo, model):
    rows = []

    for _, m in matches.iterrows():
        snap = elo.snapshot(m["Winner"], m["Loser"])

        feats = {
            **snap,
            "surface_enc": SURFACE_ENC.get(m["Surface"], 1),
            "round_enc": 1
        }

        rows.append(feats)

    if len(rows) == 0:
        return matches

    X = pd.DataFrame(rows)[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]

    matches["Prob Over 22"] = probs

    return matches.sort_values("Prob Over 22", ascending=False)


# ============================================================
# STREAMLIT APP
# ============================================================

st.title("🎾 Tennis Predictor (Stable Version)")

df = load_data()

st.write(f"Jogos carregados: {len(df)}")

elo, feat_df = build(df)

model, acc = train(feat_df)

st.metric("Accuracy", f"{acc:.2%}")

# ============================================================
# PREDICTIONS
# ============================================================

st.header("📅 Previsões")

matches = fetch_matches()

if matches.empty:
    st.warning("Sem jogos da API")
else:
    preds = predict(matches, elo, model)

    st.dataframe(preds, use_container_width=True)

    st.subheader("🔥 TOP 5")
    for i, r in preds.head(5).iterrows():
        st.write(
            f"{r['Winner']} vs {r['Loser']} → {r['Prob Over 22']:.1%}"
        )
