import pandas as pd
import numpy as np
import streamlit as st
import re
import io
import requests
import unicodedata

from datetime import datetime, timedelta
from difflib import get_close_matches
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="🎾 Tennis Predictor PRO", layout="wide")

# =========================
# 🎨 UI
# =========================
st.markdown("""
<style>
.card {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 20px;
    border-radius: 15px;
    color: white;
    margin-bottom: 15px;
}
.high {color:#00ff88; font-weight:bold;}
.medium {color:#ffd700; font-weight:bold;}
.low {color:#ff6b6b; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# =========================
# 🧠 NORMALIZAÇÃO DE NOMES
# =========================
def normalize_name(name):
    if pd.isna(name):
        return None

    name = str(name).lower().strip()

    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])

    name = re.sub(r'[^a-z\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    return name

def match_player(name, players):
    name = normalize_name(name)
    match = get_close_matches(name, players, n=1, cutoff=0.8)
    return match[0] if match else None

# =========================
# 🎾 SURFACE NORMALIZATION
# =========================
def normalize_surface(s):
    if pd.isna(s):
        return "Hard"

    s = str(s).lower()

    if "clay" in s:
        return "Clay"
    if "grass" in s:
        return "Grass"

    return "Hard"

# =========================
# 📂 LOAD DATA
# =========================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df.rename(columns={
        "winner_name": "winner",
        "loser_name": "loser",
        "tourney_date": "date",
        "t_games": "total_games"
    }, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["surface"] = df["surface"].apply(normalize_surface)

    # NORMALIZAR NOMES
    df["winner"] = df["winner"].apply(normalize_name)
    df["loser"] = df["loser"].apply(normalize_name)

    return df

# =========================
# 🎾 SURFACE ELO
# =========================
def calculate_surface_elo(df):
    players = set(df["winner"]) | set(df["loser"])

    elo = {p: {"Hard":1500,"Clay":1500,"Grass":1500} for p in players}

    df = df.sort_values("date")

    for _, row in df.iterrows():
        w, l = row["winner"], row["loser"]
        surf = row["surface"]

        if surf not in ["Hard","Clay","Grass"]:
            surf = "Hard"

        r1, r2 = elo[w][surf], elo[l][surf]

        exp = 1 / (1 + 10 ** ((r2 - r1)/400))

        elo[w][surf] += 32 * (1 - exp)
        elo[l][surf] -= 32 * (1 - exp)

    return elo

# =========================
# 📊 STATS
# =========================
def compute_stats(df):
    stats = {}
    surface_elo = calculate_surface_elo(df)

    for p in set(df["winner"]) | set(df["loser"]):
        m = df[(df["winner"]==p) | (df["loser"]==p)]

        if len(m) < 5:
            continue

        wins = (m["winner"] == p).sum()

        recent = m.sort_values("date", ascending=False).head(10)
        recent_form = (recent["winner"] == p).mean()

        streak = 0
        for _, row in m.sort_values("date", ascending=False).iterrows():
            if row["winner"] == p:
                streak += 1
            else:
                break

        stats[p] = {
            "elo": surface_elo[p],
            "win_rate": wins / len(m),
            "recent": recent_form,
            "streak": streak,
            "matches": len(m)
        }

    return stats

# =========================
# 🧩 FEATURES
# =========================
def build_features(p1, p2, surf, stats):
    if p1 not in stats or p2 not in stats:
        return None

    if surf not in ["Hard","Clay","Grass"]:
        surf = "Hard"

    s1, s2 = stats[p1], stats[p2]

    return [
        s1["elo"][surf] - s2["elo"][surf],
        s1["win_rate"] - s2["win_rate"],
        s1["recent"] - s2["recent"],
        s1["streak"] - s2["streak"],
        np.log(s1["matches"]+1) - np.log(s2["matches"]+1)
    ]

# =========================
# 🤖 TRAIN
# =========================
def train(df, stats):
    X, y = [], []

    for _, r in df.iterrows():
        f1 = build_features(r["winner"], r["loser"], r["surface"], stats)
        f2 = build_features(r["loser"], r["winner"], r["surface"], stats)

        if f1:
            X.append(f1); y.append(1)
        if f2:
            X.append(f2); y.append(0)

    X, y = np.array(X), np.array(y)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for tr, te in tscv.split(X):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        scores.append(accuracy_score(y[te], pred))

    st.write(f"📊 Accuracy: {np.mean(scores):.3f}")

    model.fit(X, y)
    return model

# =========================
# 🎯 PREDICT
# =========================
def predict(model, stats, p1, p2, surf):

    players = list(stats.keys())

    p1 = match_player(p1, players)
    p2 = match_player(p2, players)

    if not p1 or not p2:
        return None

    f = build_features(p1, p2, surf, stats)

    if f is None:
        return None

    prob = model.predict_proba([f])[0][1]

    if abs(prob - 0.5) < 0.05:
        return None

    return {
        "match": f"{p1} vs {p2}",
        "winner": p1 if prob > 0.5 else p2,
        "prob": max(prob, 1 - prob),
        "p1": prob,
        "p2": 1 - prob,
        "surface": surf
    }

# =========================
# 📡 SCRAPER (SOFASCORE)
# =========================
def get_matches(days_ahead=0):
    try:
        date = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{date}"

        r = requests.get(url, timeout=10)
        data = r.json()

        matches = []

        for ev in data.get("events", []):
            try:
                cat = ev["tournament"]["category"]["name"]
                if "WTA" in cat.upper():
                    continue

                matches.append({
                    "tournament": ev["tournament"]["name"],
                    "player1": ev["homeTeam"]["name"],
                    "player2": ev["awayTeam"]["name"],
                    "surface": "Hard"
                })
            except:
                continue

        return matches

    except:
        return []

# =========================
# 🚀 APP
# =========================
st.title("🎾 Tennis Predictor PRO FINAL")

file = st.file_uploader("Upload ATP dataset", type=["xlsx"])

if file:
    df = load_data(file)

    stats = compute_stats(df)
    model = train(df, stats)

    st.success("✅ Model Ready")

    # BOTÕES
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📅 Hoje"):
            st.session_state.matches = get_matches(0)
    with col2:
        if st.button("📅 Amanhã"):
            st.session_state.matches = get_matches(1)

    # MOSTRAR JOGOS
    if "matches" in st.session_state:
        st.header("🎯 Jogos do Dia")

        results = []

        for m in st.session_state.matches:
            pred = predict(model, stats, m["player1"], m["player2"], m["surface"])

            if pred:
                results.append(pred)

                cls = "high" if pred["prob"]>0.65 else "medium"

                st.markdown(f"""
                <div class="card">
                <b>{m['tournament']}</b><br>
                {pred['match']}<br>
                🏆 <span class="{cls}">{pred['winner']} ({pred['prob']:.1%})</span>
                </div>
                """, unsafe_allow_html=True)

        # EXPORT
        if results:
            df_exp = pd.DataFrame(results)

            buffer = io.BytesIO()
            df_exp.to_excel(buffer, index=False)
            buffer.seek(0)

            st.download_button("📥 Download Picks", buffer, "picks.xlsx")
