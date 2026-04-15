import pandas as pd
import numpy as np
import streamlit as st
import re
import io

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
# 🧠 NORMALIZE SURFACE
# =========================
def normalize_surface(s):
    if pd.isna(s):
        return "Hard"

    s = str(s).lower()

    if "clay" in s:
        return "Clay"
    if "grass" in s:
        return "Grass"
    if "carpet" in s:
        return "Hard"

    return "Hard"

# =========================
# 📂 LOAD DATA
# =========================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    # limpar colunas
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # renomear
    df.rename(columns={
        "winner_name": "winner",
        "loser_name": "loser",
        "tourney_date": "date",
        "t_games": "total_games"
    }, inplace=True)

    # validar colunas essenciais
    required = ["winner", "loser", "date"]
    for col in required:
        if col not in df.columns:
            st.error(f"❌ Coluna obrigatória não encontrada: {col}")
            st.stop()

    # converter data
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # surface limpa
    df["surface"] = df["surface"].apply(normalize_surface)

    # total games fallback
    if "total_games" not in df.columns:
        def extract_games(score):
            if pd.isna(score):
                return 22
            nums = [int(x) for x in re.findall(r'\d+', str(score))]
            return sum(nums) if nums else 22

        df["total_games"] = df["score"].apply(extract_games)

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

        # streak
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

    s1, s2 = stats[p1], stats[p2]

    if surf not in ["Hard","Clay","Grass"]:
        surf = "Hard"

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
    f = build_features(p1, p2, surf, stats)

    if f is None:
        return None

    prob = model.predict_proba([f])[0][1]

    # filtro de edge
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
# 🚀 APP
# =========================
st.title("🎾 Tennis Predictor PRO (FINAL)")

file = st.file_uploader("Upload ATP dataset", type=["xlsx"])

if file:
    df = load_data(file)

    st.write("📊 Superfícies:")
    st.write(df["surface"].value_counts())

    stats = compute_stats(df)
    model = train(df, stats)

    st.success("✅ Model trained!")

    # =========================
    # INPUT MANUAL
    # =========================
    st.header("🎯 Predict Match")

    p1 = st.text_input("Player 1")
    p2 = st.text_input("Player 2")
    surf = st.selectbox("Surface", ["Hard","Clay","Grass"])

    if st.button("Predict"):
        pred = predict(model, stats, p1, p2, surf)

        if pred:
            conf = pred["prob"]
            cls = "high" if conf > 0.65 else "medium" if conf > 0.55 else "low"

            st.markdown(f"""
            <div class="card">
            <h3>{pred['match']}</h3>
            🏆 <span class="{cls}">{pred['winner']} ({conf:.1%})</span><br>
            📊 {p1}: {pred['p1']:.1%} | {p2}: {pred['p2']:.1%}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No edge (skip)")

    # =========================
    # 🔥 TOP PICKS
    # =========================
    st.header("🔥 Top Picks")

    players = list(stats.keys())
    picks = []

    for i in range(min(40, len(players))):
        for j in range(i+1, min(40, len(players))):
            for surf in ["Hard","Clay"]:
                pred = predict(model, stats, players[i], players[j], surf)
                if pred:
                    picks.append(pred)

    picks = sorted(picks, key=lambda x: x["prob"], reverse=True)[:10]

    df_export = pd.DataFrame(picks)

    for p in picks:
        cls = "high" if p["prob"] > 0.65 else "medium"

        st.markdown(f"""
        <div class="card">
        <b>{p['surface']}</b><br>
        {p['match']}<br>
        🏆 <span class="{cls}">{p['winner']} ({p['prob']:.1%})</span>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # 📥 EXPORT
    # =========================
    buffer = io.BytesIO()
    df_export.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        "📥 Download Picks Excel",
        buffer,
        file_name="top_picks.xlsx"
    )
