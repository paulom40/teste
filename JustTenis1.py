import pandas as pd
import numpy as np
import streamlit as st
from collections import defaultdict
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="🎾 Tennis Predictor PRO", layout="wide")

# =========================
# 🎾 SURFACE DETECTION
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
# 🧠 SURFACE ELO
# =========================
def calculate_surface_elo(df, k=32):
    players = set(df['winner']) | set(df['loser'])

    elo = {p: {"Hard":1500, "Clay":1500, "Grass":1500} for p in players}

    df = df.sort_values("date")

    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        surf = row.get("surface", "Hard")

        if surf not in ["Hard","Clay","Grass"]:
            surf = "Hard"

        r1 = elo[w][surf]
        r2 = elo[l][surf]

        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))

        elo[w][surf] += k * (1 - exp1)
        elo[l][surf] -= k * (1 - exp1)

    return elo

# =========================
# 📊 STATS + MOMENTUM
# =========================
def compute_stats(df):
    stats = {}
    surface_elo = calculate_surface_elo(df)

    for p in set(df['winner']) | set(df['loser']):
        matches = df[(df['winner']==p) | (df['loser']==p)]

        if len(matches) < 5:
            continue

        wins = (matches['winner'] == p).sum()
        total = len(matches)

        recent = matches.sort_values("date", ascending=False).head(10)
        recent_form = (recent['winner'] == p).mean()

        # streak
        streak = 0
        for _, row in matches.sort_values("date", ascending=False).iterrows():
            if row['winner'] == p:
                streak += 1
            else:
                break

        stats[p] = {
            "surface_elo": surface_elo[p],
            "win_rate": wins / total,
            "recent_form": recent_form,
            "streak": streak,
            "matches": total
        }

    return stats

# =========================
# 🧩 FEATURES
# =========================
def build_features(p1, p2, surface, stats, match=None):
    if p1 not in stats or p2 not in stats:
        return None

    s1, s2 = stats[p1], stats[p2]

    surf = surface if surface in ["Hard","Clay","Grass"] else "Hard"

    feat = [
        # 🎾 SURFACE ELO (mais importante)
        s1["surface_elo"][surf] - s2["surface_elo"][surf],

        # performance
        s1["win_rate"] - s2["win_rate"],
        s1["recent_form"] - s2["recent_form"],

        # momentum
        s1["streak"] - s2["streak"],

        # experience
        np.log(s1["matches"] + 1) - np.log(s2["matches"] + 1),
    ]

    # 🎲 ODDS (se houver)
    if match and match.get("odd_p1") and match.get("odd_p2"):
        p1_imp = 1 / match["odd_p1"]
        p2_imp = 1 / match["odd_p2"]
        total = p1_imp + p2_imp
        p1_imp /= total

        feat.append(p1_imp - 0.5)
    else:
        feat.append(0)

    return feat

# =========================
# 🤖 TRAIN MODEL
# =========================
def train_model(df, stats):
    X, y = [], []

    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        surf = row.get("surface","Hard")

        f1 = build_features(w, l, surf, stats)
        f2 = build_features(l, w, surf, stats)

        if f1:
            X.append(f1)
            y.append(1)

        if f2:
            X.append(f2)
            y.append(0)

    X = np.array(X)
    y = np.array(y)

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
    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], preds))

    st.write(f"📊 Accuracy: {np.mean(scores):.3f}")

    model.fit(X, y)

    return model

# =========================
# 🎯 PREDICT
# =========================
def predict(model, stats, match):
    f = build_features(
        match["player1"],
        match["player2"],
        match["surface"],
        stats,
        match
    )

    if f is None:
        return None

    prob = model.predict_proba([f])[0][1]

    # 🔥 EDGE FILTER
    if abs(prob - 0.5) < 0.05:
        return None

    return {
        "winner": match["player1"] if prob > 0.5 else match["player2"],
        "confidence": max(prob, 1 - prob),
        "p1_prob": prob,
        "p2_prob": 1 - prob
    }

# =========================
# 🚀 STREAMLIT UI
# =========================
st.title("🎾 Tennis Predictor PRO (Surface ELO)")

uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    df.columns = [c.lower() for c in df.columns]

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['surface'] = df['surface'].apply(normalize_surface)

    st.write(f"📁 Matches: {len(df)}")

    stats = compute_stats(df)
    st.write(f"👥 Players: {len(stats)}")

    model = train_model(df, stats)

    st.success("✅ Model trained!")

    st.subheader("🎯 Predict Match")

    p1 = st.text_input("Player 1")
    p2 = st.text_input("Player 2")
    surface = st.selectbox("Surface", ["Hard","Clay","Grass"])

    if st.button("Predict"):
        match = {
            "player1": p1,
            "player2": p2,
            "surface": surface,
            "odd_p1": None,
            "odd_p2": None
        }

        pred = predict(model, stats, match)

        if pred:
            st.success(f"🏆 {pred['winner']} ({pred['confidence']:.1%})")
            st.write(f"{p1}: {pred['p1_prob']:.1%}")
            st.write(f"{p2}: {pred['p2_prob']:.1%}")
        else:
            st.warning("⚠️ No edge (skip match)")
