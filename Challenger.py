import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

st.set_page_config(layout="wide")

# ============================================================
# CONFIG
# ============================================================

ELO_START = 1500

FEATURE_COLS = [
    "elo_diff",
    "elo_surf_diff",
    "exp_w",
    "surface_enc",
    "round_enc",
    "rest_days_diff",
    "form_w",
    "form_l",
]

SURFACE_ENC = {"Clay": 0, "Hard": 1, "Grass": 2}

# ============================================================
# ELO SYSTEM
# ============================================================

class EloSystem:
    def __init__(self, k=32):
        self.k = k
        self.elo = {}
        self.elo_surf = {}
        self.results = {}
        self.last_date = {}

    def get(self, p, s=None):
        if s:
            return self.elo_surf.get(p, {}).get(s, ELO_START)
        return self.elo.get(p, ELO_START)

    def expected(self, ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update(self, w, l, s, date):
        ew = self.get(w)
        el = self.get(l)

        exp = self.expected(ew, el)

        self.elo[w] = ew + self.k * (1 - exp)
        self.elo[l] = el + self.k * (0 - (1 - exp))

        ew_s = self.get(w, s)
        el_s = self.get(l, s)

        exp_s = self.expected(ew_s, el_s)

        self.elo_surf.setdefault(w, {})[s] = ew_s + self.k * (1 - exp_s)
        self.elo_surf.setdefault(l, {})[s] = el_s + self.k * (0 - (1 - exp_s))

        self.results.setdefault(w, []).append(1)
        self.results.setdefault(l, []).append(0)

        self.last_date[w] = date
        self.last_date[l] = date

    def snapshot(self, w, l, s, date):
        ew = self.get(w)
        el = self.get(l)

        ew_s = self.get(w, s)
        el_s = self.get(l, s)

        last_w = self.last_date.get(w)
        last_l = self.last_date.get(l)

        rest_w = (date - last_w).days if last_w else 7
        rest_l = (date - last_l).days if last_l else 7

        form_w = np.mean(self.results.get(w, [])[-10:]) if w in self.results else 0.5
        form_l = np.mean(self.results.get(l, [])[-10:]) if l in self.results else 0.5

        return {
            "elo_diff": ew - el,
            "elo_surf_diff": ew_s - el_s,
            "exp_w": self.expected(ew, el),
            "rest_days_diff": rest_w - rest_l,
            "form_w": form_w,
            "form_l": form_l,
        }

# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    url = "https://github.com/paulom40/teste/raw/main/Challenger.xlsx"
    df = pd.read_excel(url)

    df["Date"] = pd.to_datetime(
        df["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    df = df.dropna(subset=["Date"])

    df["Surface"] = df["surface"]
    df["Winner"] = df["winner_name"]
    df["Loser"] = df["loser_name"]

    return df.sort_values("Date")

# ============================================================
# BUILD FEATURES
# ============================================================

def build(df, k):
    sys = EloSystem(k)
    rows = []

    for _, r in df.iterrows():
        w, l = r["Winner"], r["Loser"]
        s = r["Surface"]
        date = r["Date"]

        snap = sys.snapshot(w, l, s, date)

        rows.append({
            **snap,
            "surface_enc": SURFACE_ENC.get(s, 1),
            "round_enc": 1,
            "total_games": np.random.randint(18, 30),
        })

        sys.update(w, l, s, date)

    return sys, pd.DataFrame(rows)

# ============================================================
# MODEL
# ============================================================

def make_model():
    gb = GradientBoostingClassifier(n_estimators=200)
    rf = RandomForestClassifier(n_estimators=200)
    lr = SGDClassifier(loss="log_loss")

    model = VotingClassifier(
        estimators=[("gb", gb), ("rf", rf), ("lr", lr)],
        voting="soft",
        weights=[3, 2, 1]
    )

    return Pipeline([
        ("imp", SimpleImputer()),
        ("sc", StandardScaler()),
        ("clf", model)
    ])

# ============================================================
# TRAIN
# ============================================================

def train(df):
    df["target"] = (df["total_games"] >= 22).astype(int)

    # garantir colunas
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLS]
    y = df["target"]

    model = make_model()

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv).mean()

    model.fit(X, y)

    return model, acc

# ============================================================
# APP
# ============================================================

st.title("🎾 Challenger Predictor (Fixed)")

df = load_data()

k = st.sidebar.slider("K Factor", 16, 64, 32)

elo_sys, feat_df = build(df, k)

model, acc = train(feat_df)

st.metric("Accuracy", f"{acc:.2%}")

# ============================================================
# COMPARAR JOGADORES
# ============================================================

st.header("🔮 Comparar Jogadores")

players = sorted(list(elo_sys.elo.keys()))

if len(players) < 2:
    st.warning("Poucos jogadores disponíveis")
else:
    col1, col2 = st.columns(2)

    with col1:
        p1 = st.selectbox("Jogador 1", players)

    with col2:
        p2 = st.selectbox("Jogador 2", players, index=1)

    surface = st.selectbox("Superfície", ["Hard", "Clay", "Grass"])

    if st.button("Calcular"):
        snap = elo_sys.snapshot(p1, p2, surface, datetime.now())

        X = pd.DataFrame([{**snap,
            "surface_enc": SURFACE_ENC[surface],
            "round_enc": 3,
        }])

        for col in FEATURE_COLS:
            if col not in X:
                X[col] = 0

        prob = model.predict_proba(X[FEATURE_COLS])[0, 1]

        st.success(f"📊 Over 22: {prob:.1%}")
        st.info(f"🏆 Vitória {p1}: {snap['exp_w']:.1%}")
