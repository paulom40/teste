"""
CHALLENGER TENNIS PREDICTOR v4 — Dynamic Elo + Serve Features
================================================================
Key additions vs v3:
- Overall Elo per player (chronological, from match history)
- Surface-specific Elo (Clay / Hard / Grass)
- Dynamic K-factor: adjusts by round, player experience, and match closeness
- Rolling serve stats: 1stWon%, 2ndWon%, ace%, bp_save%  (last 15 matches)
- All Elo features are leakage-free: only data BEFORE each match is used
- Custom prediction tab for head-to-head analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import requests
import warnings
from io import BytesIO
from datetime import datetime, timedelta

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Challenger Predictor v4 — Dynamic Elo",
    page_icon="🎾",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

ELO_START    = 1500.0
DEFAULT_BASE_K = 32.0
SURFACES     = ["Clay", "Hard", "Grass"]
SERVE_WINDOW = 15   # rolling window for serve stats

# These will be updated dynamically
ROUND_K_MULT = {"R32": 0.9, "R16": 1.0, "QF": 1.1, "SF": 1.3, "F": 1.5, "Final": 1.5}
SURFACE_ENC  = {"Clay": 0, "Hard": 1, "Grass": 2}
ROUND_ENC    = {"R32": 1, "R16": 2, "QF": 3, "SF": 4, "F": 5, "Final": 5}

FEATURE_COLS = [
    # Elo features
    "elo_diff", "elo_surf_diff", "elo_abs", "elo_surf_abs",
    "elo_w", "elo_l", "elo_w_surf", "elo_l_surf",
    "exp_w", "exp_w_surf",
    # Context
    "surface_enc", "round_enc", "indoor_enc",
    # Rolling serve stats — Winner
    "w_1w_pct", "w_2w_pct", "w_ace_pct", "w_bp_save", "w_bp_faced_pg",
    # Rolling serve stats — Loser
    "l_1w_pct", "l_2w_pct", "l_ace_pct", "l_bp_save", "l_bp_faced_pg",
    # Combined
    "serve_dom_sum", "ace_sum", "bp_save_diff",
    # Experience
    "n_w", "n_l",
    # Rolling avg games
    "w_avg_games", "l_avg_games", "avg_games_combined",
    "w_surf_games", "l_surf_games",
]


# ─────────────────────────────────────────────────────────────
# HELPERS — SCORE PARSING
# ─────────────────────────────────────────────────────────────

def parse_total_games(score):
    if pd.isna(score):
        return np.nan
    s = str(score)
    if any(x in s for x in ("RET", "W/O", "DEF")):
        return np.nan
    sets = re.findall(r"(\d+)-(\d+)(?:\(\d+\))?", s)
    return float(sum(int(a) + int(b) for a, b in sets)) if sets else np.nan


def count_sets(score):
    if pd.isna(score):
        return np.nan
    s = str(score)
    if any(x in s for x in ("RET", "W/O")):
        return np.nan
    return float(len(re.findall(r"\d+-\d+", s)))


# ─────────────────────────────────────────────────────────────
# ELO ENGINE
# ─────────────────────────────────────────────────────────────

def elo_expected(ra: float, rb: float) -> float:
    """Probability that player A beats player B."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def dynamic_k(n_matches: int, round_str: str, is_3set: bool, base_k: float) -> float:
    """
    K-factor that varies by:
      - Round: Finals count more (better opponents, more context)
      - Experience: New players have higher K (faster convergence)
      - Match closeness: 3-set matches reveal more true level
    """
    round_mult = ROUND_K_MULT.get(round_str, 1.0)
    if n_matches < 10:
        exp_mult = 1.5
    elif n_matches < 30:
        exp_mult = 1.2
    else:
        exp_mult = 1.0
    close_mult = 1.2 if is_3set else 1.0
    return base_k * round_mult * exp_mult * close_mult


class EloSystem:
    """Tracks Elo ratings (overall + surface-specific) for all players."""

    def __init__(self, base_k: float = DEFAULT_BASE_K):
        self.base_k = base_k
        self.elo: dict[str, float] = {}
        self.elo_surf: dict[str, dict[str, float]] = {}
        self.n_matches: dict[str, int] = {}
        self.serve_history: dict[str, list] = {}   # player -> list of per-match serve dicts
        self.games_history: dict[str, list] = {}   # player -> list of (surface, total_games)

    def set_base_k(self, base_k: float):
        """Update base K-factor"""
        self.base_k = base_k

    def get(self, player: str, surface: str | None = None) -> float:
        if surface:
            return self.elo_surf.setdefault(player, {}).get(surface, ELO_START)
        return self.elo.get(player, ELO_START)

    def update(self, winner: str, loser: str, surface: str,
               round_str: str, is_3set: bool, total_games: float,
               w_serve: dict, l_serve: dict):
        """Process one match: update Elos and history."""
        ew = self.get(winner)
        el = self.get(loser)
        ew_s = self.get(winner, surface)
        el_s = self.get(loser, surface)

        n_w = self.n_matches.get(winner, 0)
        n_l = self.n_matches.get(loser, 0)

        kw = dynamic_k(n_w, round_str, is_3set, self.base_k)
        kl = dynamic_k(n_l, round_str, is_3set, self.base_k)

        # Overall Elo
        exp_w = elo_expected(ew, el)
        new_ew = ew + kw * (1.0 - exp_w)
        new_el = el + kl * (0.0 - (1.0 - exp_w))
        
        self.elo[winner] = new_ew
        self.elo[loser] = new_el

        # Surface Elo
        exp_w_s = elo_expected(ew_s, el_s)
        new_ew_s = ew_s + kw * (1.0 - exp_w_s)
        new_el_s = el_s + kl * (0.0 - (1.0 - exp_w_s))
        
        self.elo_surf.setdefault(winner, {})[surface] = new_ew_s
        self.elo_surf.setdefault(loser, {})[surface] = new_el_s

        # n_matches
        self.n_matches[winner] = n_w + 1
        self.n_matches[loser]  = n_l + 1

        # Serve history
        if w_serve and any(v is not None and not np.isnan(v) for v in w_serve.values()):
            self.serve_history.setdefault(winner, []).append(w_serve)
        if l_serve and any(v is not None and not np.isnan(v) for v in l_serve.values()):
            self.serve_history.setdefault(loser, []).append(l_serve)

        # Games history
        if not np.isnan(total_games):
            self.games_history.setdefault(winner, []).append((surface, total_games))
            self.games_history.setdefault(loser, []).append((surface, total_games))

    def snapshot(self, winner: str, loser: str, surface: str) -> dict:
        """Return pre-match Elo features for a given matchup."""
        ew    = self.get(winner)
        el    = self.get(loser)
        ew_s  = self.get(winner, surface)
        el_s  = self.get(loser, surface)
        return {
            "elo_w": ew, "elo_l": el,
            "elo_diff": ew - el,
            "elo_abs": abs(ew - el),
            "elo_w_surf": ew_s, "elo_l_surf": el_s,
            "elo_surf_diff": ew_s - el_s,
            "elo_surf_abs": abs(ew_s - el_s),
            "exp_w": elo_expected(ew, el),
            "exp_w_surf": elo_expected(ew_s, el_s),
            "n_w": self.n_matches.get(winner, 0),
            "n_l": self.n_matches.get(loser, 0),
        }

    def rolling_serve(self, player: str) -> dict:
        """Return rolling-average serve stats for a player (last SERVE_WINDOW matches)."""
        history = self.serve_history.get(player, [])
        recent = history[-SERVE_WINDOW:]
        def mean_key(key):
            vals = [x[key] for x in recent if x.get(key) is not None and not np.isnan(x[key])]
            return float(np.mean(vals)) if vals else np.nan
        return {
            "1w_pct": mean_key("1w_pct"),
            "2w_pct": mean_key("2w_pct"),
            "ace_pct": mean_key("ace_pct"),
            "bp_save": mean_key("bp_save"),
            "bp_faced_pg": mean_key("bp_faced_pg"),
        }

    def rolling_games(self, player: str, surface: str | None = None, window: int = 20) -> dict:
        """Return rolling average total games for a player (optionally surface-filtered)."""
        all_g = self.games_history.get(player, [])
        recent_all = [g for _, g in all_g[-window:]]
        recent_surf = [g for s, g in all_g if s == surface][-15:]
        return {
            "avg_games": float(np.mean(recent_all)) if len(recent_all) >= 3 else np.nan,
            "surf_games": float(np.mean(recent_surf)) if len(recent_surf) >= 3 else np.nan,
        }

    def final_ratings(self) -> pd.DataFrame:
        """Return a DataFrame with each player's final Elo ratings."""
        players = set(self.elo.keys())
        rows = []
        for p in players:
            rows.append({
                "Player": p,
                "Elo": round(self.get(p), 1),
                "Elo_Clay": round(self.get(p, "Clay"), 1),
                "Elo_Hard": round(self.get(p, "Hard"), 1),
                "Elo_Grass": round(self.get(p, "Grass"), 1),
                "Matches": self.n_matches.get(p, 0),
            })
        return pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# LOAD & NORMALIZE DATA
# ─────────────────────────────────────────────────────────────

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "winner_name": "Winner", "loser_name": "Loser",
        "winner_rank": "WRank", "loser_rank": "LRank",
        "winner_rank_points": "WPts", "loser_rank_points": "LPts",
        "surface": "Surface", "indoor": "Indoor", "round": "Round",
        "score": "Score", "best_of": "BestOf", "minutes": "Minutes",
        "winner_age": "WAge", "loser_age": "LAge",
        "winner_ht": "WHt", "loser_ht": "LHt",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Date
    if "tourney_date.1" in df.columns:
        df["Date"] = pd.to_datetime(df["tourney_date.1"], errors="coerce")
    elif "tourney_date" in df.columns:
        df["Date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["Date"] = pd.NaT

    for col, default in [("Surface", "Hard"), ("Indoor", "O"), ("Round", "R32")]:
        if col not in df.columns:
            df[col] = default

    df["TotalGames"] = df["Score"].apply(parse_total_games) if "Score" in df.columns else np.nan

    # Per-match serve percentages
    for pfx, svpt_col, fi_col, fw_col, sw_col, ace_col, bpf_col, bps_col, svg_col in [
        ("W", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_ace", "w_bpFaced", "w_bpSaved", "w_SvGms"),
        ("L", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_ace", "l_bpFaced", "l_bpSaved", "l_SvGms"),
    ]:
        if all(c in df.columns for c in [svpt_col, fi_col, fw_col, sw_col]):
            s = pd.to_numeric(df[svpt_col], errors="coerce").replace(0, np.nan)
            fi = pd.to_numeric(df[fi_col], errors="coerce").replace(0, np.nan)
            se = (pd.to_numeric(df[svpt_col], errors="coerce") - pd.to_numeric(df[fi_col], errors="coerce")).replace(0, np.nan)
            df[f"{pfx}1wPct"] = pd.to_numeric(df[fw_col], errors="coerce") / fi
            df[f"{pfx}2wPct"] = pd.to_numeric(df[sw_col], errors="coerce") / se
            if ace_col in df.columns:
                df[f"{pfx}AcePct"] = pd.to_numeric(df[ace_col], errors="coerce") / s
            if bpf_col in df.columns and bps_col in df.columns:
                bpf = pd.to_numeric(df[bpf_col], errors="coerce").replace(0, np.nan)
                bps = pd.to_numeric(df[bps_col], errors="coerce")
                df[f"{pfx}BpSave"] = bps / bpf
            if bpf_col in df.columns and svg_col in df.columns:
                svg = pd.to_numeric(df[svg_col], errors="coerce").replace(0, np.nan)
                df[f"{pfx}BpFacedPG"] = pd.to_numeric(df[bpf_col], errors="coerce") / svg

    return df


@st.cache_data(show_spinner=False)
def fetch_github_data():
    try:
        url = "https://github.com/paulom40/teste/raw/main/Challenger.xlsx"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_excel(BytesIO(r.content))
        return normalize_df(df), "GitHub Challenger Database"
    except Exception as e:
        st.warning(f"GitHub fetch failed: {e}")
        return None, None


def load_excel(uploaded):
    try:
        df = pd.read_excel(uploaded)
        return normalize_df(df), uploaded.name
    except Exception as e:
        st.sidebar.error(f"Load error: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────
# BUILD ELO + FEATURE MATRIX  (leakage-free)
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_elo_and_features(_df: pd.DataFrame, base_k: float):
    """
    Processes matches chronologically.
    For every match, records PRE-match Elo + serve features, then updates Elo.
    Returns: (EloSystem with final state, feature DataFrame)
    """
    df = _df.copy().sort_values("Date").reset_index(drop=True)
    sys = EloSystem(base_k=base_k)
    rows = []
    processed = 0
    skipped = 0

    for idx, row in df.iterrows():
        winner  = row.get("Winner")
        loser   = row.get("Loser")
        surface = row.get("Surface", "Hard")
        rnd     = row.get("Round", "R32")
        indoor  = row.get("Indoor", "O")
        tg      = row.get("TotalGames", np.nan)
        score   = str(row.get("Score", ""))

        if pd.isna(winner) or pd.isna(loser):
            skipped += 1
            continue

        is_3set = len(re.findall(r"\d+-\d+", score)) >= 3

        # ── Pre-match snapshot (these are the training features) ──
        elo_feats = sys.snapshot(winner, loser, surface)
        w_serve = sys.rolling_serve(winner)
        l_serve = sys.rolling_serve(loser)
        w_games = sys.rolling_games(winner, surface)
        l_games = sys.rolling_games(loser, surface)

        row_feats = {
            **elo_feats,
            "surface_enc": SURFACE_ENC.get(surface, 1),
            "round_enc": ROUND_ENC.get(rnd, 1),
            "indoor_enc": 1 if indoor == "I" else 0,
            # Serve — winner
            "w_1w_pct": w_serve["1w_pct"],
            "w_2w_pct": w_serve["2w_pct"],
            "w_ace_pct": w_serve["ace_pct"],
            "w_bp_save": w_serve["bp_save"],
            "w_bp_faced_pg": w_serve["bp_faced_pg"],
            # Serve — loser
            "l_1w_pct": l_serve["1w_pct"],
            "l_2w_pct": l_serve["2w_pct"],
            "l_ace_pct": l_serve["ace_pct"],
            "l_bp_save": l_serve["bp_save"],
            "l_bp_faced_pg": l_serve["bp_faced_pg"],
            # Combined serve
            "serve_dom_sum": (
                (w_serve["1w_pct"] or 0) * (w_serve["ace_pct"] or 0) +
                (l_serve["1w_pct"] or 0) * (l_serve["ace_pct"] or 0)
            ),
            "ace_sum": (w_serve["ace_pct"] or 0) + (l_serve["ace_pct"] or 0),
            "bp_save_diff": abs((w_serve["bp_save"] or 0.5) - (l_serve["bp_save"] or 0.5)),
            # Avg games
            "w_avg_games": w_games["avg_games"],
            "l_avg_games": l_games["avg_games"],
            "avg_games_combined": float(np.nanmean([w_games["avg_games"], l_games["avg_games"]])),
            "w_surf_games": w_games["surf_games"],
            "l_surf_games": l_games["surf_games"],
            # Target
            "total_games": tg,
        }
        rows.append(row_feats)
        processed += 1

        # ── Serve stats for this match (to feed into history) ──
        w_serve_this = {
            "1w_pct": row.get("W1wPct", np.nan),
            "2w_pct": row.get("W2wPct", np.nan),
            "ace_pct": row.get("WAcePct", np.nan),
            "bp_save": row.get("WBpSave", np.nan),
            "bp_faced_pg": row.get("WBpFacedPG", np.nan),
        }
        l_serve_this = {
            "1w_pct": row.get("L1wPct", np.nan),
            "2w_pct": row.get("L2wPct", np.nan),
            "ace_pct": row.get("LAcePct", np.nan),
            "bp_save": row.get("LBpSave", np.nan),
            "bp_faced_pg": row.get("LBpFacedPG", np.nan),
        }

        sys.update(winner, loser, surface, rnd, is_3set,
                   tg if not pd.isna(tg) else 0.0,
                   w_serve_this, l_serve_this)

    # Debug info
    if processed == 0:
        st.warning(f"Nenhum jogo processado! Skipped: {skipped}")
    else:
        st.sidebar.info(f"Processados: {processed} jogos | Skipped: {skipped}")

    return sys, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────────

def make_pipeline():
    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=15, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=15,
        n_jobs=-1, random_state=42
    )
    lr = LogisticRegression(C=0.5, max_iter=500)
    ens = VotingClassifier(
        estimators=[("gbm", gbm), ("rf", rf), ("lr", lr)],
        voting="soft", weights=[2, 1, 1]
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", ens),
    ])


@st.cache_resource(show_spinner=False)
def train_model(_feat_df: pd.DataFrame, threshold: int):
    df = _feat_df.dropna(subset=["total_games"]).copy()
    df["target"] = (df["total_games"] > threshold).astype(int)

    if len(df) < 50:
        return None, 0.0, 0.0, 0.0, 0.0, None

    X = df[FEATURE_COLS].copy()
    y = df["target"]

    # Global medians for fallback at prediction time
    global_medians = X.median().to_dict()

    pipe = make_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()
        cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    except Exception:
        cv_acc, cv_auc = 0.0, 0.0

    pipe.fit(X, y)
    avg_games = float(df["total_games"].mean())
    over_pct  = float(y.mean() * 100)

    return pipe, avg_games, over_pct, cv_acc, cv_auc, global_medians


# ─────────────────────────────────────────────────────────────
# API — TODAY & TOMORROW MATCHES
# ─────────────────────────────────────────────────────────────

def surf_from_name(name):
    if not isinstance(name, str):
        return "Hard"
    n = name.lower()
    if "clay" in n:
        return "Clay"
    if "grass" in n or "wimbledon" in n:
        return "Grass"
    return "Hard"


def fetch_api_matches() -> pd.DataFrame:
    API_URL = "https://api.api-tennis.com/tennis/"
    API_KEY = "7e3c6125ceaf5442372a487f9948c083a8778bb9604f49d8b33efc0e005f275c"
    today    = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        with st.spinner(f"Fetching matches {today} → {tomorrow}…"):
            resp = requests.get(
                API_URL,
                params={"method": "get_fixtures", "APIkey": API_KEY,
                        "date_start": today, "date_stop": tomorrow},
                timeout=15,
            )
        if resp.status_code != 200 or not resp.text:
            st.error(f"API status {resp.status_code}")
            return pd.DataFrame()
        data = resp.json()
        if data.get("success") != 1:
            st.error("API error response")
            return pd.DataFrame()
        matches = data.get("result", [])
        if not matches:
            st.info("No matches found for today/tomorrow.")
            return pd.DataFrame()

        df_api = pd.DataFrame(matches)
        df_api["Date"]    = pd.to_datetime(df_api.get("event_date", pd.Series(dtype=str)))
        df_api["Winner"]  = df_api.get("event_first_player", "")
        df_api["Loser"]   = df_api.get("event_second_player", "")
        df_api["Surface"] = df_api.get("tournament_name", pd.Series(dtype=str)).apply(surf_from_name)
        df_api["Indoor"]  = "O"
        df_api["Round"]   = df_api.get("event_round", pd.Series(dtype=str)).fillna("R32")

        if "event_status" in df_api.columns:
            df_api = df_api[df_api["event_status"] == ""]

        result = df_api[["Date", "Winner", "Loser", "Surface", "Indoor", "Round"]].copy()
        result = result.drop_duplicates().dropna(subset=["Winner", "Loser"])
        result = result[result["Winner"].str.strip() != ""]
        result = result[result["Loser"].str.strip() != ""]

        today_ts    = pd.Timestamp.now().normalize()
        tomorrow_ts = today_ts + pd.Timedelta(days=1)
        result["Date"] = pd.to_datetime(result["Date"]).dt.normalize()
        result = result[(result["Date"] == today_ts) | (result["Date"] == tomorrow_ts)]

        if len(result) > 0:
            st.success(f"✅ {len(result)} matches found")
        return result

    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# PREDICT UPCOMING MATCHES
# ─────────────────────────────────────────────────────────────

def predict_matches(upcoming: pd.DataFrame, elo_sys: EloSystem,
                    model, global_medians: dict, threshold: int) -> pd.DataFrame:
    rows = []
    for _, match in upcoming.iterrows():
        p1      = match["Winner"]
        p2      = match["Loser"]
        surface = match.get("Surface", "Hard")
        indoor  = match.get("Indoor", "O")
        rnd     = match.get("Round", "R32")

        elo_feats = elo_sys.snapshot(p1, p2, surface)
        w_serve   = elo_sys.rolling_serve(p1)
        l_serve   = elo_sys.rolling_serve(p2)
        w_games   = elo_sys.rolling_games(p1, surface)
        l_games   = elo_sys.rolling_games(p2, surface)

        feats = {
            **elo_feats,
            "surface_enc": SURFACE_ENC.get(surface, 1),
            "round_enc":   ROUND_ENC.get(rnd, 1),
            "indoor_enc":  1 if indoor == "I" else 0,
            "w_1w_pct":   w_serve["1w_pct"],
            "w_2w_pct":   w_serve["2w_pct"],
            "w_ace_pct":  w_serve["ace_pct"],
            "w_bp_save":  w_serve["bp_save"],
            "w_bp_faced_pg": w_serve["bp_faced_pg"],
            "l_1w_pct":   l_serve["1w_pct"],
            "l_2w_pct":   l_serve["2w_pct"],
            "l_ace_pct":  l_serve["ace_pct"],
            "l_bp_save":  l_serve["bp_save"],
            "l_bp_faced_pg": l_serve["bp_faced_pg"],
            "serve_dom_sum": (
                (w_serve["1w_pct"] or 0) * (w_serve["ace_pct"] or 0) +
                (l_serve["1w_pct"] or 0) * (l_serve["ace_pct"] or 0)
            ),
            "ace_sum": (w_serve["ace_pct"] or 0) + (l_serve["ace_pct"] or 0),
            "bp_save_diff": abs((w_serve["bp_save"] or 0.5) - (l_serve["bp_save"] or 0.5)),
            "w_avg_games": w_games["avg_games"],
            "l_avg_games": l_games["avg_games"],
            "avg_games_combined": float(np.nanmean([w_games["avg_games"], l_games["avg_games"]])),
            "w_surf_games": w_games["surf_games"],
            "l_surf_games": l_games["surf_games"],
        }
        rows.append(feats)

    if not rows:
        return upcoming

    feat_df = pd.DataFrame(rows)
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = global_medians.get(col, 0.0)
        else:
            feat_df[col] = feat_df[col].fillna(global_medians.get(col, 0.0))

    X = feat_df[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]

    result = upcoming.copy().reset_index(drop=True)
    result[f"prob_over_{threshold}"] = probs

    # Context columns
    result["elo_p1"]      = feat_df["elo_w"].values
    result["elo_p2"]      = feat_df["elo_l"].values
    result["elo_diff"]    = feat_df["elo_abs"].values
    result["exp_p1"]      = feat_df["exp_w"].values
    result["p1_in_hist"]  = result["Winner"].map(lambda p: p in elo_sys.elo)
    result["p2_in_hist"]  = result["Loser"].map(lambda p: p in elo_sys.elo)
    result["both_known"]  = result["p1_in_hist"] & result["p2_in_hist"]
    result["n_p1"]        = feat_df["n_w"].values
    result["n_p2"]        = feat_df["n_l"].values
    result["w_avg_games"] = feat_df["w_avg_games"].values
    result["l_avg_games"] = feat_df["l_avg_games"].values

    return result.sort_values(f"prob_over_{threshold}", ascending=False)


# ─────────────────────────────────────────────────────────────
# CUSTOM PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────

def calculate_custom_prediction(player1, player2, surface, elo_sys, model, global_medians, threshold=22):
    """
    Calcula previsões personalizadas para dois jogadores e uma superfície específica
    Retorna: (prob_over, prob_winner_win, elo_diff, exp_win, elo_p1, elo_p2)
    """
    # Obter features pré-jogo
    elo_feats = elo_sys.snapshot(player1, player2, surface)
    w_serve = elo_sys.rolling_serve(player1)
    l_serve = elo_sys.rolling_serve(player2)
    w_games = elo_sys.rolling_games(player1, surface)
    l_games = elo_sys.rolling_games(player2, surface)
    
    # Construir feature vector
    feats = {
        **elo_feats,
        "surface_enc": SURFACE_ENC.get(surface, 1),
        "round_enc": ROUND_ENC.get("QF", 1),  # Default para previsão personalizada
        "indoor_enc": 0,  # Default outdoor
        "w_1w_pct": w_serve["1w_pct"],
        "w_2w_pct": w_serve["2w_pct"],
        "w_ace_pct": w_serve["ace_pct"],
        "w_bp_save": w_serve["bp_save"],
        "w_bp_faced_pg": w_serve["bp_faced_pg"],
        "l_1w_pct": l_serve["1w_pct"],
        "l_2w_pct": l_serve["2w_pct"],
        "l_ace_pct": l_serve["ace_pct"],
        "l_bp_save": l_serve["bp_save"],
        "l_bp_faced_pg": l_serve["bp_faced_pg"],
        "serve_dom_sum": (
            (w_serve["1w_pct"] or 0) * (w_serve["ace_pct"] or 0) +
            (l_serve["1w_pct"] or 0) * (l_serve["ace_pct"] or 0)
        ),
        "ace_sum": (w_serve["ace_pct"] or 0) + (l_serve["ace_pct"] or 0),
        "bp_save_diff": abs((w_serve["bp_save"] or 0.5) - (l_serve["bp_save"] or 0.5)),
        "w_avg_games": w_games["avg_games"],
        "l_avg_games": l_games["avg_games"],
        "avg_games_combined": float(np.nanmean([w_games["avg_games"], l_games["avg_games"]])),
        "w_surf_games": w_games["surf_games"],
        "l_surf_games": l_games["surf_games"],
    }
    
    # Criar DataFrame com uma linha
    feat_df = pd.DataFrame([feats])
    
    # Preencher valores em falta com medianas globais
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = global_medians.get(col, 0.0)
        else:
            feat_df[col] = feat_df[col].fillna(global_medians.get(col, 0.0))
    
    X = feat_df[FEATURE_COLS]
    
    # Previsão de Over games
    prob_over = model.predict_proba(X)[0, 1]
    
    # Probabilidade do vencedor (baseada em Elo)
    prob_winner_win = elo_feats["exp_w"]
    
    # Diferença de Elo
    elo_diff = elo_feats["elo_diff"]
    
    # Elos individuais
    elo_p1 = elo_feats["elo_w"]
    elo_p2 = elo_feats["elo_l"]
    
    return prob_over, prob_winner_win, elo_diff, elo_feats["exp_w"], elo_p1, elo_p2


# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────

def export_excel(preds: pd.DataFrame, threshold: int, elo_ratings: pd.DataFrame) -> BytesIO:
    out = preds.copy()
    prob_col = f"prob_over_{threshold}"
    rename = {
        "Date": "Data", "Winner": "Jogador_1", "Loser": "Jogador_2",
        "Surface": "Superfície", "Round": "Ronda", "Indoor": "Indoor",
        prob_col: f"Prob_Over_{threshold}",
        "elo_p1": "Elo_J1", "elo_p2": "Elo_J2",
        "elo_diff": "Elo_Dif_Abs", "exp_p1": "Prob_Vitoria_J1",
        "n_p1": "N_Jogos_J1", "n_p2": "N_Jogos_J2",
        "w_avg_games": "Avg_Games_J1", "l_avg_games": "Avg_Games_J2",
        "both_known": "Ambos_no_Historico",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    desired = [
        "Data", "Jogador_1", "Jogador_2", "Superfície", "Ronda",
        f"Prob_Over_{threshold}",
        "Elo_J1", "Elo_J2", "Elo_Dif_Abs", "Prob_Vitoria_J1",
        "N_Jogos_J1", "N_Jogos_J2", "Avg_Games_J1", "Avg_Games_J2",
        "Ambos_no_Historico",
    ]
    existing = [c for c in desired if c in out.columns]
    out = out[existing]
    if "Data" in out.columns:
        out["Data"] = pd.to_datetime(out["Data"]).dt.strftime("%Y-%m-%d")
    pc = f"Prob_Over_{threshold}"
    if pc in out.columns:
        out[pc] = out[pc].apply(lambda x: f"{x:.1%}")
    for fc in ["Prob_Vitoria_J1"]:
        if fc in out.columns:
            out[fc] = out[fc].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    for fc in ["Elo_J1", "Elo_J2", "Elo_Dif_Abs"]:
        if fc in out.columns:
            out[fc] = out[fc].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="Previsões", index=False)

        # Elo rankings sheet
        elo_ratings.head(100).to_excel(writer, sheet_name="Elo_Rankings", index=False)

        # Summary
        pc_raw = preds[f"prob_over_{threshold}"]
        summary = pd.DataFrame({
            "Métrica": [
                "Gerado em", "Jogos analisados",
                f"Média Prob Over {threshold}",
                f"Mediana Prob Over {threshold}",
                "Spread (std)", "Jogos >60%", "Jogos >65%", "Jogos >70%",
            ],
            "Valor": [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(preds),
                f"{pc_raw.mean():.1%}",
                f"{pc_raw.median():.1%}",
                f"{pc_raw.std():.1%}",
                int((pc_raw > 0.60).sum()),
                int((pc_raw > 0.65).sum()),
                int((pc_raw > 0.70).sum()),
            ],
        })
        summary.to_excel(writer, sheet_name="Resumo", index=False)
        out.head(10).to_excel(writer, sheet_name="TOP_10", index=False)

        for sheet in writer.sheets.values():
            for col in sheet.columns:
                w = max((len(str(c.value)) for c in col if c.value), default=8)
                sheet.column_dimensions[col[0].column_letter].width = min(w + 2, 40)

    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────
# ─── SIDEBAR ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────

st.sidebar.header("📂 Histórico")
uploaded = st.sidebar.file_uploader("Excel (.xlsx)", type=["xlsx"])

if uploaded:
    df_hist, source = load_excel(uploaded)
    if df_hist is not None:
        st.sidebar.success(f"✅ {uploaded.name}: {len(df_hist)} jogos")
else:
    with st.spinner("A carregar histórico do GitHub…"):
        df_hist, source = fetch_github_data()

if df_hist is None or df_hist.empty:
    st.error("Não foi possível carregar histórico.")
    st.stop()

st.sidebar.info(f"Fonte: {source} | {len(df_hist)} jogos")

st.sidebar.header("⚙️ Configurações")
threshold_games = st.sidebar.slider("Over/Under threshold (games)", 15, 30, 22, 1)

st.sidebar.header("🔢 Parâmetros do Elo")
base_k_value = st.sidebar.slider(
    "K base", min_value=16, max_value=64, value=32, step=4,
    help="K-factor base. Valores maiores = Elo muda mais rápido por jogo"
)
with st.sidebar.expander("Multiplicadores K por ronda"):
    k_r32  = st.slider("R32",   0.5, 2.0, 0.9, 0.1)
    k_r16  = st.slider("R16",   0.5, 2.0, 1.0, 0.1)
    k_qf   = st.slider("QF",    0.5, 2.0, 1.1, 0.1)
    k_sf   = st.slider("SF",    0.5, 2.0, 1.3, 0.1)
    k_f    = st.slider("Final", 0.5, 2.0, 1.5, 0.1)
    ROUND_K_MULT.update({"R32": k_r32, "R16": k_r16, "QF": k_qf, "SF": k_sf, "F": k_f, "Final": k_f})


# ─────────────────────────────────────────────────────────────
# ─── MAIN ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────

st.title("🎾 Challenger Predictor v4 — Dynamic Elo + Serve Stats")
st.caption(f"Fonte: {source} | {len(df_hist)} jogos | Modelo: Ensemble (GBM + RF + LR)")

# Build Elo + features (passando o base_k)
with st.spinner("A processar histórico e construir Elo…"):
    elo_sys, feat_df = build_elo_and_features(df_hist, base_k_value)

# Atualizar o base_k do sistema
elo_sys.set_base_k(base_k_value)

# Debug: mostrar alguns Elos
if len(elo_sys.elo) > 0:
    sample_players = list(elo_sys.elo.keys())[:5]
    st.sidebar.write("**Exemplo de Elos:**")
    for p in sample_players:
        st.sidebar.write(f"{p}: {elo_sys.get(p):.0f}")

n_valid = feat_df["total_games"].notna().sum()
st.sidebar.write(f"Jogos com features: {n_valid}")

# Train model
with st.spinner("A treinar modelo…"):
    result = train_model(feat_df, threshold_games)

if result[0] is None:
    st.error("Dados insuficientes para treinar.")
    st.stop()

model, avg_games, over_pct, cv_acc, cv_auc, global_medians = result

# Final Elo ratings
elo_ratings_df = elo_sys.final_ratings()

# ── Metrics row ──────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jogadores", len(elo_ratings_df))
c2.metric(f"Over {threshold_games} histórico", f"{over_pct:.1f}%")
c3.metric("Avg Games histórico", f"{avg_games:.1f}")
c4.metric("CV Accuracy", f"{cv_acc:.1%}")
c5.metric("CV AUC", f"{cv_auc:.3f}")

# ── Tabs ─────────────────────────────────────────────────────
tab_pred, tab_elo, tab_info, tab_custom = st.tabs(["📅 Previsões", "🏆 Elo Rankings", "ℹ️ Sobre o Modelo", "🔮 Comparar 2 Jogadores"])

# ============================================================
# TAB 1: PREVISÕES
# ============================================================
with tab_pred:
    st.header("📅 Jogos de hoje e amanhã")

    api_matches = fetch_api_matches()

    if api_matches.empty:
        st.warning("⚠️ API sem resultados.")
        st.subheader("📤 Carrega ficheiro de jogos manualmente")
        st.markdown("""
        Colunas necessárias: **Date**, **Winner**, **Loser**
        Opcionais: **Surface** (Clay/Hard/Grass), **Indoor** (I/O), **Round** (R32/R16/QF/SF/F)
        """)
        manual = st.file_uploader("Excel (.xlsx)", type=["xlsx"], key="manual")
        if manual:
            try:
                mdf = pd.read_excel(manual)
                mdf["Date"] = pd.to_datetime(mdf["Date"])
                for c, d in [("Surface","Hard"),("Indoor","O"),("Round","R32")]:
                    if c not in mdf.columns:
                        mdf[c] = d
                upcoming = mdf
                st.success(f"✅ {len(upcoming)} jogos carregados")
            except Exception as e:
                st.error(f"Erro: {e}")
                st.stop()
        else:
            st.info("👆 Carrega um ficheiro ou aguarda a API.")
            st.stop()
    else:
        upcoming = api_matches

    if upcoming.empty:
        st.info("Nenhum jogo disponível.")
        st.stop()

    # Predict
    with st.spinner("A calcular previsões com Elo dinâmico…"):
        preds = predict_matches(upcoming, elo_sys, model, global_medians, threshold_games)

    prob_col = f"prob_over_{threshold_games}"
    probs_all = preds[prob_col]

    # Prob spread health check
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Prob mín", f"{probs_all.min():.1%}")
    cB.metric("Prob média", f"{probs_all.mean():.1%}")
    cC.metric("Prob máx", f"{probs_all.max():.1%}")
    cD.metric("Spread (std)", f"{probs_all.std():.1%}")

    if probs_all.std() < 0.02:
        st.warning("⚠️ Baixo spread de probabilidades — jogadores provavelmente não estão no histórico (usadas medianas globais)")
    else:
        st.success("✅ Spread saudável — modelo a usar dados reais dos jogadores")

    # Results table
    st.subheader(f"📋 Previsões — Over {threshold_games} Games")
    display_map = {
        "Date": "Data", "Winner": "Jogador 1", "Loser": "Jogador 2",
        "Surface": "Superfície", "Round": "Ronda",
        prob_col: f"Over {threshold_games}",
        "elo_p1": "Elo J1", "elo_p2": "Elo J2",
        "elo_diff": "Δ Elo", "exp_p1": "Prob Vitória J1",
        "n_p1": "N J1", "n_p2": "N J2",
        "w_avg_games": "AvgGames J1", "l_avg_games": "AvgGames J2",
        "both_known": "Histórico",
    }
    existing = {k: v for k, v in display_map.items() if k in preds.columns}
    disp = preds[list(existing.keys())].rename(columns=existing)

    # Adicionar coluna de aviso para jogadores sem histórico
    if "Histórico" in disp.columns:
        disp["Status"] = disp["Histórico"].apply(lambda x: "✅ Com histórico" if x else "⚠️ Sem histórico")
        # Mover para posição adequada
        cols = disp.columns.tolist()
        if "Jogador 2" in cols:
            idx = cols.index("Jogador 2") + 1
            cols.insert(idx, cols.pop(cols.index("Status")))
            disp = disp[cols]

    fmt = {
        f"Over {threshold_games}": "{:.1%}",
        "Prob Vitória J1": "{:.1%}",
        "Elo J1": "{:.0f}", "Elo J2": "{:.0f}", "Δ Elo": "{:.0f}",
        "AvgGames J1": "{:.1f}", "AvgGames J2": "{:.1f}",
    }

    st.dataframe(disp.style.format(fmt, na_rep="-"), use_container_width=True)

    # TOP 5
    st.subheader(f"🔥 TOP 5 — Over {threshold_games} Games")
    for i, (_, row) in enumerate(preds.head(5).iterrows(), 1):
        known = "✅" if row.get("both_known") else "⚠️ sem histórico"
        elo1  = row.get("elo_p1", np.nan)
        elo2  = row.get("elo_p2", np.nan)
        exp1  = row.get("exp_p1", np.nan)
        elo_str = f"Elo: {elo1:.0f} vs {elo2:.0f}" if not np.isnan(elo1) else ""
        exp_str = f"| Win%: {exp1:.0%}" if not np.isnan(exp1) else ""
        with st.container():
            st.markdown(
                f"**{i}. {row['Winner']} vs {row['Loser']}** {known}  \n"
                f"📅 {pd.Timestamp(row['Date']).date()} | 🎾 {row.get('Surface','?')} | "
                f"{row.get('Round','?')} | {elo_str} {exp_str}"
            )
            st.progress(
                min(float(row[prob_col]), 1.0),
                text=f"Over {threshold_games}: **{row[prob_col]:.1%}**"
            )

    # Export
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("📥 Exportar para Excel", type="primary", use_container_width=True):
            with st.spinner("A gerar Excel…"):
                try:
                    buf = export_excel(preds, threshold_games, elo_ratings_df)
                    fname = f"challenger_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    st.success("✅ Pronto!")
                    st.download_button(
                        "💾 Descarregar Excel", data=buf, file_name=fname,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Erro ao exportar: {e}")

# ============================================================
# TAB 2: ELO RANKINGS
# ============================================================
with tab_elo:
    st.subheader("🏆 Elo Rankings — Top 500")
    st.caption("Elo calculado cronologicamente com K dinâmico (round + experiência + equilíbrio do jogo)")

    col_filter = st.selectbox("Ordenar por", ["Elo Geral", "Elo Clay", "Elo Hard", "Elo Grass"])
    sort_col = {"Elo Geral": "Elo", "Elo Clay": "Elo_Clay", "Elo Hard": "Elo_Hard", "Elo Grass": "Elo_Grass"}[col_filter]

    top50 = elo_ratings_df.sort_values(sort_col, ascending=False).head(50).reset_index(drop=True)
    top50.index = top50.index + 1

    st.dataframe(
        top50.style.format({
            "Elo": "{:.0f}", "Elo_Clay": "{:.0f}",
            "Elo_Hard": "{:.0f}", "Elo_Grass": "{:.0f}",
        }).background_gradient(subset=[sort_col], cmap="RdYlGn"),
        use_container_width=True
    )

    # Quick Elo lookup
    st.subheader("🔍 Consultar jogador")
    search = st.text_input("Nome do jogador")
    if search:
        matches_search = elo_ratings_df[elo_ratings_df["Player"].str.contains(search, case=False, na=False)]
        if len(matches_search) > 0:
            st.dataframe(matches_search.style.format({
                "Elo": "{:.0f}", "Elo_Clay": "{:.0f}",
                "Elo_Hard": "{:.0f}", "Elo_Grass": "{:.0f}",
            }), use_container_width=True)
        else:
            st.warning(f"Jogador '{search}' não encontrado no histórico.")

# ============================================================
# TAB 3: INFORMAÇÕES DO MODELO
# ============================================================
with tab_info:
    st.subheader("Como funciona o Dynamic K-Factor Elo")
    st.markdown(f"""
    **K-factor dinâmico:** O quanto o Elo de um jogador muda por jogo não é fixo — varia por:

    | Fator | Lógica |
    |---|---|
    | **Ronda** | Finais valem mais (K×{k_f}) porque os adversários são mais fortes |
    | **Experiência** | Novos jogadores (<10 jogos) têm K×1.5 para convergir rápido |
    | **Equilíbrio** | Jogos de 3 sets têm K×1.2 — revelam mais sobre o nível real |

    **Surface Elo separado:** cada jogador tem 3 Elos independentes (Clay / Hard / Grass) para capturar especialização de superfície.

    **Features do modelo ({len(FEATURE_COLS)} total):**
    - Elo geral + surface: elo_diff, elo_surf_diff, exp_w (probabilidade esperada)
    - Serve stats rolling (últimos {SERVE_WINDOW} jogos): 1stWon%, 2ndWon%, ace%, bp_save%
    - Contexto: superfície, indoor, ronda
    - Avg games histórico por jogador e por superfície

    **Por que ~57% é o teto realista:**
    Total de games em Challenger tem std≈5.8 em torno de média≈22.7.
    A diferença por superfície é pequena.
    Com AUC {cv_auc:.3f}, o modelo adiciona sinal real acima do acaso — o valor está no spread de probabilidades entre jogos, não em ultrapassar 60%.
    """)

# ============================================================
# TAB 4: COMPARAR 2 JOGADORES (PREVISÃO PERSONALIZADA)
# ============================================================
with tab_custom:
    st.header("🔮 Comparar 2 Jogadores")
    st.markdown("""
    Selecione dois jogadores do histórico e uma superfície para simular o confronto:
    - **Probabilidade de Over 22 games** (jogo com mais de 22 games)
    - **Probabilidade de vitória do Jogador 1** (baseada em Elo)
    - **Diferença de Elo** entre os jogadores
    - **Elo específico da superfície** selecionada
    """)
    
    # Obter lista de jogadores com Elo registado e com pelo menos alguns jogos
    players_with_matches = elo_ratings_df[elo_ratings_df["Matches"] > 0]["Player"].tolist()
    players_with_elo = sorted(players_with_matches)
    
    if len(players_with_elo) < 2:
        st.warning(f"⚠️ É necessário pelo menos 2 jogadores com histórico para fazer previsões personalizadas.")
        st.info(f"Atualmente existem {len(players_with_elo)} jogadores com pelo menos 1 jogo no histórico.")
        
        # Mostrar alguns jogadores disponíveis
        if len(players_with_elo) > 0:
            st.write("**Jogadores disponíveis:**")
            st.write(", ".join(players_with_elo[:20]))
        else:
            st.error("Nenhum jogador encontrado no histórico. Verifique se o ficheiro Excel tem dados válidos.")
    else:
        # Layout de duas colunas para seleção
        col1, col2 = st.columns(2)
        
        with col1:
            player1 = st.selectbox(
                "🎾 Jogador 1",
                options=players_with_elo,
                index=0,
                help="Selecione o primeiro jogador"
            )
            
            # Mostrar informações do jogador 1
            player1_data = elo_ratings_df[elo_ratings_df["Player"] == player1].iloc[0]
            st.caption(f"📊 Elo Geral: **{player1_data['Elo']:.0f}** | Jogos: {player1_data['Matches']:.0f}")
            st.caption(f"🏆 Clay: {player1_data['Elo_Clay']:.0f} | Hard: {player1_data['Elo_Hard']:.0f} | Grass: {player1_data['Elo_Grass']:.0f}")
        
        with col2:
            # Evitar selecionar o mesmo jogador
            default_index = 1 if len(players_with_elo) > 1 else 0
            player2 = st.selectbox(
                "🎾 Jogador 2",
                options=players_with_elo,
                index=min(default_index, len(players_with_elo)-1),
                help="Selecione o segundo jogador"
            )
            
            # Mostrar informações do jogador 2
            player2_data = elo_ratings_df[elo_ratings_df["Player"] == player2].iloc[0]
            st.caption(f"📊 Elo Geral: **{player2_data['Elo']:.0f}** | Jogos: {player2_data['Matches']:.0f}")
            st.caption(f"🏆 Clay: {player2_data['Elo_Clay']:.0f} | Hard: {player2_data['Elo_Hard']:.0f} | Grass: {player2_data['Elo_Grass']:.0f}")
        
        # Seleção de superfície
        st.markdown("---")
        surface_options = ["Hard", "Clay", "Grass"]
        surface = st.radio(
            "🎾 Superfície",
            options=surface_options,
            index=0,
            horizontal=True,
            help="Selecione a superfície onde o jogo será disputado"
        )
        
        # Mostrar Elo específico da superfície selecionada
        elo_p1_surf = player1_data[f'Elo_{surface}']
        elo_p2_surf = player2_data[f'Elo_{surface}']
        st.info(f"💡 **Elo na superfície {surface}:** {player1} = **{elo_p1_surf:.0f}** | {player2} = **{elo_p2_surf:.0f}**")
        
        # Botão para calcular
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            calculate_btn = st.button(
                "🔮 Calcular Previsão",
                type="primary",
                use_container_width=True
            )
        
        if calculate_btn:
            if player1 == player2:
                st.error("❌ Os jogadores devem ser diferentes!")
            else:
                with st.spinner("A calcular previsões..."):
                    try:
                        prob_over, prob_win, elo_diff, exp_win, elo_p1, elo_p2 = calculate_custom_prediction(
                            player1, player2, surface, elo_sys, model, global_medians, threshold_games
                        )
                        
                        # Mostrar resultados
                        st.markdown("---")
                        st.subheader("📊 Resultados da Previsão")
                        
                        # Métricas principais
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "🎯 Over 22 Games",
                                f"{prob_over:.1%}",
                                delta=f"{prob_over - 0.5:.1%}" if prob_over != 0.5 else None,
                                help=f"Probabilidade do jogo ter mais de {threshold_games} games"
                            )
                        with col2:
                            st.metric(
                                "🏆 Vitória do Jogador 1",
                                f"{prob_win:.1%}",
                                delta=f"{prob_win - 0.5:.1%}" if prob_win != 0.5 else None,
                                help="Probabilidade baseada em Elo"
                            )
                        with col3:
                            st.metric(
                                "⚡ Diferença de Elo",
                                f"{elo_diff:.0f}",
                                help="Elo_Jogador1 - Elo_Jogador2"
                            )
                        
                        # Gráfico de barras para probabilidades
                        st.markdown("---")
                        st.subheader("📈 Visualização das Probabilidades")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Probabilidade de Over 22 Games**")
                            st.progress(prob_over, text=f"{prob_over:.1%}")
                            
                            # Interpretação
                            if prob_over > 0.65:
                                st.success("✅ Alta probabilidade de jogo longo (>22 games)")
                            elif prob_over > 0.55:
                                st.info("📊 Probabilidade moderada de jogo longo")
                            else:
                                st.warning("⚠️ Baixa probabilidade de jogo longo")
                        
                        with col2:
                            st.write("**Probabilidade de Vitória**")
                            st.progress(prob_win, text=f"{prob_win:.1%}")
                            
                            # Mostrar favorito
                            if prob_win > 0.6:
                                st.success(f"✅ {player1} é o favorito")
                            elif prob_win < 0.4:
                                st.warning(f"⚠️ {player2} é o favorito")
                            else:
                                st.info("⚖️ Jogo equilibrado")
                        
                        # Detalhes adicionais
                        st.markdown("---")
                        with st.expander("📋 Detalhes Adicionais"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**{player1}**")
                                st.write(f"- Jogos no histórico: {elo_sys.n_matches.get(player1, 0)}")
                                # Calcular média real de games do jogador
                                p1_games = elo_sys.rolling_games(player1, surface)
                                if not np.isnan(p1_games["avg_games"]):
                                    st.write(f"- Média de games (geral): {p1_games['avg_games']:.1f}")
                                if not np.isnan(p1_games["surf_games"]):
                                    st.write(f"- Média de games ({surface}): {p1_games['surf_games']:.1f}")
                                
                                # Serve stats
                                p1_serve = elo_sys.rolling_serve(player1)
                                if not np.isnan(p1_serve["1w_pct"]):
                                    st.write(f"- 1st Serve%: {p1_serve['1w_pct']:.1%}")
                                if not np.isnan(p1_serve["ace_pct"]):
                                    st.write(f"- Ace%: {p1_serve['ace_pct']:.1%}")
                            
                            with col2:
                                st.write(f"**{player2}**")
                                st.write(f"- Jogos no histórico: {elo_sys.n_matches.get(player2, 0)}")
                                p2_games = elo_sys.rolling_games(player2, surface)
                                if not np.isnan(p2_games["avg_games"]):
                                    st.write(f"- Média de games (geral): {p2_games['avg_games']:.1f}")
                                if not np.isnan(p2_games["surf_games"]):
                                    st.write(f"- Média de games ({surface}): {p2_games['surf_games']:.1f}")
                                
                                p2_serve = elo_sys.rolling_serve(player2)
                                if not np.isnan(p2_serve["1w_pct"]):
                                    st.write(f"- 1st Serve%: {p2_serve['1w_pct']:.1%}")
                                if not np.isnan(p2_serve["ace_pct"]):
                                    st.write(f"- Ace%: {p2_serve['ace_pct']:.1%}")
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao calcular previsão: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Mostrar TOP jogadores para referência
        with st.expander("🏆 TOP 10 Jogadores (Elo Geral)"):
            top_players = elo_ratings_df.head(10)[["Player", "Elo", "Matches", "Elo_Clay", "Elo_Hard", "Elo_Grass"]].copy()
            top_players["Elo"] = top_players["Elo"].round(0).astype(int)
            top_players["Elo_Clay"] = top_players["Elo_Clay"].round(0).astype(int)
            top_players["Elo_Hard"] = top_players["Elo_Hard"].round(0).astype(int)
            top_players["Elo_Grass"] = top_players["Elo_Grass"].round(0).astype(int)
            st.dataframe(top_players, use_container_width=True)
        
        # Dica de uso
        st.markdown("---")
        st.info("💡 **Dica:** Use esta ferramenta para simular confrontos entre qualquer par de jogadores do histórico e ver a probabilidade de jogo longo e de vitória baseada em Elo dinâmico!")
