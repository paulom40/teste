"""
CHALLENGER TENNIS PREDICTOR v5
================================
Fixes vs v4:
- FIX: Elo always 1500 — API returns abbreviated names ("S. Jones") while history
  has full names ("Sebastian Jones"). Now uses a fuzzy last-name + initial resolver
  to match API names to history records.
- FIX: 🏆 Elo Rankings tab now visible and working
- FIX: K-factor sidebar sliders now actually affect the model
- NEW: Each prediction shows Elo confidence (High/Medium/Low/Unknown)
- NEW: Separate section for "unknown players" so user knows which are reliable
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
from difflib import SequenceMatcher

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Challenger Predictor v5 — Dynamic Elo",
    page_icon="🎾",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS  (mutable — overridden by sidebar sliders)
# ─────────────────────────────────────────────────────────────

ELO_START    = 1500.0
SERVE_WINDOW = 15

# These are defaults; sidebar sliders replace them at runtime
_BASE_K      = 32.0
_ROUND_K     = {"R32": 0.9, "R16": 1.0, "QF": 1.1, "SF": 1.3, "F": 1.5, "Final": 1.5}

SURFACE_ENC  = {"Clay": 0, "Hard": 1, "Grass": 2}
ROUND_ENC    = {"R32": 1, "R16": 2, "QF": 3, "SF": 4, "F": 5, "Final": 5}

FEATURE_COLS = [
    "elo_diff", "elo_surf_diff", "elo_abs", "elo_surf_abs",
    "elo_w", "elo_l", "elo_w_surf", "elo_l_surf",
    "exp_w", "exp_w_surf",
    "surface_enc", "round_enc", "indoor_enc",
    "w_1w_pct", "w_2w_pct", "w_ace_pct", "w_bp_save", "w_bp_faced_pg",
    "l_1w_pct", "l_2w_pct", "l_ace_pct", "l_bp_save", "l_bp_faced_pg",
    "serve_dom_sum", "ace_sum", "bp_save_diff",
    "n_w", "n_l",
    "w_avg_games", "l_avg_games", "avg_games_combined",
    "w_surf_games", "l_surf_games",
]


# ─────────────────────────────────────────────────────────────
# NAME RESOLVER  (fixes the Elo=1500 for all API matches)
# ─────────────────────────────────────────────────────────────

def build_name_index(players: list[str]) -> dict:
    """
    Build a lookup: last_name_lower -> [(full_name, first_name_lower), ...]
    Used to match abbreviated API names (e.g. "S. Jones") to full history names.
    """
    idx: dict[str, list] = {}
    for p in players:
        parts = str(p).strip().split()
        if not parts:
            continue
        last  = parts[-1].lower()
        first = parts[0].lower()
        idx.setdefault(last, []).append((p, first))
    return idx


def resolve_player_name(
    api_name: str,
    name_idx: dict,
    all_players: list[str],
) -> tuple[str | None, str]:
    """
    Try to match an API player name (possibly abbreviated) to a full history name.

    Returns (full_name | None, confidence)
    confidence: 'high' | 'medium' | 'low' | 'none'

    Strategy:
      1. Exact full-name match
      2. last-name lookup + initial filter
      3. Fuzzy full-string match (SequenceMatcher ≥ 0.78)
    """
    api_name = str(api_name).strip()
    api_lower = api_name.lower()

    # 1. Exact match
    if api_name in all_players:
        return api_name, "high"

    # Parse the API name
    clean = api_lower.replace(".", "").strip()
    parts = clean.split()
    if len(parts) < 2:
        return None, "none"

    last       = parts[-1]
    first_part = parts[0]          # could be "s", "ri", "mitchell", etc.
    initial    = first_part[0]

    candidates = name_idx.get(last, [])

    if candidates:
        # Initial filter
        init_matches = [(fn, f) for fn, f in candidates if f.startswith(initial)]

        if len(init_matches) == 1:
            # Unique match on last name + initial
            full, first_hist = init_matches[0]
            # Confidence: medium if abbreviated ("S."), high if first_part is 3+ chars
            conf = "high" if len(first_part) >= 3 and first_hist.startswith(first_part) else "medium"
            return full, conf

        if len(init_matches) > 1 and len(first_part) >= 2:
            # Try two-char prefix to disambiguate
            two_matches = [(fn, f) for fn, f in init_matches if f.startswith(first_part[:2])]
            if len(two_matches) == 1:
                return two_matches[0][0], "medium"

        if len(candidates) == 1:
            # Only one player with that last name, initial mismatch — probably still right
            full, first_hist = candidates[0]
            if first_hist[0] == initial:
                return full, "medium"

    # 3. Fuzzy full-string match
    best_score, best_match = 0.0, None
    for p in all_players:
        score = SequenceMatcher(None, api_lower, p.lower()).ratio()
        if score > best_score:
            best_score, best_match = score, p
    if best_score >= 0.78:
        return best_match, "low"

    return None, "none"


# ─────────────────────────────────────────────────────────────
# ELO ENGINE
# ─────────────────────────────────────────────────────────────

def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def dynamic_k(n_matches: int, round_str: str, is_3set: bool,
              base_k: float, round_k: dict) -> float:
    """
    K = base_k × round_mult × experience_mult × closeness_mult
    - round_mult:  later rounds (Finals, SF) count more
    - experience_mult: new players converge faster
    - closeness_mult: 3-set matches reveal more about true level
    """
    round_mult = round_k.get(round_str, 1.0)
    if n_matches < 10:
        exp_mult = 1.5
    elif n_matches < 30:
        exp_mult = 1.2
    else:
        exp_mult = 1.0
    close_mult = 1.2 if is_3set else 1.0
    return base_k * round_mult * exp_mult * close_mult


class EloSystem:
    def __init__(self, base_k: float = 32.0, round_k: dict | None = None):
        self.base_k   = base_k
        self.round_k  = round_k or dict(_ROUND_K)
        self.elo:     dict[str, float] = {}
        self.elo_surf:dict[str, dict]  = {}
        self.n_matches: dict[str, int] = {}
        self.serve_history: dict[str, list] = {}
        self.games_history: dict[str, list] = {}

    def get(self, player: str, surface: str | None = None) -> float:
        if surface:
            return self.elo_surf.setdefault(player, {}).get(surface, ELO_START)
        return self.elo.get(player, ELO_START)

    def _k(self, player: str, round_str: str, is_3set: bool) -> float:
        return dynamic_k(
            self.n_matches.get(player, 0), round_str, is_3set,
            self.base_k, self.round_k
        )

    def update(self, winner: str, loser: str, surface: str,
               round_str: str, is_3set: bool, total_games: float,
               w_serve: dict, l_serve: dict):
        ew   = self.get(winner);      el   = self.get(loser)
        ew_s = self.get(winner, surface); el_s = self.get(loser, surface)

        kw = self._k(winner, round_str, is_3set)
        kl = self._k(loser,  round_str, is_3set)

        # Overall Elo
        exp_w = elo_expected(ew, el)
        self.elo[winner] = ew + kw * (1.0 - exp_w)
        self.elo[loser]  = el + kl * (0.0 - (1.0 - exp_w))

        # Surface Elo
        exp_ws = elo_expected(ew_s, el_s)
        self.elo_surf.setdefault(winner, {})[surface] = ew_s + kw * (1.0 - exp_ws)
        self.elo_surf.setdefault(loser,  {})[surface] = el_s + kl * (0.0 - (1.0 - exp_ws))

        self.n_matches[winner] = self.n_matches.get(winner, 0) + 1
        self.n_matches[loser]  = self.n_matches.get(loser,  0) + 1

        for player, serve in [(winner, w_serve), (loser, l_serve)]:
            if serve:
                self.serve_history.setdefault(player, []).append(serve)
        if not np.isnan(total_games):
            self.games_history.setdefault(winner, []).append((surface, total_games))
            self.games_history.setdefault(loser,  []).append((surface, total_games))

    def snapshot(self, player1: str, player2: str, surface: str) -> dict:
        """Pre-match Elo features — uses current (post-training) ratings."""
        ew   = self.get(player1);          el   = self.get(player2)
        ew_s = self.get(player1, surface); el_s = self.get(player2, surface)
        return {
            "elo_w": ew,  "elo_l": el,
            "elo_diff": ew - el,   "elo_abs": abs(ew - el),
            "elo_w_surf": ew_s,    "elo_l_surf": el_s,
            "elo_surf_diff": ew_s - el_s, "elo_surf_abs": abs(ew_s - el_s),
            "exp_w":     elo_expected(ew, el),
            "exp_w_surf":elo_expected(ew_s, el_s),
            "n_w": self.n_matches.get(player1, 0),
            "n_l": self.n_matches.get(player2, 0),
        }

    def rolling_serve(self, player: str) -> dict:
        history = self.serve_history.get(player, [])[-SERVE_WINDOW:]
        def mean_key(k):
            vals = [x[k] for x in history if x.get(k) is not None and not np.isnan(x[k])]
            return float(np.mean(vals)) if vals else np.nan
        return {
            "1w_pct": mean_key("1w_pct"), "2w_pct": mean_key("2w_pct"),
            "ace_pct": mean_key("ace_pct"), "bp_save": mean_key("bp_save"),
            "bp_faced_pg": mean_key("bp_faced_pg"),
        }

    def rolling_games(self, player: str, surface: str | None = None, window: int = 20) -> dict:
        all_g = self.games_history.get(player, [])
        recent_all  = [g for _, g in all_g[-window:]]
        recent_surf = [g for s, g in all_g if s == surface][-15:]
        return {
            "avg_games":  float(np.mean(recent_all))  if len(recent_all) >= 3  else np.nan,
            "surf_games": float(np.mean(recent_surf)) if len(recent_surf) >= 3 else np.nan,
        }

    def ratings_df(self) -> pd.DataFrame:
        rows = []
        for p in self.elo:
            rows.append({
                "Player":    p,
                "Elo":       round(self.get(p), 1),
                "Elo_Clay":  round(self.get(p, "Clay"),  1),
                "Elo_Hard":  round(self.get(p, "Hard"),  1),
                "Elo_Grass": round(self.get(p, "Grass"), 1),
                "Matches":   self.n_matches.get(p, 0),
            })
        return (pd.DataFrame(rows)
                .sort_values("Elo", ascending=False)
                .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────
# SCORE PARSING
# ─────────────────────────────────────────────────────────────

def parse_total_games(score) -> float:
    if pd.isna(score):
        return np.nan
    s = str(score)
    if any(x in s for x in ("RET", "W/O", "DEF")):
        return np.nan
    sets = re.findall(r"(\d+)-(\d+)(?:\(\d+\))?", s)
    return float(sum(int(a) + int(b) for a, b in sets)) if sets else np.nan


# ─────────────────────────────────────────────────────────────
# LOAD & NORMALIZE
# ─────────────────────────────────────────────────────────────

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "winner_name": "Winner",  "loser_name":  "Loser",
        "winner_rank": "WRank",   "loser_rank":  "LRank",
        "winner_rank_points": "WPts", "loser_rank_points": "LPts",
        "surface": "Surface",  "indoor": "Indoor",  "round": "Round",
        "score": "Score",      "best_of": "BestOf", "minutes": "Minutes",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

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

    # Serve percentages
    for pfx, svpt_c, fi_c, fw_c, sw_c, ace_c, bpf_c, bps_c, svg_c in [
        ("W", "w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_ace","w_bpFaced","w_bpSaved","w_SvGms"),
        ("L", "l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_ace","l_bpFaced","l_bpSaved","l_SvGms"),
    ]:
        needed = [svpt_c, fi_c, fw_c, sw_c]
        if not all(c in df.columns for c in needed):
            continue
        s   = pd.to_numeric(df[svpt_c], errors="coerce").replace(0, np.nan)
        fi  = pd.to_numeric(df[fi_c],   errors="coerce").replace(0, np.nan)
        se  = (pd.to_numeric(df[svpt_c], errors="coerce")
               - pd.to_numeric(df[fi_c], errors="coerce")).replace(0, np.nan)
        df[f"{pfx}1wPct"]      = pd.to_numeric(df[fw_c], errors="coerce") / fi
        df[f"{pfx}2wPct"]      = pd.to_numeric(df[sw_c], errors="coerce") / se
        if ace_c in df.columns:
            df[f"{pfx}AcePct"] = pd.to_numeric(df[ace_c], errors="coerce") / s
        if bpf_c in df.columns and bps_c in df.columns:
            bpf = pd.to_numeric(df[bpf_c], errors="coerce").replace(0, np.nan)
            df[f"{pfx}BpSave"]     = pd.to_numeric(df[bps_c], errors="coerce") / bpf
        if bpf_c in df.columns and svg_c in df.columns:
            svg = pd.to_numeric(df[svg_c], errors="coerce").replace(0, np.nan)
            df[f"{pfx}BpFacedPG"] = pd.to_numeric(df[bpf_c], errors="coerce") / svg
    return df


@st.cache_data(show_spinner=False)
def fetch_github() -> tuple:
    try:
        url = "https://github.com/paulom40/teste/raw/main/Challenger.xlsx"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return normalize_df(pd.read_excel(BytesIO(r.content))), "GitHub Challenger Database"
    except Exception as e:
        st.warning(f"GitHub fetch failed: {e}")
        return None, None


def load_excel(uploaded) -> tuple:
    try:
        return normalize_df(pd.read_excel(uploaded)), uploaded.name
    except Exception as e:
        st.sidebar.error(f"Load error: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────
# BUILD ELO + TRAINING FEATURES  (leakage-free)
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_elo_and_features(_df: pd.DataFrame, base_k: float, round_k_tuple: tuple) -> tuple:
    """
    Processes matches in chronological order.
    Before each match: records pre-match Elo + serve features.
    After each match: updates Elo ratings.
    Returns (EloSystem with FINAL ratings, feature DataFrame for training).

    round_k_tuple is a hashable version of the round_k dict for caching.
    """
    round_k = dict(round_k_tuple)
    df = _df.copy().sort_values("Date").reset_index(drop=True)
    sys_ = EloSystem(base_k=base_k, round_k=round_k)
    rows = []

    for _, row in df.iterrows():
        winner  = row.get("Winner")
        loser   = row.get("Loser")
        surface = row.get("Surface", "Hard")
        rnd     = row.get("Round", "R32")
        indoor  = row.get("Indoor", "O")
        tg      = row.get("TotalGames", np.nan)
        score   = str(row.get("Score", ""))

        if pd.isna(winner) or pd.isna(loser):
            continue

        is_3set = len(re.findall(r"\d+-\d+", score)) >= 3

        # ── Pre-match snapshot ────────────────────────────────
        elo_f  = sys_.snapshot(winner, loser, surface)
        ws     = sys_.rolling_serve(winner)
        ls     = sys_.rolling_serve(loser)
        wg     = sys_.rolling_games(winner, surface)
        lg     = sys_.rolling_games(loser, surface)

        rows.append({
            **elo_f,
            "surface_enc":  SURFACE_ENC.get(surface, 1),
            "round_enc":    ROUND_ENC.get(rnd, 1),
            "indoor_enc":   1 if indoor == "I" else 0,
            "w_1w_pct":     ws["1w_pct"],  "w_2w_pct": ws["2w_pct"],
            "w_ace_pct":    ws["ace_pct"], "w_bp_save": ws["bp_save"],
            "w_bp_faced_pg":ws["bp_faced_pg"],
            "l_1w_pct":     ls["1w_pct"],  "l_2w_pct": ls["2w_pct"],
            "l_ace_pct":    ls["ace_pct"], "l_bp_save": ls["bp_save"],
            "l_bp_faced_pg":ls["bp_faced_pg"],
            "serve_dom_sum": ((ws["1w_pct"] or 0) * (ws["ace_pct"] or 0) +
                              (ls["1w_pct"] or 0) * (ls["ace_pct"] or 0)),
            "ace_sum":       (ws["ace_pct"] or 0) + (ls["ace_pct"] or 0),
            "bp_save_diff":  abs((ws["bp_save"] or 0.5) - (ls["bp_save"] or 0.5)),
            "w_avg_games":   wg["avg_games"], "l_avg_games": lg["avg_games"],
            "avg_games_combined": float(np.nanmean([wg["avg_games"], lg["avg_games"]])),
            "w_surf_games":  wg["surf_games"], "l_surf_games": lg["surf_games"],
            "total_games":   tg,
        })

        # ── Update Elo after match ───────────────────────────
        w_serve = {
            "1w_pct":     row.get("W1wPct", np.nan),
            "2w_pct":     row.get("W2wPct", np.nan),
            "ace_pct":    row.get("WAcePct", np.nan),
            "bp_save":    row.get("WBpSave", np.nan),
            "bp_faced_pg":row.get("WBpFacedPG", np.nan),
        }
        l_serve = {
            "1w_pct":     row.get("L1wPct", np.nan),
            "2w_pct":     row.get("L2wPct", np.nan),
            "ace_pct":    row.get("LAcePct", np.nan),
            "bp_save":    row.get("LBpSave", np.nan),
            "bp_faced_pg":row.get("LBpFacedPG", np.nan),
        }
        sys_.update(winner, loser, surface, rnd, is_3set,
                    tg if not pd.isna(tg) else 0.0,
                    w_serve, l_serve)

    return sys_, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────────

def make_pipeline():
    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=15, random_state=42)
    rf  = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=15,
        n_jobs=-1, random_state=42)
    lr  = LogisticRegression(C=0.5, max_iter=500)
    ens = VotingClassifier(
        [("gbm", gbm), ("rf", rf), ("lr", lr)],
        voting="soft", weights=[2, 1, 1])
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", ens),
    ])


@st.cache_resource(show_spinner=False)
def train_model(_feat_df: pd.DataFrame, threshold: int):
    df  = _feat_df.dropna(subset=["total_games"]).copy()
    df["target"] = (df["total_games"] > threshold).astype(int)
    if len(df) < 100:
        return None, 0.0, 0.0, 0.0, 0.0, {}

    X = df[FEATURE_COLS].copy()
    y = df["target"]
    global_medians = X.median().to_dict()

    pipe = make_pipeline()
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()
        cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    except Exception:
        cv_acc, cv_auc = 0.0, 0.0

    pipe.fit(X, y)
    return pipe, float(df["total_games"].mean()), float(y.mean()*100), cv_acc, cv_auc, global_medians


# ─────────────────────────────────────────────────────────────
# API — TODAY & TOMORROW
# ─────────────────────────────────────────────────────────────

def surf_from_name(name) -> str:
    if not isinstance(name, str):
        return "Hard"
    n = name.lower()
    if "clay" in n: return "Clay"
    if "grass" in n or "wimbledon" in n: return "Grass"
    return "Hard"


def fetch_api_matches() -> pd.DataFrame:
    today    = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        with st.spinner(f"A buscar jogos {today} → {tomorrow}…"):
            resp = requests.get(
                "https://api.api-tennis.com/tennis/",
                params={"method": "get_fixtures",
                        "APIkey": "7e3c6125ceaf5442372a487f9948c083a8778bb9604f49d8b33efc0e005f275c",
                        "date_start": today, "date_stop": tomorrow},
                timeout=15)
        if resp.status_code != 200 or not resp.text:
            st.error(f"API status {resp.status_code}")
            return pd.DataFrame()
        data = resp.json()
        if data.get("success") != 1:
            st.error("API error")
            return pd.DataFrame()
        matches = data.get("result", [])
        if not matches:
            st.info("Nenhum jogo encontrado.")
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

        result = (df_api[["Date","Winner","Loser","Surface","Indoor","Round"]]
                  .drop_duplicates()
                  .dropna(subset=["Winner","Loser"]))
        result = result[result["Winner"].str.strip() != ""]
        result = result[result["Loser"].str.strip() != ""]

        today_ts    = pd.Timestamp.now().normalize()
        tomorrow_ts = today_ts + pd.Timedelta(days=1)
        result["Date"] = pd.to_datetime(result["Date"]).dt.normalize()
        result = result[(result["Date"] == today_ts) | (result["Date"] == tomorrow_ts)]

        if len(result):
            st.success(f"✅ {len(result)} jogos encontrados")
        return result

    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

def predict_matches(upcoming: pd.DataFrame, elo_sys: EloSystem,
                    model, global_medians: dict, threshold: int,
                    name_idx: dict, all_players: list) -> pd.DataFrame:
    feat_rows = []
    resolved_p1 = []
    resolved_p2 = []
    conf_p1     = []
    conf_p2     = []

    for _, match in upcoming.iterrows():
        raw_p1  = match["Winner"]
        raw_p2  = match["Loser"]
        surface = match.get("Surface", "Hard")
        indoor  = match.get("Indoor", "O")
        rnd     = match.get("Round", "R32")

        # Resolve abbreviated names to full history names
        p1, c1 = resolve_player_name(raw_p1, name_idx, all_players)
        p2, c2 = resolve_player_name(raw_p2, name_idx, all_players)

        # Use resolved name if found, else original (will return Elo=1500)
        lookup_p1 = p1 if p1 else raw_p1
        lookup_p2 = p2 if p2 else raw_p2

        resolved_p1.append(p1 or raw_p1)
        resolved_p2.append(p2 or raw_p2)
        conf_p1.append(c1)
        conf_p2.append(c2)

        elo_f = elo_sys.snapshot(lookup_p1, lookup_p2, surface)
        ws    = elo_sys.rolling_serve(lookup_p1)
        ls    = elo_sys.rolling_serve(lookup_p2)
        wg    = elo_sys.rolling_games(lookup_p1, surface)
        lg    = elo_sys.rolling_games(lookup_p2, surface)

        feat_rows.append({
            **elo_f,
            "surface_enc":  SURFACE_ENC.get(surface, 1),
            "round_enc":    ROUND_ENC.get(rnd, 1),
            "indoor_enc":   1 if indoor == "I" else 0,
            "w_1w_pct":     ws["1w_pct"],  "w_2w_pct": ws["2w_pct"],
            "w_ace_pct":    ws["ace_pct"], "w_bp_save": ws["bp_save"],
            "w_bp_faced_pg":ws["bp_faced_pg"],
            "l_1w_pct":     ls["1w_pct"],  "l_2w_pct": ls["2w_pct"],
            "l_ace_pct":    ls["ace_pct"], "l_bp_save": ls["bp_save"],
            "l_bp_faced_pg":ls["bp_faced_pg"],
            "serve_dom_sum":((ws["1w_pct"] or 0)*(ws["ace_pct"] or 0) +
                             (ls["1w_pct"] or 0)*(ls["ace_pct"] or 0)),
            "ace_sum":       (ws["ace_pct"] or 0) + (ls["ace_pct"] or 0),
            "bp_save_diff":  abs((ws["bp_save"] or 0.5) - (ls["bp_save"] or 0.5)),
            "w_avg_games":   wg["avg_games"], "l_avg_games": lg["avg_games"],
            "avg_games_combined": float(np.nanmean([wg["avg_games"], lg["avg_games"]])),
            "w_surf_games":  wg["surf_games"], "l_surf_games": lg["surf_games"],
        })

    if not feat_rows:
        return upcoming

    feat_df = pd.DataFrame(feat_rows)
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = global_medians.get(col, 0.0)
        else:
            feat_df[col] = feat_df[col].fillna(global_medians.get(col, 0.0))

    probs = model.predict_proba(feat_df[FEATURE_COLS])[:, 1]

    result = upcoming.copy().reset_index(drop=True)
    result[f"prob_over_{threshold}"] = probs
    result["resolved_p1"]  = resolved_p1
    result["resolved_p2"]  = resolved_p2
    result["conf_p1"]      = conf_p1
    result["conf_p2"]      = conf_p2
    result["elo_conf"]     = result.apply(
        lambda r: "✅ High"   if r.conf_p1 in ("high","medium") and r.conf_p2 in ("high","medium")
             else "⚠️ Partial" if r.conf_p1 != "none" or r.conf_p2 != "none"
             else "❌ Unknown", axis=1)
    result["elo_p1"]       = feat_df["elo_w"].values
    result["elo_p2"]       = feat_df["elo_l"].values
    result["elo_diff"]     = feat_df["elo_abs"].values
    result["exp_p1"]       = feat_df["exp_w"].values
    result["n_p1"]         = feat_df["n_w"].values
    result["n_p2"]         = feat_df["n_l"].values
    result["w_avg_games"]  = feat_df["w_avg_games"].values
    result["l_avg_games"]  = feat_df["l_avg_games"].values

    return result.sort_values(f"prob_over_{threshold}", ascending=False)


# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────

def export_excel(preds: pd.DataFrame, threshold: int, elo_df: pd.DataFrame) -> BytesIO:
    out = preds.copy()
    pc  = f"prob_over_{threshold}"
    rename = {
        "Date":"Data", "Winner":"API_Nome_J1", "Loser":"API_Nome_J2",
        "resolved_p1":"Nome_Historico_J1", "resolved_p2":"Nome_Historico_J2",
        "elo_conf":"Elo_Confianca",
        "Surface":"Superfície", "Round":"Ronda",
        pc: f"Prob_Over_{threshold}",
        "elo_p1":"Elo_J1", "elo_p2":"Elo_J2",
        "elo_diff":"Elo_Dif", "exp_p1":"Prob_Vitoria_J1",
        "n_p1":"N_J1", "n_p2":"N_J2",
        "w_avg_games":"AvgGames_J1", "l_avg_games":"AvgGames_J2",
    }
    out = out.rename(columns={k:v for k,v in rename.items() if k in out.columns})
    order = [
        "Data", "API_Nome_J1", "API_Nome_J2",
        "Nome_Historico_J1", "Nome_Historico_J2", "Elo_Confianca",
        "Superfície", "Ronda", f"Prob_Over_{threshold}",
        "Elo_J1", "Elo_J2", "Elo_Dif", "Prob_Vitoria_J1",
        "N_J1", "N_J2", "AvgGames_J1", "AvgGames_J2",
    ]
    out = out[[c for c in order if c in out.columns]]
    if "Data" in out.columns:
        out["Data"] = pd.to_datetime(out["Data"]).dt.strftime("%Y-%m-%d")
    for col in [f"Prob_Over_{threshold}", "Prob_Vitoria_J1"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    for col in ["Elo_J1","Elo_J2","Elo_Dif"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="Previsões", index=False)
        elo_df.head(200).to_excel(writer, sheet_name="Elo_Rankings", index=False)
        pc_raw = preds[f"prob_over_{threshold}"]
        pd.DataFrame({
            "Métrica": ["Gerado em","Jogos","Média Prob",f"Over {threshold} histórico",
                        "Spread std",">60%",">65%",">70%"],
            "Valor":   [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(preds),
                        f"{pc_raw.mean():.1%}", "-", f"{pc_raw.std():.1%}",
                        int((pc_raw>0.60).sum()), int((pc_raw>0.65).sum()),
                        int((pc_raw>0.70).sum())],
        }).to_excel(writer, sheet_name="Resumo", index=False)
        out.head(10).to_excel(writer, sheet_name="TOP_10", index=False)

        for sheet in writer.sheets.values():
            for col in sheet.columns:
                w = max((len(str(c.value)) for c in col if c.value), default=8)
                sheet.column_dimensions[col[0].column_letter].width = min(w+2, 42)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────
# ─── SIDEBAR ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────

st.sidebar.header("📂 Histórico")
uploaded = st.sidebar.file_uploader("Excel (.xlsx)", type=["xlsx"])

if uploaded:
    df_hist, source = load_excel(uploaded)
    if df_hist is not None:
        st.sidebar.success(f"✅ {uploaded.name}: {len(df_hist)} jogos")
else:
    with st.spinner("A carregar histórico do GitHub…"):
        df_hist, source = fetch_github()

if df_hist is None or df_hist.empty:
    st.error("Não foi possível carregar histórico.")
    st.stop()

st.sidebar.info(f"Fonte: {source} | {len(df_hist)} jogos")

st.sidebar.header("⚙️ Configurações")
threshold_games = st.sidebar.slider("Over/Under threshold", 15, 30, 22, 1)

st.sidebar.header("🔢 Parâmetros do Elo")
base_k_slider = st.sidebar.slider(
    "K base", 16, 64, 32, 4,
    help="K maior = Elo muda mais rápido por jogo"
)
with st.sidebar.expander("Multiplicadores K por ronda"):
    k_r32  = st.sidebar.slider("R32",   0.5, 2.0, 0.9, 0.1)
    k_r16  = st.sidebar.slider("R16",   0.5, 2.0, 1.0, 0.1)
    k_qf   = st.sidebar.slider("QF",    0.5, 2.0, 1.1, 0.1)
    k_sf   = st.sidebar.slider("SF",    0.5, 2.0, 1.3, 0.1)
    k_f    = st.sidebar.slider("Final", 0.5, 2.0, 1.5, 0.1)

# Build round_k dict from sliders — passed into cached function as a tuple
round_k_dict  = {"R32": k_r32, "R16": k_r16, "QF": k_qf, "SF": k_sf, "F": k_f, "Final": k_f}
round_k_tuple = tuple(sorted(round_k_dict.items()))  # hashable for @st.cache_data

# ─────────────────────────────────────────────────────────────
# ─── BUILD ELO + TRAIN ───────────────────────────────────────
# ─────────────────────────────────────────────────────────────

st.title("🎾 Challenger Predictor v5 — Dynamic Elo + Serve Stats")
st.caption(f"Fonte: {source} | {len(df_hist)} jogos | Modelo: Ensemble (GBM + RF + LR)")

with st.spinner("A construir Elo e features (processamento cronológico)…"):
    elo_sys, feat_df = build_elo_and_features(df_hist, float(base_k_slider), round_k_tuple)

with st.spinner("A treinar modelo…"):
    train_result = train_model(feat_df, threshold_games)

if train_result[0] is None:
    st.error("Dados insuficientes para treinar o modelo.")
    st.stop()

model, avg_games, over_pct, cv_acc, cv_auc, global_medians = train_result
elo_ratings_df = elo_sys.ratings_df()

# Name index for fuzzy resolver
all_players  = elo_ratings_df["Player"].tolist()
name_idx     = build_name_index(all_players)

# ── Metrics ────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jogadores no Elo", len(elo_ratings_df))
c2.metric(f"Over {threshold_games} histórico", f"{over_pct:.1f}%")
c3.metric("Avg Games", f"{avg_games:.1f}")
c4.metric("CV Accuracy", f"{cv_acc:.1%}")
c5.metric("CV AUC", f"{cv_auc:.3f}")

# ── Tabs ───────────────────────────────────────────────────────
tab_pred, tab_elo, tab_info = st.tabs(["📅 Previsões", "🏆 Elo Rankings", "ℹ️ Sobre o Modelo"])

# ══════════════════════════════════════════════════════════════
with tab_elo:
    st.subheader("🏆 Elo Rankings — Top 100")
    st.caption(
        f"K base: {base_k_slider} | Multiplicadores: "
        f"R32×{k_r32} R16×{k_r16} QF×{k_qf} SF×{k_sf} Final×{k_f} | "
        "Jogos de 3 sets: K×1.2 | Novos jogadores (<10 jogos): K×1.5"
    )

    sort_opt = st.selectbox("Ordenar por", ["Elo Geral", "Elo Clay", "Elo Hard", "Elo Grass"])
    sort_col = {"Elo Geral":"Elo","Elo Clay":"Elo_Clay","Elo Hard":"Elo_Hard","Elo Grass":"Elo_Grass"}[sort_opt]

    top100 = elo_ratings_df.sort_values(sort_col, ascending=False).head(100).reset_index(drop=True)
    top100.index += 1

    st.dataframe(
        top100.style
            .format({"Elo":"{:.0f}","Elo_Clay":"{:.0f}","Elo_Hard":"{:.0f}","Elo_Grass":"{:.0f}"})
            .background_gradient(subset=[sort_col], cmap="RdYlGn"),
        use_container_width=True,
        height=600,
    )

    st.subheader("🔍 Pesquisar jogador")
    search = st.text_input("Nome (parcial ou completo)")
    if search:
        found = elo_ratings_df[elo_ratings_df["Player"].str.contains(search, case=False, na=False)]
        if len(found):
            st.dataframe(
                found.style.format({"Elo":"{:.0f}","Elo_Clay":"{:.0f}","Elo_Hard":"{:.0f}","Elo_Grass":"{:.0f}"}),
                use_container_width=True,
            )
        else:
            # Try fuzzy resolve
            resolved, conf = resolve_player_name(search, name_idx, all_players)
            if resolved:
                st.info(f"Sugestão: **{resolved}** (confiança: {conf})")
                found2 = elo_ratings_df[elo_ratings_df["Player"] == resolved]
                st.dataframe(
                    found2.style.format({"Elo":"{:.0f}","Elo_Clay":"{:.0f}","Elo_Hard":"{:.0f}","Elo_Grass":"{:.0f}"}),
                    use_container_width=True,
                )
            else:
                st.warning(f"'{search}' não encontrado no histórico.")

    # K-factor explanation
    with st.expander("📐 Como funciona o K-factor dinâmico"):
        ex_data = []
        for rnd in ["R32", "R16", "QF", "SF", "Final"]:
            ex_data.append({
                "Ronda": rnd,
                "K (veterano, 2 sets)": f"{dynamic_k(50, rnd, False, base_k_slider, round_k_dict):.1f}",
                "K (veterano, 3 sets)": f"{dynamic_k(50, rnd, True,  base_k_slider, round_k_dict):.1f}",
                "K (novo, 2 sets)":     f"{dynamic_k(5,  rnd, False, base_k_slider, round_k_dict):.1f}",
                "K (novo, 3 sets)":     f"{dynamic_k(5,  rnd, True,  base_k_slider, round_k_dict):.1f}",
            })
        st.dataframe(pd.DataFrame(ex_data).set_index("Ronda"), use_container_width=True)
        st.markdown("""
        **Veterano** = 50+ jogos no histórico | **Novo** = 5 jogos  
        3 sets = jogo mais equilibrado → K×1.2 (revela mais sobre o nível real)
        """)

# ══════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("Sobre o modelo e o Elo")
    st.markdown(f"""
    **Features utilizadas ({len(FEATURE_COLS)} total):**

    | Grupo | Features |
    |---|---|
    | Elo geral | elo_diff, elo_abs, elo_w, elo_l, exp_w |
    | Elo surface | elo_surf_diff, elo_surf_abs, elo_w_surf, elo_l_surf, exp_w_surf |
    | Contexto | surface (Clay/Hard/Grass), indoor, round |
    | Serve Winner | 1stWon%, 2ndWon%, ace%, bp_save%, bp_faced/game |
    | Serve Loser | idem |
    | Combinado | serve_dom_sum, ace_sum, bp_save_diff |
    | Histórico de jogos | avg_games, surf_games por jogador |
    | Experiência | n_matches W, n_matches L |

    **Por que a API retorna Elo=1500 para muitos jogos:**
    A API devolve abreviações ("S. Jones") enquanto o histórico Challenger usa nomes completos
    ("Sebastian Jones"). O v5 inclui um resolver fuzzy que tenta fazer o match por inicial + apelido.
    Jogadores que genuinamente não estão no histórico (ITF, sub-Challenger) ficam com 1500.
    Esses jogos são sinalizados como "❌ Unknown" na coluna Elo_Confiança.

    **Accuracy ceiling:**
    Total de games em Challenger tem std≈5.8 em torno de média≈22.7. AUC realista ≈ 0.54.
    O valor está no spread: jogos com prob >65% são genuinamente mais longos que jogos <35%.
    """)

# ══════════════════════════════════════════════════════════════
with tab_pred:
    st.header("📅 Jogos de hoje e amanhã")

    api_matches = fetch_api_matches()

    if api_matches.empty:
        st.warning("⚠️ API sem resultados.")
        st.subheader("📤 Carrega ficheiro de jogos manualmente")
        st.markdown("""
        Colunas obrigatórias: **Date**, **Winner**, **Loser**  
        Opcionais: **Surface** (Clay/Hard/Grass) | **Indoor** (I/O) | **Round** (R32/R16/QF/SF/F)
        """)
        manual = st.file_uploader("Excel (.xlsx)", type=["xlsx"], key="manual")
        if manual:
            try:
                mdf = pd.read_excel(manual)
                mdf["Date"] = pd.to_datetime(mdf["Date"])
                for c, d in [("Surface","Hard"),("Indoor","O"),("Round","R32")]:
                    if c not in mdf.columns: mdf[c] = d
                upcoming = mdf
                st.success(f"✅ {len(upcoming)} jogos carregados")
            except Exception as e:
                st.error(f"Erro: {e}"); st.stop()
        else:
            st.info("👆 Carrega um ficheiro ou aguarda a API.")
            st.stop()
    else:
        upcoming = api_matches

    if upcoming.empty:
        st.info("Nenhum jogo disponível.")
        st.stop()

    with st.spinner("A calcular previsões com Elo dinâmico e name resolver…"):
        preds = predict_matches(
            upcoming, elo_sys, model, global_medians,
            threshold_games, name_idx, all_players
        )

    prob_col = f"prob_over_{threshold_games}"
    probs_all = preds[prob_col]

    # Health metrics
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Prob mín",  f"{probs_all.min():.1%}")
    c2.metric("Prob média", f"{probs_all.mean():.1%}")
    c3.metric("Prob máx",  f"{probs_all.max():.1%}")
    c4.metric("Spread std", f"{probs_all.std():.1%}")

    # Name resolution stats
    n_known   = (preds["elo_conf"] == "✅ High").sum()
    n_partial = (preds["elo_conf"] == "⚠️ Partial").sum()
    n_unknown = (preds["elo_conf"] == "❌ Unknown").sum()
    st.info(
        f"Resolução de nomes: "
        f"✅ Ambos conhecidos: **{n_known}** | "
        f"⚠️ Um conhecido: **{n_partial}** | "
        f"❌ Nenhum no histórico (Elo=1500): **{n_unknown}**"
    )

    if probs_all.std() < 0.02:
        st.warning("⚠️ Spread muito baixo — quase todos os jogadores são desconhecidos no histórico Challenger")

    # ── Results table ──────────────────────────────────────────
    st.subheader(f"📋 Previsões — Over {threshold_games} Games")

    disp_cols = {
        "Date":          "Data",
        "Winner":        "Jogador 1 (API)",
        "Loser":         "Jogador 2 (API)",
        "resolved_p1":   "J1 (histórico)",
        "resolved_p2":   "J2 (histórico)",
        "elo_conf":      "Confiança Elo",
        "Surface":       "Sup",
        "Round":         "Ronda",
        prob_col:        f"Over {threshold_games}",
        "elo_p1":        "Elo J1",
        "elo_p2":        "Elo J2",
        "elo_diff":      "Δ Elo",
        "exp_p1":        "Win% J1",
    }
    exist = {k: v for k, v in disp_cols.items() if k in preds.columns}
    disp  = preds[list(exist.keys())].rename(columns=exist)
    fmt   = {
        f"Over {threshold_games}": "{:.1%}",
        "Win% J1": "{:.1%}",
        "Elo J1": "{:.0f}", "Elo J2": "{:.0f}", "Δ Elo": "{:.0f}",
    }
    st.dataframe(
        disp.style.format(fmt, na_rep="-")
            .background_gradient(subset=[f"Over {threshold_games}"], cmap="RdYlGn"),
        use_container_width=True,
        height=500,
    )

    # ── Filter to known-only ──────────────────────────────────
    known_preds = preds[preds["elo_conf"].isin(["✅ High","⚠️ Partial"])]
    if len(known_preds) > 0:
        st.subheader(f"🎯 Jogos com Elo conhecido ({len(known_preds)})")
        disp2 = known_preds[list(exist.keys())].rename(columns=exist)
        st.dataframe(
            disp2.style.format(fmt, na_rep="-")
                .background_gradient(subset=[f"Over {threshold_games}"], cmap="RdYlGn"),
            use_container_width=True,
        )

    # ── TOP 5 ─────────────────────────────────────────────────
    st.subheader(f"🔥 TOP 5 — Over {threshold_games} Games")
    for i, (_, row) in enumerate(preds.head(5).iterrows(), 1):
        elo1 = row.get("elo_p1", np.nan)
        elo2 = row.get("elo_p2", np.nan)
        exp1 = row.get("exp_p1", np.nan)
        conf = row.get("elo_conf", "❌ Unknown")
        elo_str = f"Elo: {elo1:.0f} vs {elo2:.0f}" if not np.isnan(elo1) else "Elo: N/A"
        exp_str = f"| Win%: {exp1:.0%}" if not np.isnan(exp1) else ""
        r1 = row.get("resolved_p1", row["Winner"])
        r2 = row.get("resolved_p2", row["Loser"])
        with st.container():
            st.markdown(
                f"**{i}. {r1} vs {r2}** {conf}  \n"
                f"📅 {pd.Timestamp(row['Date']).date()} | 🎾 {row.get('Surface','?')} | "
                f"{row.get('Round','?')} | {elo_str} {exp_str}"
            )
            st.progress(
                min(float(row[prob_col]), 1.0),
                text=f"Over {threshold_games}: **{row[prob_col]:.1%}**"
            )

    # ── Export ────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("📥 Exportar para Excel", type="primary", use_container_width=True):
            with st.spinner("A gerar Excel…"):
                try:
                    buf   = export_excel(preds, threshold_games, elo_ratings_df)
                    fname = f"challenger_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    st.success("✅ Pronto!")
                    st.download_button(
                        "💾 Descarregar Excel", data=buf, file_name=fname,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Erro: {e}")
