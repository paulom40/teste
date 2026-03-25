import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from io import BytesIO
from datetime import datetime, timedelta
import requests
import json
import re
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Ténis — Predição de Jogos Competitivos",
    page_icon="🎾",
    layout="wide"
)

# ============================================================
# 1. NORMALIZAÇÃO E PARSING DO SCORE
# ============================================================

def normalize_columns(df):
    col_map = {
        "winner_name": "Winner", "loser_name": "Loser",
        "winner_rank": "WRank",  "loser_rank": "LRank",
        "winner_rank_points": "WPts", "loser_rank_points": "LPts",
        "tourney_date": "Date",  "score": "Score",
        "surface": "Surface"
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "Score" in df.columns:
        df = parse_score(df)

    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except:
            df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")

    if "Surface" not in df.columns:
        df["Surface"] = "Hard"

    return df


def parse_score(df):
    def _parse(score):
        if pd.isna(score):
            return [np.nan] * 11
        sets = re.findall(r"(\d+)-(\d+)", str(score))
        w = [int(s[0]) for s in sets]
        l = [int(s[1]) for s in sets]
        while len(w) < 5: w.append(np.nan)
        while len(l) < 5: l.append(np.nan)
        wsets = sum(1 for a, b in zip(w, l) if not pd.isna(a) and not pd.isna(b) and a > b)
        return w[:5] + l[:5] + [wsets]

    parsed = df["Score"].apply(_parse)
    cols = ["W1","W2","W3","W4","W5","L1","L2","L3","L4","L5","Wsets"]
    for i, col in enumerate(cols):
        df[col] = parsed.apply(lambda x: x[i])
    return df


def calculate_total_games(df):
    total = pd.Series(0.0, index=df.index)
    for i in range(1, 6):
        wc, lc = f"W{i}", f"L{i}"
        if wc in df.columns and lc in df.columns:
            w = pd.to_numeric(df[wc], errors="coerce").fillna(-1)
            l = pd.to_numeric(df[lc], errors="coerce").fillna(-1)
            valid = (w >= 0) & (l >= 0)
            total += np.where(valid, w + l, 0)
    return total.where(total > 0, np.nan)

# ============================================================
# 2. ELO GLOBAL + ELO POR SUPERFÍCIE
# ============================================================

def build_elo_ratings(df_hist):
    df = df_hist.sort_values("Date").copy()
    elo = {}
    matches_count = {}
    default_elo = 1500

    surface_map = {"Clay": "elo_clay", "Hard": "elo_hard", "Grass": "elo_grass"}

    def get_k(player):
        n = matches_count.get(player, 0)
        if n < 20: return 40
        if n < 50: return 32
        if n < 100: return 24
        return 16

    for _, row in df.iterrows():
        w, l = row["Winner"], row["Loser"]
        surf = row.get("Surface", "Hard")

        for p in [w, l]:
            if p not in elo:
                elo[p] = {
                    "elo": default_elo,
                    "elo_clay": default_elo,
                    "elo_hard": default_elo,
                    "elo_grass": default_elo
                }
                matches_count[p] = 0

        k_w, k_l = get_k(w), get_k(l)

        Ew = 1 / (1 + 10 ** ((elo[l]["elo"] - elo[w]["elo"]) / 400))
        El = 1 - Ew
        elo[w]["elo"] += k_w * (1 - Ew)
        elo[l]["elo"] += k_l * (0 - El)

        key = surface_map.get(surf, "elo_hard")
        Ew_surf = 1 / (1 + 10 ** ((elo[l][key] - elo[w][key]) / 400))
        El_surf = 1 - Ew_surf
        elo[w][key] += k_w * (1 - Ew_surf)
        elo[l][key] += k_l * (0 - El_surf)

        matches_count[w] += 1
        matches_count[l] += 1

    return elo

# ============================================================
# 3. H2H GLOBAL
# ============================================================

def build_h2h_stats(df_hist):
    df = df_hist.copy()
    df["Total_Games"] = calculate_total_games(df)
    df = df.sort_values("Date")

    h2h = {}
    df["pair"] = df.apply(lambda r: tuple(sorted([r["Winner"], r["Loser"]])), axis=1)

    for pair, group in df.groupby("pair"):
        games = group["Total_Games"].dropna()
        avg_g = float(games.mean()) if len(games) else np.nan
        p1, p2 = pair
        p1_wins = (group["Winner"] == p1).sum()
        p1_wr = p1_wins / len(group)
        h2h[pair] = {
            "avg_games": avg_g,
            "n_h2h": len(group),
            "p1_win_rate": p1_wr,
        }

    return h2h
# ============================================================
# 4. STATS RECENTES (ÚLTIMOS 30 JOGOS) — GLOBAIS E POR SUPERFÍCIE
# ============================================================

def compute_player_match_stats(row):
    # Winner service stats
    w_spw = (row["w_1stWon"] + row["w_2ndWon"]) / row["w_svpt"] if row.get("w_svpt", 0) > 0 else np.nan
    w_rpw = 1 - ((row["l_1stWon"] + row["l_2ndWon"]) / row["l_svpt"]) if row.get("l_svpt", 0) > 0 else np.nan
    w_sgw = 1 - ((row["w_bpFaced"] - row["w_bpSaved"]) / row["w_SvGms"]) if row.get("w_SvGms", 0) > 0 else np.nan
    w_rgw = ((row["l_bpFaced"] - row["l_bpSaved"]) / row["l_SvGms"]) if row.get("l_SvGms", 0) > 0 else np.nan

    # Loser service stats
    l_spw = (row["l_1stWon"] + row["l_2ndWon"]) / row["l_svpt"] if row.get("l_svpt", 0) > 0 else np.nan
    l_rpw = 1 - ((row["w_1stWon"] + row["w_2ndWon"]) / row["w_svpt"]) if row.get("w_svpt", 0) > 0 else np.nan
    l_sgw = 1 - ((row["l_bpFaced"] - row["l_bpSaved"]) / row["l_SvGms"]) if row.get("l_SvGms", 0) > 0 else np.nan
    l_rgw = ((row["w_bpFaced"] - row["w_bpSaved"]) / row["w_SvGms"]) if row.get("w_SvGms", 0) > 0 else np.nan

    # Games won
    total_games = calculate_total_games(pd.DataFrame([row])).iloc[0]
    w_games = sum([row.get(f"W{i}", 0) for i in range(1, 6) if not pd.isna(row.get(f"W{i}", np.nan))])
    l_games = sum([row.get(f"L{i}", 0) for i in range(1, 6) if not pd.isna(row.get(f"L{i}", np.nan))])

    return {
        "winner": {
            "spw": w_spw, "rpw": w_rpw, "sgw": w_sgw, "rgw": w_rgw,
            "games_won": w_games, "games_played": total_games,
            "aces": row.get("w_ace", np.nan),
            "df": row.get("w_df", np.nan),
            "minutes": row.get("minutes", np.nan)
        },
        "loser": {
            "spw": l_spw, "rpw": l_rpw, "sgw": l_sgw, "rgw": l_rgw,
            "games_won": l_games, "games_played": total_games,
            "aces": row.get("l_ace", np.nan),
            "df": row.get("l_df", np.nan),
            "minutes": row.get("minutes", np.nan)
        }
    }


def build_recent_stats(df_hist, window=30):
    df = df_hist.sort_values("Date").copy()
    df["Total_Games"] = calculate_total_games(df)

    match_stats = df.apply(compute_player_match_stats, axis=1)

    players = set(df["Winner"]).union(df["Loser"])
    recent_stats = {p: {} for p in players}

    history = {p: [] for p in players}
    history_surf = {p: {"Clay": [], "Hard": [], "Grass": []} for p in players}

    for idx, row in df.iterrows():
        stats = match_stats[idx]
        surf = row.get("Surface", "Hard")

        w = row["Winner"]
        history[w].append(stats["winner"])
        history_surf[w][surf].append(stats["winner"])

        l = row["Loser"]
        history[l].append(stats["loser"])
        history_surf[l][surf].append(stats["loser"])

    def avg_last(stats_list, key):
        vals = [s[key] for s in stats_list[-window:] if not pd.isna(s[key])]
        return np.mean(vals) if len(vals) else np.nan

    for p in players:
        recent_stats[p] = {
            "spw_30": avg_last(history[p], "spw"),
            "rpw_30": avg_last(history[p], "rpw"),
            "sgw_30": avg_last(history[p], "sgw"),
            "rgw_30": avg_last(history[p], "rgw"),
            "games_won_30": avg_last(history[p], "games_won"),
            "games_played_30": avg_last(history[p], "games_played"),
            "aces_30": avg_last(history[p], "aces"),
            "df_30": avg_last(history[p], "df"),
            "minutes_30": avg_last(history[p], "minutes"),
            "spw_30_surf": {},
            "rpw_30_surf": {},
            "sgw_30_surf": {},
            "rgw_30_surf": {},
            "games_won_30_surf": {}
        }

        for surf in ["Clay", "Hard", "Grass"]:
            recent_stats[p]["spw_30_surf"][surf] = avg_last(history_surf[p][surf], "spw")
            recent_stats[p]["rpw_30_surf"][surf] = avg_last(history_surf[p][surf], "rpw")
            recent_stats[p]["sgw_30_surf"][surf] = avg_last(history_surf[p][surf], "sgw")
            recent_stats[p]["rgw_30_surf"][surf] = avg_last(history_surf[p][surf], "rgw")
            recent_stats[p]["games_won_30_surf"][surf] = avg_last(history_surf[p][surf], "games_won")

    return recent_stats

# ============================================================
# 5. FEATURE ENGINEERING (COM FALLBACK STATS + FALLBACK ELO)
# ============================================================

def engineer_features(row, recent_stats, elo, h2h_stats):
    p1, p2 = row["Winner"], row["Loser"]
    surf = row.get("Surface", "Hard")

    # Fallback stats
    default_stats = {
        "spw_30": np.nan, "rpw_30": np.nan, "sgw_30": np.nan, "rgw_30": np.nan,
        "games_won_30": np.nan, "games_played_30": np.nan,
        "aces_30": np.nan, "df_30": np.nan, "minutes_30": np.nan,
        "spw_30_surf": {"Clay": np.nan, "Hard": np.nan, "Grass": np.nan},
        "rpw_30_surf": {"Clay": np.nan, "Hard": np.nan, "Grass": np.nan},
        "sgw_30_surf": {"Clay": np.nan, "Hard": np.nan, "Grass": np.nan},
        "rgw_30_surf": {"Clay": np.nan, "Hard": np.nan, "Grass": np.nan},
        "games_won_30_surf": {"Clay": np.nan, "Hard": np.nan, "Grass": np.nan},
    }

    s1 = recent_stats.get(p1, default_stats)
    s2 = recent_stats.get(p2, default_stats)

    def diff(a, b): return (a - b) if not (pd.isna(a) or pd.isna(b)) else np.nan
    def summ(a, b): return (a + b) if not (pd.isna(a) or pd.isna(b)) else np.nan

    feats = {
        "spw_diff_30": diff(s1["spw_30"], s2["spw_30"]),
        "rpw_diff_30": diff(s1["rpw_30"], s2["rpw_30"]),
        "sgw_diff_30": diff(s1["sgw_30"], s2["sgw_30"]),
        "rgw_diff_30": diff(s1["rgw_30"], s2["rgw_30"]),
        "games_won_diff_30": diff(s1["games_won_30"], s2["games_won_30"]),
        "spw_sum_30": summ(s1["spw_30"], s2["spw_30"]),
        "rpw_sum_30": summ(s1["rpw_30"], s2["rpw_30"]),
        "sgw_sum_30": summ(s1["sgw_30"], s2["sgw_30"]),
        "rgw_sum_30": summ(s1["rgw_30"], s2["rgw_30"]),
        "games_won_sum_30": summ(s1["games_won_30"], s2["games_won_30"]),
    }

    # Fallback Elo
    default_elo = {
        "elo": 1500,
        "elo_clay": 1500,
        "elo_hard": 1500,
        "elo_grass": 1500
    }

    e1 = elo.get(p1, default_elo)
    e2 = elo.get(p2, default_elo)

    feats["elo_diff"] = diff(e1["elo"], e2["elo"])
    feats["elo_surf_diff"] = diff(
        e1.get(f"elo_{surf.lower()}", 1500),
        e2.get(f"elo_{surf.lower()}", 1500)
    )

    # H2H
    pair = tuple(sorted([p1, p2]))
    h2h = h2h_stats.get(pair, {})
    feats["h2h_avg_games"] = h2h.get("avg_games", np.nan)
    feats["h2h_balance"] = abs(h2h.get("p1_win_rate", 0.5) - 0.5)
    feats["h2h_n"] = h2h.get("n_h2h", 0)

    # Surface encoding
    feats["surface_enc"] = {"Clay": 0, "Hard": 1, "Grass": 2}.get(surf, 1)
    feats["surface_bonus"] = {"Clay": 1.5, "Hard": 0.0, "Grass": -1.5}.get(surf, 0)

    return feats
# ============================================================
# 6. COLUNAS DE FEATURES
# ============================================================

FEATURE_COLS = [
    "spw_diff_30", "rpw_diff_30", "sgw_diff_30", "rgw_diff_30",
    "games_won_diff_30",
    "spw_sum_30", "rpw_sum_30", "sgw_sum_30", "rgw_sum_30",
    "games_won_sum_30",
    "elo_diff", "elo_surf_diff",
    "h2h_avg_games", "h2h_balance", "h2h_n",
    "surface_enc", "surface_bonus"
]

# ============================================================
# 7. MODELO ENSEMBLE
# ============================================================

def make_ensemble():
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        min_samples_leaf=20,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.8,
            max_iter=800,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    ensemble = VotingClassifier(
        estimators=[("gb", gb), ("rf", rf), ("lr", lr)],
        voting="soft",
        weights=[3, 2, 1]
    )

    return ensemble

# ============================================================
# 8. CONSTRUÇÃO DO DATASET DE TREINO
# ============================================================

def build_training_dataset(df_hist, recent_stats, elo, h2h_stats, threshold_games=22):
    df = df_hist.sort_values("Date").copy()
    df["Total_Games"] = calculate_total_games(df)

    feature_rows = []
    labels_3sets = []
    labels_over = []

    for _, row in df.iterrows():
        if pd.isna(row["Total_Games"]):
            continue

        feats = engineer_features(row, recent_stats, elo, h2h_stats)

        # Label 3+ sets
        sets_played = 0
        for j in range(1, 6):
            w = row.get(f"W{j}", np.nan)
            l = row.get(f"L{j}", np.nan)
            if not (pd.isna(w) or pd.isna(l)):
                sets_played += 1

        three_sets = 1 if sets_played >= 3 else 0

        # Label Over threshold
        over = 1 if row["Total_Games"] > threshold_games else 0

        feature_rows.append(feats)
        labels_3sets.append(three_sets)
        labels_over.append(over)

    feat_df = pd.DataFrame(feature_rows)
    feat_df["three_sets"] = labels_3sets
    feat_df["over_threshold"] = labels_over

    return feat_df

# ============================================================
# 9. TREINO DO MODELO 3+ SETS
# ============================================================

def train_three_sets_model(feat_df):
    dfm = feat_df.dropna(subset=FEATURE_COLS + ["three_sets"]).copy()

    if len(dfm) < 100:
        st.sidebar.warning(f"Apenas {len(dfm)} jogos com features completas para 3+ Sets.")
        return None

    X = dfm[FEATURE_COLS]
    y = dfm["three_sets"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_ensemble()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    test_score = model.score(X_test, y_test)

    st.sidebar.success(f"Modelo 3+ Sets treinado com {len(dfm)} jogos")
    st.sidebar.info(f"CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%} | Teste: {test_score:.1%}")

    return model

# ============================================================
# 10. TREINO DO MODELO OVER GAMES
# ============================================================

def train_over_games_model(feat_df, threshold=22):
    dfm = feat_df.dropna(subset=FEATURE_COLS + ["over_threshold"]).copy()

    if len(dfm) < 100:
        st.sidebar.warning(f"Apenas {len(dfm)} jogos com features completas para Over {threshold}.")
        return None

    X = dfm[FEATURE_COLS]
    y = dfm["over_threshold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_ensemble()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    test_score = model.score(X_test, y_test)

    st.sidebar.success(f"Modelo Over {threshold} treinado com {len(dfm)} jogos")
    st.sidebar.info(f"CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%} | Teste: {test_score:.1%}")

    return model
# ============================================================
# 11. PREVISÕES PARA JOGOS FUTUROS (COM TODOS OS PATCHES)
# ============================================================

def predict_for_upcoming(upcoming_df, df_hist, model_3sets, model_over, threshold=22):
    df_up = upcoming_df.copy()

    # Construir estruturas globais
    elo = build_elo_ratings(df_hist)
    h2h_stats = build_h2h_stats(df_hist)
    recent_stats = build_recent_stats(df_hist, window=30)

    feature_list = []
    meta_info = []

    for _, row in df_up.iterrows():
        p1, p2 = row["Winner"], row["Loser"]

        feats = engineer_features(row, recent_stats, elo, h2h_stats)
        feature_list.append(feats)

        s1 = recent_stats.get(p1, {})
        s2 = recent_stats.get(p2, {})

        n1 = 0 if np.isnan(s1.get("games_played_30", np.nan)) else 30
        n2 = 0 if np.isnan(s2.get("games_played_30", np.nan)) else 30

        pair = tuple(sorted([p1, p2]))
        h2h = h2h_stats.get(pair, {})
        h2h_n = h2h.get("n_h2h", 0)

        meta_info.append({"n1": n1, "n2": n2, "h2h_n": h2h_n})

    feat_df = pd.DataFrame(feature_list)
    meta_df = pd.DataFrame(meta_info)

    # ============================================================
    # PATCH DEFINITIVO — garantir que não há NaN, inf ou colunas faltantes
    # ============================================================

    # Criar colunas em falta
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = np.nan

    # Manter apenas as colunas do modelo
    feat_df = feat_df[FEATURE_COLS]

    # Converter tudo para float
    feat_df = feat_df.astype(float)

    # Remover inf e substituir NaN por 0
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.fillna(0)

    X = feat_df

    # ============================================================
    # PREVISÕES
    # ============================================================

    if model_3sets is not None:
        prob_3 = model_3sets.predict_proba(X)[:, 1]
    else:
        prob_3 = np.full(len(X), 0.33)

    if model_over is not None:
        prob_over = model_over.predict_proba(X)[:, 1]
    else:
        prob_over = np.full(len(X), 0.5)

    df_up["prob_3_sets"] = prob_3
    df_up[f"prob_over_{threshold}_games"] = prob_over
    df_up["prob_competitive_match"] = 0.5 * prob_3 + 0.5 * prob_over

    # ============================================================
    # CONFIANÇA
    # ============================================================

    data_conf = (
        (meta_df["n1"].clip(0, 30) / 30) * 0.4 +
        (meta_df["n2"].clip(0, 30) / 30) * 0.4 +
        (meta_df["h2h_n"].clip(0, 5) / 5) * 0.2
    )

    prob_conf = (np.abs(df_up["prob_competitive_match"] - 0.5) * 2).clip(0, 1)

    df_up["confiança_modelo"] = (0.6 * data_conf + 0.4 * prob_conf)

    # Guardar algumas features úteis
    df_up["elo_diff"] = feat_df["elo_diff"].values
    df_up["rank_diff_dummy"] = np.nan

    return df_up.sort_values("prob_competitive_match", ascending=False)
# ============================================================
# 12. FILTROS E SELEÇÃO DOS MELHORES JOGOS
# ============================================================

def filtrar_por_confianca(df, limiar=0.6):
    """Filtra jogos com confiança mínima."""
    return df[df["confiança_modelo"] >= limiar].copy()


def selecionar_melhores_jogos(df, top_n=10, peso_prob=0.6, peso_conf=0.4):
    """Seleciona os jogos mais promissores com base em probabilidade e confiança."""
    df = df.copy()
    df["score_final"] = (
        peso_prob * df["prob_competitive_match"] +
        peso_conf * df["confiança_modelo"]
    )
    return df.sort_values("score_final", ascending=False).head(top_n)


# ============================================================
# 13. API — EXTRAÇÃO DE SUPERFÍCIE
# ============================================================

def extract_surface_from_tournament(tournament_name):
    """Tenta inferir a superfície a partir do nome do torneio."""
    if not isinstance(tournament_name, str):
        return "Hard"

    t = tournament_name.lower()

    if "clay" in t or "terra" in t:
        return "Clay"
    if "grass" in t or "wimbledon" in t:
        return "Grass"
    if "hard" in t or "cement" in t:
        return "Hard"

    return "Hard"


# ============================================================
# 14. API — OBTENÇÃO DE JOGOS (HOJE E AMANHÃ)
# ============================================================

def fetch_matches_from_api():
    API_URL = "https://api.api-tennis.com/tennis/"
    API_KEY = "7e3c6125ceaf5442372a487f9948c083a8778bb9604f49d8b33efc0e005f275c"

    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    params = {
        "method": "get_fixtures",
        "APIkey": API_KEY,
        "date_start": today,
        "date_stop": tomorrow,
    }

    try:
        with st.spinner(f"A buscar jogos de {today} a {tomorrow}..."):
            response = requests.get(API_URL, params=params, timeout=15)

            if response.status_code != 200 or not response.text:
                return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])

            try:
                data = response.json()
            except json.JSONDecodeError:
                return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])

        if data.get("success") != 1:
            return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])

        matches = data.get("result", [])
        if not matches:
            return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])

        df_api = pd.DataFrame(matches)

        required_cols = ["event_date", "event_first_player", "event_second_player"]
        if any(col not in df_api.columns for col in required_cols):
            return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])

        df_api["Date"] = pd.to_datetime(df_api["event_date"])
        df_api["Winner"] = df_api["event_first_player"]
        df_api["Loser"] = df_api["event_second_player"]

        if "tournament_name" in df_api.columns:
            df_api["Surface"] = df_api["tournament_name"].apply(extract_surface_from_tournament)
        else:
            df_api["Surface"] = "Hard"

        # Filtrar jogos já terminados
        if "event_status" in df_api.columns:
            df_api = df_api[df_api["event_status"] == ""]

        result_df = df_api[["Date", "Winner", "Loser", "Surface"]].copy()
        result_df = result_df.drop_duplicates().dropna(subset=["Winner", "Loser"])

        return result_df

    except Exception:
        return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])
# ============================================================
# 15. EXPORTAÇÃO EXCEL PRO
# ============================================================

def export_to_excel(predictions_df, threshold_games, top_jogos_df):
    export_df = predictions_df.copy()

    # Renomear colunas para o relatório
    export_df = export_df.rename(columns={
        "Date": "Data",
        "Winner": "Jogador_A",
        "Loser": "Jogador_B",
        "Surface": "Superfície",
        "prob_3_sets": "Probabilidade_3_Sets",
        f"prob_over_{threshold_games}_games": f"Probabilidade_Over_{threshold_games}_Games",
        "prob_competitive_match": "Probabilidade_Jogo_Competitivo",
        "confiança_modelo": "Confianca_Modelo",
        "score_final": "Score_Final",
        "elo_diff": "Diferenca_Elo",
        "rank_diff_dummy": "Diferenca_Ranking_Aproximada",
    })

    # Formatar datas
    export_df["Data"] = pd.to_datetime(export_df["Data"]).dt.strftime("%Y-%m-%d")

    # Formatar percentagens
    for col in [
        "Probabilidade_3_Sets",
        f"Probabilidade_Over_{threshold_games}_Games",
        "Probabilidade_Jogo_Competitivo",
        "Confianca_Modelo",
        "Score_Final"
    ]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(lambda x: f"{x:.1%}")

    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Aba principal
        export_df.to_excel(writer, sheet_name='Previsoes', index=False)

        # Aba Resumo
        summary_df = pd.DataFrame({
            "Métrica": [
                "Data de Geração",
                "Total de Jogos",
                "Média Probabilidade 3+ Sets",
                f"Média Probabilidade Over {threshold_games}",
                "Média Probabilidade Competitivo",
                "Média Confiança",
                "Jogos com Confiança ≥ 0.6",
                "Jogos com Score_Final ≥ 0.7",
            ],
            "Valor": [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(predictions_df),
                f"{predictions_df['prob_3_sets'].mean():.1%}",
                f"{predictions_df[f'prob_over_{threshold_games}_games'].mean():.1%}",
                f"{predictions_df['prob_competitive_match'].mean():.1%}",
                f"{predictions_df['confiança_modelo'].mean():.1%}",
                len(predictions_df[predictions_df['confiança_modelo'] >= 0.6]),
                len(top_jogos_df[top_jogos_df['score_final'] >= 0.7]),
            ]
        })
        summary_df.to_excel(writer, sheet_name='Resumo', index=False)

        # Aba Top Jogos
        top_jogos_df.to_excel(writer, sheet_name='Top_Jogos', index=False)

        # Aba Estatísticas por superfície
        stats_df = pd.DataFrame({
            "Superfície": predictions_df["Surface"].value_counts().index,
            "Jogos": predictions_df["Surface"].value_counts().values,
            "Média_Prob_Competitivo": predictions_df.groupby("Surface")["prob_competitive_match"].mean().values,
            "Média_Confiança": predictions_df.groupby("Surface")["confiança_modelo"].mean().values,
            "Média_Elo_Diff": predictions_df.groupby("Surface")["elo_diff"].mean().values,
        })
        stats_df.to_excel(writer, sheet_name='Estatisticas_Modelo', index=False)

    output.seek(0)
    return output
# ============================================================
# 16. INTERFACE STREAMLIT — PARTE 1 (HISTÓRICO + TREINO)
# ============================================================

st.title("🎾 Ténis — Predição de Jogos Competitivos (Stats Recentes + Elo + H2H)")

# -----------------------------
# Upload do histórico
# -----------------------------
st.sidebar.header("📂 Histórico de jogos")
hist_file = st.sidebar.file_uploader(
    "Escolhe um ficheiro Excel de histórico (.xlsx)",
    type=["xlsx"]
)

if hist_file is None:
    st.warning("Carrega primeiro um ficheiro de histórico com jogos (Winner, Loser, Date, Surface, Score, stats...).")
    st.stop()

df_hist = pd.read_excel(hist_file)
df_hist = normalize_columns(df_hist)

st.sidebar.info(f"Total de jogos históricos carregados: {len(df_hist)}")

# -----------------------------
# Slider para threshold Over
# -----------------------------
threshold_games = st.sidebar.slider(
    "Threshold para total de games (Over)",
    min_value=15,
    max_value=30,
    value=22,
    step=1
)

st.sidebar.header("⚙️ Treino do modelo")

# -----------------------------
# Construção das estruturas
# -----------------------------
with st.spinner("A preparar stats recentes, Elo e H2H..."):
    elo = build_elo_ratings(df_hist)
    h2h_stats = build_h2h_stats(df_hist)
    recent_stats = build_recent_stats(df_hist, window=30)
    feat_df = build_training_dataset(df_hist, recent_stats, elo, h2h_stats, threshold_games)

# -----------------------------
# Treino dos modelos
# -----------------------------
with st.spinner("A treinar modelos..."):
    model_3sets = train_three_sets_model(feat_df)
    model_over = train_over_games_model(feat_df, threshold_games)

if model_3sets is None and model_over is None:
    st.error("Não foi possível treinar nenhum modelo. Verifica se o histórico tem dados suficientes.")
    st.stop()

st.markdown("---")
st.header("📅 Jogos para previsão")
# ============================================================
# 17. INTERFACE STREAMLIT — PARTE 2 (PREVISÕES + DOWNLOAD)
# ============================================================

# Escolha da fonte dos jogos
modo = st.radio(
    "Fonte dos jogos:",
    ["API (hoje e amanhã)", "Ficheiro Excel"],
    horizontal=True
)

df_upcoming = None

# -----------------------------
# MODO API
# -----------------------------
if modo == "API (hoje e amanhã)":
    df_api = fetch_matches_from_api()

    if df_api.empty:
        st.warning("⚠️ A API não devolveu jogos. Podes usar um ficheiro Excel em alternativa.")
    else:
        st.success(f"Encontrados {len(df_api)} jogos via API.")
        st.dataframe(df_api)
        df_upcoming = df_api

# -----------------------------
# MODO EXCEL
# -----------------------------
else:
    st.write("Carrega um ficheiro Excel com jogos futuros (colunas: Date, Winner, Loser, Surface opcional).")

    upcoming_file = st.file_uploader(
        "Ficheiro de jogos futuros (.xlsx)",
        type=["xlsx"],
        key="upcoming"
    )

    if upcoming_file is not None:
        df_upcoming = pd.read_excel(upcoming_file)

        required_cols = ["Date", "Winner", "Loser"]
        missing = [c for c in required_cols if c not in df_upcoming.columns]

        if missing:
            st.error(f"Colunas em falta: {missing}")
            df_upcoming = None
        else:
            df_upcoming["Date"] = pd.to_datetime(df_upcoming["Date"])

            if "Surface" not in df_upcoming.columns:
                df_upcoming["Surface"] = "Hard"

            st.dataframe(df_upcoming)

# -----------------------------
# GERAR PREVISÕES
# -----------------------------
if df_upcoming is not None and not df_upcoming.empty:

    with st.spinner("🔮 A gerar previsões..."):
        preds = predict_for_upcoming(
            df_upcoming,
            df_hist,
            model_3sets,
            model_over,
            threshold_games
        )

    st.subheader("📊 Todas as previsões")
    st.dataframe(preds)

    # -----------------------------
    # FILTROS E TOP JOGOS
    # -----------------------------
    preds_filtradas = filtrar_por_confianca(preds, limiar=0.6)
    top_jogos = selecionar_melhores_jogos(preds_filtradas, top_n=10)

    st.subheader("🎯 Top jogos recomendados (confiança ≥ 0.6)")
    st.dataframe(top_jogos)

    # -----------------------------
    # DOWNLOAD EXCEL
    # -----------------------------
    excel_file = export_to_excel(preds, threshold_games, top_jogos)

    st.download_button(
        label="📥 Baixar Relatório Excel",
        data=excel_file,
        file_name="Relatorio_Tenis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
