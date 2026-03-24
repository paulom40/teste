import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import requests
from io import BytesIO
from datetime import datetime, timedelta
import re
import warnings
import json

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CHALLENGER 3+ Sets & Over 22 Games Predictor",
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
        # serve stats (may be present in richer datasets)
        "w_ace": "WAce", "l_ace": "LAce",
        "w_df": "WDf", "l_df": "LDf",
        "w_svpt": "WSvpt", "l_svpt": "LSvpt",
        "w_1stin": "W1stIn", "l_1stin": "L1stIn",
        "w_1stwon": "W1stWon", "l_1stwon": "L1stWon",
        "w_2ndwon": "W2ndWon", "l_2ndwon": "L2ndWon",
        "w_bpfaced": "WBpFaced", "l_bpfaced": "LBpFaced",
        "w_bpsaved": "WBpSaved", "l_bpsaved": "LBpSaved",
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
        while len(w) < 5:
            w.append(np.nan)
        while len(l) < 5:
            l.append(np.nan)
        wsets = sum(
            1 for a, b in zip(w[:5], l[:5])
            if not (pd.isna(a) or pd.isna(b)) and a > b
        )
        return w[:5] + l[:5] + [wsets]

    parsed = df["Score"].apply(_parse)
    cols = ["W1","W2","W3","W4","W5","L1","L2","L3","L4","L5","Wsets"]
    for i, col in enumerate(cols):
        df[col] = [row[i] for row in parsed]
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
    return total.where(total > 0, other=np.nan)


def extract_surface_from_tournament(tournament_name):
    if not isinstance(tournament_name, str):
        return "Hard"
    t = tournament_name.lower()
    if "clay" in t:
        return "Clay"
    elif "grass" in t or "wimbledon" in t:
        return "Grass"
    elif "hard" in t:
        return "Hard"
    return "Hard"


# ============================================================
# 2. FEATURE ENGINEERING (THE KEY IMPROVEMENT)
# ============================================================

def build_player_stats(df_hist):
    """
    Build a lookup of per-player stats computed from historical data.
    Returns a dict of player -> stats dict.
    This is the core of the improvement: rolling form, avg games, surface splits.
    """
    df = df_hist.copy()
    df = df.sort_values("Date")
    df["Total_Games"] = calculate_total_games(df)

    # Surface encoding
    surface_map = {"Clay": 0, "Hard": 1, "Grass": 2, "Carpet": 1}

    stats = {}

    all_players = set(df["Winner"].dropna().unique()) | set(df["Loser"].dropna().unique())

    for player in all_players:
        w_mask = df["Winner"] == player
        l_mask = df["Loser"] == player
        all_mask = w_mask | l_mask

        p_df = df[all_mask].copy()

        if len(p_df) < 3:
            continue

        # --- Basic ranking ---
        last_w = df[w_mask]["WRank"].dropna()
        last_l = df[l_mask]["LRank"].dropna()
        last_w_pts = df[w_mask]["WPts"].dropna()
        last_l_pts = df[l_mask]["LPts"].dropna()

        rank_vals = pd.concat([last_w, last_l])
        pts_vals = pd.concat([last_w_pts, last_l_pts])

        rank = float(rank_vals.iloc[-1]) if len(rank_vals) > 0 else 300.0
        pts = float(pts_vals.iloc[-1]) if len(pts_vals) > 0 else 30.0

        # --- Win rate (last 20 matches) ---
        recent = p_df.tail(20)
        win_rate = (recent["Winner"] == player).mean()

        # --- Average total games per match (last 20) ---
        recent_games = recent["Total_Games"].dropna()
        avg_games = float(recent_games.mean()) if len(recent_games) > 0 else 21.0

        # --- Average games per match by surface ---
        surface_avg = {}
        for surf in ["Clay", "Hard", "Grass"]:
            s_df = p_df[p_df["Surface"] == surf]["Total_Games"].dropna()
            surface_avg[surf] = float(s_df.mean()) if len(s_df) >= 3 else avg_games

        # --- Serve stats (if available) ---
        # Ace rate
        w_aces = df[w_mask]["WAce"].dropna()
        l_aces = df[l_mask]["LAce"].dropna()
        all_aces = pd.concat([w_aces, l_aces])
        avg_aces = float(all_aces.tail(20).mean()) if len(all_aces) > 0 else np.nan

        # Break points faced rate (proxy for serve solidity)
        w_bpf = df[w_mask]["WBpFaced"].dropna()
        l_bpf = df[l_mask]["LBpFaced"].dropna()
        all_bpf = pd.concat([w_bpf, l_bpf])
        avg_bpf = float(all_bpf.tail(20).mean()) if len(all_bpf) > 0 else np.nan

        # --- Tiebreak rate (proxy for serve dominance — more tiebreaks = stronger servers = fewer games) ---
        def count_tbs(row, is_winner):
            tb_count = 0
            score = str(row.get("Score", ""))
            tbs = re.findall(r"\(\d+\)", score)
            return len(tbs)

        tiebreak_rate = p_df.tail(20).apply(lambda r: count_tbs(r, r["Winner"] == player), axis=1).mean()

        stats[player] = {
            "rank": rank,
            "pts": pts,
            "win_rate": win_rate,
            "avg_games": avg_games,
            "surface_avg_Clay": surface_avg.get("Clay", avg_games),
            "surface_avg_Hard": surface_avg.get("Hard", avg_games),
            "surface_avg_Grass": surface_avg.get("Grass", avg_games),
            "avg_aces": avg_aces,
            "avg_bpf": avg_bpf,
            "tiebreak_rate": tiebreak_rate,
            "n_matches": len(p_df),
        }

    return stats


def build_h2h_stats(df_hist):
    """
    Build head-to-head average games and win rates between player pairs.
    Returns dict of (p1, p2) -> {avg_games, n_matches, win_rate_p1}
    """
    df = df_hist.copy()
    df["Total_Games"] = calculate_total_games(df)
    df = df.sort_values("Date")

    h2h = {}

    pairs = df.apply(lambda r: tuple(sorted([r["Winner"], r["Loser"]])), axis=1)
    df["pair"] = pairs

    for pair, group in df.groupby("pair"):
        games = group["Total_Games"].dropna()
        avg_g = float(games.mean()) if len(games) > 0 else np.nan
        p1, p2 = pair
        p1_wins = (group["Winner"] == p1).sum()
        p1_wr = p1_wins / len(group)
        h2h[pair] = {
            "avg_games": avg_g,
            "n_h2h": len(group),
            "p1_win_rate": p1_wr,
        }

    return h2h


def engineer_features(row, player_stats, h2h_stats, surface):
    """
    Given a (Winner, Loser) row and precomputed stats, produce a feature vector.
    This replaces the old 4-feature approach.
    """
    p1, p2 = row["Winner"], row["Loser"]

    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})

    # --- Surface key ---
    surf_key = f"surface_avg_{surface}" if surface in ["Clay", "Hard", "Grass"] else "surface_avg_Hard"
    surface_enc = {"Clay": 0, "Hard": 1, "Grass": 2}.get(surface, 1)

    # --- Ranking features ---
    r1 = s1.get("rank", 300)
    r2 = s2.get("rank", 300)
    rank_diff = abs(r1 - r2)
    rank_ratio = min(r1, r2) / max(r1, r2) if max(r1, r2) > 0 else 1.0  # Close to 1 = evenly matched

    pts1 = s1.get("pts", 30)
    pts2 = s2.get("pts", 30)
    pts_diff = abs(pts1 - pts2)

    # --- Form features ---
    wr1 = s1.get("win_rate", 0.5)
    wr2 = s2.get("win_rate", 0.5)
    win_rate_diff = abs(wr1 - wr2)        # Small diff = more competitive match
    win_rate_sum = wr1 + wr2              # Both in form = competitive

    # --- Avg games per player (surface-specific) ---
    ag1 = s1.get(surf_key, s1.get("avg_games", 21))
    ag2 = s2.get(surf_key, s2.get("avg_games", 21))
    avg_games_sum = (ag1 + ag2) / 2       # Both tend toward long matches = over likely

    # --- Tiebreak rate (high = strong servers = fewer breaks = fewer games potentially) ---
    tb1 = s1.get("tiebreak_rate", 0)
    tb2 = s2.get("tiebreak_rate", 0)
    tb_sum = tb1 + tb2                    # Many tiebreaks = more sets go full

    # --- Serve stats ---
    aces1 = s1.get("avg_aces", np.nan)
    aces2 = s2.get("avg_aces", np.nan)
    # If missing, use a neutral value
    aces_sum = (aces1 if not np.isnan(aces1 if aces1 is not None else np.nan) else 5) + \
               (aces2 if not np.isnan(aces2 if aces2 is not None else np.nan) else 5)

    bpf1 = s1.get("avg_bpf", np.nan)
    bpf2 = s2.get("avg_bpf", np.nan)
    bpf_sum = (bpf1 if bpf1 and not np.isnan(bpf1) else 4) + \
              (bpf2 if bpf2 and not np.isnan(bpf2) else 4)

    # --- H2H features ---
    pair_key = tuple(sorted([p1, p2]))
    h2h = h2h_stats.get(pair_key, {})
    h2h_avg_games = h2h.get("avg_games", np.nan)
    h2h_n = h2h.get("n_h2h", 0)
    h2h_wr = abs(h2h.get("p1_win_rate", 0.5) - 0.5)  # 0 = perfectly balanced H2H

    # Fill missing H2H games with average of both player averages
    h2h_avg_games_filled = h2h_avg_games if h2h_avg_games and not np.isnan(h2h_avg_games) else avg_games_sum

    # --- Surface factor ---
    # Clay matches tend to have more games; grass fewer
    surface_bonus = {"Clay": 1.5, "Hard": 0.0, "Grass": -1.5}.get(surface, 0.0)

    return {
        "rank_diff": rank_diff,
        "rank_ratio": rank_ratio,          # KEY: close to 1 = evenly matched
        "pts_diff": pts_diff,
        "win_rate_diff": win_rate_diff,    # KEY: low = close match
        "win_rate_sum": win_rate_sum,
        "avg_games_sum": avg_games_sum,    # KEY: both tend to long matches
        "tb_sum": tb_sum,
        "aces_sum": aces_sum,
        "bpf_sum": bpf_sum,
        "h2h_avg_games": h2h_avg_games_filled,  # KEY: history between these two
        "h2h_n": min(h2h_n, 10),          # Cap at 10
        "h2h_balance": h2h_wr,            # KEY: close H2H = more games
        "surface_enc": surface_enc,
        "surface_bonus": surface_bonus,    # KEY: clay/grass effect
        "r1": r1,
        "r2": r2,
        "pts1": pts1,
        "pts2": pts2,
    }


FEATURE_COLS = [
    "rank_diff", "rank_ratio", "pts_diff",
    "win_rate_diff", "win_rate_sum",
    "avg_games_sum", "tb_sum", "aces_sum", "bpf_sum",
    "h2h_avg_games", "h2h_n", "h2h_balance",
    "surface_enc", "surface_bonus",
    "r1", "r2", "pts1", "pts2",
]


def build_features_for_hist(df_hist):
    """
    For each row in historical data, engineer features using ONLY data available
    BEFORE that match (to avoid leakage). Uses an expanding window approach.
    """
    df = df_hist.copy().sort_values("Date").reset_index(drop=True)
    df["Total_Games"] = calculate_total_games(df)

    feature_rows = []
    global_avg_games = df["Total_Games"].dropna().mean()

    for i, row in df.iterrows():
        # Only use data before this match to build stats (no leakage)
        past = df.iloc[:i]

        if len(past) < 20:
            # Skip rows with very little history — too noisy
            feature_rows.append(None)
            continue

        p_stats = build_player_stats(past)
        h2h = build_h2h_stats(past)

        surface = row.get("Surface", "Hard")
        feats = engineer_features(row, p_stats, h2h, surface)
        feats["Total_Games"] = row["Total_Games"]
        feats["three_sets"] = 1 if row.get("Wsets", 0) >= 2 else 0
        feature_rows.append(feats)

    result = pd.DataFrame([r for r in feature_rows if r is not None])
    return result


# ============================================================
# 3. CARREGAR HISTÓRICO
# ============================================================

@st.cache_data(show_spinner=False)
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
# 4. API — JOGOS DE HOJE E AMANHÃ
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
            if response.status_code != 200:
                st.error(f"API retornou status {response.status_code}")
                return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])
            if not response.text:
                st.error("API retornou resposta vazia")
                return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                st.error(f"Erro ao decodificar JSON: {e}")
                return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])

        if data.get("success") != 1:
            st.error("API retornou erro")
            return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])

        matches = data.get("result", [])
        if not matches:
            st.info(f"Nenhum jogo encontrado para {today} e {tomorrow}.")
            return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])

        df_api = pd.DataFrame(matches)
        required_cols = ["event_date", "event_first_player", "event_second_player"]
        missing_cols = [col for col in required_cols if col not in df_api.columns]
        if missing_cols:
            st.error(f"Colunas em falta na resposta da API: {missing_cols}")
            return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])

        df_api["Date"] = pd.to_datetime(df_api["event_date"])
        df_api["Winner"] = df_api["event_first_player"]
        df_api["Loser"] = df_api["event_second_player"]

        if "tournament_name" in df_api.columns:
            df_api["Surface"] = df_api["tournament_name"].apply(extract_surface_from_tournament)
        else:
            df_api["Surface"] = "Hard"

        if "event_status" in df_api.columns:
            df_api = df_api[df_api["event_status"] == ""]

        result_df = df_api[["Date", "Winner", "Loser", "Surface"]].copy()
        result_df = result_df.drop_duplicates().dropna(subset=["Winner", "Loser"])

        if len(result_df) > 0:
            st.success(f"✅ Encontrados {len(result_df)} jogos para previsão")

        return result_df

    except Exception as e:
        st.error(f"Erro ao buscar jogos: {e}")
        return pd.DataFrame(columns=["Date", "Winner", "Loser", "Surface"])


def get_today_and_tomorrow_matches(df_matches):
    if df_matches.empty:
        return df_matches
    dfm = df_matches.copy()
    dfm["Date"] = pd.to_datetime(dfm["Date"]).dt.normalize()
    today = pd.Timestamp.now().normalize()
    tomorrow = today + pd.Timedelta(days=1)
    return dfm[(dfm["Date"] == today) | (dfm["Date"] == tomorrow)]


# ============================================================
# 5. TREINAR MODELOS COM FEATURES MELHORADAS
# ============================================================

@st.cache_data(show_spinner=True)
def build_training_features(_df_hist):
    """Build full feature matrix from historical data (cached)."""
    with st.spinner("A engenheirar features avançadas (isto pode demorar um momento)..."):
        feat_df = build_features_for_hist(_df_hist)
    return feat_df


def make_ensemble():
    """Return an ensemble of 3 models for better accuracy."""
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42))
    ])
    ensemble = VotingClassifier(
        estimators=[("gb", gb), ("rf", rf), ("lr", lr)],
        voting="soft",
        weights=[2, 1, 1]   # GBM gets more weight
    )
    return ensemble


def train_three_sets_model(feat_df):
    dfm = feat_df.dropna(subset=FEATURE_COLS + ["three_sets"]).copy()

    if len(dfm) < 50:
        st.warning(f"Apenas {len(dfm)} jogos com features completas para 3+ Sets. Mínimo recomendado: 50")
        return None

    X = dfm[FEATURE_COLS]
    y = dfm["three_sets"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = make_ensemble()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    test_score = model.score(X_test, y_test)

    st.sidebar.success(f"✅ Modelo 3+ Sets treinado com {len(dfm)} jogos")
    st.sidebar.info(f"📊 CV: {cv_scores.mean():.1%} ±{cv_scores.std():.1%} | Teste: {test_score:.1%}")

    return model


def train_over_games_model(feat_df, df_hist, threshold=22):
    dfm = feat_df.dropna(subset=FEATURE_COLS + ["Total_Games"]).copy()
    dfm["over_threshold"] = (dfm["Total_Games"] > threshold).astype(int)
    dfm = dfm.dropna(subset=["over_threshold"])

    avg_games = dfm["Total_Games"].mean()
    over_percentage = dfm["over_threshold"].mean() * 100

    if len(dfm) < 50:
        st.warning(f"Apenas {len(dfm)} jogos com features completas. Mínimo recomendado: 50")
        return None, avg_games, over_percentage

    X = dfm[FEATURE_COLS]
    y = dfm["over_threshold"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = make_ensemble()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    test_score = model.score(X_test, y_test)

    st.sidebar.success(f"✅ Modelo Over {threshold} Games treinado com {len(dfm)} jogos")
    st.sidebar.info(f"📊 CV: {cv_scores.mean():.1%} ±{cv_scores.std():.1%} | Teste: {test_score:.1%}")
    st.sidebar.info(f"📊 Média de games: {avg_games:.1f} | Over {threshold}: {over_percentage:.1f}%")

    return model, avg_games, over_percentage


# ============================================================
# 6. PREVISÕES PARA JOGOS FUTUROS
# ============================================================

def predict_for_upcoming(upcoming_df, hist_df, model_3sets, model_over, threshold=22):
    df_up = upcoming_df.copy()

    # Build player stats from FULL historical data for prediction
    player_stats = build_player_stats(hist_df)
    h2h_stats = build_h2h_stats(hist_df)

    feature_list = []
    for _, row in df_up.iterrows():
        surface = row.get("Surface", "Hard")
        feats = engineer_features(row, player_stats, h2h_stats, surface)
        feature_list.append(feats)

    feat_df = pd.DataFrame(feature_list)

    # Fill any remaining NaNs with neutral values
    feat_df = feat_df.fillna(feat_df.median(numeric_only=True))

    X = feat_df[FEATURE_COLS]

    if model_3sets is not None:
        df_up["prob_3_sets"] = model_3sets.predict_proba(X)[:, 1]
    else:
        df_up["prob_3_sets"] = 0.33

    if model_over is not None:
        df_up[f"prob_over_{threshold}_games"] = model_over.predict_proba(X)[:, 1]
    else:
        df_up[f"prob_over_{threshold}_games"] = 0.5

    # Add interpretable context
    df_up["rank_diff"] = feat_df["rank_diff"].values
    df_up["rank_ratio"] = feat_df["rank_ratio"].values
    df_up["avg_games_sum"] = feat_df["avg_games_sum"].values
    df_up["h2h_n"] = feat_df["h2h_n"].values

    # Combined score (evenly weighted now — both signals matter equally)
    df_up["prob_competitive_match"] = (
        df_up["prob_3_sets"] * 0.5 +
        df_up[f"prob_over_{threshold}_games"] * 0.5
    )

    return df_up.sort_values("prob_competitive_match", ascending=False)


# ============================================================
# 7. EXPORT TO EXCEL
# ============================================================

def export_to_excel(predictions_df, threshold_games):
    export_df = predictions_df.copy()

    export_df = export_df.rename(columns={
        "Date": "Data",
        "Winner": "Vencedor",
        "Loser": "Derrotado",
        "Surface": "Superfície",
        "prob_3_sets": "Probabilidade_3_Sets",
        f"prob_over_{threshold_games}_games": f"Probabilidade_Over_{threshold_games}_Games",
        "prob_competitive_match": "Probabilidade_Jogo_Competitivo",
        "rank_diff": "Diferença_Ranking",
        "rank_ratio": "Proximidade_Ranking",
        "avg_games_sum": "Media_Games_Historica",
        "h2h_n": "Jogos_H2H",
    })

    export_df["Data"] = pd.to_datetime(export_df["Data"]).dt.strftime("%Y-%m-%d")

    base_cols = [
        "Data", "Vencedor", "Derrotado", "Superfície",
        "Probabilidade_3_Sets", f"Probabilidade_Over_{threshold_games}_Games",
        "Probabilidade_Jogo_Competitivo",
        "Diferença_Ranking", "Proximidade_Ranking", "Media_Games_Historica", "Jogos_H2H",
    ]
    existing_cols = [c for c in base_cols if c in export_df.columns]
    export_df = export_df[existing_cols]

    for col in ["Probabilidade_3_Sets", f"Probabilidade_Over_{threshold_games}_Games", "Probabilidade_Jogo_Competitivo"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(lambda x: f"{x:.1%}")

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='Previsões', index=False)

        summary_df = pd.DataFrame({
            "Métrica": [
                "Data de Geração",
                "Total de Jogos Analisados",
                "Média Probabilidade 3+ Sets",
                f"Média Probabilidade Over {threshold_games} Games",
                "Média Probabilidade Jogo Competitivo",
                "Jogos com >60% Competitivo",
                "Jogos com >70% Competitivo",
                "Jogos com >80% Competitivo"
            ],
            "Valor": [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(predictions_df),
                f"{predictions_df['prob_3_sets'].mean():.1%}",
                f"{predictions_df[f'prob_over_{threshold_games}_games'].mean():.1%}",
                f"{predictions_df['prob_competitive_match'].mean():.1%}",
                len(predictions_df[predictions_df['prob_competitive_match'] > 0.6]),
                len(predictions_df[predictions_df['prob_competitive_match'] > 0.7]),
                len(predictions_df[predictions_df['prob_competitive_match'] > 0.8])
            ]
        })
        summary_df.to_excel(writer, sheet_name='Resumo', index=False)
        export_df.head(10).to_excel(writer, sheet_name='TOP_10_Jogos', index=False)

        for sheet in writer.sheets.values():
            for column in sheet.columns:
                max_length = max((len(str(cell.value)) for cell in column if cell.value), default=10)
                sheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)

    output.seek(0)
    return output


# ============================================================
# 8. SIDEBAR — CARREGAR HISTÓRICO
# ============================================================

st.sidebar.header("📂 Histórico de jogos")

uploaded_file = st.sidebar.file_uploader("Escolhe um ficheiro Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df, source_name = load_custom_excel(uploaded_file)
else:
    df, source_name = fetch_challenger_github_data()

if df is None or df.empty:
    st.error("Não foi possível carregar nenhum histórico.")
    st.stop()

st.sidebar.info(f"Fonte atual: {source_name}")
st.sidebar.write(f"Total de jogos: {len(df)}")

st.sidebar.header("⚙️ Configurações")
threshold_games = st.sidebar.slider(
    "Threshold para total de games",
    min_value=15, max_value=30, value=22, step=1,
    help="Número mínimo de games para considerar 'Over'"
)

# ============================================================
# 9. UI PRINCIPAL
# ============================================================

st.title("🎾 CHALLENGER — Predição de jogos competitivos (v2 — Features Avançadas)")
st.caption(f"Fonte de dados: {source_name} | Total de jogos históricos: {len(df)}")

# Build training features with leakage-free expanding window
with st.spinner("A calcular features avançadas do histórico (sem data leakage)..."):
    feat_df = build_training_features(df)

st.sidebar.header("📈 Features utilizadas no modelo")
st.sidebar.markdown("""
**Novas features (vs versão anterior):**
- ✅ Rank ratio (proximidade entre jogadores)
- ✅ Média de games por jogador (surface-specific)
- ✅ Win rate recente (últimos 20 jogos)
- ✅ H2H avg games entre estes dois jogadores
- ✅ H2H balanço (quem domina)
- ✅ Tiebreak rate (serve dominance)
- ✅ Aces + Break Points (se disponível)
- ✅ Surface encoding (Clay/Hard/Grass)
- ✅ Ensemble de 3 modelos (GBM + RF + LR)
- ✅ Cross-validation para métricas reais
""")

# Train models
with st.spinner("A treinar modelos (ensemble de 3 algoritmos)..."):
    model_3sets = train_three_sets_model(feat_df)
    model_over, avg_games, over_percentage = train_over_games_model(feat_df, df, threshold_games)

if model_3sets is None and model_over is None:
    st.error("Não foi possível treinar os modelos. Verifica se o histórico tem dados suficientes.")
    st.stop()

st.markdown("---")
st.header("📅 Jogos de hoje e amanhã")

# Fetch matches from API
api_matches = fetch_matches_from_api()

if api_matches.empty:
    st.warning("⚠️ Não foi possível obter jogos da API.")

    st.subheader("📤 Carrega um ficheiro com jogos para prever")
    st.markdown("""
    O ficheiro Excel deve conter as colunas:
    - **Date** (data do jogo)
    - **Winner** (nome do jogador favorito)
    - **Loser** (nome do adversário)
    - **Surface** (opcional: Clay, Grass, Hard)
    """)

    manual_file = st.file_uploader("Escolhe um ficheiro Excel (.xlsx)", type=["xlsx"], key="manual_matches")

    if manual_file is not None:
        try:
            manual_df = pd.read_excel(manual_file)
            required_cols = ["Date", "Winner", "Loser"]
            missing = [col for col in required_cols if col not in manual_df.columns]
            if missing:
                st.error(f"Colunas em falta: {missing}")
                st.stop()
            manual_df["Date"] = pd.to_datetime(manual_df["Date"])
            if "Surface" not in manual_df.columns:
                manual_df["Surface"] = "Hard"
            upcoming = manual_df
            st.success(f"✅ Carregados {len(upcoming)} jogos para previsão")
        except Exception as e:
            st.error(f"Erro ao carregar ficheiro: {e}")
            st.stop()
    else:
        st.info("👆 Carrega um ficheiro Excel para fazer previsões ou verifica a ligação à API.")
        st.stop()
else:
    upcoming = get_today_and_tomorrow_matches(api_matches)
    if upcoming.empty:
        st.info("📭 Nenhum jogo encontrado para hoje ou amanhã.")
        st.stop()

# Make predictions
with st.spinner("A calcular previsões com features avançadas..."):
    preds = predict_for_upcoming(upcoming, df, model_3sets, model_over, threshold_games)

# Show results
st.subheader("📋 Jogos ordenados por probabilidade de ser competitivo")

display_df = preds[[
    "Date", "Winner", "Loser", "Surface",
    "rank_diff", "h2h_n",
    "prob_3_sets", f"prob_over_{threshold_games}_games", "prob_competitive_match"
]].copy()

display_df.columns = [
    "Date", "Winner", "Loser", "Surface",
    "Δ Ranking", "H2H",
    "3+ Sets", f"Over {threshold_games} Games", "Competitivo"
]

st.dataframe(
    display_df.style.format({
        "3+ Sets": "{:.1%}",
        f"Over {threshold_games} Games": "{:.1%}",
        "Competitivo": "{:.1%}",
        "Δ Ranking": "{:.0f}",
        "H2H": "{:.0f}",
    }).background_gradient(subset=["Competitivo"], cmap="RdYlGn"),
    use_container_width=True
)

# Export
st.markdown("---")
st.subheader("📊 Exportar Resultados")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    export_clicked = st.button("📥 Exportar para Excel", type="primary", use_container_width=True)

if export_clicked:
    with st.spinner("A gerar ficheiro Excel..."):
        try:
            excel_file = export_to_excel(preds, threshold_games)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"previsoes_tenis_{timestamp}.xlsx"
            st.success(f"✅ Ficheiro Excel gerado! ({len(preds)} jogos)")
            st.download_button(
                label="💾 Descarregar Ficheiro Excel",
                data=excel_file,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_excel"
            )
        except Exception as e:
            st.error(f"❌ Erro ao gerar ficheiro Excel: {e}")

# TOP matches
st.markdown("---")
st.subheader("🔥 TOP 5 jogos mais prováveis de serem competitivos")
top_n = min(5, len(preds))

for idx, (_, row) in enumerate(preds.head(top_n).iterrows(), 1):
    with st.container():
        rank_diff = row.get("rank_diff", "?")
        h2h_n = int(row.get("h2h_n", 0))
        h2h_label = f"{h2h_n} jogos" if h2h_n > 0 else "Sem H2H"

        st.markdown(f"""
        **{idx}. {row['Winner']} vs {row['Loser']}**  
        📅 {row['Date'].date()} | 🎾 {row['Surface']} | ΔRanking: {rank_diff:.0f} | H2H: {h2h_label}

        | Probabilidade | Valor |
        |--------------|-------|
        | 🎯 3+ Sets | **{row['prob_3_sets']:.1%}** |
        | 📊 Over {threshold_games} Games | **{row[f'prob_over_{threshold_games}_games']:.1%}** |
        | ⭐ Competitivo | **{row['prob_competitive_match']:.1%}** |
        """)
        st.progress(row['prob_competitive_match'], text="Probabilidade de jogo competitivo")

# Stats
st.markdown("---")
st.subheader("📊 Estatísticas")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Jogos no histórico", len(df))
with col2:
    st.metric("Jogos com features", len(feat_df))
with col3:
    st.metric("Média de games (histórico)", f"{avg_games:.1f}")
with col4:
    st.metric(f"Over {threshold_games} (histórico)", f"{over_percentage:.1f}%")

# Recommendations
st.markdown("---")
st.subheader("💡 Recomendações")
high_value_matches = preds[preds["prob_competitive_match"] > 0.6]
very_high_value_matches = preds[preds["prob_competitive_match"] > 0.7]

if len(high_value_matches) > 0:
    st.success(f"🎯 Encontrados {len(high_value_matches)} jogos com alta probabilidade de serem competitivos!")
    if len(very_high_value_matches) > 0:
        st.info(f"🔥 Destes, {len(very_high_value_matches)} jogos têm probabilidade >70% — são os mais promissores!")
    st.markdown(f"""
    **Estes jogos têm maior probabilidade de:**
    - ✅ Irem a 3 sets
    - ✅ Terem mais de {threshold_games} games no total
    - ✅ Serem emocionantes e equilibrados
    """)
else:
    st.info("Nenhum jogo com probabilidade > 60% de ser competitivo no momento.")
