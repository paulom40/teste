import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from io import BytesIO
from datetime import datetime
import re
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Ténis — 3+ Sets & Over Predictor",
    page_icon="🎾",
    layout="wide"
)

# ============================================================
# 0. NORMALIZAÇÃO / PARSING BÁSICO
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
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "Score" in df.columns:
        df = parse_score(df)

    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
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

# ============================================================
# 1. ELO + K DINÂMICO + H2H
# ============================================================

def build_elo_ratings(df_hist):
    df = df_hist.sort_values("Date").copy()

    elo = {}
    matches_count = {}
    default_elo = 1500

    surface_map = {
        "Clay": "elo_clay",
        "Hard": "elo_hard",
        "Grass": "elo_grass"
    }

    def get_k(player):
        n = matches_count.get(player, 0)
        if n < 20:
            return 40
        elif n < 50:
            return 32
        elif n < 100:
            return 24
        else:
            return 16

    for _, row in df.iterrows():
        w = row["Winner"]
        l = row["Loser"]
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

        k_w = get_k(w)
        k_l = get_k(l)

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


def build_h2h_stats(df_hist):
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

# ============================================================
# 2. PLAYER STATS + FEATURES + CORREÇÃO 3 SETS
# ============================================================

def build_player_stats(df_hist):
    df = df_hist.copy()
    df = df.sort_values("Date")
    df["Total_Games"] = calculate_total_games(df)

    elo_ratings = build_elo_ratings(df_hist)

    stats = {}
    all_players = set(df["Winner"].dropna().unique()) | set(df["Loser"].dropna().unique())

    for player in all_players:
        w_mask = df["Winner"] == player
        l_mask = df["Loser"] == player
        all_mask = w_mask | l_mask

        p_df = df[all_mask].copy()

        last_w = df[w_mask]["WRank"].dropna()
        last_l = df[l_mask]["LRank"].dropna()
        rank_vals = pd.concat([last_w, last_l])
        rank = float(rank_vals.iloc[-1]) if len(rank_vals) > 0 else 300.0

        last_w_pts = df[w_mask]["WPts"].dropna()
        last_l_pts = df[l_mask]["LPts"].dropna()
        pts_vals = pd.concat([last_w_pts, last_l_pts])
        pts = float(pts_vals.iloc[-1]) if len(pts_vals) > 0 else 30.0

        recent = p_df.tail(20)
        win_rate = (recent["Winner"] == player).mean()

        recent_games = recent["Total_Games"].dropna()
        avg_games = float(recent_games.mean()) if len(recent_games) > 0 else 21.0

        surface_avg = {}
        for surf in ["Clay", "Hard", "Grass"]:
            s_df = p_df[p_df["Surface"] == surf]["Total_Games"].dropna()
            surface_avg[surf] = float(s_df.mean()) if len(s_df) >= 3 else avg_games

        stats[player] = {
            "rank": rank,
            "pts": pts,
            "win_rate": win_rate,
            "avg_games": avg_games,
            "surface_avg_Clay": surface_avg["Clay"],
            "surface_avg_Hard": surface_avg["Hard"],
            "surface_avg_Grass": surface_avg["Grass"],
            "n_matches": len(p_df)
        }

        stats[player]["elo"] = elo_ratings[player]["elo"]
        stats[player]["elo_clay"] = elo_ratings[player]["elo_clay"]
        stats[player]["elo_hard"] = elo_ratings[player]["elo_hard"]
        stats[player]["elo_grass"] = elo_ratings[player]["elo_grass"]

    return stats


def engineer_features(row, player_stats, h2h_stats, surface):
    p1, p2 = row["Winner"], row["Loser"]

    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})

    r1 = s1.get("rank", 300)
    r2 = s2.get("rank", 300)
    rank_diff = abs(r1 - r2)
    rank_ratio = min(r1, r2) / max(r1, r2) if max(r1, r2) > 0 else 1.0

    pts1 = s1.get("pts", 30)
    pts2 = s2.get("pts", 30)
    pts_diff = abs(pts1 - pts2)

    wr1 = s1.get("win_rate", 0.5)
    wr2 = s2.get("win_rate", 0.5)
    win_rate_diff = abs(wr1 - wr2)
    win_rate_sum = wr1 + wr2

    surf_key = f"surface_avg_{surface}"
    ag1 = s1.get(surf_key, s1.get("avg_games", 21))
    ag2 = s2.get(surf_key, s2.get("avg_games", 21))
    avg_games_sum = (ag1 + ag2) / 2

    elo1 = s1.get("elo", 1500)
    elo2 = s2.get("elo", 1500)
    elo_diff = abs(elo1 - elo2)
    elo_ratio = min(elo1, elo2) / max(elo1, elo2) if max(elo1, elo2) > 0 else 1.0

    surf_elo_key = f"elo_{surface.lower()}"
    elo_surf1 = s1.get(surf_elo_key, elo1)
    elo_surf2 = s2.get(surf_elo_key, elo2)
    elo_surf_diff = abs(elo_surf1 - elo_surf2)

    pair_key = tuple(sorted([p1, p2]))
    h2h = h2h_stats.get(pair_key, {})
    h2h_avg_games = h2h.get("avg_games", avg_games_sum)
    h2h_n = min(h2h.get("n_h2h", 0), 10)
    h2h_balance = abs(h2h.get("p1_win_rate", 0.5) - 0.5)

    surface_enc = {"Clay": 0, "Hard": 1, "Grass": 2}.get(surface, 1)
    surface_bonus = {"Clay": 1.5, "Hard": 0.0, "Grass": -1.5}.get(surface, 0.0)

    return {
        "rank_diff": rank_diff,
        "rank_ratio": rank_ratio,
        "pts_diff": pts_diff,
        "win_rate_diff": win_rate_diff,
        "win_rate_sum": win_rate_sum,
        "avg_games_sum": avg_games_sum,
        "h2h_avg_games": h2h_avg_games,
        "h2h_n": h2h_n,
        "h2h_balance": h2h_balance,
        "surface_enc": surface_enc,
        "surface_bonus": surface_bonus,
        "r1": r1,
        "r2": r2,
        "pts1": pts1,
        "pts2": pts2,
        "elo_diff": elo_diff,
        "elo_ratio": elo_ratio,
        "elo_surf_diff": elo_surf_diff
    }


FEATURE_COLS = [
    "rank_diff", "rank_ratio", "pts_diff",
    "win_rate_diff", "win_rate_sum",
    "avg_games_sum",
    "h2h_avg_games", "h2h_n", "h2h_balance",
    "surface_enc", "surface_bonus",
    "r1", "r2", "pts1", "pts2",
    "elo_diff", "elo_ratio", "elo_surf_diff"
]

def build_features_for_hist(df_hist):
    df = df_hist.copy().sort_values("Date").reset_index(drop=True)
    df["Total_Games"] = calculate_total_games(df)

    feature_rows = []

    for i, row in df.iterrows():
        past = df.iloc[:i]
        if len(past) < 20:
            feature_rows.append(None)
            continue

        p_stats = build_player_stats(past)
        h2h = build_h2h_stats(past)
        surface = row.get("Surface", "Hard")
        feats = engineer_features(row, p_stats, h2h, surface)
        feats["Total_Games"] = row["Total_Games"]

        sets_played = 0
        for j in range(1, 6):
            w = row.get(f"W{j}", np.nan)
            l = row.get(f"L{j}", np.nan)
            if not (pd.isna(w) or pd.isna(l)):
                sets_played += 1
        feats["three_sets"] = 1 if sets_played >= 3 else 0

        feature_rows.append(feats)

    result = pd.DataFrame([r for r in feature_rows if r is not None])
    return result

# ============================================================
# 3. MODELOS + ENSEMBLE OTIMIZADO + CONFIANÇA
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


def train_three_sets_model(feat_df):
    dfm = feat_df.dropna(subset=FEATURE_COLS + ["three_sets"]).copy()

    if len(dfm) < 50:
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
    st.sidebar.info(f"CV: {cv_scores.mean():.1%} ±{cv_scores.std():.1%} | Teste: {test_score:.1%}")

    return model


def train_over_games_model(feat_df, threshold=22):
    dfm = feat_df.dropna(subset=FEATURE_COLS + ["Total_Games"]).copy()
    dfm["over_threshold"] = (dfm["Total_Games"] > threshold).astype(int)
    dfm = dfm.dropna(subset=["over_threshold"])

    if len(dfm) < 50:
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
    st.sidebar.info(f"CV: {cv_scores.mean():.1%} ±{cv_scores.std():.1%} | Teste: {test_score:.1%}")

    return model


def predict_for_upcoming(upcoming_df, hist_df, model_3sets, model_over, threshold=22):
    df_up = upcoming_df.copy()

    player_stats = build_player_stats(hist_df)
    h2h_stats = build_h2h_stats(hist_df)

    feature_list = []
    meta_info = []

    for _, row in df_up.iterrows():
        p1, p2 = row["Winner"], row["Loser"]
        surface = row.get("Surface", "Hard")

        s1 = player_stats.get(p1, {})
        s2 = player_stats.get(p2, {})

        n1 = s1.get("n_matches", 0)
        n2 = s2.get("n_matches", 0)

        feats = engineer_features(row, player_stats, h2h_stats, surface)
        feature_list.append(feats)

        pair_key = tuple(sorted([p1, p2]))
        h2h = h2h_stats.get(pair_key, {})
        h2h_n = h2h.get("n_h2h", 0)

        meta_info.append({
            "n1": n1,
            "n2": n2,
            "h2h_n": h2h_n
        })

    feat_df = pd.DataFrame(feature_list)
    meta_df = pd.DataFrame(meta_info)

    feat_df = feat_df.fillna(feat_df.median(numeric_only=True))
    X = feat_df[FEATURE_COLS]

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

    data_conf = (
        (meta_df["n1"].clip(0, 50) / 50) * 0.4 +
        (meta_df["n2"].clip(0, 50) / 50) * 0.4 +
        (meta_df["h2h_n"].clip(0, 5) / 5) * 0.2
    )

    prob_conf = (np.abs(df_up["prob_competitive_match"] - 0.5) * 2).clip(0, 1)

    df_up["confiança_modelo"] = (0.6 * data_conf + 0.4 * prob_conf)

    return df_up.sort_values("prob_competitive_match", ascending=False)


def filtrar_por_confianca(df, limiar=0.6):
    return df[df["confiança_modelo"] >= limiar].copy()


def selecionar_melhores_jogos(df, top_n=10, peso_prob=0.6, peso_conf=0.4):
    df = df.copy()
    df["score_final"] = (
        peso_prob * df["prob_competitive_match"] +
        peso_conf * df["confiança_modelo"]
    )
    return df.sort_values("score_final", ascending=False).head(top_n)

# ============================================================
# 4. RELATÓRIO EXCEL
# ============================================================

def export_to_excel(predictions_df, threshold_games, top_jogos_df):
    export_df = predictions_df.copy()

    export_df = export_df.rename(columns={
        "Date": "Data",
        "Winner": "Vencedor",
        "Loser": "Derrotado",
        "Surface": "Superfície",
        "prob_3_sets": "Probabilidade_3_Sets",
        f"prob_over_{threshold_games}_games": f"Probabilidade_Over_{threshold_games}_Games",
        "prob_competitive_match": "Probabilidade_Jogo_Competitivo",
        "confiança_modelo": "Confianca_Modelo",
        "score_final": "Score_Final",
        "elo_diff": "Diferenca_Elo",
        "rank_diff": "Diferenca_Ranking",
        "rank_ratio": "Proximidade_Ranking",
        "avg_games_sum": "Media_Games_Historica",
        "h2h_n": "Jogos_H2H",
    })

    export_df["Data"] = pd.to_datetime(export_df["Data"]).dt.strftime("%Y-%m-%d")

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

        export_df.to_excel(writer, sheet_name='Previsoes', index=False)

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
                len(predictions_df[predictions_df['score_final'] >= 0.7]),
            ]
        })
        summary_df.to_excel(writer, sheet_name='Resumo', index=False)

        top_jogos_df.to_excel(writer, sheet_name='Top_Jogos', index=False)

        stats_df = pd.DataFrame({
            "Superfície": predictions_df["Surface"].value_counts().index,
            "Jogos": predictions_df["Surface"].value_counts().values,
            "Média_Prob_Competitivo": predictions_df.groupby("Surface")["prob_competitive_match"].mean().values,
            "Média_Confiança": predictions_df.groupby("Surface")["confiança_modelo"].mean().values,
            "Média_Elo_Diff": predictions_df.groupby("Surface")["elo_diff"].mean().values,
        })
        stats_df.to_excel(writer, sheet_name='Estatisticas_Modelo', index=False)

        for sheet in writer.sheets.values():
            for column in sheet.columns:
                max_length = max((len(str(cell.value)) for cell in column if cell.value), default=10)
                sheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)

    output.seek(0)
    return output

# ============================================================
# 5. STREAMLIT UI
# ============================================================

st.title("🎾 Ténis — Predição de Jogos Competitivos (Elo + Features Avançadas)")
st.sidebar.header("📂 Histórico de jogos")

hist_file = st.sidebar.file_uploader("Escolhe um ficheiro Excel de histórico (.xlsx)", type=["xlsx"])

if hist_file is None:
    st.warning("Carrega primeiro um ficheiro de histórico com jogos (Winner, Loser, Date, Surface, Score, etc.).")
    st.stop()

df_hist = pd.read_excel(hist_file)
df_hist = normalize_columns(df_hist)

st.sidebar.info(f"Total de jogos históricos: {len(df_hist)}")

threshold_games = st.sidebar.slider(
    "Threshold para total de games (Over)",
    min_value=15, max_value=30, value=22, step=1
)

st.sidebar.header("⚙️ Treino do modelo")

with st.spinner("A engenheirar features do histórico..."):
    feat_df = build_features_for_hist(df_hist)

with st.spinner("A treinar modelos..."):
    model_3sets = train_three_sets_model(feat_df)
    model_over = train_over_games_model(feat_df, threshold_games)

if model_3sets is None and model_over is None:
    st.error("Não foi possível treinar nenhum modelo. Verifica se o histórico tem dados suficientes.")
    st.stop()

st.markdown("---")
st.header("📅 Jogos para previsão")

st.write("Carrega um ficheiro Excel com jogos futuros (colunas: Date, Winner, Loser, Surface opcional).")

upcoming_file = st.file_uploader("Ficheiro de jogos futuros (.xlsx)", type=["xlsx"], key="upcoming")

if upcoming_file is not None:
    df_upcoming = pd.read_excel(upcoming_file)
    required_cols = ["Date", "Winner", "Loser"]
    missing = [c for c in required_cols if c not in df_upcoming.columns]
    if missing:
        st.error(f"Colunas em falta: {missing}")
        st.stop()

    df_upcoming["Date"] = pd.to_datetime(df_upcoming["Date"])
    if "Surface" not in df_upcoming.columns:
        df_upcoming["Surface"] = "Hard"

    with st.spinner("A gerar previsões..."):
        preds = predict_for_upcoming(df_upcoming, df_hist, model_3sets, model_over, threshold_games)

    st.subheader("📊 Todas as previsões")
    st.dataframe(preds)

    preds_filtradas = filtrar_por_confianca(preds, limiar=0.6)
    top_jogos = selecionar_melhores_jogos(preds_filtradas, top_n=10)

    st.subheader("🎯 Top jogos recomendados (confiança ≥ 0.6)")
    st.dataframe(top_jogos)

    excel_file = export_to_excel(preds, threshold_games, top_jogos)

    st.download_button(
        label="📥 Baixar Relatório Excel",
        data=excel_file,
        file_name="Relatorio_Tenis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
