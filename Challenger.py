import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from io import BytesIO
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

app_api = FastAPI()

FEATURE_COLS = [
    "elo_diff",
    "elo_surf_diff",
    "h2h_winrate",
    "h2h_avg_games",
    "spw_diff_30",
    "bpw_diff_30",
    "rpw_diff_30",
    "games_played_diff_30"
]

def normalize_columns(df):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df

def build_elo_ratings(df):
    elo = {}
    K = 32
    for _, row in df.iterrows():
        w, l = row["Winner"], row["Loser"]
        elo.setdefault(w, 1500)
        elo.setdefault(l, 1500)
        ew = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
        elo[w] += K * (1 - ew)
        elo[l] += K * (0 - (1 - ew))
    return elo

def build_h2h_stats(df):
    h2h = {}
    for _, row in df.iterrows():
        w, l = row["Winner"], row["Loser"]
        pair = tuple(sorted([w, l]))
        h2h.setdefault(pair, {"wins": 0, "losses": 0, "games": [], "n_h2h": 0})
        h2h[pair]["n_h2h"] += 1
        if row["Winner"] == pair[0]:
            h2h[pair]["wins"] += 1
        else:
            h2h[pair]["losses"] += 1
        if "Score" in row and isinstance(row["Score"], str):
            try:
                total = sum(int(x) for x in row["Score"].replace(" ", "").replace("-", " ").split())
                h2h[pair]["games"].append(total)
            except:
                pass
    return h2h

def build_recent_stats(df, window=30):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff = df["Date"].max() - timedelta(days=window)
    df = df[df["Date"] >= cutoff]
    stats = {}
    for _, row in df.iterrows():
        w, l = row["Winner"], row["Loser"]
        stats.setdefault(w, {"spw": [], "bpw": [], "rpw": [], "games": []})
        stats.setdefault(l, {"spw": [], "bpw": [], "rpw": [], "games": []})
        if "Score" in row and isinstance(row["Score"], str):
            try:
                total = sum(int(x) for x in row["Score"].replace(" ", "").replace("-", " ").split())
                stats[w]["games"].append(total)
                stats[l]["games"].append(total)
            except:
                pass
    final = {}
    for p, s in stats.items():
        final[p] = {
            "spw_30": np.mean(s["spw"]) if s["spw"] else np.nan,
            "bpw_30": np.mean(s["bpw"]) if s["bpw"] else np.nan,
            "rpw_30": np.mean(s["rpw"]) if s["rpw"] else np.nan,
            "games_played_30": len(s["games"])
        }
    return final
def engineer_features(row, recent_stats, elo, h2h):
    p1, p2 = row["Winner"], row["Loser"]
    e1 = elo.get(p1, 1500)
    e2 = elo.get(p2, 1500)
    elo_diff = e1 - e2
    elo_surf_diff = elo_diff
    pair = tuple(sorted([p1, p2]))
    h = h2h.get(pair, {"wins": 0, "losses": 0, "games": [], "n_h2h": 0})
    wins = h["wins"] if pair[0] == p1 else h["losses"]
    losses = h["losses"] if pair[0] == p1 else h["wins"]
    total = wins + losses
    h2h_winrate = wins / total if total > 0 else 0.5
    h2h_avg_games = np.mean(h["games"]) if h["games"] else np.nan
    s1 = recent_stats.get(p1, {})
    s2 = recent_stats.get(p2, {})
    spw_diff = (s1.get("spw_30", np.nan) - s2.get("spw_30", np.nan))
    bpw_diff = (s1.get("bpw_30", np.nan) - s2.get("bpw_30", np.nan))
    rpw_diff = (s1.get("rpw_30", np.nan) - s2.get("rpw_30", np.nan))
    games_diff = (s1.get("games_played_30", 0) - s2.get("games_played_30", 0))
    return {
        "elo_diff": elo_diff,
        "elo_surf_diff": elo_surf_diff,
        "h2h_winrate": h2h_winrate,
        "h2h_avg_games": h2h_avg_games,
        "spw_diff_30": spw_diff,
        "bpw_diff_30": bpw_diff,
        "rpw_diff_30": rpw_diff,
        "games_played_diff_30": games_diff
    }

def build_training_dataset(df, recent_stats, elo, h2h, threshold):
    rows = []
    for _, row in df.iterrows():
        feats = engineer_features(row, recent_stats, elo, h2h)
        feats["target_3sets"] = 1 if row.get("BestOf", 3) == 3 else 0
        if "Score" in row and isinstance(row["Score"], str):
            try:
                total = sum(int(x) for x in row["Score"].replace(" ", "").replace("-", " ").split())
                feats["target_over"] = 1 if total >= threshold else 0
            except:
                feats["target_over"] = 0
        else:
            feats["target_over"] = 0
        rows.append(feats)
    df_feat = pd.DataFrame(rows)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.fillna(df_feat.median(numeric_only=True))
    df_feat = df_feat.fillna(0)
    return df_feat

def train_three_sets_model(df):
    if df["target_3sets"].nunique() < 2:
        return None
    X = df[FEATURE_COLS]
    y = df["target_3sets"]
    m1 = GradientBoostingClassifier()
    m2 = RandomForestClassifier()
    model = VotingClassifier([("gb", m1), ("rf", m2)], voting="soft")
    model.fit(X, y)
    return model

def train_over_games_model(df, threshold):
    if df["target_over"].nunique() < 2:
        return None
    X = df[FEATURE_COLS]
    y = df["target_over"]
    m1 = GradientBoostingClassifier()
    m2 = RandomForestClassifier()
    model = VotingClassifier([("gb", m1), ("rf", m2)], voting="soft")
    model.fit(X, y)
    return model
def predict_for_upcoming(upcoming_df, df_hist, model_3sets, model_over, threshold=22):
    df_up = upcoming_df.copy()
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
        h2h_n = h2h_stats.get(pair, {}).get("n_h2h", 0)
        meta_info.append({"n1": n1, "n2": n2, "h2h_n": h2h_n})
    feat_df = pd.DataFrame(feature_list)
    meta_df = pd.DataFrame(meta_info)
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = np.nan
    feat_df = feat_df[FEATURE_COLS].astype(float)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    medianas = feat_df.median(numeric_only=True)
    medianas = medianas.fillna(0)
    feat_df = feat_df.fillna(medianas)
    feat_df = feat_df.fillna(0)
    X = feat_df
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
        (meta_df["n1"].clip(0, 30) / 30) * 0.4 +
        (meta_df["n2"].clip(0, 30) / 30) * 0.4 +
        (meta_df["h2h_n"].clip(0, 5) / 5) * 0.2
    )
    prob_conf = (np.abs(df_up["prob_competitive_match"] - 0.5) * 2).clip(0, 1)
    df_up["confiança_modelo"] = (0.6 * data_conf + 0.4 * prob_conf)
    df_up["elo_diff"] = feat_df["elo_diff"].values
    df_up["rank_diff_dummy"] = np.nan
    return df_up.sort_values("prob_competitive_match", ascending=False)
def export_to_excel(preds, threshold_games, top_jogos):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    preds.to_excel(writer, index=False, sheet_name="Todas as Previsões")
    top_jogos.to_excel(writer, index=False, sheet_name="Top Jogos")
    workbook = writer.book
    worksheet = writer.sheets["Top Jogos"]
    fmt = workbook.add_format({"num_format": "0.00"})
    for col in range(len(top_jogos.columns)):
        worksheet.set_column(col, col, 18, fmt)
    summary = pd.DataFrame({
        "Métrica": [
            "Total de Jogos",
            "Jogos com prob >= 0.7",
            "Média probabilidade competitiva",
            "Média confiança"
        ],
        "Valor": [
            len(preds),
            len(top_jogos[top_jogos["score_final"] >= 0.7]),
            preds["prob_competitive_match"].mean(),
            preds["confiança_modelo"].mean()
        ]
    })
    summary.to_excel(writer, index=False, sheet_name="Resumo")
    writer.close()
    output.seek(0)
    return output

def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = normalize_columns(df)
    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def prepare_upcoming(df):
    df = df.copy()
    df = normalize_columns(df)
    if "Winner" not in df or "Loser" not in df:
        return pd.DataFrame()
    return df[["Winner", "Loser"]]
def app_streamlit():
    st.title("🎾 Challenger Predictor — Versão TURBO + API")

    st.sidebar.header("Carregar Dados")
    hist_file = st.sidebar.file_uploader("Histórico (CSV)", type=["csv"])
    upcoming_file = st.sidebar.file_uploader("Jogos Futuros (CSV)", type=["csv"])

    threshold_games = st.sidebar.number_input(
        "Threshold para Over Games", min_value=10, max_value=40, value=22
    )

    if hist_file is not None:
        df_hist = load_csv(hist_file)
        st.subheader("Histórico Carregado")
        st.dataframe(df_hist.head())

        elo = build_elo_ratings(df_hist)
        h2h = build_h2h_stats(df_hist)
        recent = build_recent_stats(df_hist)

        st.sidebar.success("Histórico carregado com sucesso!")

        st.subheader("Treinar Modelos")
        df_train = build_training_dataset(df_hist, recent, elo, h2h, threshold_games)

        model_3 = train_three_sets_model(df_train)
        model_over = train_over_games_model(df_train, threshold_games)

        st.sidebar.success("Modelos treinados!")

        if upcoming_file is not None:
            df_up = load_csv(upcoming_file)
            df_up = prepare_upcoming(df_up)

            st.subheader("Jogos Futuros")
            st.dataframe(df_up)

            preds = predict_for_upcoming(df_up, df_hist, model_3, model_over, threshold_games)

            st.subheader("Previsões")
            st.dataframe(preds)

            preds["score_final"] = (
                0.5 * preds["prob_competitive_match"] +
                0.5 * preds["confiança_modelo"]
            )

            top = preds.sort_values("score_final", ascending=False).head(10)

            st.subheader("Top Jogos")
            st.dataframe(top)

            excel_file = export_to_excel(preds, threshold_games, top)

            st.download_button(
                label="📥 Baixar Excel",
                data=excel_file,
                file_name="previsoes_challenger.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
class MatchInput(BaseModel):
    Winner: str
    Loser: str

class PredictRequest(BaseModel):
    matches: list[MatchInput]
    threshold: int = 22

global_hist = None
global_model_3 = None
global_model_over = None
global_elo = None
global_h2h = None
global_recent = None

@app_api.get("/health")
def health():
    return {"status": "ok"}

@app_api.get("/info")
def info():
    return {
        "model_3sets": global_model_3 is not None,
        "model_over": global_model_over is not None,
        "historico_carregado": global_hist is not None
    }
@app_api.post("/predict")
def predict_api(req: PredictRequest):
    global global_hist, global_model_3, global_model_over
    global global_elo, global_h2h, global_recent

    if global_hist is None:
        return {"error": "Histórico não carregado no servidor."}

    if global_model_3 is None or global_model_over is None:
        return {"error": "Modelos ainda não foram treinados."}

    df_up = pd.DataFrame([m.dict() for m in req.matches])
    df_up = normalize_columns(df_up)

    preds = predict_for_upcoming(
        df_up,
        global_hist,
        global_model_3,
        global_model_over,
        req.threshold
    )

    preds["score_final"] = (
        0.5 * preds["prob_competitive_match"] +
        0.5 * preds["confiança_modelo"]
    )

    return preds.to_dict(orient="records")
class TrainRequest(BaseModel):
    csv_data: str
    threshold: int = 22

@app_api.post("/train")
def train_api(req: TrainRequest):
    import base64
    import io

    global global_hist, global_model_3, global_model_over
    global global_elo, global_h2h, global_recent

    decoded = base64.b64decode(req.csv_data)
    df = pd.read_csv(io.BytesIO(decoded))
    df = normalize_columns(df)

    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    global_hist = df
    global_elo = build_elo_ratings(df)
    global_h2h = build_h2h_stats(df)
    global_recent = build_recent_stats(df)

    df_train = build_training_dataset(df, global_recent, global_elo, global_h2h, req.threshold)

    global_model_3 = train_three_sets_model(df_train)
    global_model_over = train_over_games_model(df_train, req.threshold)

    return {
        "status": "Modelos treinados",
        "3sets_model": global_model_3 is not None,
        "over_model": global_model_over is not None,
        "historico_registos": len(df)
    }
@app_api.post("/validate_upcoming")
def validate_upcoming(req: PredictRequest):
    df_up = pd.DataFrame([m.dict() for m in req.matches])
    df_up = normalize_columns(df_up)

    if "Winner" not in df_up or "Loser" not in df_up:
        return {"error": "Formato inválido. Campos necessários: Winner, Loser"}

    return {
        "status": "ok",
        "total_matches": len(df_up),
        "sample": df_up.head(5).to_dict(orient="records")
    }
def initialize_api_models(csv_path: str = None, threshold: int = 22):
    global global_hist, global_model_3, global_model_over
    global global_elo, global_h2h, global_recent

    if csv_path is None:
        return False

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    global_hist = df
    global_elo = build_elo_ratings(df)
    global_h2h = build_h2h_stats(df)
    global_recent = build_recent_stats(df)

    df_train = build_training_dataset(df, global_recent, global_elo, global_h2h, threshold)

    global_model_3 = train_three_sets_model(df_train)
    global_model_over = train_over_games_model(df_train, threshold)

    return True
def run_all():
    import threading
    import uvicorn

    def start_api():
        uvicorn.run(app_api, host="0.0.0.0", port=8000)

    t = threading.Thread(target=start_api, daemon=True)
    t.start()

    app_streamlit()
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "api":
        import uvicorn
        uvicorn.run(app_api, host="0.0.0.0", port=8000)
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        run_all()
    else:
        app_streamlit()
