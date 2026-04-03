# TennisJD.py
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import io

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Tennis JD – Elo & ML", layout="wide")

DATA_FILE = "Challenger1_2026.xlsx"   # tem de estar no repo ou ser carregado
SHEET_MATCHES = "Sheet1"

RAPIDAPI_KEY = "bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a44"
RAPIDAPI_HOST = "sportscore1.p.rapidapi.com"
RAPIDAPI_URL = "https://sportscore1.p.rapidapi.com/events/search"

# -----------------------
# FUNÇÕES AUXILIARES
# -----------------------

def total_games_from_score(score_str):
    try:
        if pd.isna(score_str):
            return np.nan
        s = str(score_str).replace("RET", "").replace("W/O", "").strip()
        sets = s.split()
        total = 0
        for set_str in sets:
            if "(" in set_str:
                set_str = set_str.split("(")[0]
            if "-" in set_str:
                a, b = set_str.split("-")
                total += int(a) + int(b)
        return total
    except Exception:
        return np.nan

@st.cache_data
def load_data_from_file(path: str, sheet_name: str):
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    if "T Games" not in df.columns:
        df["T Games"] = df["score"].apply(total_games_from_score)
    return df

def load_data():
    try:
        df = load_data_from_file(DATA_FILE, SHEET_MATCHES)
        return df
    except FileNotFoundError:
        st.warning("Ficheiro não encontrado no repositório. Carrega o Excel manualmente.")
        uploaded = st.file_uploader("Carregar Challenger1_2026.xlsx", type=["xlsx"])
        if uploaded is None:
            st.stop()
        df = pd.read_excel(uploaded, sheet_name=SHEET_MATCHES)
        df.columns = [str(c).strip() for c in df.columns]
        if "tourney_date" in df.columns:
            df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
        if "T Games" not in df.columns:
            df["T Games"] = df["score"].apply(total_games_from_score)
        return df

def init_elo_dict():
    return {}

def update_elo(elo_dict, p1, p2, winner, K):
    R1 = elo_dict.get(p1, 1500.0)
    R2 = elo_dict.get(p2, 1500.0)
    E1 = 1 / (1 + 10 ** ((R2 - R1) / 400))
    E2 = 1 - E1
    S1 = 1.0 if winner == p1 else 0.0
    S2 = 1.0 - S1
    elo_dict[p1] = R1 + K * (S1 - E1)
    elo_dict[p2] = R2 + K * (S2 - E2)

@st.cache_data
def compute_elo_by_surface(df, K_global: int, K_surface: int):
    elo_global = init_elo_dict()
    elo_surface = {}

    df_sorted = df.sort_values("tourney_date")
    for _, row in df_sorted.iterrows():
        w = row["winner_name"]
        l = row["loser_name"]
        surf = row.get("surface", "Unknown")

        update_elo(elo_global, w, l, w, K_global)

        if surf not in elo_surface:
            elo_surface[surf] = init_elo_dict()
        update_elo(elo_surface[surf], w, l, w, K_surface)

    elo_global_df = (
        pd.DataFrame([{"player": p, "elo_global": r} for p, r in elo_global.items()])
        .sort_values("elo_global", ascending=False)
        .reset_index(drop=True)
    )

    rows = []
    for surf, d in elo_surface.items():
        for p, r in d.items():
            rows.append({"surface": surf, "player": p, "elo_surface": r})
    elo_surface_df = (
        pd.DataFrame(rows)
        .sort_values(["surface", "elo_surface"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return elo_global_df, elo_surface_df

@st.cache_data
def compute_service_return_elo(df, K: int):
    svc_elo = {}
    ret_elo = {}

    def get_elo(d, p):
        return d.get(p, 1500.0)

    def upd(d, p, delta):
        d[p] = d.get(p, 1500.0) + delta

    df_sorted = df.sort_values("tourney_date")
    for _, row in df_sorted.iterrows():
        w = row["winner_name"]
        l = row["loser_name"]

        w_svpt = row.get("w_svpt", np.nan)
        w_1stWon = row.get("w_1stWon", np.nan)
        w_2ndWon = row.get("w_2ndWon", np.nan)
        l_svpt = row.get("l_svpt", np.nan)
        l_1stWon = row.get("l_1stWon", np.nan)
        l_2ndWon = row.get("l_2ndWon", np.nan)

        w_svc_pts_won = (w_1stWon or 0) + (w_2ndWon or 0)
        l_svc_pts_won = (l_1stWon or 0) + (l_2ndWon or 0)

        w_ret_pts_won = (l_svpt or 0) - l_svc_pts_won
        l_ret_pts_won = (w_svpt or 0) - w_svc_pts_won

        for player, pts_won, svpt in [(w, w_svc_pts_won, w_svpt), (l, l_svc_pts_won, l_svpt)]:
            if svpt and svpt > 0:
                pct = pts_won / svpt
                delta = K * (pct - 0.5)
                upd(svc_elo, player, delta)

        for player, pts_won, opp_svpt in [(w, w_ret_pts_won, l_svpt), (l, l_ret_pts_won, w_svpt)]:
            if opp_svpt and opp_svpt > 0:
                pct = pts_won / opp_svpt
                delta = K * (pct - 0.4)
                upd(ret_elo, player, delta)

    svc_df = (
        pd.DataFrame([{"player": p, "service_elo": r} for p, r in svc_elo.items()])
        .sort_values("service_elo", ascending=False)
        .reset_index(drop=True)
    )
    ret_df = (
        pd.DataFrame([{"player": p, "return_elo": r} for p, r in ret_elo.items()])
        .sort_values("return_elo", ascending=False)
        .reset_index(drop=True)
    )
    return svc_df, ret_df

def build_training_dataset(df, elo_global_df, elo_surface_df, svc_df, ret_df):
    base = df.copy()
    base = base.dropna(subset=["winner_name", "loser_name"])

    eg = elo_global_df.set_index("player")["elo_global"]
    base["w_elo_global"] = base["winner_name"].map(eg)
    base["l_elo_global"] = base["loser_name"].map(eg)

    es = elo_surface_df.set_index(["surface", "player"])["elo_surface"]
    base["w_elo_surface"] = base.apply(
        lambda r: es.get((r.get("surface", "Unknown"), r["winner_name"]), np.nan), axis=1
    )
    base["l_elo_surface"] = base.apply(
        lambda r: es.get((r.get("surface", "Unknown"), r["loser_name"]), np.nan), axis=1
    )

    svc = svc_df.set_index("player")["service_elo"]
    ret = ret_df.set_index("player")["return_elo"]
    base["w_service_elo"] = base["winner_name"].map(svc)
    base["l_service_elo"] = base["loser_name"].map(svc)
    base["w_return_elo"] = base["winner_name"].map(ret)
    base["l_return_elo"] = base["loser_name"].map(ret)

    base["elo_global_diff"] = base["w_elo_global"] - base["l_elo_global"]
    base["elo_surface_diff"] = base["w_elo_surface"] - base["l_elo_surface"]
    base["service_elo_diff"] = base["w_service_elo"] - base["l_service_elo"]
    base["return_elo_diff"] = base["w_return_elo"] - base["l_return_elo"]

    base["over_21_5"] = (base["T Games"] > 21.5).astype(int)

    feats = ["elo_global_diff", "elo_surface_diff", "service_elo_diff", "return_elo_diff"]
    base = base.dropna(subset=feats + ["T Games"])

    X = base[feats]
    y_over = base["over_21_5"]
    return X, y_over, feats

@st.cache_resource
def train_models(df, elo_global_df, elo_surface_df, svc_df, ret_df):
    X, y_over, feats = build_training_dataset(df, elo_global_df, elo_surface_df, svc_df, ret_df)
    if len(X) < 50:
        st.warning("Poucos dados para treinar o modelo. Resultados podem ser fracos.")
    dtrain_over = xgb.DMatrix(X, label=y_over, feature_names=feats)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": 42,
    }

    model_over = xgb.train(params, dtrain_over, num_boost_round=150)
    return model_over, feats

def build_feature_row(playerA, playerB, surface, elo_global_df, elo_surface_df, svc_df, ret_df):
    eg = elo_global_df.set_index("player")["elo_global"]
    es = elo_surface_df.set_index(["surface", "player"])["elo_surface"]
    svc = svc_df.set_index("player")["service_elo"]
    ret = ret_df.set_index("player")["return_elo"]

    def get_es(surf, p):
        try:
            return es.loc[(surf, p)]
        except KeyError:
            return np.nan

    row = {}
    row["elo_global_diff"] = eg.get(playerA, 1500) - eg.get(playerB, 1500)
    row["elo_surface_diff"] = get_es(surface, playerA) - get_es(surface, playerB)
    row["service_elo_diff"] = svc.get(playerA, 1500) - svc.get(playerB, 1500)
    row["return_elo_diff"] = ret.get(playerA, 1500) - ret.get(playerB, 1500)

    return pd.DataFrame([row])

def predict_match(playerA, playerB, surface, model_over, feats,
                  elo_global_df, elo_surface_df, svc_df, ret_df):
    X = build_feature_row(playerA, playerB, surface, elo_global_df, elo_surface_df, svc_df, ret_df)
    dtest = xgb.DMatrix(X[feats], feature_names=feats)
    p_over = float(model_over.predict(dtest)[0])
    return p_over, 1 - p_over

def call_sportscore_api():
    querystring = {
        "challenge_id": "663",
        "venue_id": "6",
        "referee_id": "26",
        "league_id": "317",
        "page": "1",
        "away_team_id": "138",
        "home_team_id": "6",
        "status": "postponed",
        "season_id": "1",
        "date_start": "2018-09-15",
        "date_end": "2020-11-14",
        "sport_id": "1",
    }
    headers = {
        "Content-Type": "application/json",
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }
    try:
        resp = requests.post(RAPIDAPI_URL, headers=headers, params=querystring, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": resp.status_code, "text": resp.text}
    except Exception as e:
        return {"error": str(e)}

# -----------------------
# APP
# -----------------------

df = load_data()

st.title("Tennis JD – Elo, Serviço/Devolução e Over 21.5")

with st.sidebar:
    st.header("Parâmetros Elo")
    K_global = st.slider("K Elo Global", 8, 64, 32, 2)
    K_surface = st.slider("K Elo por Superfície", 8, 64, 24, 2)
    K_srv_ret = st.slider("K Elo Serviço/Devolução", 4, 32, 16, 2)

    st.markdown("---")
    st.header("API Sportscore")
    if st.button("Testar chamada API (exemplo)"):
        api_result = call_sportscore_api()
        st.json(api_result)

elo_global_df, elo_surface_df = compute_elo_by_surface(df, K_global=K_global, K_surface=K_surface)
svc_df, ret_df = compute_service_return_elo(df, K=K_srv_ret)
model_over, feats = train_models(df, elo_global_df, elo_surface_df, svc_df, ret_df)

players = sorted(set(df["winner_name"]).union(set(df["loser_name"])))

tab1, tab2, tab3, tab4 = st.tabs([
    "Ranking Elo por Superfície",
    "Elo Serviço & Devolução",
    "Comparar Jogadores & Over 21.5",
    "Explorar Dados & Exportar"
])

with tab1:
    st.subheader("Ranking Elo por Superfície")
    surfaces = sorted(elo_surface_df["surface"].unique())
    surf_sel = st.selectbox("Superfície", surfaces)
    elo_surf_sel = elo_surface_df[elo_surface_df["surface"] == surf_sel].reset_index(drop=True)
    elo_surf_sel["Rank"] = np.arange(1, len(elo_surf_sel) + 1)
    elo_surf_sel = elo_surf_sel[["Rank", "player", "elo_surface"]]

    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(elo_surf_sel, use_container_width=True)
    with col2:
        st.bar_chart(
            elo_surf_sel.set_index("player")["elo_surface"].head(20),
            use_container_width=True
        )

with tab2:
    st.subheader("Elo de Serviço e Devolução")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 20 – Service Elo**")
        st.dataframe(svc_df.head(20).reset_index(drop=True), use_container_width=True)
    with col2:
        st.markdown("**Top 20 – Return Elo**")
        st.dataframe(ret_df.head(20).reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.markdown("**Gráfico comparativo (Top 20 Service vs Return)**")
    top_players = list(set(svc_df.head(20)["player"]).union(set(ret_df.head(20)["player"])))
    svc_plot = svc_df[svc_df["player"].isin(top_players)].set_index("player")
    ret_plot = ret_df[ret_df["player"].isin(top_players)].set_index("player")
    combo = svc_plot.join(ret_plot, how="outer")
    st.bar_chart(combo, use_container_width=True)

with tab3:
    st.subheader("Comparar 2 Jogadores e Prever Over 21.5")

    colA, colB = st.columns(2)
    with colA:
        playerA = st.selectbox("Jogador A", players, index=0)
    with colB:
        playerB = st.selectbox("Jogador B", players, index=1)

    surfaces_all = sorted(df["surface"].dropna().unique())
    surface_sel = st.selectbox("Superfície do jogo", surfaces_all)

    if playerA == playerB:
        st.warning("Escolhe dois jogadores diferentes.")
    else:
        def get_elo_row(p):
            eg = elo_global_df.set_index("player")["elo_global"]
            es = elo_surface_df.set_index(["surface", "player"])["elo_surface"]
            svc = svc_df.set_index("player")["service_elo"]
            ret = ret_df.set_index("player")["return_elo"]

            def get_es(surf, pl):
                try:
                    return es.loc[(surf, pl)]
                except KeyError:
                    return np.nan

            return {
                "Elo Global": eg.get(p, 1500),
                f"Elo {surface_sel}": get_es(surface_sel, p),
                "Service Elo": svc.get(p, 1500),
                "Return Elo": ret.get(p, 1500),
            }

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {playerA}")
            st.json(get_elo_row(playerA))
        with col2:
            st.markdown(f"### {playerB}")
            st.json(get_elo_row(playerB))

        st.markdown("---")
        if st.button("Calcular previsão Over/Under 21.5 para este confronto"):
            p_over, p_under = predict_match(
                playerA, playerB, surface_sel,
                model_over, feats,
                elo_global_df, elo_surface_df, svc_df, ret_df
            )
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prob. Over 21.5 jogos", f"{p_over*100:.1f}%")
            with col2:
                st.metric("Prob. Under 21.5 jogos", f"{p_under*100:.1f}%")

with tab4:
    st.subheader("Explorar dados históricos")

    col1, col2, col3 = st.columns(3)
    with col1:
        surf_filter = st.multiselect("Superfície", sorted(df["surface"].unique()),
                                     default=list(sorted(df["surface"].unique())))
    with col2:
        years = df["tourney_date"].dt.year.dropna().unique() if "tourney_date" in df.columns else []
        year_filter = st.multiselect("Ano", sorted(years), default=list(sorted(years)))
    with col3:
        player_filter = st.multiselect("Jogador (winner/loser)", players)

    df_filt = df.copy()
    if surf_filter:
        df_filt = df_filt[df_filt["surface"].isin(surf_filter)]
    if year_filter and "tourney_date" in df_filt.columns:
        df_filt = df_filt[df_filt["tourney_date"].dt.year.isin(year_filter)]
    if player_filter:
        df_filt = df_filt[
            df_filt["winner_name"].isin(player_filter) |
            df_filt["loser_name"].isin(player_filter)
        ]

    st.dataframe(df_filt.head(200), use_container_width=True)

    st.markdown("---")
    st.subheader("Percentagem de jogos Over/Under 21.5 (dados filtrados)")

    df_filt = df_filt.dropna(subset=["T Games"])
    if len(df_filt) > 0:
        over = (df_filt["T Games"] > 21.5).mean()
        under = 1 - over
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Over 21.5 (%)", f"{over*100:.1f}%")
        with col2:
            st.metric("Under 21.5 (%)", f"{under*100:.1f}%")
    else:
        st.info("Sem jogos após filtros para calcular Over/Under.")

    st.markdown("---")
    st.subheader("Exportar ficheiro completo (dados + rankings)")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Matches", index=False)
        elo_global_df.to_excel(writer, sheet_name="Elo_Global", index=False)
        elo_surface_df.to_excel(writer, sheet_name="Elo_Surface", index=False)
        svc_df.to_excel(writer, sheet_name="Service_Elo", index=False)
        ret_df.to_excel(writer, sheet_name="Return_Elo", index=False)

    st.download_button(
        label="Download Excel completo",
        data=output.getvalue(),
        file_name="Challenger_Elo_ML_2026.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
