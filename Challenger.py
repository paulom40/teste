import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    """
    Calcula o total de games de uma partida baseado nos sets
    """
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
    """
    Extrai o tipo de superfície do nome do torneio
    """
    if not isinstance(tournament_name, str):
        return "Hard"
    
    tournament_lower = tournament_name.lower()
    
    if "clay" in tournament_lower:
        return "Clay"
    elif "grass" in tournament_lower or "wimbledon" in tournament_lower:
        return "Grass"
    elif "hard" in tournament_lower:
        return "Hard"
    else:
        return "Hard"


# ============================================================
# 2. CARREGAR HISTÓRICO (GitHub ou Excel)
# ============================================================

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
# 3. API — JOGOS DE HOJE E AMANHÃ
# ============================================================

def fetch_matches_from_api():
    """
    Busca os jogos da API Tennis com tratamento de erros melhorado.
    """
    
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
            st.error(f"API retornou erro: {data.get('error', 'Erro desconhecido')}")
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
        result_df = result_df.drop_duplicates()
        result_df = result_df.dropna(subset=["Winner", "Loser"])
        
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
# 4. MODELOS PARA PREVISÕES
# ============================================================

def train_three_sets_model(df_hist):
    """
    Treina modelo para prever probabilidade de 3+ sets
    """
    dfm = df_hist.copy()

    if "Wsets" not in dfm.columns:
        st.error("Erro: coluna Wsets não existe.")
        return None

    dfm["three_sets"] = (dfm["Wsets"] >= 2).astype(int)
    
    features = ["WRank", "LRank", "WPts", "LPts"]
    
    missing_features = [f for f in features if f not in dfm.columns]
    if missing_features:
        st.error(f"Features em falta: {missing_features}")
        return None
    
    dfm = dfm.dropna(subset=features + ["three_sets"])

    if len(dfm) < 30:
        st.warning(f"Apenas {len(dfm)} jogos. Mínimo: 30")
        return None

    X = dfm[features]
    y = dfm["three_sets"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test) if len(X_test) > 0 else 0
    
    st.sidebar.success(f"✅ Modelo 3+ Sets treinado com {len(dfm)} jogos")
    st.sidebar.info(f"📊 Acurácia - Treino: {train_score:.1%} | Teste: {test_score:.1%}")

    return model


def train_over_games_model(df_hist, threshold=22):
    """
    Treina modelo para prever probabilidade de total de games > threshold
    """
    dfm = df_hist.copy()

    # Calcular total de games
    dfm["Total_Games"] = calculate_total_games(dfm)
    
    if dfm["Total_Games"].isna().all():
        st.error("Não foi possível calcular total de games dos jogos históricos.")
        return None, 0, 0
    
    dfm["over_threshold"] = (dfm["Total_Games"] > threshold).astype(int)
    
    # Features para o modelo
    features = ["WRank", "LRank", "WPts", "LPts"]
    
    missing_features = [f for f in features if f not in dfm.columns]
    if missing_features:
        st.error(f"Features em falta: {missing_features}")
        return None, 0, 0
    
    # Remover linhas com valores nulos
    dfm = dfm.dropna(subset=features + ["over_threshold", "Total_Games"])
    
    if len(dfm) < 30:
        st.warning(f"Apenas {len(dfm)} jogos com dados completos. Mínimo: 30")
        return None, 0, 0
    
    X = dfm[features]
    y = dfm["over_threshold"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test) if len(X_test) > 0 else 0
    
    # Calcular estatísticas adicionais
    avg_games = dfm["Total_Games"].mean()
    over_percentage = dfm["over_threshold"].mean() * 100
    
    st.sidebar.success(f"✅ Modelo Over {threshold} Games treinado com {len(dfm)} jogos")
    st.sidebar.info(f"📊 Média de games: {avg_games:.1f} | Over {threshold}: {over_percentage:.1f}%")
    st.sidebar.info(f"📊 Acurácia - Treino: {train_score:.1%} | Teste: {test_score:.1%}")
    
    return model, avg_games, over_percentage


def predict_for_upcoming(upcoming_df, hist_df, model_3sets, model_over, threshold=22):
    """
    Faz previsões para os jogos futuros
    """
    df_up = upcoming_df.copy()

    # Obter últimos rankings conhecidos
    hist_sorted = hist_df.sort_values("Date")

    w_rank_map = hist_sorted.groupby("Winner")["WRank"].last().to_dict()
    l_rank_map = hist_sorted.groupby("Loser")["LRank"].last().to_dict()
    w_pts_map  = hist_sorted.groupby("Winner")["WPts"].last().to_dict()
    l_pts_map  = hist_sorted.groupby("Loser")["LPts"].last().to_dict()

    # Preencher features
    df_up["WRank"] = df_up["Winner"].map(w_rank_map).fillna(300)
    df_up["LRank"] = df_up["Loser"].map(l_rank_map).fillna(300)
    df_up["WPts"]  = df_up["Winner"].map(w_pts_map).fillna(30)
    df_up["LPts"]  = df_up["Loser"].map(l_pts_map).fillna(30)

    for col in ["WRank", "LRank", "WPts", "LPts"]:
        df_up[col] = pd.to_numeric(df_up[col], errors="coerce").fillna(300)

    features = ["WRank", "LRank", "WPts", "LPts"]
    
    # Previsões
    if model_3sets is not None:
        df_up["prob_3_sets"] = model_3sets.predict_proba(df_up[features])[:, 1]
    else:
        df_up["prob_3_sets"] = 0.33
    
    if model_over is not None:
        df_up[f"prob_over_{threshold}_games"] = model_over.predict_proba(df_up[features])[:, 1]
    else:
        df_up[f"prob_over_{threshold}_games"] = 0.5
    
    # Calcular probabilidade combinada (jogo competitivo)
    df_up["prob_competitive_match"] = (
        df_up["prob_3_sets"] * 0.6 + 
        df_up[f"prob_over_{threshold}_games"] * 0.4
    )
    
    return df_up.sort_values("prob_competitive_match", ascending=False)


def export_to_excel(predictions_df, threshold_games):
    """
    Exporta as previsões para um ficheiro Excel
    """
    # Preparar dados para exportação
    export_df = predictions_df.copy()
    
    # Renomear colunas para português
    export_df = export_df.rename(columns={
        "Date": "Data",
        "Winner": "Vencedor",
        "Loser": "Derrotado",
        "Surface": "Superfície",
        "prob_3_sets": "Probabilidade_3_Sets",
        f"prob_over_{threshold_games}_games": f"Probabilidade_Over_{threshold_games}_Games",
        "prob_competitive_match": "Probabilidade_Jogo_Competitivo"
    })
    
    # Formatar datas
    export_df["Data"] = pd.to_datetime(export_df["Data"]).dt.strftime("%Y-%m-%d")
    
    # Adicionar colunas adicionais com informações
    export_df["Ranking_Vencedor"] = export_df["Vencedor"].map(
        predictions_df.groupby("Winner")["WRank"].first() if "WRank" in predictions_df.columns else pd.Series()
    ).fillna("N/A")
    
    export_df["Ranking_Derrotado"] = export_df["Derrotado"].map(
        predictions_df.groupby("Loser")["LRank"].first() if "LRank" in predictions_df.columns else pd.Series()
    ).fillna("N/A")
    
    # Reordenar colunas
    column_order = [
        "Data", "Vencedor", "Derrotado", "Ranking_Vencedor", "Ranking_Derrotado",
        "Superfície", "Probabilidade_3_Sets", f"Probabilidade_Over_{threshold_games}_Games",
        "Probabilidade_Jogo_Competitivo"
    ]
    
    export_df = export_df[column_order]
    
    # Formatar percentagens
    for col in ["Probabilidade_3_Sets", f"Probabilidade_Over_{threshold_games}_Games", "Probabilidade_Jogo_Competitivo"]:
        export_df[col] = export_df[col].apply(lambda x: f"{x:.1%}")
    
    # Criar ficheiro Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet principal com previsões
        export_df.to_excel(writer, sheet_name='Previsões', index=False)
        
        # Sheet com resumo estatístico
        summary_df = pd.DataFrame({
            "Métrica": [
                "Total de Jogos Analisados",
                f"Média Probabilidade 3+ Sets",
                f"Média Probabilidade Over {threshold_games} Games",
                "Média Probabilidade Jogo Competitivo",
                f"Jogos com >60% Competitivo",
                f"Jogos com >70% Competitivo"
            ],
            "Valor": [
                len(predictions_df),
                f"{predictions_df['prob_3_sets'].mean():.1%}",
                f"{predictions_df[f'prob_over_{threshold_games}_games'].mean():.1%}",
                f"{predictions_df['prob_competitive_match'].mean():.1%}",
                len(predictions_df[predictions_df['prob_competitive_match'] > 0.6]),
                len(predictions_df[predictions_df['prob_competitive_match'] > 0.7])
            ]
        })
        summary_df.to_excel(writer, sheet_name='Resumo', index=False)
        
        # Sheet com TOP 10 jogos
        top10_df = export_df.head(10).copy()
        top10_df.to_excel(writer, sheet_name='TOP_10_Jogos', index=False)
        
        # Ajustar largura das colunas
        for sheet in writer.sheets.values():
            for column in sheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                sheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return output


# ============================================================
# 5. SIDEBAR — CARREGAR HISTÓRICO
# ============================================================

st.sidebar.header("📂 Histórico de jogos")

uploaded_file = st.sidebar.file_uploader(
    "Escolhe um ficheiro Excel (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:
    df, source_name = load_custom_excel(uploaded_file)
else:
    df, source_name = fetch_challenger_github_data()

if df is None or df.empty:
    st.error("Não foi possível carregar nenhum histórico.")
    st.stop()

st.sidebar.info(f"Fonte atual: {source_name}")
st.sidebar.write(f"Total de jogos: {len(df)}")

# Configuração do threshold
st.sidebar.header("⚙️ Configurações")
threshold_games = st.sidebar.slider(
    "Threshold para total de games",
    min_value=15,
    max_value=30,
    value=22,
    step=1,
    help="Número mínimo de games para considerar 'Over'"
)


# ============================================================
# 6. UI PRINCIPAL — PREVISÕES
# ============================================================

st.title("🎾 CHALLENGER — Predição de jogos competitivos")
st.caption(f"Fonte de dados: {source_name} | Total de jogos históricos: {len(df)}")

# Treinar modelos
with st.spinner("A treinar modelos de previsão..."):
    model_3sets = train_three_sets_model(df)
    model_over, avg_games, over_percentage = train_over_games_model(df, threshold_games)

if model_3sets is None and model_over is None:
    st.error("Não foi possível treinar os modelos.")
    st.stop()

st.markdown("---")
st.header("📅 Jogos de hoje e amanhã")

# Buscar jogos da API
api_matches = fetch_matches_from_api()

if api_matches.empty:
    st.warning("⚠️ Não foi possível obter jogos da API.")
    
    st.subheader("📤 Carrega um ficheiro com jogos para prever")
    st.markdown(f"""
    O ficheiro Excel deve conter as colunas:
    - **Date** (data do jogo)
    - **Winner** (nome do jogador favorito)
    - **Loser** (nome do adversário)
    - **Surface** (opcional: Clay, Grass, Hard)
    """)
    
    manual_file = st.file_uploader(
        "Escolhe um ficheiro Excel (.xlsx)",
        type=["xlsx"],
        key="manual_matches"
    )
    
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

# Fazer previsões
preds = predict_for_upcoming(upcoming, df, model_3sets, model_over, threshold_games)

# Mostrar resultados
st.subheader(f"📋 Jogos ordenados por probabilidade de ser competitivo")

# Criar dataframe para exibição
display_df = preds[["Date", "Winner", "Loser", "Surface", "prob_3_sets", f"prob_over_{threshold_games}_games", "prob_competitive_match"]].copy()
display_df.columns = ["Date", "Winner", "Loser", "Surface", "3+ Sets", f"Over {threshold_games} Games", "Competitivo"]

st.dataframe(
    display_df.style.format({
        "3+ Sets": "{:.1%}",
        f"Over {threshold_games} Games": "{:.1%}",
        "Competitivo": "{:.1%}"
    })
)

# Botão para exportar para Excel
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    export_button = st.button(
        "📊 Exportar para Excel",
        type="primary",
        use_container_width=True,
        help="Exporta as previsões para um ficheiro Excel com múltiplas sheets"
    )

if export_button:
    with st.spinner("A gerar ficheiro Excel..."):
        excel_file = export_to_excel(preds, threshold_games)
        
        # Nome do ficheiro com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"previsoes_tenis_{timestamp}.xlsx"
        
        st.success("✅ Ficheiro Excel gerado com sucesso!")
        
        st.download_button(
            label="📥 Descarregar Excel",
            data=excel_file,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# TOP jogos
st.markdown("---")
st.subheader(f"🔥 TOP 5 jogos mais prováveis de serem competitivos")
top_n = min(5, len(preds))

for idx, (_, row) in enumerate(preds.head(top_n).iterrows(), 1):
    with st.container():
        st.markdown(f"""
        **{idx}. {row['Winner']} vs {row['Loser']}**  
        📅 {row['Date'].date()} | 🎾 {row['Surface']}
        
        | Probabilidade | Valor |
        |--------------|-------|
        | 🎯 3+ Sets | **{row['prob_3_sets']:.1%}** |
        | 📊 Over {threshold_games} Games | **{row[f'prob_over_{threshold_games}_games']:.1%}** |
        | ⭐ Competitivo | **{row['prob_competitive_match']:.1%}** |
        """)
        
        # Adicionar barra de progresso visual
        st.progress(row['prob_competitive_match'], text="Probabilidade de jogo competitivo")

# Estatísticas
st.markdown("---")
st.subheader("📊 Estatísticas")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Jogos no histórico", len(df))
with col2:
    st.metric("Jogos para prever", len(preds))
with col3:
    st.metric("Média de games (histórico)", f"{avg_games:.1f}")
with col4:
    st.metric(f"Over {threshold_games} (histórico)", f"{over_percentage:.1f}%")

# Distribuição das probabilidades
if len(preds) > 1:
    st.subheader("📈 Distribuição das probabilidades")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Probabilidade de 3+ Sets**")
        st.bar_chart(preds["prob_3_sets"].value_counts(bins=10).sort_index())
    
    with col2:
        st.write(f"**Probabilidade de Over {threshold_games} Games**")
        st.bar_chart(preds[f"prob_over_{threshold_games}_games"].value_counts(bins=10).sort_index())

# Recomendações
st.markdown("---")
st.subheader("💡 Recomendações")

high_value_matches = preds[preds["prob_competitive_match"] > 0.6]
very_high_value_matches = preds[preds["prob_competitive_match"] > 0.7]

if len(high_value_matches) > 0:
    st.success(f"🎯 Encontrados {len(high_value_matches)} jogos com alta probabilidade de serem competitivos!")
    
    if len(very_high_value_matches) > 0:
        st.info(f"🔥 Destes, {len(very_high_value_matches)} jogos têm probabilidade >70% - são os mais promissores!")
    
    st.markdown("**Estes jogos têm maior probabilidade de:**")
    st.markdown("- ✅ Irem a 3 sets")
    st.markdown(f"- ✅ Terem mais de {threshold_games} games no total")
    st.markdown("- ✅ Serem emocionantes e equilibrados")
else:
    st.info("Nenhum jogo com probabilidade > 60% de ser competitivo no momento.")

# Informação sobre o ficheiro exportado
st.markdown("---")
with st.expander("ℹ️ Sobre o ficheiro Excel exportado"):
    st.markdown("""
    O ficheiro Excel exportado contém 3 sheets:
    
    1. **Previsões** - Todos os jogos com as probabilidades detalhadas
    2. **Resumo** - Estatísticas gerais das previsões
    3. **TOP 10 Jogos** - Os 10 jogos com maior probabilidade de serem competitivos
    
    **Colunas incluídas:**
    - Data do jogo
    - Nome dos jogadores
    - Rankings (quando disponíveis)
    - Superfície
    - Probabilidade de 3+ Sets
    - Probabilidade de Over X Games
    - Probabilidade de jogo competitivo (combinação)
    """)
