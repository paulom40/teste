import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Tennis Value Bets PRO", layout="wide")

st.title("🎾 Tennis Value Bets PRO")
st.caption("Modelo com Forma + Superfície + Value Bets")

# ================= UPLOAD =================
uploaded_file = st.file_uploader("📁 Carregar histórico ATP (Excel)", type=["xlsx"])

# ================= FEATURES =================
def calcular_forma(df):
    df = df.sort_values("tourney_date")

    historico = {}
    forma_w, forma_l = [], []

    for _, row in df.iterrows():
        w, l = row['winner_name'], row['loser_name']

        historico.setdefault(w, [])
        historico.setdefault(l, [])

        fw = sum(historico[w][-5:]) / max(1, len(historico[w][-5:]))
        fl = sum(historico[l][-5:]) / max(1, len(historico[l][-5:]))

        forma_w.append(fw)
        forma_l.append(fl)

        historico[w].append(1)
        historico[l].append(0)

    df['forma_w'] = forma_w
    df['forma_l'] = forma_l
    return df


def calcular_surface(df):
    stats = {}
    surf_w, surf_l = [], []

    for _, row in df.iterrows():
        w, l, s = row['winner_name'], row['loser_name'], row['surface']

        stats.setdefault(w, {})
        stats.setdefault(l, {})

        stats[w].setdefault(s, [0,0])
        stats[l].setdefault(s, [0,0])

        w_wins, w_tot = stats[w][s]
        l_wins, l_tot = stats[l][s]

        surf_w.append(w_wins / max(1, w_tot))
        surf_l.append(l_wins / max(1, l_tot))

        stats[w][s][0] += 1
        stats[w][s][1] += 1
        stats[l][s][1] += 1

    df['surf_w'] = surf_w
    df['surf_l'] = surf_l
    return df


def preparar_dados(df):
    df = calcular_forma(df)
    df = calcular_surface(df)

    df_w = pd.DataFrame({
        'rank_diff': df['winner_rank'] - df['loser_rank'],
        'age_diff': df['winner_age'] - df['loser_age'],
        'ht_diff': df['winner_ht'] - df['loser_ht'],
        'forma_diff': df['forma_w'] - df['forma_l'],
        'surf_diff': df['surf_w'] - df['surf_l'],
        'target': 1
    })

    df_l = pd.DataFrame({
        'rank_diff': df['loser_rank'] - df['winner_rank'],
        'age_diff': df['loser_age'] - df['winner_age'],
        'ht_diff': df['loser_ht'] - df['winner_ht'],
        'forma_diff': df['forma_l'] - df['forma_w'],
        'surf_diff': df['surf_l'] - df['surf_w'],
        'target': 0
    })

    df_final = pd.concat([df_w, df_l])

    X = df_final[['rank_diff','age_diff','ht_diff','forma_diff','surf_diff']].fillna(0)
    y = df_final['target']

    return X, y


def calcular_value(prob, odd):
    if odd <= 1:
        return 0
    return (prob - 1/odd) * 100


# ================= MAIN =================
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.success(f"✅ {len(df)} jogos carregados")

    # Treinar modelo
    X, y = preparar_dados(df)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    st.success("🤖 Modelo treinado com sucesso")

    # ================= INPUT =================
    st.subheader("🎯 Prever Novo Jogo")

    col1, col2 = st.columns(2)

    with col1:
        rank1 = st.number_input("Rank Jogador 1", 1, 2000, 100)
        age1 = st.number_input("Idade Jogador 1", 15, 45, 25)
        ht1 = st.number_input("Altura Jogador 1", 150, 220, 180)
        forma1 = st.slider("Forma J1", 0.0, 1.0, 0.5)
        surf1 = st.slider("Winrate Superfície J1", 0.0, 1.0, 0.5)

    with col2:
        rank2 = st.number_input("Rank Jogador 2", 1, 2000, 120)
        age2 = st.number_input("Idade Jogador 2", 15, 45, 26)
        ht2 = st.number_input("Altura Jogador 2", 150, 220, 182)
        forma2 = st.slider("Forma J2", 0.0, 1.0, 0.5)
        surf2 = st.slider("Winrate Superfície J2", 0.0, 1.0, 0.5)

    # ================= PREDIÇÃO =================
    input_data = np.array([[
        rank1 - rank2,
        age1 - age2,
        ht1 - ht2,
        forma1 - forma2,
        surf1 - surf2
    ]])

    prob = model.predict_proba(input_data)[0][1]

    st.metric("📊 Probabilidade Jogador 1", f"{prob:.2%}")

    # ================= ODDS =================
    st.subheader("💰 Odds")

    odd1 = st.number_input("Odd Jogador 1", value=2.0)
    odd2 = st.number_input("Odd Jogador 2", value=2.0)

    value1 = calcular_value(prob, odd1)
    value2 = calcular_value(1 - prob, odd2)

    colv1, colv2 = st.columns(2)
    colv1.metric("Value J1", f"{value1:.2f}%")
    colv2.metric("Value J2", f"{value2:.2f}%")

    # ================= ALERTAS =================
    if value1 > 5:
        st.warning("🔥 VALUE BET FORTE: Jogador 1")
    elif value2 > 5:
        st.warning("🔥 VALUE BET FORTE: Jogador 2")
    elif value1 > 2 or value2 > 2:
        st.info("⚠️ Value leve encontrado")

else:
    st.info("📁 Carrega o ficheiro para começar")
