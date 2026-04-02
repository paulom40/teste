import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.title("🎾 Modelo Preditivo de Ténis (PRO)")

uploaded_file = st.file_uploader("📁 Carregar histórico ATP", type=["xlsx"])

# ================= FEATURE ENGINEERING =================
def preparar_dados(df):
    df = df.copy()

    df['rank_diff'] = df['winner_rank'] - df['loser_rank']
    df['age_diff'] = df['winner_age'] - df['loser_age']
    df['ht_diff'] = df['winner_ht'] - df['loser_ht']

    # target = 1 (winner ganhou)
    df['target'] = 1

    features = df[['rank_diff', 'age_diff', 'ht_diff']].fillna(0)

    return features, df['target']

# ================= TREINO =================
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.success(f"{len(df)} jogos carregados")

    X, y = preparar_dados(df)

    model = LogisticRegression()
    model.fit(X, y)

    st.success("✅ Modelo treinado")

    # ================= INPUT NOVO JOGO =================
    st.subheader("🎯 Prever Novo Jogo")

    col1, col2 = st.columns(2)

    with col1:
        rank1 = st.number_input("Rank Jogador 1", value=100)
        age1 = st.number_input("Idade Jogador 1", value=25)
        ht1 = st.number_input("Altura Jogador 1", value=180)

    with col2:
        rank2 = st.number_input("Rank Jogador 2", value=120)
        age2 = st.number_input("Idade Jogador 2", value=26)
        ht2 = st.number_input("Altura Jogador 2", value=182)

    # features do jogo
    input_data = np.array([[
        rank1 - rank2,
        age1 - age2,
        ht1 - ht2
    ]])

    prob = model.predict_proba(input_data)[0][1]

    st.metric("📊 Probabilidade Jogador 1", f"{prob:.2%}")

    # ================= ODDS =================
    st.subheader("💰 Odds")

    odd1 = st.number_input("Odd Jogador 1", value=2.0)
    odd2 = st.number_input("Odd Jogador 2", value=2.0)

    def value(prob, odd):
        return (prob - 1/odd) * 100

    value1 = value(prob, odd1)
    value2 = value(1-prob, odd2)

    st.write(f"Value J1: {value1:.2f}%")
    st.write(f"Value J2: {value2:.2f}%")

    # ================= ALERTA =================
    if value1 > 5:
        st.warning("🔥 VALUE BET: Jogador 1")
    elif value2 > 5:
        st.warning("🔥 VALUE BET: Jogador 2")
