import streamlit as st
import pandas as pd
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
from io import BytesIO

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Tennis Predictor Pro - Stats + Modeling Strategy")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Todos os Jogos", 
    "🔍 Previsão Personalizada", 
    "📈 Recommended Modeling Strategy",
    "ℹ️ Sobre o Modelo"
])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])

# ====================== CARREGAR DADOS ======================
@st.cache_data
def load_data(file):
    if not file:
        return pd.DataFrame(), []
    try:
        df = pd.read_excel(file)
        
        def norm(name):
            if not isinstance(name, str): return ""
            n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            return ''.join(filter(str.isalnum, n.lower().strip()))
        
        df['winner_clean'] = df['winner_name'].apply(norm)
        df['loser_clean'] = df['loser_name'].apply(norm)
        
        all_players = pd.concat([df['winner_name'], df['loser_name']]).drop_duplicates().sort_values().tolist()
        
        st.sidebar.success(f"✅ {len(df)} jogos | {len(all_players)} jogadores")
        return df, all_players
    except Exception as e:
        st.sidebar.error(f"Erro: {e}")
        return pd.DataFrame(), []

df_raw, player_list = load_data(uploaded_file)

# ====================== FUNÇÃO DE MATCHING ======================
def find_best_player_stats(player_name, df):
    clean_name = norm(player_name)
    best_match = None
    best_score = 0.0

    for _, row in df.iterrows():
        for col in ['winner_clean', 'loser_clean']:
            clean_db = row.get(col, "")
            if not clean_db: continue
            similarity = SequenceMatcher(None, clean_name, clean_db).ratio()
            if clean_name in clean_db or clean_db in clean_name:
                similarity = max(similarity, 0.95)
            score = similarity * 100
            if score > best_score:
                best_score = score
                best_match = row
    return best_match if best_score >= 60 else pd.Series(dtype='object')

def norm(name):
    if not isinstance(name, str): return ""
    n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return ''.join(filter(str.isalnum, n.lower().strip()))

# ====================== PREDICTOR ======================
def predict_match(jogador_a, jogador_b, superficie="Hard"):
    p1_stats = find_best_player_stats(jogador_a, df_raw)
    p2_stats = find_best_player_stats(jogador_b, df_raw)
    
    if p1_stats.empty or p2_stats.empty:
        return None
    
    def safe(v):
        try: return float(v) if pd.notna(v) else 0.0
        except: return 0.0

    def serve_win(stats):
        svpt = safe(stats.get('w_svpt', 0))
        if svpt == 0: return 0.65
        return (safe(stats.get('w_1stWon', 0)) + safe(stats.get('w_2ndWon', 0))) / svpt

    serve1 = serve_win(p1_stats)
    serve2 = serve_win(p2_stats)
    return1 = 1 - serve2
    return2 = 1 - serve1

    p1_point_win = (serve1 + return1) / 2
    p2_point_win = (serve2 + return2) / 2

    surface_factor = {'Clay': 1.08, 'Hard': 1.0, 'Grass': 0.93, 'Indoor': 1.02}.get(superficie, 1.0)

    diff = (p1_point_win - p2_point_win) * 100
    prob_a_win = 1 / (1 + 10 ** (-diff / 38))

    hold1 = serve1 ** 1.85
    hold2 = serve2 ** 1.85
    break_prob = (1 - hold1 + 1 - hold2) / 2
    games_per_set = 9.6 + 4.2 * break_prob
    total_esperado = round(games_per_set * 2.15 * surface_factor, 2)

    prob_over = max(0.38, min(0.78, 0.5 + (total_esperado - 21.5) * 0.085))
    prob_under = 100 - prob_over

    return {
        "Jogador_A": jogador_a,
        "Jogador_B": jogador_b,
        "Superficie": superficie,
        "Prob_A_Vitória_%": round(prob_a_win * 100, 1),
        "Prob_B_Vitória_%": round(100 - prob_a_win * 100, 1),
        "Total_Esperado": total_esperado,
        "Prob_Over_21.5_%": round(prob_over, 1),
        "Prob_Under_21.5_%": round(prob_under, 1),
        "Serve_A_%": round(serve1 * 100, 1),
        "BP_Saved_A_%": round(safe(p1_stats.get('w_bpSaved',0)) / max(safe(p1_stats.get('w_bpFaced',1)), 1) * 100, 1),
    }

# ====================== ABA 1 - TODOS OS JOGOS ======================
with tab1:
    st.header("Todos os Jogos")
    if not df_raw.empty:
        st.dataframe(df_raw.head(20), use_container_width=True)

# ====================== ABA 2 - PREVISÃO PERSONALIZADA ======================
with tab2:
    st.header("🔍 Previsão Personalizada")
    
    col1, col2 = st.columns(2)
    with col1:
        jogador_a = st.selectbox("Jogador A", options=player_list, key="ja")
    with col2:
        jogador_b = st.selectbox("Jogador B", options=player_list, key="jb")

    superficie = st.selectbox("Superfície", ["Hard", "Clay", "Grass", "Indoor"], index=0)

    if st.button("🚀 Calcular Previsão", type="primary"):
        if jogador_a == jogador_b:
            st.error("Escolha dois jogadores diferentes!")
        else:
            result = predict_match(jogador_a, jogador_b, superficie)
            if result:
                st.success("Previsão Calculada com Sucesso!")
                
                c1, c2 = st.columns(2)
                with c1: st.metric(f"{result['Jogador_A']} vence", f"{result['Prob_A_Vitória_%']}%")
                with c2: st.metric(f"{result['Jogador_B']} vence", f"{result['Prob_B_Vitória_%']}%")
                
                st.metric("Total de Jogos Esperado", f"{result['Total_Esperado']} jogos")
                
                o1, o2 = st.columns(2)
                with o1: st.metric("Over 21.5", f"{result['Prob_Over_21.5_%']}%")
                with o2: st.metric("Under 21.5", f"{result['Prob_Under_21.5_%']}%")

                # Exportar
                result_df = pd.DataFrame([result])
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False)
                output.seek(0)

                st.download_button(
                    "📥 Exportar Previsão para Excel",
                    data=output,
                    file_name=f"Previsao_{jogador_a}_vs_{jogador_b}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# ====================== NOVA ABA 3 - RECOMMENDED MODELING STRATEGY ======================
with tab3:
    st.header("📈 Recommended Modeling Strategy")
    st.markdown("""
    ### Estratégia Recomendada para Modelar Tênis (Vitória + Total Jogos)

    Para obter os **melhores resultados** em previsões de vitória e especialmente no mercado **Over/Under 21.5 jogos**, a abordagem mais eficaz é uma **estratégia híbrida**:

    #### 1. Pré-processamento de Dados (Feature Engineering)
    - Diferença de Ranking (`Rank Difference`)
    - Diferença de Pontos (`Points Difference`)
    - Média de jogos dos últimos 5–10 jogos (`Average Total Games`)
    - Percentagem de serviço e retorno por superfície
    - Taxa de break points salvos/convertidos
    - Forma recente (últimos 3–5 jogos)

    #### 2. Modelo Híbrido Recomendado
    | Objetivo                  | Modelo Recomendado              | Porquê? |
    |---------------------------|----------------------------------|--------|
    | Probabilidade de Vitória  | Logistic Regression / XGBoost    | Rápido e interpretável |
    | Total de Jogos (O/U 21.5) | **Markov Chain Simulation**      | Melhor precisão para contagem de jogos |
    | Combinação Final          | **Hybrid Approach**              | Combina força dos dois |

    #### 3. Validação Robusta
    - 10-fold Cross-Validation (repetido 5–10 vezes)
    - Time-based validation (não aleatório)
    - Teste em diferentes superfícies e níveis de torneio

    #### 4. Abordagem Avançada (State-of-the-Art)
    - **Markov Chains** para simular pontos → jogos → sets → match
    - **Poisson / Negative Binomial** para modelar distribuição de jogos
    - **XGBoost / CatBoost** para win probability
    - **Monte Carlo Simulation** para gerar distribuição de totais

    ### Fontes e Referências
    - BrandoPolistirolo/Tennis-Betting-ML (GitHub)
    - ATP Tennis Dataset (2000–2025) – Kaggle
    - "Modelling a Game of Tennis" – Mark Jamison (Medium)
    - Estudos académicos sobre Point-Based Modeling

    **Conclusão dos especialistas:**
    > Enquanto algoritmos de Machine Learning são excelentes para prever o vencedor, **simulações baseadas em Markov Chains** são atualmente a melhor abordagem para o mercado de **Total de Jogos**.
    """)

    st.info("Esta estratégia combina o teu modelo atual (baseado em stats de serve/return) com as melhores práticas da comunidade de modelação de ténis.")

# ====================== ABA 4 - SOBRE ======================
with tab4:
    st.header("ℹ️ Sobre o Modelo Atual")
    st.write("""
    O modelo atual utiliza:
    - Stats reais de serviço, retorno e break points do teu ficheiro
    - Ajuste por superfície
    - Cálculo de pontos ganhos esperados
    - Simulação simplificada de total de jogos

    **Limitações atuais:**
    - Não inclui ranking ou forma recente (podes adicionar)
    - Matching de nomes ainda pode ser melhorado
    - Não faz simulação completa de sets (Markov Chain)

    Queres que eu evolua este modelo para incluir **Markov Chain** ou **XGBoost** no futuro?
    """)

st.caption("Predictor de Tênis • Estratégia Híbrida Recomendada • Exportação para Excel")
