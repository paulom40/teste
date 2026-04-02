import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random

# ====================== CONFIGURAÇÃO ======================
st.set_page_config(page_title="Tênis Hoje - WELO + Total", page_icon="🎾", layout="wide")

st.title("🎾 Partidas de Tênis Hoje + WELO + Linha Total")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger.xlsm")
    st.warning("⚠️ Como não tem o ficheiro WELO, a demonstração usa valores estimados")
    
    uploaded_file = st.file_uploader("Quando tiver o ficheiro, carregue aqui", type=["xlsm", "xlsx"])

# ====================== DADOS DE DEMONSTRAÇÃO ======================
@st.cache_data
def get_demo_matches():
    """Gera partidas de demonstração realistas"""
    today = datetime.now()
    
    torneios = [
        "ATP Masters 1000 Monte Carlo", "WTA 500 Stuttgart", "ATP 250 Houston",
        "WTA 1000 Madrid", "ATP Challenger Oeiras", "ITF M15 Lisbon"
    ]
    
    jogadores = [
        ("Novak Djokovic", "Carlos Alcaraz", 1980, 1870),
        ("Jannik Sinner", "Daniil Medvedev", 1950, 1850),
        ("Iga Swiatek", "Elena Rybakina", 1930, 1820),
        ("Coco Gauff", "Jessica Pegula", 1880, 1790),
        ("Alexander Zverev", "Andrey Rublev", 1860, 1770),
        ("Holger Rune", "Stefanos Tsitsipas", 1820, 1750),
        ("Nuno Borges", "Arthur Fils", 1650, 1600),
        ("João Sousa", "Gastao Elias", 1550, 1480)
    ]
    
    superficies = ['Clay', 'Hard', 'Clay', 'Hard', 'Clay', 'Hard', 'Clay', 'Hard']
    
    matches = []
    for i, (j1, j2, elo1, elo2) in enumerate(jogadores):
        # Horário progressivo
        hora = 10 + i
        minuto = random.choice([0, 30])
        
        matches.append({
            'torneio': torneios[i % len(torneios)],
            'jogador_1': j1,
            'jogador_2': j2,
            'horario': f"{hora:02d}:{minuto:02d}",
            'superficie': superficies[i % len(superficies)],
            'WELO_J1_Demo': elo1,
            'WELO_J2_Demo': elo2
        })
    
    return pd.DataFrame(matches)

# ====================== CÁLCULO DA LINHA TOTAL ======================
def calcular_linha_total(welo1: float, welo2: float, superficie: str) -> tuple:
    dif = abs(welo1 - welo2)
    
    base_jogos = {
        'Clay': 22.8,
        'Hard': 22.4,
        'Grass': 21.9,
        'Indoor': 22.6
    }.get(superficie, 22.5)
    
    ajuste_dif = -0.035 * dif
    total_esperado = base_jogos + ajuste_dif
    total_esperado = max(18.5, min(27.0, total_esperado))
    prob_mais_21_5 = max(0.35, min(0.78, 0.5 + (total_esperado - 22.0) * 0.08))
    
    return round(total_esperado, 2), round(prob_mais_21_5 * 100, 1)

# ====================== EXECUÇÃO ======================
if st.button("🔄 Buscar Partidas Hoje + Calcular Linha Total", type="primary"):
    with st.spinner("Gerando partidas e calculando totais..."):
        df = get_demo_matches()
        
        if not df.empty:
            # Calcular linha total
            resultados = df.apply(
                lambda row: calcular_linha_total(
                    row['WELO_J1_Demo'], 
                    row['WELO_J2_Demo'], 
                    row['superficie']
                ), axis=1
            )
            
            df['Total_Esperado'] = [r[0] for r in resultados]
            df['Prob_Mais_21.5'] = [r[1] for r in resultados]
            df['Dif_WELO'] = abs(df['WELO_J1_Demo'] - df['WELO_J2_Demo'])
            
            st.success(f"✅ {len(df)} partidas analisadas!")
            
            # Mostrar tabela
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "torneio": "🏆 Torneio",
                    "jogador_1": "🎾 Jogador 1",
                    "jogador_2": "🎾 Jogador 2",
                    "horario": "⏰ Horário",
                    "superficie": "🏟️ Superfície",
                    "WELO_J1_Demo": st.column_config.NumberColumn("WELO J1", format="%.0f"),
                    "WELO_J2_Demo": st.column_config.NumberColumn("WELO J2", format="%.0f"),
                    "Dif_WELO": st.column_config.NumberColumn("Dif WELO", format="%.0f"),
                    "Total_Esperado": st.column_config.NumberColumn("Total Esperado", format="%.2f"),
                    "Prob_Mais_21.5": st.column_config.NumberColumn("Prob >21.5 (%)", format="%.1f"),
                }
            )
            
            # Dicas para Over/Under
            st.subheader("🎯 Recomendações")
            for _, row in df.iterrows():
                if row['Prob_Mais_21.5'] > 65:
                    st.success(f"🔴 **{row['jogador_1']} vs {row['jogador_2']}** - {row['Prob_Mais_21.5']}% >21.5 (Total Esperado: {row['Total_Esperado']})")
                elif row['Prob_Mais_21.5'] < 45:
                    st.info(f"🔵 **{row['jogador_1']} vs {row['jogador_2']}** - {100-row['Prob_Mais_21.5']}% <21.5 (Total Esperado: {row['Total_Esperado']})")

else:
    st.info("""
    ### 🎾 Como usar esta demonstração:
    
    **Nota:** Como a API está com erro 401 e não tem o ficheiro WELO, estou a usar **dados de demonstração**.
    
    ### Para usar com dados reais:
    
    1. **Obter chave API válida** no RapidAPI:
       - Crie conta em [RapidAPI](https://rapidapi.com/)
       - Subscreva o plano gratuito do **SportScore1**
       - Copie a chave API correta
       
    2. **Obter ficheiro WELO** (Challenger.xlsm):
       - Este ficheiro contém os ratings ELO dos jogadores
       - É essencial para os cálculos precisos
       
    ### Enquanto não tem os dados reais:
    - Use esta demonstração para testar a lógica
    - Os valores WELO são aproximados para exemplo
    - A metodologia de cálculo é a mesma
    """)

st.caption("🎾 Versão de Demonstração • Aguarda integração com API real e ficheiro WELO")
