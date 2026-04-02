import streamlit as st
import pandas as pd
from extractor import get_tennis_matches_today
from datetime import datetime

# --- Configuração da Página ---
st.set_page_config(
    page_title="Tênis Hoje - FlashScore",
    page_icon="🎾",
    layout="centered"
)

# --- Título e Botão ---
st.title("🎾 Partidas de Tênis - Hoje")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

if st.button("🔄 Atualizar Calendário de Hoje"):
    with st.spinner("Buscando dados no FlashScore... Isso pode levar alguns segundos."):
        df = get_tennis_matches_today()
        
        if not df.empty:
            st.success(f"{len(df)} partidas encontradas!")
            st.dataframe(
                df[['torneio', 'jogador_1', 'jogador_2', 'data_hora', 'status']],
                use_container_width=True,
                hide_index=True
            )
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar como CSV",
                data=csv,
                file_name=f'tenis_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )
        else:
            st.warning("Nenhuma partida encontrada para hoje. Verifique se o site está acessível ou tente mais tarde.")
else:
    st.info("Clique em 'Atualizar Calendário de Hoje' para carregar as partidas.")

# --- Rodapé ---
st.markdown("---")
st.caption("Dados fornecidos por FlashScore")
