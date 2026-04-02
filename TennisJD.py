import streamlit as st
import pandas as pd
from datetime import datetime
import re

# Configuração da página
st.set_page_config(
    page_title="Tênis Hoje - Challenger Tour",
    page_icon="🎾",
    layout="centered"
)

st.title("🎾 Partidas de Tênis Agendadas para Hoje")
st.caption(f"Circuito Challenger | Data: {datetime.now().strftime('%d/%m/%Y')}")

@st.cache_data
def load_and_filter_matches():
    """
    Carrega o Excel e filtra partidas agendadas para hoje
    """
    try:
        # Carregar o Excel
        df = pd.read_excel('Challenger.xlsx')
        
        # Converter a coluna de data
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
        
        # Filtrar apenas a data atual
        today = datetime.now().date()
        df_today = df[df['tourney_date'].dt.date == today].copy()
        
        # Identificar partidas sem resultado (agendadas)
        # Score vazio ou com formato de partida não iniciada
        df_today['is_scheduled'] = df_today['score'].apply(
            lambda x: pd.isna(x) or x == '' or x == 'RET' or 'RET' not in str(x)
        )
        
        # Filtrar apenas agendadas (sem resultado final)
        scheduled_matches = df_today[df_today['is_scheduled'] == True].copy()
        
        # Se não houver agendadas, mostrar todas da data com status
        if scheduled_matches.empty:
            # Mostrar todas mas marcar as que já terminaram
            df_today['status'] = df_today['score'].apply(
                lambda x: '✅ Finalizado' if pd.notna(x) and x != '' else '⏰ Agendado'
            )
            return df_today
        else:
            scheduled_matches['status'] = '⏰ Agendado'
            return scheduled_matches
            
    except Exception as e:
        st.error(f"Erro ao carregar o ficheiro: {str(e)}")
        return pd.DataFrame()

# Carregar dados
df = load_and_filter_matches()

if not df.empty:
    # Mostrar estatísticas
    st.success(f"✅ {len(df)} partida(s) encontrada(s) para hoje")
    
    # Preparar dados para exibição
    display_df = df[[
        'tourney_name', 'round', 'winner_name', 'loser_name', 
        'tourney_date', 'status', 'surface'
    ]].copy()
    
    # Renomear colunas para português
    display_df.columns = ['Torneio', 'Ronda', 'Jogador 1', 'Jogador 2', 'Data', 'Status', 'Superfície']
    
    # Formatar data
    display_df['Data'] = pd.to_datetime(display_df['Data']).dt.strftime('%d/%m/%Y')
    
    # Ordenar por torneio e ronda
    display_df = display_df.sort_values(['Torneio', 'Ronda'])
    
    # Exibir tabela
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Torneio": "🏆 Torneio",
            "Ronda": "📋 Ronda",
            "Jogador 1": "🎾 Jogador 1",
            "Jogador 2": "🎾 Jogador 2",
            "Data": "📅 Data",
            "Status": "📌 Status",
            "Superfície": "🎯 Superfície"
        }
    )
    
    # Botão para download
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar CSV",
        data=csv,
        file_name=f'challenger_hoje_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )
    
    # Exibir detalhes adicionais
    with st.expander("📊 Detalhes por Torneio"):
        tourney_stats = df.groupby('tourney_name').size().reset_index(name='Partidas')
        tourney_stats.columns = ['Torneio', 'Nº Partidas']
        st.dataframe(tourney_stats, use_container_width=True, hide_index=True)
        
else:
    st.info("📅 Nenhuma partida agendada para hoje no circuito Challenger")
    
    # Sugestão de próximos dias
    try:
        df_all = pd.read_excel('Challenger.xlsx')
        df_all['tourney_date'] = pd.to_datetime(df_all['tourney_date'], format='%Y%m%d')
        next_dates = df_all[df_all['tourney_date'].dt.date > datetime.now().date()]
        
        if not next_dates.empty:
            st.subheader("📆 Próximas partidas disponíveis:")
            next_matches = next_dates.head(5)[['tourney_name', 'tourney_date']].drop_duplicates()
            next_matches['tourney_date'] = pd.to_datetime(next_matches['tourney_date']).dt.strftime('%d/%m/%Y')
            next_matches.columns = ['Torneio', 'Data']
            st.dataframe(next_matches, use_container_width=True, hide_index=True)
    except:
        pass

# Rodapé
st.markdown("---")
st.caption("Dados do circuito Challenger | ATP Tour")
