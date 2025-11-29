import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(
    page_title="LaLiga Live - Mallorca vs Opponent",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match the screenshot style
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #ff0000, #8b0000);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .match-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #333;
        margin: 1rem 0;
    }
    .match-time {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .team-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #e9ecef;
    }
    .attack-moment {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .tab-content {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-top: 1rem;
    }
    .odds-card {
        background-color: #2ecc71;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .fullscreen-btn {
        background-color: #3498db;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        display: block;
        margin: 1rem auto;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# Header with league information
st.markdown("""
<div class="main-header">
    <h2>🇪🇸 Espanha ► LaLiga, Rodada 11+</h2>
</div>
""", unsafe_allow_html=True)

# Match information columns
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown("""
    <div class="team-card">
        <h3>RCD Mallorca</h3>
        <div style="font-size: 2rem; color: #c00;">🏠</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="match-score">
        0 - 0
    </div>
    <div class="match-time">
        ⏱️ 60:34
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="team-card">
        <h3>Real Sociedad</h3>
        <div style="font-size: 2rem; color: #00c;">✈️</div>
    </div>
    """, unsafe_allow_html=True)

# Fullscreen button
st.markdown("""
<div style="text-align: center;">
    <div class="fullscreen-btn">
        📺 VISUALIZAÇÃO EM TELA CHEIA
    </div>
</div>
""", unsafe_allow_html=True)

# Attack moment indicator
st.markdown("""
<div class="attack-moment">
    ⚡ Momento de ataque
</div>
""", unsafe_allow_html=True)

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Detalhes", "Formações", "Estatísticas", "Comentário", "Classificação"])

with tab1:
    st.markdown("""
    <div class="tab-content">
        <h4>📋 Detalhes da Partida</h4>
        
        **📍 Estádio:** Visit Mallorca Stadium<br>
        **🏆 Competição:** LaLiga EA Sports<br>
        **📅 Data:** 19 de Novembro, 2024<br>
        **⏰ Horário:** 21:00 (Local)<br>
        **👨‍⚖️ Árbitro:** José Luis Munuera<br>
        
        <h5>📊 Situação Atual:</h5>
        • Posse de Bola: Mallorca 48% - 52% Real Sociedad<br>
        • Finalizações: 8 - 12<br>
        • Finalizações no Gol: 3 - 5<br>
        • Escanteios: 4 - 6<br>
        • Faltas: 12 - 9<br>
        
        <h5>🎯 Últimos Lances:</h5>
        • 58': ⚽ Finalização perigosa do Mallorca (Fora)<br>
        • 56': 🟨 Cartão amarelo para Merino<br>
        • 53': 🔄 Substituição no Mallorca<br>
        • 49': 🥅 Defesa difícil do goleiro<br>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tab-content">
            <h4>🏠 Mallorca - 4-4-2</h4>
            
            **Goleiro:**<br>
            • 1. Rajković<br><br>
            
            **Defensores:**<br>
            • 2. Nastasić<br>
            • 21. Raíllo<br>
            • 24. Valjent<br>
            • 15. Maffeo<br><br>
            
            **Meio-campo:**<br>
            • 14. Rodríguez<br>
            • 10. Darder<br>
            • 12. Costa<br>
            • 18. Sánchez<br><br>
            
            **Atacantes:**<br>
            • 7. Muriqi<br>
            • 17. Larin<br>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tab-content">
            <h4>✈️ Real Sociedad - 4-3-3</h4>
            
            **Goleiro:**<br>
            • 1. Remiro<br><br>
            
            **Defensores:**<br>
            • 18. Hamari<br>
            • 5. Zubeldia<br>
            • 24. Le Normand<br>
            • 17. Tierney<br><br>
            
            **Meio-campo:**<br>
            • 4. Zubimendi<br>
            • 8. Merino<br>
            • 23. Méndez<br><br>
            
            **Atacantes:**<br>
            • 7. Barrenetxea<br>
            • 10. Oyarzabal<br>
            • 19. Sadiq<br>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    # Create match statistics
    stats_data = {
        'Estatística': ['Posse de Bola', 'Finalizações', 'Finalizações no Gol', 'Escanteios', 
                       'Faltas', 'Cartões Amarelos', 'Cartões Vermelhos', 'Impedimentos',
                       'Defesas', 'Cruzamentos', 'Passes Completos'],
        'Mallorca': [48, 8, 3, 4, 12, 2, 0, 1, 4, 15, 285],
        'Real Sociedad': [52, 12, 5, 6, 9, 1, 0, 2, 3, 18, 312]
    }
    
    df_stats = pd.DataFrame(stats_data)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Possession chart
        fig_possession = go.Figure(data=[
            go.Pie(labels=['Mallorca', 'Real Sociedad'], 
                  values=[48, 52], 
                  hole=.3,
                  marker_colors=['#FF6B6B', '#4ECDC4'])
        ])
        fig_possession.update_layout(title_text='Posse de Bola (%)', showlegend=True)
        st.plotly_chart(fig_possession, use_container_width=True)
    
    with col2:
        # Shots comparison
        fig_shots = go.Figure(data=[
            go.Bar(name='Mallorca', x=['Total', 'No Gol'], y=[8, 3], marker_color='#FF6B6B'),
            go.Bar(name='Real Sociedad', x=['Total', 'No Gol'], y=[12, 5], marker_color='#4ECDC4')
        ])
        fig_shots.update_layout(title_text='Finalizações', barmode='group')
        st.plotly_chart(fig_shots, use_container_width=True)
    
    # Statistics table
    st.markdown("### 📊 Estatísticas Detalhadas")
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("""
    <div class="tab-content">
        <h4>🗣️ Comentário ao Vivo</h4>
        
        <div style="background: #f0f8ff; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
            <strong>60:34</strong> - MOMENTO DE ATAQUE! Mallorca avança com perigo...<br>
            <small>⚡ Pressão ofensiva do time da casa</small>
        </div>
        
        <div style="background: #fff0f0; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
            <strong>59:12</strong> - Defesa importante da Real Sociedad!<br>
            <small>🥅 Remiro faz grande defesa</small>
        </div>
        
        <div style="background: #f8f8f8; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
            <strong>58:30</strong> - Finalização de Muriqi! Passa perto do gol...<br>
            <small>📏 A bola passa a centímetros da trave</small>
        </div>
        
        <div style="background: #f8f8f8; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
            <strong>56:45</strong> - Cartão amarelo para Merino<br>
            <small>🟨 Falta dura no meio-campo</small>
        </div>
        
        <div style="background: #f8f8f8; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
            <strong>53:20</strong> - Substituição no Mallorca<br>
            <small>🔄 Entra Sánchez, sai Rodríguez</small>
        </div>
        
        <div style="background: #f0f8ff; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
            <strong>49:15</strong> - Grande chance da Real Sociedad!<br>
            <small>🎯 Oyarzabal finaliza por cima</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab5:
    # LaLiga standings
    standings_data = {
        'Pos': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'Time': ['Real Madrid', 'Barcelona', 'Girona', 'Atlético Madrid', 'Athletic Club', 
                'Real Sociedad', 'Real Betis', 'Valencia', 'Getafe', 'Mallorca',
                'Osasuna', 'Sevilla'],
        'PJ': [11, 11, 11, 10, 11, 11, 11, 11, 11, 11, 11, 11],
        'V': [9, 8, 8, 7, 6, 5, 5, 5, 4, 3, 3, 2],
        'E': [2, 3, 1, 1, 3, 4, 3, 2, 5, 6, 3, 5],
        'D': [0, 0, 2, 2, 2, 2, 3, 4, 2, 2, 5, 4],
        'GP': [25, 22, 24, 23, 19, 18, 16, 15, 17, 13, 12, 14],
        'GC': [6, 8, 14, 11, 12, 14, 14, 13, 14, 12, 18, 16],
        'SG': [19, 14, 10, 12, 7, 4, 2, 2, 3, 1, -6, -2],
        'Pts': [29, 27, 25, 22, 21, 19, 18, 17, 17, 15, 12, 11]
    }
    
    df_standings = pd.DataFrame(standings_data)
    
    # Highlight Mallorca
    def highlight_mallorca(row):
        if row['Time'] == 'Mallorca':
            return ['background-color: #fffacd'] * len(row)
        return [''] * len(row)
    
    st.markdown("### 📈 Classificação da LaLiga")
    st.dataframe(df_standings.style.apply(highlight_mallorca, axis=1), 
                use_container_width=True, hide_index=True)

# Odds section
st.markdown("---")
st.markdown("### 🎰 Odds em destaque ►")

odds_col1, odds_col2, odds_col3, odds_col4 = st.columns(4)

with odds_col1:
    st.markdown("""
    <div class="odds-card">
        <strong>🥅 Vitória Mallorca</strong><br>
        <h3>3.25</h3>
    </div>
    """, unsafe_allow_html=True)

with odds_col2:
    st.markdown("""
    <div class="odds-card">
        <strong>🤝 Empate</strong><br>
        <h3>3.10</h3>
    </div>
    """, unsafe_allow_html=True)

with odds_col3:
    st.markdown("""
    <div class="odds-card">
        <strong>✈️ Vitória Real Sociedad</strong><br>
        <h3>2.30</h3>
    </div>
    """, unsafe_allow_html=True)

with odds_col4:
    st.markdown("""
    <div class="odds-card">
        <strong>⚽ Ambos Marcam</strong><br>
        <h3>1.85</h3>
    </div>
    """, unsafe_allow_html=True)

# Bottom match status
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 1.2rem;">
    <strong>2º tempo 0 - 0</strong>
</div>
""", unsafe_allow_html=True)

# Auto-refresh functionality
if st.button("🔄 Atualizar Dados"):
    st.rerun()

# Auto-refresh every 30 seconds
st.markdown("""
<script>
function refreshPage() {
    setTimeout(function() {
        window.location.reload();
    }, 30000); // 30 seconds
}
refreshPage();
</script>
""", unsafe_allow_html=True)
