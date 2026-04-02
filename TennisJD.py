import streamlit as st
import pandas as pd
from datetime import datetime
import time
import re
from urllib.request import urlopen
from bs4 import BeautifulSoup
import json

# Tentar importar bibliotecas opcionais
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    st.warning("⚠️ Playwright não instalado. Usando método alternativo...")

def get_tennis_matches_alternative():
    """
    Método alternativo usando requests + BeautifulSoup
    (mais simples, mas pode ser bloqueado)
    """
    matches = []
    
    try:
        # URL do FlashScore Tênis
        url = "https://www.flashscore.pt/tenis/"
        
        # Headers para simular um navegador
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = urlopen(url)
        html = response.read().decode('utf-8')
        
        # Buscar por padrões de jogadores no HTML
        # (FlashScore carrega dados via JavaScript, então isso é limitado)
        player_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+)\s+vs\s+([A-Z][a-z]+ [A-Z][a-z]+)'
        found_matches = re.findall(player_pattern, html)
        
        for player1, player2 in found_matches[:20]:  # Limitar a 20 partidas
            matches.append({
                'torneio': 'Tênis Profissional',
                'jogador_1': player1,
                'jogador_2': player2,
                'data_hora': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'status': 'Agendado'
            })
            
        return pd.DataFrame(matches)
        
    except Exception as e:
        st.error(f"Erro no método alternativo: {str(e)}")
        return pd.DataFrame()

def get_tennis_matches_playwright():
    """
    Método usando Playwright (mais robusto, mas requer instalação)
    """
    if not PLAYWRIGHT_AVAILABLE:
        return pd.DataFrame()
    
    matches = []
    
    try:
        with sync_playwright() as p:
            # Usar Chromium
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = context.new_page()
            
            # Ir para a página de tênis
            page.goto("https://www.flashscore.pt/tenis/", timeout=30000)
            
            # Esperar um pouco para o JavaScript carregar
            time.sleep(3)
            
            # Tentar encontrar os elementos das partidas
            # (seletores podem mudar - é preciso ajustar)
            try:
                # Clicar na aba "Agendados" se existir
                scheduled_tab = page.query_selector("text=Agendados")
                if scheduled_tab:
                    scheduled_tab.click()
                    time.sleep(2)
            except:
                pass
            
            # Extrair informações das partidas
            matches_elements = page.query_selector_all(".event__match")
            
            for element in matches_elements[:30]:  # Limitar a 30 partidas
                try:
                    home_team = element.query_selector(".event__homeParticipant")
                    away_team = element.query_selector(".event__awayParticipant")
                    status = element.query_selector(".event__status")
                    tournament = element.query_selector(".event__tournament")
                    
                    if home_team and away_team:
                        matches.append({
                            'torneio': tournament.inner_text() if tournament else 'Não informado',
                            'jogador_1': home_team.inner_text(),
                            'jogador_2': away_team.inner_text(),
                            'data_hora': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'status': status.inner_text() if status else 'Agendado'
                        })
                except Exception as e:
                    continue
            
            browser.close()
            
    except Exception as e:
        st.error(f"Erro no Playwright: {str(e)}")
        return pd.DataFrame()
    
    return pd.DataFrame(matches)

def get_tennis_matches_today():
    """
    Função principal - tenta múltiplos métodos
    """
    st.info("🔍 Buscando partidas de tênis...")
    
    # Método 1: Playwright (mais confiável)
    if PLAYWRIGHT_AVAILABLE:
        st.text("📡 Método 1: Usando Playwright...")
        df = get_tennis_matches_playwright()
        if not df.empty:
            return df
    
    # Método 2: Alternativo (mais simples)
    st.text("📡 Método 2: Usando método alternativo...")
    df = get_tennis_matches_alternative()
    
    if df.empty:
        # Dados de exemplo para demonstração
        st.warning("⚠️ Não foi possível obter dados ao vivo. Mostrando dados de exemplo.")
        df = pd.DataFrame([
            {'torneio': 'ATP Masters 1000 Monte Carlo', 'jogador_1': 'Novak Djokovic', 'jogador_2': 'Carlos Alcaraz', 'data_hora': '2026-04-02 14:30', 'status': 'Agendado'},
            {'torneio': 'WTA 1000 Madrid', 'jogador_1': 'Iga Swiatek', 'jogador_2': 'Elena Rybakina', 'data_hora': '2026-04-02 16:00', 'status': 'Agendado'},
            {'torneio': 'ATP 500 Barcelona', 'jogador_1': 'Daniil Medvedev', 'jogador_2': 'Jannik Sinner', 'data_hora': '2026-04-02 18:30', 'status': 'Agendado'},
        ])
    
    return df

# --- Interface Streamlit ---
st.set_page_config(
    page_title="Tênis Hoje - FlashScore",
    page_icon="🎾",
    layout="centered"
)

# --- Título ---
st.title("🎾 Partidas de Tênis - Hoje")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

# --- Botão de Atualização ---
if st.button("🔄 Atualizar Calendário de Hoje", type="primary"):
    with st.spinner("Buscando dados no FlashScore... Isso pode levar alguns segundos."):
        df = get_tennis_matches_today()
        
        if not df.empty:
            st.success(f"✅ {len(df)} partidas encontradas!")
            
            # Exibir tabela
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "torneio": "🏆 Torneio",
                    "jogador_1": "🎾 Jogador 1",
                    "jogador_2": "🎾 Jogador 2",
                    "data_hora": "📅 Data/Hora",
                    "status": "📊 Status"
                }
            )
            
            # Botão para download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar como CSV",
                data=csv,
                file_name=f'tenis_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
            )
        else:
            st.error("❌ Nenhuma partida encontrada. O site pode estar bloqueando a requisição.")
            st.info("💡 Dica: O FlashScore protege seus dados. Considere usar uma API paga ou serviço especializado.")

# --- Informações Adicionais ---
with st.expander("ℹ️ Sobre este app"):
    st.markdown("""
    **Como funciona:**
    - Este app tenta extrair dados de partidas de tênis do FlashScore
    - Usa múltiplos métodos para tentar obter os dados
    - O FlashScore tem proteção anti-bot, então o sucesso pode variar
    
    **Limitações:**
    - Pode não funcionar sempre devido a bloqueios do site
    - Para uso profissional, considere APIs oficiais
    
    **Alternativas:**
    - [API do SportsRadar](https://developer.sportradar.com/)
    - [API do TheSportsDB](https://www.thesportsdb.com/)
    """)

st.markdown("---")
st.caption("Dados de demonstração | Para dados reais, é necessário ajustar os seletores ou usar uma API oficial")
