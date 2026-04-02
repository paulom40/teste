import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
import os
import sys

# Instalar browsers do Playwright automaticamente
if not os.path.exists("/home/appuser/.cache/ms-playwright"):
    os.system("playwright install chromium")

# Configuração da página
st.set_page_config(
    page_title="Tênis Hoje - FlashScore",
    page_icon="🎾",
    layout="centered"
)

st.title("🎾 Partidas de Tênis Agendadas para Hoje")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

async def get_flashscore_matches():
    """
    Extrai as partidas agendadas do FlashScore usando Playwright
    """
    matches = []
    
    async with async_playwright() as p:
        # Lançar browser (modo headless)
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        try:
            # Acessar o FlashScore Tênis
            await page.goto("https://www.flashscore.pt/tenis/", timeout=30000)
            
            # Aguardar o carregamento inicial
            await page.wait_for_timeout(5000)
            
            # Tentar clicar na aba "Agendados"
            try:
                # FlashScore carrega as abas dinamicamente
                scheduled_tab = await page.query_selector("text=Agendados")
                if scheduled_tab:
                    await scheduled_tab.click()
                    await page.wait_for_timeout(3000)
            except:
                pass
            
            # Encontrar todos os elementos de partida
            match_elements = await page.query_selector_all(".event__match")
            
            for match in match_elements[:30]:  # Limitar a 30 partidas
                try:
                    # Extrair nome do torneio
                    tournament_elem = await match.query_selector(".event__tournament")
                    tournament = await tournament_elem.inner_text() if tournament_elem else "Torneio não informado"
                    
                    # Extrair jogador da casa
                    home_elem = await match.query_selector(".event__participant--home")
                    home = await home_elem.inner_text() if home_elem else "?"
                    
                    # Extrair jogador visitante
                    away_elem = await match.query_selector(".event__participant--away")
                    away = await away_elem.inner_text() if away_elem else "?"
                    
                    # Extrair horário/status
                    time_elem = await match.query_selector(".event__time")
                    match_time = await time_elem.inner_text() if time_elem else "Horário não informado"
                    
                    # Verificar se é uma partida agendada (não iniciada)
                    if match_time not in ["AO VIVO", "Terminado", "Cancelado"]:
                        matches.append({
                            'torneio': tournament,
                            'jogador_1': home,
                            'jogador_2': away,
                            'horario': match_time,
                            'status': 'Agendado'
                        })
                except Exception as e:
                    continue
            
            await browser.close()
            
        except Exception as e:
            st.error(f"Erro ao carregar a página: {str(e)}")
            await browser.close()
    
    return pd.DataFrame(matches)

# Botão para buscar partidas
if st.button("🔄 Buscar Partidas de Hoje", type="primary"):
    with st.spinner("Conectando ao FlashScore e buscando partidas agendadas..."):
        df = asyncio.run(get_flashscore_matches())
        
        if not df.empty:
            st.success(f"✅ {len(df)} partidas agendadas encontradas para hoje!")
            
            # Exibir tabela formatada
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "torneio": "🏆 Torneio",
                    "jogador_1": "🎾 Jogador 1",
                    "jogador_2": "🎾 Jogador 2",
                    "horario": "⏰ Horário",
                    "status": "📌 Status"
                }
            )
            
            # Botão para download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar CSV",
                data=csv,
                file_name=f'tenis_agendados_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
            )
        else:
            st.warning("⚠️ Nenhuma partida agendada encontrada para hoje.")
            st.info("💡 Dica: Tente novamente mais tarde ou verifique se o FlashScore está acessível.")
else:
    st.info("👆 Clique no botão acima para buscar as partidas agendadas para hoje no FlashScore")

# Rodapé
st.markdown("---")
st.caption("Dados fornecidos por FlashScore.pt | Atualização em tempo real")
