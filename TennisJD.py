import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
from playwright.async_api import async_playwright
import os

# ====================== CONFIGURAÇÃO ======================
st.set_page_config(
    page_title="Tênis Hoje - FlashScore + WELO",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 Partidas de Tênis Hoje + WELO")
st.caption(f"Data: {datetime.now().strftime('%d/%m/%Y')}")

# ====================== SIDEBAR - UPLOAD DO FICHEIRO ======================
with st.sidebar:
    st.header("📁 Carregar Dados WELO")
    st.markdown("Carregue o ficheiro **Challenger.xlsm**")
    
    uploaded_file = st.file_uploader(
        "Selecione o ficheiro Challenger.xlsm",
        type=["xlsm", "xlsx"],
        help="O ficheiro deve conter a aba 'Jogadores>20'"
    )
    
    st.divider()
    st.caption("O ficheiro será processado apenas quando carregado.")

# ====================== CARREGAR DADOS WELO ======================
@st.cache_data(show_spinner=False)
def load_welo_data(file):
    try:
        xls = pd.ExcelFile(file)
        df_welo = pd.read_excel(xls, sheet_name="Jogadores>20")
        
        # Normalizar nomes para matching mais robusto
        df_welo['Jogador_clean'] = (
            df_welo['Jogador']
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'[^a-z0-9\s]', '', regex=True)
            .str.replace(r'\s+', '', regex=True)
        )
        
        st.sidebar.success(f"✅ {len(df_welo)} jogadores carregados com sucesso!")
        return df_welo
    except Exception as e:
        st.sidebar.error(f"Erro ao ler o ficheiro: {e}")
        return pd.DataFrame()

# Carrega os dados se o ficheiro for enviado
df_welo = pd.DataFrame()
if uploaded_file is not None:
    df_welo = load_welo_data(uploaded_file)
else:
    st.sidebar.info("👆 Carregue o ficheiro Challenger.xlsm para ativar o cálculo de WELO")

# ====================== FUNÇÃO PARA CALCULAR WELO ======================
def calculate_welo(row, df_welo):
    if df_welo.empty:
        return 1484.0, 1484.0
    
    j1 = str(row['jogador_1']).strip().lower()
    j2 = str(row['jogador_2']).strip().lower()
    
    # Limpeza para matching
    j1_clean = ''.join(filter(str.isalnum, j1))
    j2_clean = ''.join(filter(str.isalnum, j2))
    
    superficie = str(row.get('Superficie', 'Hard')).strip().lower()
    
    surface_map = {
        'hard': 'ELO Hard',
        'clay': 'ELO Clay',
        'grass': 'ELO Grass',
        'indoor': 'ELO Indoor'
    }
    col_surface = surface_map.get(superficie)
    
    def get_player_welo(jogador_clean):
        match = df_welo[df_welo['Jogador_clean'] == jogador_clean]
        if match.empty:
            return 1484.0
        
        if col_surface and col_surface in match.columns:
            val = match[col_surface].iloc[0]
            if pd.notna(val) and val != '':
                return float(val)
        
        # Fallback: média dos ELOs disponíveis
        elo_cols = ['ELO Hard', 'ELO Clay', 'ELO Grass', 'ELO Indoor']
        available = []
        for col in elo_cols:
            if col in match.columns:
                val = match[col].iloc[0]
                if pd.notna(val) and val != '':
                    available.append(float(val))
        
        return sum(available) / len(available) if available else 1484.0
    
    welo1 = get_player_welo(j1_clean)
    welo2 = get_player_welo(j2_clean)
    
    return round(welo1, 1), round(welo2, 1)

# ====================== BUSCAR PARTIDAS ======================
async def get_flashscore_matches():
    matches = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
        context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
        page = await context.new_page()
        
        try:
            await page.goto("https://www.flashscore.pt/tenis/", timeout=30000)
            await page.wait_for_timeout(6000)
            
            try:
                scheduled_tab = await page.query_selector("text=Agendados")
                if scheduled_tab:
                    await scheduled_tab.click()
                    await page.wait_for_timeout(4000)
            except:
                pass
            
            match_elements = await page.query_selector_all(".event__match")
            
            for match in match_elements[:60]:
                try:
                    tournament = await (await match.query_selector(".event__tournament")).inner_text() if await match.query_selector(".event__tournament") else "Desconhecido"
                    home = await (await match.query_selector(".event__participant--home")).inner_text() if await match.query_selector(".event__participant--home") else "?"
                    away = await (await match.query_selector(".event__participant--away")).inner_text() if await match.query_selector(".event__participant--away") else "?"
                    time_str = await (await match.query_selector(".event__time")).inner_text() if await match.query_selector(".event__time") else "?"
                    
                    if time_str not in ["AO VIVO", "Terminado", "Cancelado", ""]:
                        matches.append({
                            'torneio': tournament.strip(),
                            'jogador_1': home.strip(),
                            'jogador_2': away.strip(),
                            'horario': time_str.strip(),
                            'status': 'Agendado'
                        })
                except:
                    continue
        except Exception as e:
            st.error(f"Erro ao acessar FlashScore: {str(e)}")
        finally:
            await browser.close()
    
    return pd.DataFrame(matches)

# ====================== INTERFACE PRINCIPAL ======================
if st.button("🔄 Buscar Partidas de Hoje e Calcular WELO", type="primary"):
    if df_welo.empty and uploaded_file is None:
        st.warning("⚠️ Por favor, carregue primeiro o ficheiro Challenger.xlsm na barra lateral.")
    else:
        with st.spinner("A aceder ao FlashScore e a calcular WELO..."):
            df_matches = asyncio.run(get_flashscore_matches())
            
            if not df_matches.empty:
                # Inferir superfície (pode ser melhorado depois)
                df_matches['Superficie'] = df_matches['torneio'].apply(
                    lambda x: 'Clay' if any(word in str(x).lower() for word in ['clay', 'saibro', 'challenger clay']) else
                              'Grass' if 'grass' in str(x).lower() else 'Hard'
                )
                
                # Calcular WELO
                welo_results = df_matches.apply(lambda row: calculate_welo(row, df_welo), axis=1)
                df_matches['WELO_J1'] = [w[0] for w in welo_results]
                df_matches['WELO_J2'] = [w[1] for w in welo_results]
                
                st.success(f"✅ {len(df_matches)} partidas encontradas!")
                
                # Mostrar tabela
                st.dataframe(
                    df_matches,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "torneio": "🏆 Torneio",
                        "jogador_1": "🎾 Jogador 1",
                        "jogador_2": "🎾 Jogador 2",
                        "horario": "⏰ Horário",
                        "Superficie": "🏟️ Superfície",
                        "WELO_J1": st.column_config.NumberColumn("WELO Jogador 1", format="%.1f"),
                        "WELO_J2": st.column_config.NumberColumn("WELO Jogador 2", format="%.1f"),
                        "status": "📌 Status"
                    }
                )
                
                # ====================== DOWNLOADS ======================
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = df_matches.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Baixar como CSV",
                        data=csv,
                        file_name=f"tenis_hoje_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Exportar para Excel
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_matches.to_excel(writer, index=False, sheet_name='Partidas')
                    output.seek(0)
                    
                    st.download_button(
                        label="📊 Baixar como Excel (.xlsx)",
                        data=output,
                        file_name=f"tenis_hoje_com_welo_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("Não foram encontradas partidas agendadas no momento.")

else:
    st.info("👈 Carregue o ficheiro na barra lateral e clique no botão acima para começar.")

st.markdown("---")
st.caption("FlashScore.pt • WELO calculado a partir do Challenger.xlsm")
