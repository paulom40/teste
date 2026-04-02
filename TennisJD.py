import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from io import BytesIO
import unicodedata

st.title("🎾 Tênis Hoje - WELO (Sem Playwright)")

with st.sidebar:
    uploaded_file = st.file_uploader("Carregue Challenger.xlsm", type=["xlsm", "xlsx"])

# Carregar WELO (mesma função de antes)
@st.cache_data
def load_welo_data(file):
    # ... (mesma função normalize_name e load_welo_data que te dei antes)
    pass   # mantenha a função que já tem

df_welo = pd.DataFrame()
if uploaded_file:
    df_welo = load_welo_data(uploaded_file)

# ====================== SCRAPING LEVE COM REQUESTS ======================
def get_flashscore_matches_requests():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        r = requests.get("https://www.flashscore.pt/tenis/", headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        matches = []
        # Este seletor pode precisar de ajuste com o tempo
        games = soup.find_all("div", class_="event__match")
        
        for game in games[:60]:
            try:
                tournament = game.find("div", class_="event__tournament")
                tour_name = tournament.get_text(strip=True) if tournament else "Desconhecido"
                
                home = game.find("div", class_="event__participant--home")
                j1 = home.get_text(strip=True) if home else "?"
                
                away = game.find("div", class_="event__participant--away")
                j2 = away.get_text(strip=True) if away else "?"
                
                time_el = game.find("div", class_="event__time")
                horario = time_el.get_text(strip=True) if time_el else "?"
                
                if horario not in ["AO VIVO", "Terminado", "Cancelado"]:
                    superficie = "Clay" if any(x in tour_name.lower() for x in ['clay','saibro','kigali','santiago']) else "Hard"
                    matches.append({
                        'torneio': tour_name,
                        'jogador_1': j1,
                        'jogador_2': j2,
                        'horario': horario,
                        'superficie': superficie
                    })
            except:
                continue
                
        return pd.DataFrame(matches)
    except Exception as e:
        st.error(f"Erro no scraping: {e}")
        return pd.DataFrame()

# ====================== RESTO DO CÓDIGO (WELO + Linha Total) ======================
# ... (mantenha as funções get_player_welo, calcular_linha_total, detect_surface que te dei antes)

if st.button("🔄 Tentar Buscar Jogos (requests)"):
    with st.spinner("Tentando buscar jogos..."):
        df = get_flashscore_matches_requests()
        if not df.empty:
            # calcular WELO, Total, etc.
            df['WELO_J1'] = df.apply(lambda row: get_player_welo(row['jogador_1'], row['superficie'], df_welo), axis=1)
            # ... resto igual
            st.dataframe(df)
        else:
            st.warning("Não conseguiu extrair jogos automaticamente.")

# Opção manual sempre disponível
st.subheader("Ou cole os jogos manualmente (recomendado)")
# ... (mesma text_area que te dei antes)
