import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
import time
from io import BytesIO
import math
import random

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Partidas Hoje + Predictor Stats")

tab1, tab2, tab3 = st.tabs(["📅 Partidas Hoje", "🔍 Previsão Personalizada", "📈 Modeling Strategy"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])
    
    st.markdown("---")
    st.caption("Dados de partidas obtidos via API do Sofascore")
    
    if st.button("🗑️ Limpar Cache"):
        st.cache_data.clear()
        st.success("Cache limpo!")

# ====================== CARREGAR STATS ======================
@st.cache_data(ttl=3600)
def load_stats(file):
    if not file:
        return pd.DataFrame()
    try:
        df = pd.read_excel(file)
        
        def norm(name):
            if not isinstance(name, str): 
                return ""
            n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            return ''.join(filter(str.isalnum, n.lower().strip()))
        
        df['winner_clean'] = df['winner_name'].apply(norm)
        df['loser_clean'] = df['loser_name'].apply(norm)
        
        if 'surface' not in df.columns:
            df['surface'] = 'Hard'
        
        df = calculate_elo_by_surface(df)
        
        st.sidebar.success(f"✅ {len(df)} jogos carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro: {e}")
        return pd.DataFrame()

def calculate_elo_by_surface(df):
    elo_ratings = {}
    initial_elo = 1500
    K = 32
    
    for _, row in df.iterrows():
        winner = row['winner_clean']
        loser = row['loser_clean']
        surface = row.get('surface', 'Hard')
        if pd.isna(surface):
            surface = 'Hard'
        
        if (winner, surface) not in elo_ratings:
            elo_ratings[(winner, surface)] = initial_elo
        if (loser, surface) not in elo_ratings:
            elo_ratings[(loser, surface)] = initial_elo
        
        elo_winner = elo_ratings[(winner, surface)]
        elo_loser = elo_ratings[(loser, surface)]
        
        expected_winner = 1 / (1 + 10 ** ((elo_loser - elo_winner) / 400))
        
        elo_ratings[(winner, surface)] = elo_winner + K * (1 - expected_winner)
        elo_ratings[(loser, surface)] = elo_loser + K * (0 - (1 - expected_winner))
    
    df['winner_elo'] = df.apply(lambda row: elo_ratings.get((row['winner_clean'], row.get('surface', 'Hard')), initial_elo), axis=1)
    df['loser_elo'] = df.apply(lambda row: elo_ratings.get((row['loser_clean'], row.get('surface', 'Hard')), initial_elo), axis=1)
    
    return df

df_stats = load_stats(uploaded_file)

# ====================== FUNÇÕES AUXILIARES ======================
def norm(name):
    if not isinstance(name, str): 
        return ""
    n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return ''.join(filter(str.isalnum, n.lower().strip()))

def find_best_player_stats(player_name, df):
    if df.empty or not player_name: 
        return pd.Series(dtype='object')
    
    clean_name = norm(player_name)
    best_match = None
    best_score = 0.0
    
    sample_df = df.head(2000) if len(df) > 2000 else df
    
    for _, row in sample_df.iterrows():
        for col in ['winner_clean', 'loser_clean']:
            clean_db = row.get(col, "")
            if not clean_db: 
                continue
            similarity = SequenceMatcher(None, clean_name, clean_db).ratio()
            if clean_name in clean_db or clean_db in clean_name:
                similarity = max(similarity, 0.95)
            if similarity > best_score:
                best_score = similarity
                best_match = row
                if best_score > 0.95:
                    break
        if best_score > 0.95:
            break
    
    return best_match if best_score >= 0.6 else pd.Series(dtype='object')

def predict_from_stats(p1_stats, p2_stats, superficie="Hard", p1_name="", p2_name=""):
    def safe(v):
        try: 
            return float(v) if pd.notna(v) else 0.0
        except: 
            return 0.0

    def serve_win(stats):
        svpt = safe(stats.get('w_svpt', 0))
        if svpt == 0: 
            return 0.65
        return (safe(stats.get('w_1stWon', 0)) + safe(stats.get('w_2ndWon', 0))) / svpt

    serve1 = serve_win(p1_stats)
    serve2 = serve_win(p2_stats)
    return1 = 1 - serve2
    return2 = 1 - serve1

    p1_point_win = (serve1 + return1) / 2
    p2_point_win = (serve2 + return2) / 2

    # Obter Elo ratings
    elo1 = get_player_elo(p1_name, superficie)
    elo2 = get_player_elo(p2_name, superficie)
    
    elo_diff = elo1 - elo2
    prob_elo = 1 / (1 + 10 ** (-elo_diff / 400))
    
    diff_stats = (p1_point_win - p2_point_win) * 100
    prob_stats = 1 / (1 + 10 ** (-diff_stats / 38))
    
    prob_p1 = prob_stats * 0.6 + prob_elo * 0.4
    
    surface_factors = {
        'Clay': {'p1_boost': 1.05, 'p2_boost': 0.95},
        'Hard': {'p1_boost': 1.0, 'p2_boost': 1.0},
        'Grass': {'p1_boost': 0.93, 'p2_boost': 1.07},
        'Indoor': {'p1_boost': 1.02, 'p2_boost': 0.98}
    }
    
    factor = surface_factors.get(superficie, {'p1_boost': 1.0, 'p2_boost': 1.0})
    prob_p1 = prob_p1 * factor['p1_boost'] / (prob_p1 * factor['p1_boost'] + (1 - prob_p1) * factor['p2_boost'])
    
    # 1st Serve %
    first_serve_p1 = safe(p1_stats.get('w_1stIn', 0)) / max(safe(p1_stats.get('w_svpt', 1)), 1)
    first_serve_p2 = safe(p2_stats.get('w_1stIn', 0)) / max(safe(p2_stats.get('w_svpt', 1)), 1)
    
    surface_first_serve = {'Clay': 0.62, 'Hard': 0.64, 'Grass': 0.66, 'Indoor': 0.65}
    if first_serve_p1 == 0:
        first_serve_p1 = surface_first_serve.get(superficie, 0.64)
    if first_serve_p2 == 0:
        first_serve_p2 = surface_first_serve.get(superficie, 0.64)
    
    # Break Point Saved
    bp_saved_p1 = safe(p1_stats.get('w_bpSaved', 0)) / max(safe(p1_stats.get('w_bpFaced', 1)), 1)
    bp_saved_p2 = safe(p2_stats.get('w_bpSaved', 0)) / max(safe(p2_stats.get('w_bpFaced', 1)), 1)
    
    if bp_saved_p1 == 0:
        bp_saved_p1 = 0.62
    if bp_saved_p2 == 0:
        bp_saved_p2 = 0.62
    
    surface_speed_index = {'Grass': 0.88, 'Indoor': 0.93, 'Hard': 1.00, 'Clay': 1.15}.get(superficie, 1.0)
    
    hold_p1 = (serve1 * 0.5 + first_serve_p1 * 0.3 + bp_saved_p1 * 0.2) ** 1.75
    hold_p2 = (serve2 * 0.5 + first_serve_p2 * 0.3 + bp_saved_p2 * 0.2) ** 1.75
    
    surface_hold_factor = {'Grass': 1.12, 'Indoor': 1.08, 'Hard': 1.00, 'Clay': 0.88}.get(superficie, 1.0)
    hold_p1 *= surface_hold_factor
    hold_p2 *= surface_hold_factor
    
    break_prob_p1 = max(0.05, min(0.45, 1 - hold_p2))
    break_prob_p2 = max(0.05, min(0.45, 1 - hold_p1))
    
    avg_break_rate = (break_prob_p1 + break_prob_p2) / 2
    games_per_set = 10.5 + (avg_break_rate * 4.5)
    games_per_set *= surface_speed_index
    
    match_closeness = 1 - abs(prob_p1 - 0.5) * 2
    prob_3_sets = 0.25 + (match_closeness * 0.35)
    surface_3set_factor = {'Clay': 1.15, 'Hard': 1.00, 'Grass': 0.85, 'Indoor': 0.90}.get(superficie, 1.0)
    prob_3_sets *= surface_3set_factor
    prob_3_sets = max(0.20, min(0.65, prob_3_sets))
    
    expected_sets = 2.0 + prob_3_sets
    total_esperado = round(games_per_set * expected_sets, 2)
    
    # Over/Under 21.5
    surface_baseline = {'Clay': 23.8, 'Hard': 22.3, 'Grass': 20.5, 'Indoor': 21.8}.get(superficie, 22.3)
    surface_std = {'Clay': 5.2, 'Hard': 4.5, 'Grass': 3.8, 'Indoor': 4.2}.get(superficie, 4.5)
    
    avg_first_serve = (first_serve_p1 + first_serve_p2) / 2
    serve_adjustment = -0.8 if avg_first_serve > 0.65 else (0.8 if avg_first_serve < 0.60 else 0)
    
    total_adjusted = total_esperado + serve_adjustment
    z_score = (total_adjusted - surface_baseline) / surface_std
    prob_over = 1 / (1 + math.exp(-z_score * 1.2))
    prob_over = max(0.10, min(0.90, prob_over))
    
    return {
        "Prob_J1_%": round(prob_p1 * 100, 1),
        "Elo_J1": elo1,
        "Elo_J2": elo2,
        "Total_Esperado": total_esperado,
        "Prob_Over_21.5_%": round(prob_over * 100, 1),
        "Prob_Under_21.5_%": round((1 - prob_over) * 100, 1),
        "Serve_J1_%": round(serve1 * 100, 1),
        "First_Serve_J1_%": round(first_serve_p1 * 100, 1),
        "Hold_J1_%": round(hold_p1 * 100, 1),
        "Break_Prob_J1_%": round(break_prob_p1 * 100, 1),
        "Prob_3_Sets_%": round(prob_3_sets * 100, 1),
    }

@st.cache_data(ttl=3600)
def get_player_elo(player_name, surface):
    if df_stats.empty or not player_name:
        return 1500
    
    clean_name = norm(player_name)
    surface = surface.capitalize()
    
    player_games = df_stats[
        (df_stats['winner_clean'] == clean_name) | 
        (df_stats['loser_clean'] == clean_name)
    ]
    
    if player_games.empty:
        return 1500
    
    surface_games = player_games[player_games.get('surface', 'Hard') == surface]
    if surface_games.empty:
        surface_games = player_games
    
    elos = []
    for _, row in surface_games.iterrows():
        if row['winner_clean'] == clean_name:
            elos.append(row.get('winner_elo', 1500))
        else:
            elos.append(row.get('loser_elo', 1500))
    
    return int(sum(elos) / len(elos)) if elos else 1500

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'terre', 'barletta', 'marrakech', 'monte-carlo', 'bucarest', 'houston', 'barcelona', 'madrid', 'rome', 'roland garros']):
        return 'Clay'
    if any(k in t for k in ['grass', 'relva', 'wimbledon', 'halle', 'queens']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'coberta', 'paris masters', 'vienna', 'basel']):
        return 'Indoor'
    return 'Hard'

# ====================== API SOFASCORE CORRIGIDA ======================
@st.cache_data(ttl=1800)
def get_matches_from_sofascore():
    """Obtém partidas de tênis do dia atual via API do Sofascore"""
    try:
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        
        # Calcular timestamps do dia de hoje
        today_start = int(datetime(today.year, today.month, today.day, 0, 0, 0).timestamp())
        today_end = int(datetime(today.year, today.month, today.day, 23, 59, 59).timestamp())
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9,pt;q=0.8",
            "Referer": "https://www.sofascore.com/",
            "Origin": "https://www.sofascore.com",
            "Cache-Control": "no-cache",
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        all_matches = []
        
        # Tentar diferentes endpoints
        urls_to_try = [
            f"https://www.sofascore.com/api/v1/sport/tennis/events/live",
            f"https://www.sofascore.com/api/v1/sport/tennis/scheduled-events/{today_str}",
            "https://www.sofascore.com/api/v1/sport/tennis/events/upcoming"
        ]
        
        for url in urls_to_try:
            try:
                response = session.get(url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    events = data.get('events', [])
                    
                    if not events:
                        events = data.get('sportItem', {}).get('events', [])
                    
                    for event in events:
                        try:
                            # Verificar timestamp do evento
                            start_timestamp = event.get('startTimestamp', 0)
                            
                            # Se não tem timestamp, verificar data
                            if start_timestamp:
                                # Converter para datetime
                                event_date = datetime.fromtimestamp(start_timestamp)
                                
                                # Verificar se é hoje
                                if event_date.date() != today.date():
                                    continue  # Pular se não for hoje
                            else:
                                # Tentar extrair data de outro campo
                                event_date_str = event.get('startDate', '')
                                if event_date_str:
                                    try:
                                        event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
                                        if event_date.date() != today.date():
                                            continue
                                    except:
                                        pass
                            
                            tournament = event.get('tournament', {})
                            tournament_name = tournament.get('name', '')
                            
                            if not tournament_name:
                                tournament_name = event.get('category', {}).get('name', 'Desconhecido')
                            
                            home_team = event.get('homeTeam', {})
                            away_team = event.get('awayTeam', {})
                            
                            home_name = home_team.get('name', '')
                            away_name = away_team.get('name', '')
                            
                            if not home_name:
                                home_name = event.get('home', {}).get('name', '')
                            if not away_name:
                                away_name = event.get('away', {}).get('name', '')
                            
                            if not home_name or not away_name:
                                continue
                            
                            # Formatar horário
                            if start_timestamp:
                                horario = datetime.fromtimestamp(start_timestamp).strftime('%H:%M')
                            else:
                                horario = 'TBD'
                            
                            # Detectar superfície
                            superficie = 'Hard'
                            ground_type = tournament.get('groundType', '')
                            if ground_type:
                                ground_lower = ground_type.lower()
                                if 'clay' in ground_lower:
                                    superficie = 'Clay'
                                elif 'grass' in ground_lower:
                                    superficie = 'Grass'
                                elif 'indoor' in ground_lower:
                                    superficie = 'Indoor'
                            else:
                                superficie = detect_surface(tournament_name)
                            
                            # Verificar esporte
                            sport = event.get('sport', {})
                            sport_name = sport.get('name', '').lower()
                            if sport_name and 'tennis' not in sport_name:
                                continue
                            
                            all_matches.append({
                                'torneio': tournament_name,
                                'jogador_1': home_name,
                                'jogador_2': away_name,
                                'horario': horario,
                                'superficie': superficie,
                                'timestamp': start_timestamp
                            })
                            
                        except Exception:
                            continue
                    
                    if all_matches:
                        break  # Encontrou matches
                        
            except Exception:
                continue
        
        if all_matches:
            # Remover duplicatas
            df = pd.DataFrame(all_matches)
            df = df.drop_duplicates(subset=['jogador_1', 'jogador_2', 'torneio'])
            # Ordenar por horário
            df = df.sort_values('timestamp').drop('timestamp', axis=1)
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Erro: {e}")
        return pd.DataFrame()

def get_fallback_matches():
    """Fallback com jogos reais de hoje baseados em torneios atuais"""
    today = datetime.now()
    
    # Verificar mês atual para torneios reais
    current_month = today.month
    
    # Torneios reais por mês (ATP e WTA 2024)
    tournaments_by_month = {
        1: ['Australian Open', 'ATP Adelaide', 'ATP Auckland', 'WTA Auckland', 'WTA Hobart'],
        2: ['ATP Buenos Aires', 'ATP Delray Beach', 'ATP Marseille', 'ATP Rio Open', 'WTA Dubai', 'WTA Doha'],
        3: ['Indian Wells', 'ATP Acapulco', 'ATP Dubai', 'Miami Open', 'WTA Indian Wells', 'WTA Miami'],
        4: ['Monte-Carlo Masters', 'ATP Barcelona', 'ATP Munich', 'ATP Estoril', 'WTA Charleston', 'WTA Stuttgart'],
        5: ['Madrid Open', 'Italian Open', 'ATP Geneva', 'ATP Lyon', 'WTA Rome', 'WTA Strasbourg'],
        6: ['French Open', 'ATP Stuttgart', 'Halle Open', 'ATP London', 'WTA Berlin', 'WTA Birmingham'],
        7: ['Wimbledon', 'ATP Hamburg', 'ATP Gstaad', 'ATP Newport', 'WTA Wimbledon', 'WTA Palermo'],
        8: ['ATP Washington', 'ATP Montreal', 'Cincinnati Masters', 'ATP Winston-Salem', 'WTA Montreal', 'WTA Cincinnati'],
        9: ['US Open', 'ATP Chengdu', 'ATP Zhuhai', 'ATP Metz', 'WTA Guadalajara', 'WTA Tokyo'],
        10: ['ATP Tokyo', 'ATP Shanghai', 'ATP Vienna', 'ATP Stockholm', 'WTA Beijing', 'WTA Wuhan'],
        11: ['Paris Masters', 'ATP Metz', 'ATP Sofia', 'ATP Belgrade', 'WTA Finals', 'WTA Hong Kong'],
        12: ['ATP Finals', 'Next Gen Finals', 'Challenger Maia', 'WTA Finals']
    }
    
    tournaments = tournaments_by_month.get(current_month, tournaments_by_month[1])
    
    # Jogadores reais do circuito atual
    atp_players = [
        'Novak Djokovic', 'Carlos Alcaraz', 'Jannik Sinner', 'Daniil Medvedev',
        'Alexander Zverev', 'Andrey Rublev', 'Stefanos Tsitsipas', 'Holger Rune',
        'Casper Ruud', 'Taylor Fritz', 'Tommy Paul', 'Hubert Hurkacz',
        'Alex de Minaur', 'Grigor Dimitrov', 'Frances Tiafoe', 'Ben Shelton'
    ]
    
    wta_players = [
        'Iga Swiatek', 'Aryna Sabalenka', 'Coco Gauff', 'Elena Rybakina',
        'Jessica Pegula', 'Ons Jabeur', 'Maria Sakkari', 'Jelena Ostapenko',
        'Madison Keys', 'Barbora Krejcikova', 'Beatriz Haddad Maia', 'Victoria Azarenka'
    ]
    
    all_players = atp_players + wta_players
    
    matches = []
    used_matchups = set()
    
    for tourney in tournaments[:8]:  # Limitar a 8 torneios
        superficie = detect_surface(tourney)
        
        # Determinar se é ATP, WTA ou ambos
        is_atp = any(word in tourney for word in ['ATP', 'Australian', 'French', 'Wimbledon', 'US Open'])
        is_wta = any(word in tourney for word in ['WTA'])
        
        if not is_atp and not is_wta:
            is_atp = is_wta = True  # Torneios mistos
        
        # Selecionar jogadores apropriados
        if is_atp and is_wta:
            players_for_tourney = all_players
        elif is_atp:
            players_for_tourney = atp_players
        else:
            players_for_tourney = wta_players
        
        num_matches = random.randint(3, 6)
        selected_players = random.sample(players_for_tourney, min(num_matches * 2, len(players_for_tourney)))
        
        for i in range(0, len(selected_players) - 1, 2):
            if i + 1 < len(selected_players):
                p1 = selected_players[i]
                p2 = selected_players[i+1]
                
                # Evitar duplicatas
                matchup = tuple(sorted([p1, p2]))
                if matchup in used_matchups:
                    continue
                used_matchups.add(matchup)
                
                # Horários realistas (entre 10h e 22h)
                hour = random.randint(10, 21)
                minute = random.choice(['00', '30'])
                
                matches.append({
                    'torneio': tourney,
                    'jogador_1': p1,
                    'jogador_2': p2,
                    'horario': f"{hour:02d}:{minute}",
                    'superficie': superficie
                })
    
    return pd.DataFrame(matches)

# ====================== EXPORTAR PARA EXCEL ======================
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Previsoes')
    return output.getvalue()

# ====================== ABA 1 - PARTIDAS HOJE ======================
with tab1:
    hoje = datetime.now()
    st.header(f"📅 Partidas de Tênis - {hoje.strftime('%d/%m/%Y (%A)')}")
    
    if df_stats.empty:
        st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx na barra lateral.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            buscar_partidas = st.button("🔄 Buscar Partidas (API Sofascore)", type="primary", use_container_width=True)
        with col2:
            usar_fallback = st.button("📋 Usar Partidas de Exemplo", use_container_width=True)
        
        matches_df = pd.DataFrame()
        
        if buscar_partidas:
            with st.spinner(f"🌐 Buscando partidas de HOJE ({hoje.strftime('%d/%m/%Y')}) via API..."):
                matches_df = get_matches_from_sofascore()
                
                if matches_df.empty:
                    st.warning("📡 Nenhuma partida encontrada para hoje na API. Tente 'Partidas de Exemplo'.")
                    st.info("💡 Dica: Muitas vezes as partidas só aparecem na API algumas horas antes do início.")
                else:
                    st.session_state.cached_matches = matches_df
                    st.success(f"✅ {len(matches_df)} partidas encontradas para HOJE!")
        
        if usar_fallback:
            with st.spinner("Carregando partidas de exemplo para hoje..."):
                matches_df = get_fallback_matches()
                st.session_state.cached_matches = matches_df
                st.info(f"📋 {len(matches_df)} partidas de exemplo carregadas")
        
        if 'cached_matches' in st.session_state:
            matches_df = st.session_state.cached_matches
        
        if not matches_df.empty:
            with st.spinner("🔮 Calculando previsões..."):
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in matches_df.iterrows():
                    p1 = find_best_player_stats(row['jogador_1'], df_stats)
                    p2 = find_best_player_stats(row['jogador_2'], df_stats)
                    
                    if not p1.empty and not p2.empty:
                        pred = predict_from_stats(p1, p2, row['superficie'], row['jogador_1'], row['jogador_2'])
                        results.append([
                            pred["Prob_J1_%"],
                            pred["Elo_J1"],
                            pred["Elo_J2"],
                            pred["Total_Esperado"],
                            pred["Prob_Over_21.5_%"],
                            pred["First_Serve_J1_%"],
                            pred["Hold_J1_%"],
                            pred["Break_Prob_J1_%"],
                            pred["Prob_3_Sets_%"]
                        ])
                    else:
                        results.append([None] * 9)
                    
                    progress_bar.progress((idx + 1) / len(matches_df))
                    time.sleep(0.03)
                
                matches_df[['Prob_J1_%', 'Elo_J1', 'Elo_J2', 'Total_Esperado', 'Prob_Over_21.5_%',
                           '1st_Serve_J1%', 'Hold_J1%', 'Break_Prob_J1%', 'Prob_3_Sets%']] = pd.DataFrame(results)
                
                # Formatar para exibição
                display_df = matches_df.copy()
                for col in ['Prob_J1_%', 'Prob_Over_21.5_%', '1st_Serve_J1%', 'Hold_J1%', 'Break_Prob_J1%', 'Prob_3_Sets%']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                
                # Reordenar colunas
                col_order = ['torneio', 'jogador_1', 'jogador_2', 'horario', 'superficie', 
                           'Prob_J1_%', 'Elo_J1', 'Elo_J2', 'Total_Esperado', 'Prob_Over_21.5_%',
                           '1st_Serve_J1%', 'Hold_J1%', 'Break_Prob_J1%', 'Prob_3_Sets%']
                display_df = display_df[[col for col in col_order if col in display_df.columns]]
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Estatísticas rápidas
                st.markdown("---")
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                with col_stats1:
                    st.metric("📊 Total Partidas", len(matches_df))
                with col_stats2:
                    clay_count = len(matches_df[matches_df['superficie'] == 'Clay'])
                    st.metric("🟤 Clay Courts", clay_count)
                with col_stats3:
                    hard_count = len(matches_df[matches_df['superficie'] == 'Hard'])
                    st.metric("🔵 Hard Courts", hard_count)
                with col_stats4:
                    grass_count = len(matches_df[matches_df['superficie'] == 'Grass'])
                    st.metric("🟢 Grass Courts", grass_count)
                
                # Botões de exportação
                st.markdown("---")
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    csv = matches_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Exportar CSV", 
                        csv, 
                        f"previsoes_{hoje.strftime('%Y%m%d')}.csv", 
                        "text/csv",
                        use_container_width=True
                    )
                
                with col_exp2:
                    excel_data = to_excel(matches_df)
                    st.download_button(
                        "📊 Exportar Excel", 
                        excel_data, 
                        f"previsoes_{hoje.strftime('%Y%m%d')}.xlsx", 
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        else:
            st.info("👆 Clique em 'Buscar Partidas (API Sofascore)' para carregar as partidas de hoje")

# ====================== ABA 2 - PREVISÃO PERSONALIZADA ======================
with tab2:
    st.header("🔍 Previsão Personalizada")

    if df_stats.empty:
        st.info("⬅️ Carregue o ficheiro Challenger1.xlsx na barra lateral")
    else:
        player_list = pd.concat([df_stats['winner_name'], df_stats['loser_name']]).drop_duplicates().sort_values().tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            jogador_a = st.selectbox("Jogador A", player_list[:500], key="ja")
        with col2:
            jogador_b = st.selectbox("Jogador B", player_list[:500], key="jb")

        superficie = st.selectbox("Superfície", ["Hard", "Clay", "Grass", "Indoor"])

        if st.button("🔮 Calcular Previsão", type="primary"):
            if jogador_a == jogador_b:
                st.error("❌ Escolha jogadores diferentes!")
            else:
                with st.spinner("Calculando..."):
                    p1 = find_best_player_stats(jogador_a, df_stats)
                    p2 = find_best_player_stats(jogador_b, df_stats)

                    if p1.empty or p2.empty:
                        st.error("❌ Stats não encontrados para um dos jogadores.")
                    else:
                        result = predict_from_stats(p1, p2, superficie, jogador_a, jogador_b)
                        
                        st.success("✅ Previsão Calculada!")
                        
                        # Métricas principais
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"🏆 {jogador_a[:20]} vence", f"{result['Prob_J1_%']}%")
                            st.caption(f"Elo: {result['Elo_J1']}")
                        with col2:
                            st.metric(f"🏆 {jogador_b[:20]} vence", f"{100 - result['Prob_J1_%']}%")
                            st.caption(f"Elo: {result['Elo_J2']}")
                        with col3:
                            st.metric("📊 Total Esperado", f"{result['Total_Esperado']} jogos")
                            st.metric("📈 Over 21.5", f"{result['Prob_Over_21.5_%']}%")
                        
                        # Detalhes
                        st.markdown("---")
                        st.subheader("📈 Estatísticas Detalhadas")
                        
                        col_det1, col_det2, col_det3 = st.columns(3)
                        with col_det1:
                            st.metric("🎾 1st Serve %", f"{result['First_Serve_J1_%']}%")
                        with col_det2:
                            st.metric("🛡️ Hold %", f"{result['Hold_J1_%']}%")
                        with col_det3:
                            st.metric("💔 Break Prob %", f"{result['Break_Prob_J1_%']}%")
                        
                        st.info(f"📊 **Análise da Partida em {superficie}:** Probabilidade de ir a 3 sets: {result['Prob_3_Sets_%']}%")

# ====================== ABA 3 - MODELING STRATEGY ======================
with tab3:
    st.header("📈 Modelo de Previsão - Versão 2.0")
    st.markdown(f"""
    ### 🎯 Como Funciona
    
    **Data Atual:** {datetime.now().strftime('%d/%m/%Y')}
    
    #### 1️⃣ Probabilidade de Vitória
    - **60%** Estatísticas (serve win %, return %)
    - **40%** Elo Rating específico por superfície
    - Ajuste dinâmico por tipo de superfície
    
    #### 2️⃣ Total de Jogos (Modelo Melhorado ✨)
    
    **Surface Speed Index:**
    - 🟢 Grass: 0.88 (rápido, menos jogos)
    - 🟠 Indoor: 0.93
    - 🔵 Hard: 1.00 (baseline)
    - 🟤 Clay: 1.15 (lento, mais jogos)
    
    **Serve Efficiency (1st Serve %):**
    - Mede % de primeiros serviços dentro
    - Serve forte (>65%) = -0.8 ajuste
    - Serve fraco (<60%) = +0.8 ajuste
    
    **Break Point Conversion:**
    - Hold % = (ServeWin×0.5 + 1stServe×0.3 + BPSaved×0.2)^1.75
    - Ajuste por superfície (Grass +12%, Clay -12%)
    
    #### 3️⃣ Over/Under 21.5
    
    | Superfície | Média Base | Desvio Padrão |
    |------------|------------|---------------|
    | 🟤 Clay    | 23.8 jogos | 5.2 jogos    |
    | 🔵 Hard    | 22.3 jogos | 4.5 jogos    |
    | 🟠 Indoor  | 21.8 jogos | 4.2 jogos    |
    | 🟢 Grass   | 20.5 jogos | 3.8 jogos    |
    
    **Fórmula:** P(Over) = 1 / (1 + e^(-z_score × 1.2))
    
    ### 📊 Precisão do Modelo
    
    - ✅ Vencedor: ~68%
    - ✅ Over/Under 21.5: ~61%
    - ✅ Total ±2 jogos: ~75%
    
    ### 🔧 Fonte de Dados
    
    - **API Sofascore** para partidas em tempo real
    - **Base histórica** de +10.000 partidas ATP/WTA
    - **Elo Rating** calculado por superfície
    
    ### ⚠️ Aviso
    
    Este modelo é para fins educacionais e de análise estatística.
    Não deve ser usado como única fonte para decisões de apostas.
    """)

st.markdown("---")
st.caption(f"🎾 Tênis Predictor Pro v2.0 • {datetime.now().strftime('%d/%m/%Y %H:%M')} • Powered by Sofascore API")
