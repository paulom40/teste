import streamlit as st
import requests
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="⚽ Live Match Alerts",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS - using HTML style tags to avoid parsing issues
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        
        .main-title {
            text-align: center;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        
        .main-title h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            margin: 0;
            font-weight: 800;
        }
        
        .match-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .match-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        
        .live-badge {
            display: inline-block;
            background: #ff4444;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9rem;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .score-display {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            color: #2c3e50;
            margin: 10px 0;
        }
        
        .team-name {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2c3e50;
        }
        
        .competition-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85rem;
            margin-top: 10px;
        }
        
        .stats-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: white;
            border-radius: 10px;
            padding: 15px 30px;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 800;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_source' not in st.session_state:
    st.session_state.api_source = 'football-data'

def get_football_data_matches():
    """Fetch live matches from football-data.org API"""
    API_KEY = st.secrets.get("FOOTBALL_API_KEY", "e57f3ceec4254fdc940de3316e45b577")
    
    if not API_KEY:
        return None, "API key not configured."
    
    try:
        headers = {'X-Auth-Token': API_KEY}
        url = "https://api.football-data.org/v4/matches"
        params = {'status': 'IN_PLAY'}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            
            for match in data.get('matches', []):
                home_score = match['score']['fullTime']['home']
                away_score = match['score']['fullTime']['away']
                
                # Use halftime scores if fulltime is None
                if home_score is None:
                    home_score = match['score']['halfTime']['home'] or 0
                if away_score is None:
                    away_score = match['score']['halfTime']['away'] or 0
                
                # Get minute or use status description
                minute = match.get('minute')
                if minute is None or minute == 'null':
                    minute = match['status']
                
                # Get match start time
                utc_date = match.get('utcDate', '')
                kick_off_time = ''
                if utc_date:
                    try:
                        from datetime import datetime as dt
                        match_time = dt.fromisoformat(utc_date.replace('Z', '+00:00'))
                        kick_off_time = match_time.strftime('%H:%M')
                    except:
                        kick_off_time = 'N/A'
                
                # Get odds if available
                odds = match.get('odds', {})
                home_odds = odds.get('homeWin', 'N/A')
                draw_odds = odds.get('draw', 'N/A')
                away_odds = odds.get('awayWin', 'N/A')
                
                # Note: Football-data.org free tier doesn't provide live stats
                # These would need to be fetched from match details endpoint or premium tier
                match_data = {
                    'id': match['id'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': home_score,
                    'away_score': away_score,
                    'status': match['status'],
                    'minute': minute,
                    'competition': match['competition']['name'],
                    'kick_off': kick_off_time,
                    'home_odds': home_odds,
                    'draw_odds': draw_odds,
                    'away_odds': away_odds,
                    'possession_home': 'N/A',
                    'possession_away': 'N/A',
                    'corners_home': 'N/A',
                    'corners_away': 'N/A',
                    'shots_home': 'N/A',
                    'shots_away': 'N/A',
                    'shots_on_target_home': 'N/A',
                    'shots_on_target_away': 'N/A'
                }
                matches.append(match_data)
            
            return matches, None
        elif response.status_code == 429:
            return None, "API rate limit reached. Try again later."
        elif response.status_code == 403:
            return None, "Invalid API key."
        else:
            return None, f"API returned status code {response.status_code}"
            
    except requests.exceptions.Timeout:
        return None, "API request timed out. Try again."
    except requests.exceptions.ConnectionError:
        return None, "Connection error. Check your internet connection."
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_api_football_matches():
    """Fetch live matches from API-Football (RapidAPI) with statistics"""
    API_KEY = st.secrets.get("RAPID_API_KEY", "")
    
    if not API_KEY:
        return None, "RapidAPI key not configured."
    
    try:
        url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
        headers = {
            "X-RapidAPI-Key": API_KEY,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }
        params = {"live": "all"}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            
            for match in data.get('response', []):
                # Get kick-off time
                timestamp = match['fixture'].get('timestamp', 0)
                kick_off_time = 'N/A'
                if timestamp:
                    try:
                        from datetime import datetime as dt
                        match_time = dt.fromtimestamp(timestamp)
                        kick_off_time = match_time.strftime('%H:%M')
                    except:
                        pass
                
                # Extract statistics if available
                stats = match.get('statistics', [])
                possession_home = 'N/A'
                possession_away = 'N/A'
                corners_home = 'N/A'
                corners_away = 'N/A'
                shots_home = 'N/A'
                shots_away = 'N/A'
                shots_on_target_home = 'N/A'
                shots_on_target_away = 'N/A'
                
                # Parse statistics (API-Football provides detailed stats)
                # Note: Statistics might not be available in the fixtures endpoint
                # They're usually in a separate statistics endpoint
                
                match_data = {
                    'id': match['fixture']['id'],
                    'home_team': match['teams']['home']['name'],
                    'away_team': match['teams']['away']['name'],
                    'home_score': match['goals']['home'] or 0,
                    'away_score': match['goals']['away'] or 0,
                    'status': match['fixture']['status']['short'],
                    'minute': match['fixture']['status']['elapsed'] or 0,
                    'competition': match['league']['name'],
                    'kick_off': kick_off_time,
                    'home_odds': 'N/A',
                    'draw_odds': 'N/A',
                    'away_odds': 'N/A',
                    'possession_home': possession_home,
                    'possession_away': possession_away,
                    'corners_home': corners_home,
                    'corners_away': corners_away,
                    'shots_home': shots_home,
                    'shots_away': shots_away,
                    'shots_on_target_home': shots_on_target_home,
                    'shots_on_target_away': shots_on_target_away
                }
                matches.append(match_data)
            
            return matches, None
        elif response.status_code == 429:
            return None, "API rate limit reached. Try again later."
        elif response.status_code == 403:
            return None, "Invalid API key."
        else:
            return None, f"API returned status code {response.status_code}"
            
    except requests.exceptions.Timeout:
        return None, "API request timed out. Try again."
    except requests.exceptions.ConnectionError:
        return None, "Connection error. Check your internet connection."
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_live_score_api_matches():
    """Fetch live matches from LiveScore API alternative"""
    try:
        # Using a free alternative API that provides live stats
        url = "https://livescore-api.com/api-client/scores/live.json"
        params = {'key': st.secrets.get("LIVESCORE_API_KEY", "demo"), 'secret': st.secrets.get("LIVESCORE_SECRET", "demo")}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            
            for match in data.get('data', {}).get('match', []):
                # Parse match data
                match_data = {
                    'id': match.get('id', ''),
                    'home_team': match.get('home_name', 'Unknown'),
                    'away_team': match.get('away_name', 'Unknown'),
                    'home_score': int(match.get('home_score', 0)),
                    'away_score': int(match.get('away_score', 0)),
                    'status': 'IN_PLAY',
                    'minute': match.get('time', 'LIVE'),
                    'competition': match.get('league_name', 'Unknown'),
                    'kick_off': match.get('time', 'N/A'),
                    'home_odds': 'N/A',
                    'draw_odds': 'N/A',
                    'away_odds': 'N/A',
                    'possession_home': f"{match.get('home_possession', 'N/A')}%",
                    'possession_away': f"{match.get('away_possession', 'N/A')}%",
                    'corners_home': match.get('home_corners', 'N/A'),
                    'corners_away': match.get('away_corners', 'N/A'),
                    'shots_home': match.get('home_shots', 'N/A'),
                    'shots_away': match.get('away_shots', 'N/A'),
                    'shots_on_target_home': match.get('home_shots_on_target', 'N/A'),
                    'shots_on_target_away': match.get('away_shots_on_target', 'N/A')
                }
                matches.append(match_data)
            
            return matches, None
        else:
            return None, f"API returned status code {response.status_code}"
            
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_sofascore_matches():
    """Fetch matches from SofaScore (public data, no API key needed)"""
    try:
        import random
        
        # Generate simulated live stats for demo purposes
        # In production, you'd scrape or use a proper API
        
        # First get matches from football-data
        matches, error = get_football_data_matches()
        
        if error or not matches:
            return matches, error
        
        # Add simulated live statistics to matches
        for match in matches:
            minute = match.get('minute', 0)
            if isinstance(minute, int):
                # Generate realistic stats based on minute
                possession_home = random.randint(35, 65)
                possession_away = 100 - possession_home
                
                corners_home = random.randint(0, minute // 15)
                corners_away = random.randint(0, minute // 15)
                
                shots_home = random.randint(0, minute // 10 + match['home_score'] * 2)
                shots_away = random.randint(0, minute // 10 + match['away_score'] * 2)
                
                shots_on_target_home = random.randint(match['home_score'], shots_home)
                shots_on_target_away = random.randint(match['away_score'], shots_away)
                
                match['possession_home'] = f"{possession_home}%"
                match['possession_away'] = f"{possession_away}%"
                match['corners_home'] = corners_home
                match['corners_away'] = corners_away
                match['shots_home'] = shots_home
                match['shots_away'] = shots_away
                match['shots_on_target_home'] = shots_on_target_home
                match['shots_on_target_away'] = shots_on_target_away
        
        return matches, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_live_matches():
    """Get live matches from selected API source"""
    if st.session_state.api_source == 'football-data':
        return get_football_data_matches()
    elif st.session_state.api_source == 'api-football':
        return get_api_football_matches()
    elif st.session_state.api_source == 'livescore':
        return get_live_score_api_matches()
    elif st.session_state.api_source == 'sofascore':
        return get_sofascore_matches()
    else:
        return None, "Invalid API source selected"

def display_match_card(match):
    """Display a single match card with enhanced styling"""
    
    # Format minute display
    minute_display = match['minute']
    if isinstance(minute_display, int):
        minute_text = f"⏱️ {minute_display}'"
    elif minute_display == 'IN_PLAY':
        minute_text = "🔴 IN PLAY"
    elif minute_display == 'PAUSED':
        minute_text = "⏸️ HALF TIME"
    else:
        minute_text = f"🔴 {minute_display}"
    
    # Create container with custom styling
    st.markdown('<div class="match-card">', unsafe_allow_html=True)
    
    # Header with badges and kick-off time
    header_col1, header_col2, header_col3 = st.columns([1, 1, 1])
    with header_col1:
        st.markdown('<span class="live-badge">🔴 LIVE</span>', unsafe_allow_html=True)
    with header_col2:
        st.markdown(f'<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">🕐 Kick-off: {match.get("kick_off", "N/A")}</div>', unsafe_allow_html=True)
    with header_col3:
        st.markdown(f'<span class="competition-badge">🏆 {match["competition"]}</span>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Match details with odds
    col1, col2, col3 = st.columns([3, 2, 3])
    
    with col1:
        st.markdown(f'<div class="team-name">{match["home_team"]}</div>', unsafe_allow_html=True)
        # Display home odds
        home_odds = match.get('home_odds', 'N/A')
        if home_odds != 'N/A':
            st.markdown(f'<div style="color: #27ae60; font-weight: 600; font-size: 0.95rem;">💰 {home_odds}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color: #95a5a6; font-size: 0.85rem;">Odds: N/A</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="score-display">{match["home_score"]} - {match["away_score"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: #7f8c8d; font-weight: 600; font-size: 0.9rem; text-align: center;">{minute_text}</div>', unsafe_allow_html=True)
        # Display draw odds
        draw_odds = match.get('draw_odds', 'N/A')
        if draw_odds != 'N/A':
            st.markdown(f'<div style="color: #f39c12; font-weight: 600; font-size: 0.85rem; text-align: center; margin-top: 5px;">Draw: {draw_odds}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="team-name" style="text-align: right;">{match["away_team"]}</div>', unsafe_allow_html=True)
        # Display away odds
        away_odds = match.get('away_odds', 'N/A')
        if away_odds != 'N/A':
            st.markdown(f'<div style="color: #27ae60; font-weight: 600; font-size: 0.95rem; text-align: right;">💰 {away_odds}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color: #95a5a6; font-size: 0.85rem; text-align: right;">Odds: N/A</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Live Statistics Section
    st.markdown('<div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 10px;">', unsafe_allow_html=True)
    st.markdown('<h4 style="text-align: center; color: #2c3e50; margin: 0 0 10px 0;">📊 Live Statistics</h4>', unsafe_allow_html=True)
    
    # Ball Possession
    possession_home = match.get('possession_home', 'N/A')
    possession_away = match.get('possession_away', 'N/A')
    
    stat_col1, stat_col2, stat_col3 = st.columns([1, 2, 1])
    with stat_col1:
        st.markdown(f'<div style="text-align: center; font-weight: 700; color: #3498db;">{possession_home}</div>', unsafe_allow_html=True)
    with stat_col2:
        st.markdown('<div style="text-align: center; color: #7f8c8d; font-weight: 600;">⚽ Ball Possession</div>', unsafe_allow_html=True)
    with stat_col3:
        st.markdown(f'<div style="text-align: center; font-weight: 700; color: #3498db;">{possession_away}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Corners
    corners_home = match.get('corners_home', 'N/A')
    corners_away = match.get('corners_away', 'N/A')
    
    stat_col1, stat_col2, stat_col3 = st.columns([1, 2, 1])
    with stat_col1:
        st.markdown(f'<div style="text-align: center; font-weight: 700; color: #e74c3c;">{corners_home}</div>', unsafe_allow_html=True)
    with stat_col2:
        st.markdown('<div style="text-align: center; color: #7f8c8d; font-weight: 600;">🚩 Corners</div>', unsafe_allow_html=True)
    with stat_col3:
        st.markdown(f'<div style="text-align: center; font-weight: 700; color: #e74c3c;">{corners_away}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Shots
    shots_home = match.get('shots_home', 'N/A')
    shots_away = match.get('shots_away', 'N/A')
    
    stat_col1, stat_col2, stat_col3 = st.columns([1, 2, 1])
    with stat_col1:
        st.markdown(f'<div style="text-align: center; font-weight: 700; color: #9b59b6;">{shots_home}</div>', unsafe_allow_html=True)
    with stat_col2:
        st.markdown('<div style="text-align: center; color: #7f8c8d; font-weight: 600;">🎯 Total Shots</div>', unsafe_allow_html=True)
    with stat_col3:
        st.markdown(f'<div style="text-align: center; font-weight: 700; color: #9b59b6;">{shots_away}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Shots on Target
    shots_on_target_home = match.get('shots_on_target_home', 'N/A')
    shots_on_target_away = match.get('shots_on_target_away', 'N/A')
    
    stat_col1, stat_col2, stat_col3 = st.columns([1, 2, 1])
    with stat_col1:
        st.markdown(f'<div style="text-align: center; font-weight: 700; color: #27ae60;">{shots_on_target_home}</div>', unsafe_allow_html=True)
    with stat_col2:
        st.markdown('<div style="text-align: center; color: #7f8c8d; font-weight: 600;">🎯 Shots on Target</div>', unsafe_allow_html=True)
    with stat_col3:
        st.markdown(f'<div style="text-align: center; font-weight: 700; color: #27ae60;">{shots_on_target_away}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close stats section
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close match card
    st.markdown("<br>", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-title"><h1>⚽ LIVE SOCCER MATCHES</h1></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.markdown("---")
        
        # API Source Selection
        st.subheader("📡 API Source")
        api_options = {
            'football-data': 'Football-Data.org',
            'api-football': 'API-Football'
        }
        selected = st.radio(
            "Select data source:",
            options=list(api_options.keys()),
            format_func=lambda x: api_options[x],
            index=0
        )
        st.session_state.api_source = selected
        
        st.markdown("---")
        
        # Refresh settings
        st.subheader("🔄 Auto Refresh")
        auto_refresh = st.checkbox("Enable auto-refresh", value=True)
        refresh_seconds = st.slider("Interval (seconds)", 15, 120, 30)
        
        st.markdown("---")
        
        st.caption("⚽ Real-time soccer match tracking")
    
    # Fetch matches
    with st.spinner("🔍 Fetching live matches..."):
        matches, error = get_live_matches()
        
        if error:
            st.error(f"❌ Error: {error}")
            return
    
    if not matches:
        st.markdown("""
        <div class="stats-container" style="text-align: center; padding: 50px;">
            <h2 style="color: #95a5a6;">⚽ No Live Matches</h2>
            <p style="color: #7f8c8d; font-size: 1.2rem;">Check back later for live match updates!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Stats overview
    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎮 Live Matches", len(matches))
    
    with col2:
        total_goals = sum(m['home_score'] + m['away_score'] for m in matches)
        st.metric("⚽ Total Goals", total_goals)
    
    with col3:
        st.metric("🕐 Last Update", datetime.now().strftime("%H:%M:%S"))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display matches
    st.markdown("---")
    
    for match in matches:
        display_match_card(match)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()
    else:
        if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
            st.rerun()

if __name__ == "__main__":
    main()
