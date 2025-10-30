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
                    'away_odds': away_odds
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
    """Fetch live matches from API-Football (RapidAPI)"""
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
                    'away_odds': 'N/A'
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

def get_live_matches():
    """Get live matches from selected API source"""
    if st.session_state.api_source == 'football-data':
        return get_football_data_matches()
    elif st.session_state.api_source == 'api-football':
        return get_api_football_matches()
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
    
    st.markdown('</div>', unsafe_allow_html=True)
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
