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

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2c3e50;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'favorite_teams' not in st.session_state:
    st.session_state.favorite_teams = set()
if 'api_source' not in st.session_state:
    st.session_state.api_source = 'football-data'

def get_football_data_matches():
    """Fetch live matches from football-data.org API"""
    API_KEY = st.secrets.get("FOOTBALL_API_KEY", "")
    
    if not API_KEY:
        return None, "API key not configured. Add FOOTBALL_API_KEY to Streamlit secrets."
    
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
                
                match_data = {
                    'id': match['id'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': home_score,
                    'away_score': away_score,
                    'status': match['status'],
                    'minute': match.get('minute', 'LIVE'),
                    'competition': match['competition']['name']
                }
                matches.append(match_data)
            
            return matches, None
        elif response.status_code == 429:
            return None, "API rate limit reached. Try again later."
        elif response.status_code == 403:
            return None, "Invalid API key. Check your FOOTBALL_API_KEY in secrets."
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
        return None, "RapidAPI key not configured. Add RAPID_API_KEY to Streamlit secrets."
    
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
                match_data = {
                    'id': match['fixture']['id'],
                    'home_team': match['teams']['home']['name'],
                    'away_team': match['teams']['away']['name'],
                    'home_score': match['goals']['home'] or 0,
                    'away_score': match['goals']['away'] or 0,
                    'status': match['fixture']['status']['short'],
                    'minute': match['fixture']['status']['elapsed'] or 0,
                    'competition': match['league']['name']
                }
                matches.append(match_data)
            
            return matches, None
        elif response.status_code == 429:
            return None, "API rate limit reached. Try again later."
        elif response.status_code == 403:
            return None, "Invalid API key. Check your RAPID_API_KEY in secrets."
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

def is_favorite_team(team_name):
    """Check if team is in favorites"""
    return team_name.lower() in st.session_state.favorite_teams

def check_if_losing(match):
    """Check if a favorite team is losing in this match"""
    home_is_fav = is_favorite_team(match['home_team'])
    away_is_fav = is_favorite_team(match['away_team'])
    
    if home_is_fav and match['home_score'] < match['away_score']:
        return True, match['home_team']
    elif away_is_fav and match['away_score'] < match['home_score']:
        return True, match['away_team']
    
    return False, None

def display_match_card(match):
    """Display a single match card"""
    home_is_fav = is_favorite_team(match['home_team'])
    away_is_fav = is_favorite_team(match['away_team'])
    is_losing, losing_team = check_if_losing(match)
    
    # Determine card color
    if is_losing:
        border_color = "#dc3545"  # Red
        bg_color = "#fff5f5"
    elif home_is_fav or away_is_fav:
        border_color = "#28a745"  # Green
        bg_color = "#f0fff4"
    else:
        border_color = "#e0e0e0"  # Gray
        bg_color = "#ffffff"
    
    col1, col2, col3 = st.columns([3, 2, 3])
    
    with col1:
        if home_is_fav:
            st.markdown(f"### ⭐ **{match['home_team']}**")
        else:
            st.markdown(f"### {match['home_team']}")
    
    with col2:
        st.markdown(f"<h1 style='text-align: center; color: {border_color};'>{match['home_score']} - {match['away_score']}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>⏱️ {match['minute']}'</p>", unsafe_allow_html=True)
    
    with col3:
        if away_is_fav:
            st.markdown(f"### **{match['away_team']}** ⭐")
        else:
            st.markdown(f"### {match['away_team']}")
    
    st.caption(f"🏆 {match['competition']}")
    
    if is_losing:
        st.error(f"🚨 {losing_team} is currently LOSING!")
    
    st.divider()

def main():
    st.markdown('<div class="main-title"><h1>⚽ Live Soccer Match Alerts</h1></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Source Selection
        st.subheader("📡 API Source")
        api_options = {
            'football-data': 'Football-Data.org (Free)',
            'api-football': 'API-Football (RapidAPI)'
        }
        selected = st.radio(
            "Select data source:",
            options=list(api_options.keys()),
            format_func=lambda x: api_options[x],
            index=0
        )
        st.session_state.api_source = selected
        
        if selected == 'football-data':
            st.info("📝 Get free API key at: https://www.football-data.org/")
            st.caption("Add as FOOTBALL_API_KEY in secrets")
        else:
            st.info("📝 Get API key at: https://rapidapi.com/api-sports/api/api-football")
            st.caption("Add as RAPID_API_KEY in secrets")
        
        st.divider()
        
        # Favorite teams
        st.subheader("⭐ Your Favorite Teams")
        
        with st.form("add_team_form"):
            new_team = st.text_input("Team name:", placeholder="e.g., Liverpool")
            submitted = st.form_submit_button("➕ Add Team", use_container_width=True)
            
            if submitted and new_team.strip():
                st.session_state.favorite_teams.add(new_team.lower().strip())
                st.success(f"✅ Added {new_team}!")
                time.sleep(0.5)
                st.rerun()
        
        if st.session_state.favorite_teams:
            st.write("**Current favorites:**")
            teams_to_remove = []
            for team in sorted(st.session_state.favorite_teams):
                cols = st.columns([4, 1])
                with cols[0]:
                    st.write(f"⭐ {team.title()}")
                with cols[1]:
                    if st.button("🗑️", key=f"del_{team}", help="Remove"):
                        teams_to_remove.append(team)
            
            for team in teams_to_remove:
                st.session_state.favorite_teams.remove(team)
                st.rerun()
        else:
            st.info("No favorites yet. Add teams above!")
        
        st.divider()
        
        # Refresh settings
        st.subheader("🔄 Auto Refresh")
        auto_refresh = st.checkbox("Enable", value=True)
        refresh_seconds = st.slider("Interval (seconds)", 15, 120, 30)
        
        st.divider()
        st.caption("💡 Add teams to get alerts when they're losing")
    
    # Fetch matches
    with st.spinner("🔍 Fetching live matches..."):
        matches, error = get_live_matches()
        
        if error:
            st.error(f"❌ Error: {error}")
            st.info("💡 Make sure you've added the correct API key to your Streamlit secrets.")
            st.code("""
# Add to Streamlit secrets (.streamlit/secrets.toml):
FOOTBALL_API_KEY = "your_api_key_here"
# OR
RAPID_API_KEY = "your_api_key_here"
            """)
            return
    
    if not matches:
        st.warning("⚽ No live matches at the moment. Check back later!")
        st.info("Matches typically happen during weekends and weekday evenings (local time).")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["🚨 Alerts", "📊 All Live Matches"])
    
    # Alerts tab
    with tab1:
        st.header("🚨 Match Alerts")
        
        alerts = [(m, team) for m in matches for is_losing, team in [check_if_losing(m)] if is_losing]
        
        if alerts:
            st.error(f"**{len(alerts)} ALERT(S)** - Your favorite team(s) are losing!")
            st.write("")
            
            for match, losing_team in alerts:
                with st.container():
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                                padding: 20px; border-radius: 10px; color: white; margin: 10px 0;
                                border: 3px solid #c92a2a;'>
                        <h2 style='margin: 0; color: white;'>🚨 {losing_team} IS LOSING!</h2>
                        <h3 style='margin: 10px 0; color: white;'>{match['home_team']} {match['home_score']} - {match['away_score']} {match['away_team']}</h3>
                        <p style='margin: 5px 0; color: white;'>⏱️ Minute {match['minute']} | 🏆 {match['competition']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            if st.session_state.favorite_teams:
                st.success("✅ Great news! All your favorite teams are winning or drawing!")
                st.balloons()
            else:
                st.info("💡 Add your favorite teams in the sidebar to see alerts when they're losing")
    
    # All matches tab
    with tab2:
        st.header(f"📊 Live Matches ({len(matches)})")
        st.write("")
        
        for match in matches:
            display_match_card(match)
    
    # Display last update time
    st.caption(f"🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()
    else:
        if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
            st.rerun()

if __name__ == "__main__":
    main()
