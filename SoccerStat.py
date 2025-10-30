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
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .alert-box {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 3px solid #ff0000;
        animation: pulse 2s infinite;
    }
    .match-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #ddd;
        margin: 10px 0;
    }
    .winning {
        border-left: 5px solid #28a745;
    }
    .losing {
        border-left: 5px solid #dc3545;
    }
    .score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'favorite_teams' not in st.session_state:
    st.session_state.favorite_teams = set()
if 'previous_scores' not in st.session_state:
    st.session_state.previous_scores = {}

def get_live_matches():
    """
    Fetch live matches from API-Football (football-data.org)
    You need to get a free API key from https://www.football-data.org/
    """
    # Free API: football-data.org (requires API key)
    API_KEY = st.secrets.get("FOOTBALL_API_KEY", "YOUR_API_KEY_HERE")
    
    try:
        headers = {'X-Auth-Token': API_KEY}
        url = "https://api.football-data.org/v4/matches"
        params = {'status': 'IN_PLAY'}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            
            for match in data.get('matches', []):
                match_data = {
                    'id': match['id'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': match['score']['fullTime']['home'] or 0,
                    'away_score': match['score']['fullTime']['away'] or 0,
                    'status': match['status'],
                    'minute': match.get('minute', 'LIVE'),
                    'competition': match['competition']['name']
                }
                matches.append(match_data)
            
            return matches, None
        else:
            return [], f"API Error: {response.status_code}"
            
    except Exception as e:
        return [], f"Error: {str(e)}"

def get_simulated_matches():
    """Generate simulated live matches for demo purposes"""
    import random
    
    matches = [
        {'home_team': 'Manchester City', 'away_team': 'Liverpool', 'competition': 'Premier League'},
        {'home_team': 'Real Madrid', 'away_team': 'Barcelona', 'competition': 'La Liga'},
        {'home_team': 'Bayern Munich', 'away_team': 'Borussia Dortmund', 'competition': 'Bundesliga'},
        {'home_team': 'PSG', 'away_team': 'Marseille', 'competition': 'Ligue 1'},
        {'home_team': 'Juventus', 'away_team': 'Inter Milan', 'competition': 'Serie A'},
    ]
    
    result = []
    for i, match in enumerate(matches):
        # Generate random scores
        home_score = random.randint(0, 3)
        away_score = random.randint(0, 3)
        
        result.append({
            'id': i,
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'home_score': home_score,
            'away_score': away_score,
            'status': 'IN_PLAY',
            'minute': random.randint(1, 90),
            'competition': match['competition']
        })
    
    return result, None

def check_favorite_losing(match):
    """Check if a favorite team is losing"""
    home_team_lower = match['home_team'].lower()
    away_team_lower = match['away_team'].lower()
    
    home_is_favorite = home_team_lower in st.session_state.favorite_teams
    away_is_favorite = away_team_lower in st.session_state.favorite_teams
    
    if home_is_favorite and match['home_score'] < match['away_score']:
        return True, match['home_team']
    elif away_is_favorite and match['away_score'] < match['home_score']:
        return True, match['away_team']
    
    return False, None

def main():
    st.markdown('<h1 class="main-title">⚽ Live Match Alert System</h1>', unsafe_allow_html=True)
    
    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Favorite teams management
        st.subheader("⭐ Favorite Teams")
        new_team = st.text_input("Add favorite team:", placeholder="e.g., Liverpool")
        
        if st.button("➕ Add Team", use_container_width=True):
            if new_team.strip():
                st.session_state.favorite_teams.add(new_team.lower().strip())
                st.success(f"Added {new_team}!")
                st.rerun()
        
        # Display favorite teams
        if st.session_state.favorite_teams:
            st.write("**Your favorites:**")
            for team in sorted(st.session_state.favorite_teams):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"⭐ {team.title()}")
                with col2:
                    if st.button("❌", key=f"remove_{team}"):
                        st.session_state.favorite_teams.remove(team)
                        st.rerun()
        else:
            st.info("No favorite teams added yet")
        
        st.divider()
        
        # Data source selection
        st.subheader("📡 Data Source")
        use_demo = st.checkbox("Use Demo Mode (Simulated Data)", value=True)
        
        st.divider()
        
        # Auto-refresh
        st.subheader("🔄 Auto Refresh")
        auto_refresh = st.checkbox("Enable auto-refresh", value=True)
        if auto_refresh:
            refresh_interval = st.slider("Refresh interval (seconds)", 10, 60, 30)
    
    # Main content
    st.write("---")
    
    # Fetch live matches
    if use_demo:
        live_matches, error = get_simulated_matches()
        st.info("🎮 Demo Mode: Showing simulated matches")
    else:
        live_matches, error = get_live_matches()
        if error:
            st.error(f"⚠️ {error}")
            st.info("💡 Tip: Enable Demo Mode in settings to test the app")
            return
    
    # Display alerts section
    st.header("🚨 Alerts")
    
    alerts = []
    for match in live_matches:
        is_losing, team_name = check_favorite_losing(match)
        if is_losing:
            alerts.append((match, team_name))
    
    if alerts:
        for match, team_name in alerts:
            st.markdown(f"""
            <div class="alert-box">
                <h2>🚨 ALERT: {team_name} is LOSING!</h2>
                <h3>{match['home_team']} {match['home_score']} - {match['away_score']} {match['away_team']}</h3>
                <p>⏱️ Minute: {match['minute']} | 🏆 {match['competition']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        if st.session_state.favorite_teams:
            st.success("✅ All your favorite teams are winning or drawing!")
        else:
            st.info("💡 Add favorite teams in the sidebar to receive alerts")
    
    st.write("---")
    
    # Display all live matches
    st.header("📊 All Live Matches")
    
    if not live_matches:
        st.info("No live matches at the moment")
    else:
        st.write(f"**{len(live_matches)} matches in play**")
        
        for match in live_matches:
            is_losing, losing_team = check_favorite_losing(match)
            
            # Determine if any favorite team is in this match
            home_is_fav = match['home_team'].lower() in st.session_state.favorite_teams
            away_is_fav = match['away_team'].lower() in st.session_state.favorite_teams
            
            card_class = "losing" if is_losing else ("winning" if (home_is_fav or away_is_fav) else "")
            
            st.markdown(f"""
            <div class="match-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1; text-align: left;">
                        <h3>{match['home_team']} {"⭐" if home_is_fav else ""}</h3>
                    </div>
                    <div style="flex: 0.5; text-align: center;">
                        <div class="score">{match['home_score']} - {match['away_score']}</div>
                        <small>⏱️ {match['minute']}'</small>
                    </div>
                    <div style="flex: 1; text-align: right;">
                        <h3>{"⭐ " if away_is_fav else ""}{match['away_team']}</h3>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 10px;">
                    <small>🏆 {match['competition']}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Auto-refresh logic
    if auto_refresh:
        st.write(f"🔄 Auto-refreshing in {refresh_interval} seconds...")
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
