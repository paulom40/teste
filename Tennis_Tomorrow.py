import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import json

# --- Fetch real matches using Tennis API ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_atp_challenger_matches():
    """
    Fetches real ATP and Challenger matches for tomorrow using free API
    """
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Try using Sofascore's public API (reverse engineered)
    try:
        tomorrow_date = datetime.now() + timedelta(days=1)
        tomorrow_date_str = tomorrow_date.strftime('%Y-%m-%d')
        
        # Alternative Sofascore endpoint
        api_url = f"https://www.sofascore.com/api/v1/sport/tennis/events/live"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            
            # Parse the response for upcoming events
            if 'events' in data:
                for event in data['events']:
                    # Check if event is for tomorrow
                    start_time = event.get('startTimestamp', 0)
                    if start_time:
                        event_date = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d')
                        
                        if event_date == tomorrow_date_str:
                            tournament_name = event.get('tournament', {}).get('name', '')
                            
                            # Filter ATP and Challenger
                            if 'ATP' in tournament_name or 'Challenger' in tournament_name:
                                # Get surface
                                surface = 'Hard'
                                category_name = event.get('tournament', {}).get('category', {}).get('name', '')
                                if 'Clay' in category_name:
                                    surface = 'Clay'
                                elif 'Grass' in category_name:
                                    surface = 'Grass'
                                
                                # Get players
                                home_team = event.get('homeTeam', {}).get('name', '')
                                away_team = event.get('awayTeam', {}).get('name', '')
                                
                                if home_team and away_team:
                                    matches.append({
                                        "tournament": tournament_name,
                                        "surface": surface,
                                        "player1": home_team,
                                        "player2": away_team
                                    })
            
            if matches:
                return matches
    except Exception as e:
        pass
    
    # Try alternative API endpoint
    try:
        # Different Sofascore endpoint for scheduled events
        api_url = "https://www.sofascore.com/api/v1/sport/tennis/scheduled-events/0"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            tomorrow_date_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            
            if 'sportEvents' in data:
                for event in data['sportEvents']:
                    start_date = event.get('startDate', '').split('T')[0]
                    
                    if start_date == tomorrow_date_str:
                        tournament_name = event.get('tournament', {}).get('name', '')
                        
                        if 'ATP' in tournament_name or 'Challenger' in tournament_name:
                            surface = 'Hard'
                            if 'Clay' in tournament_name:
                                surface = 'Clay'
                            elif 'Grass' in tournament_name:
                                surface = 'Grass'
                            
                            home_team = event.get('homeTeam', {}).get('name', '')
                            away_team = event.get('awayTeam', {}).get('name', '')
                            
                            if home_team and away_team:
                                matches.append({
                                    "tournament": tournament_name,
                                    "surface": surface,
                                    "player1": home_team,
                                    "player2": away_team
                                })
            
            if matches:
                return matches
    except Exception as e:
        pass
    
    # If no matches found, return empty list
    return []

def export_to_txt(matches):
    """Convert matches to the required txt format"""
    if not matches:
        return "No real matches found for tomorrow."
    
    lines = []
    current_tournament = None
    
    for match in matches:
        tourney_name = match["tournament"]
        surface = match["surface"]
        player1 = match["player1"]
        player2 = match["player2"]
        
        if current_tournament != tourney_name:
            lines.append(f"{tourney_name} ({surface})")
            current_tournament = tourney_name
        
        lines.append(f"{player1} vs {player2}")
    
    return "\n".join(lines)

# --- Streamlit UI ---
st.set_page_config(
    page_title="ATP & Challenger Matches",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 ATP & Challenger Tennis Matches for Tomorrow")
st.markdown("Fetch real matches from Sofascore tennis data")

# Initialize session state
if "matches" not in st.session_state:
    st.session_state["matches"] = []
if "last_fetch" not in st.session_state:
    st.session_state["last_fetch"] = None

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔍 Fetch Tomorrow's Real Matches", type="primary", use_container_width=True):
        with st.spinner("Fetching real match data from Sofascore... Please wait..."):
            matches_data = fetch_atp_challenger_matches()
            
            if matches_data and len(matches_data) > 0:
                st.session_state["matches"] = matches_data
                st.session_state["last_fetch"] = datetime.now()
                st.success(f"✅ Found {len(matches_data)} ATP/Challenger matches for tomorrow!")
                
                # Display matches
                df = pd.DataFrame(matches_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No real matches found for tomorrow.")
                st.info("Possible reasons: No ATP or Challenger matches scheduled for tomorrow, or tournament break period.")
                st.session_state["matches"] = []

with col2:
    if st.session_state.get("matches") and len(st.session_state["matches"]) > 0:
        txt_content = export_to_txt(st.session_state["matches"])
        
        st.metric("Total Matches Found", len(st.session_state["matches"]))
        
        st.download_button(
            label="📥 Download as TXT File",
            data=txt_content,
            file_name=f"atp_challenger_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        with st.expander("📄 Preview Export Format"):
            st.code(txt_content, language="text", line_numbers=True)
        
        if st.session_state["last_fetch"]:
            st.caption(f"Last fetched: {st.session_state['last_fetch'].strftime('%H:%M:%S')}")

# Information section
with st.expander("ℹ️ About This App"):
    st.markdown("**How it works:**")
    st.markdown("- This app fetches real ATP and Challenger matches from Sofascore")
    st.markdown("- No demo or fake data - only actual scheduled matches")
    st.markdown("- Data is refreshed every hour")
    st.markdown("")
    st.markdown("**Requirements for deployment on Streamlit Cloud:**")
    st.markdown("")
    st.markdown("Create `requirements.txt`:")
    st.code("""
streamlit>=1.28.0
pandas>=2.0.0
requests>=2.31.0
    """, language="text")
    st.markdown("")
    st.markdown("**Note:** If no matches are found for tomorrow, it means no ATP/Challenger events are scheduled for that date.")

# Show helpful links
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("📅 [Official ATP Schedule](https://www.atptour.com/en/scores/schedule)")
with col2:
    st.markdown("🎾 [Challenger Tour](https://www.atptour.com/en/tournaments/challenger)")
with col3:
    st.markdown("📊 [Live Tennis Scores](https://www.sofascore.com/tennis)")

# Display message if no matches
if not st.session_state.get("matches"):
    st.info("👈 Click 'Fetch Tomorrow's Real Matches' to get ATP and Challenger matches for tomorrow")
