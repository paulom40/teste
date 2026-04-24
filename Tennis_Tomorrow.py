import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re

def get_matches_for_date(target_date):
    """
    Fetch matches for a specific date from Tennis24
    Target date should be a datetime object
    """
    matches = []
    
    # Format the date for display
    date_str = target_date.strftime('%d/%m/%Y')
    day_name = target_date.strftime('%A')
    
    try:
        # Use a requests approach instead of Selenium (much faster)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Tennis24 URL for specific date
        url = f"https://www.tennis24.com/"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            page_text = soup.get_text()
            
            # Parse the page text to find matches for the target date
            lines = page_text.split('\n')
            
            current_tournament = None
            current_surface = None
            found_target_date = False
            
            for i, line in enumerate(lines):
                # Look for date header
                if target_date.strftime('%d/%m') in line or target_date.strftime('%d.%m') in line:
                    found_target_date = True
                
                # If we're in the target date section
                if found_target_date:
                    # Detect tournament (contains location)
                    if any(city in line for city in ['Madrid', 'Gwangju', 'Rome', 'Shymkent', 'Oeiras', 'Barcelona', 'Munich']):
                        if 'Madrid' in line:
                            current_tournament = "Madrid ATP"
                            current_surface = "Clay"
                        elif 'Gwangju' in line:
                            current_tournament = "Gwangju Challenger"
                            current_surface = "Hard"
                        elif 'Rome' in line:
                            current_tournament = "Rome Challenger"
                            current_surface = "Clay"
                        elif 'Shymkent' in line:
                            current_tournament = "Shymkent Challenger"
                            current_surface = "Clay"
                        elif 'Oeiras' in line:
                            current_tournament = "Oeiras Challenger"
                            current_surface = "Clay"
                    
                    # Look for match patterns (time followed by player names)
                    time_match = re.match(r'^(\d{2}:\d{2})\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?)', line)
                    
                    if time_match and current_tournament:
                        player1 = time_match.group(2).strip()
                        player2 = time_match.group(3).strip()
                        
                        matches.append({
                            "tournament": current_tournament,
                            "surface": current_surface,
                            "player1": player1,
                            "player2": player2,
                            "time": time_match.group(1)
                        })
                    
                    # Stop after we've passed the matches section (next date or end)
                    if 'CHALLENGER WOMEN' in line or 'WTA -' in line:
                        break
            
    except Exception as e:
        st.warning(f"Auto-fetch failed: {str(e)}")
    
    return matches, day_name

# Fallback function with pre-loaded matches for different dates
def get_preloaded_matches(target_date):
    """
    Returns pre-loaded matches based on the actual date
    """
    date_str = target_date.strftime('%Y-%m-%d')
    
    # Pre-loaded matches for known dates
    preloaded = {
        '2026-04-25': [  # Saturday, April 25
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Cerundolo F.", "player2": "Hanfmann Y.", "time": "10:00"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Davidovich Fokina A.", "player2": "Carreno-Busta P.", "time": "10:00"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Khachanov K.", "player2": "Walton A.", "time": "10:00"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Cerundolo J. M.", "player2": "Darderi L.", "time": "11:30"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Damm M.", "player2": "Mensik J.", "time": "11:30"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Munar J.", "player2": "Ruud C.", "time": "11:30"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Gaubas V.", "player2": "Auger-Aliassime F.", "time": "13:00"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Humbert U.", "player2": "Atmane T.", "time": "13:00"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Nakashima B.", "player2": "Blockx A.", "time": "13:00"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Tien L.", "player2": "Vallejo D.", "time": "14:30"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Ugo Carabelli C.", "player2": "Cobolli F.", "time": "14:30"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Navone M.", "player2": "Zverev A.", "time": "15:00"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Budkov Kjaer N.", "player2": "Shapovalov D.", "time": "16:00"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Merida Aguilar D.", "player2": "Moutet C.", "time": "16:00"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Medvedev D.", "player2": "Marozsan F.", "time": "18:00"},
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Bublik A.", "player2": "Tsitsipas S.", "time": "20:30"},
            {"tournament": "Gwangju Challenger", "surface": "Hard", "player1": "Holmgren A.", "player2": "Riedi L.", "time": "03:00"},
            {"tournament": "Gwangju Challenger", "surface": "Hard", "player1": "Kwon S.", "player2": "Hsu Y. H.", "time": "03:00"},
            {"tournament": "Rome Challenger", "surface": "Clay", "player1": "Svrcina D.", "player2": "Vasami J.", "time": "12:30"},
            {"tournament": "Shymkent Challenger", "surface": "Clay", "player1": "Skatov T.", "player2": "Fomin S.", "time": "10:00"},
        ],
        '2026-04-26': [  # Sunday, April 26
            {"tournament": "Madrid ATP", "surface": "Clay", "player1": "Round 2 Matches", "player2": "TBD", "time": "TBD"},
            {"tournament": "Danube Upper Austria Open", "surface": "Clay", "player1": "Joel Schwärzler", "player2": "Qualifier", "time": "11:00"},
            {"tournament": "Danube Upper Austria Open", "surface": "Clay", "player1": "Jurij Rodionov", "player2": "Qualifier", "time": "13:00"},
        ]
    }
    
    # Return matches for the specific date, or empty list if not found
    return preloaded.get(date_str, [])

def export_to_txt(matches, target_date):
    """Convert matches to the required txt format"""
    if not matches:
        return f"No matches found for {target_date.strftime('%A, %B %d, %Y')}."
    
    lines = [f"# ATP and Challenger Matches - {target_date.strftime('%A, %B %d, %Y')}"]
    lines.append("")
    
    current_tournament = None
    
    for match in matches:
        tourney_name = match["tournament"]
        surface = match["surface"]
        player1 = match["player1"]
        player2 = match["player2"]
        
        if current_tournament != tourney_name:
            lines.append(f"{tourney_name} ({surface})")
            current_tournament = tourney_name
        
        if "time" in match and match["time"] and match["time"] != "TBD":
            lines.append(f"{match['time']} - {player1} vs {player2}")
        else:
            lines.append(f"{player1} vs {player2}")
    
    return "\n".join(lines)

# --- Streamlit UI ---
st.set_page_config(
    page_title="ATP & Challenger Matches",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 ATP & Challenger Tennis Matches")

# Calculate tomorrow's date
tomorrow = datetime.now() + timedelta(days=1)
today = datetime.now()

st.markdown(f"**Today:** {today.strftime('%A, %B %d, %Y')}")
st.markdown(f"**Tomorrow:** {tomorrow.strftime('%A, %B %d, %Y')}")

# Initialize session state
if "matches" not in st.session_state:
    st.session_state.matches = []
if "current_date" not in st.session_state:
    st.session_state.current_date = None

# Sidebar
with st.sidebar:
    st.header("📅 Date Selection")
    
    # Allow manual date selection
    use_tomorrow = st.radio(
        "Select date:",
        ["Tomorrow", "Specific date"],
        help="Tomorrow automatically updates each day"
    )
    
    if use_tomorrow == "Tomorrow":
        target_date = tomorrow
        st.info(f"Showing matches for: {target_date.strftime('%A, %B %d, %Y')}")
    else:
        target_date = st.date_input(
            "Pick a date",
            value=tomorrow,
            min_value=today,
            max_value=today + timedelta(days=7)
        )
        target_date = datetime.combine(target_date, datetime.min.time())
    
    st.markdown("---")
    
    if st.button("🔍 Load Matches", type="primary", use_container_width=True):
        with st.spinner(f"Fetching matches for {target_date.strftime('%A, %B %d')}..."):
            # Try to fetch live first
            matches_data, day_name = get_matches_for_date(target_date)
            
            # If no matches found, use pre-loaded data for that date
            if not matches_data:
                matches_data = get_preloaded_matches(target_date)
                if matches_data:
                    st.info(f"Using pre-loaded data for {target_date.strftime('%A, %B %d')}")
            
            st.session_state.matches = matches_data
            st.session_state.current_date = target_date
            st.rerun()

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.matches and st.session_state.current_date:
        st.success(f"✅ Loaded {len(st.session_state.matches)} matches for {st.session_state.current_date.strftime('%A, %B %d, %Y')}")
        
        # Display by tournament
        tournaments = {}
        for match in st.session_state.matches:
            tourney = match["tournament"]
            if tourney not in tournaments:
                tournaments[tourney] = []
            tournaments[tourney].append(match)
        
        for tourney, matches_list in tournaments.items():
            surface = matches_list[0]["surface"]
            st.subheader(f"{tourney} ({surface})")
            
            for match in matches_list:
                if "time" in match and match["time"] and match["time"] != "TBD":
                    st.write(f"  🕐 {match['time']} - **{match['player1']}** vs **{match['player2']}**")
                else:
                    st.write(f"  • **{match['player1']}** vs **{match['player2']}**")
            st.divider()
        
        # Display as dataframe
        with st.expander("📊 View as Table"):
            df = pd.DataFrame(st.session_state.matches)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(f"👈 Select a date and click 'Load Matches' to see ATP and Challenger matches")

with col2:
    if st.session_state.matches and st.session_state.current_date:
        txt_content = export_to_txt(st.session_state.matches, st.session_state.current_date)
        
        st.metric("Total Matches", len(st.session_state.matches))
        
        st.download_button(
            label="📥 Download TXT File",
            data=txt_content,
            file_name=f"tennis_matches_{st.session_state.current_date.strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        with st.expander("📄 Preview Export"):
            st.code(txt_content, language="text", line_numbers=True)

# Show upcoming dates info
with st.expander("📅 Upcoming Tournament Schedule"):
    st.markdown("""
    ### April 2026
    
    **April 25 (Saturday)**
    - Madrid ATP (Clay) - Round 1
    - Gwangju Challenger (Hard) - Quarterfinals
    - Rome Challenger (Clay) - Round 2
    - Shymkent Challenger (Clay) - Round 2
    
    **April 26 (Sunday)**
    - Madrid ATP (Clay) - Round 2
    - Danube Upper Austria Open (Clay) - Round 1
    
    **April 27 (Monday)**
    - Madrid ATP (Clay) - Round 2 continues
    - Various Challenger tournaments continue
    """)
