import streamlit as st
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from datetime import datetime, timedelta
import time
import re

def scrape_tennis24_matches():
    """
    Fast scraper specifically for Tennis24's match structure
    """
    matches = []
    driver = None
    
    try:
        # Configure Chromium
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--page-load-strategy", "eager")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(10)
        
        # Go to Tennis24
        driver.get("https://www.tennis24.com/")
        time.sleep(3)  # Short wait for initial load
        
        # Get page text
        page_text = driver.find_element(By.TAG_NAME, "body").text
        
        # Parse line by line
        lines = page_text.split('\n')
        
        current_tournament = None
        current_surface = None
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Detect tournament header (contains location and surface)
            if any(city in line for city in ['Madrid', 'Gwangju', 'Rome', 'Shymkent', 'Oeiras']):
                # Extract tournament info
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
            
            # Detect match lines (have time and player names)
            # Match format: "10:00 Cerundolo F. Hanfmann Y."
            time_match = re.match(r'^(\d{2}:\d{2})\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?)', line)
            
            if time_match and current_tournament:
                player1 = time_match.group(2).strip()
                player2 = time_match.group(3).strip()
                
                # Only add ATP and Challenger matches (not WTA)
                if 'ATP' in current_tournament or 'CHALLENGER' in current_tournament.upper():
                    matches.append({
                        "tournament": current_tournament,
                        "surface": current_surface,
                        "player1": player1,
                        "player2": player2,
                        "time": time_match.group(1)
                    })
            
            i += 1
        
        return matches
        
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return []
    
    finally:
        if driver:
            driver.quit()

# Real matches based on the data you provided
def get_confirmed_matches():
    """Returns the confirmed matches for April 25, 2026"""
    matches = [
        # Madrid ATP - Clay
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
        
        # Gwangju Challenger - Hard
        {"tournament": "Gwangju Challenger", "surface": "Hard", "player1": "Holmgren A.", "player2": "Riedi L.", "time": "03:00"},
        {"tournament": "Gwangju Challenger", "surface": "Hard", "player1": "Kwon S.", "player2": "Hsu Y. H.", "time": "03:00"},
        
        # Rome Challenger - Clay
        {"tournament": "Rome Challenger", "surface": "Clay", "player1": "Svrcina D.", "player2": "Vasami J.", "time": "12:30"},
        
        # Shymkent Challenger - Clay
        {"tournament": "Shymkent Challenger", "surface": "Clay", "player1": "Skatov T.", "player2": "Fomin S.", "time": "10:00"},
    ]
    return matches

def export_to_txt(matches):
    """Convert matches to the required txt format"""
    if not matches:
        return "No matches found for tomorrow."
    
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
        
        # Add time if available
        if "time" in match and match["time"]:
            lines.append(f"{match['time']} - {player1} vs {player2}")
        else:
            lines.append(f"{player1} vs {player2}")
    
    return "\n".join(lines)

# --- Streamlit UI ---
st.set_page_config(
    page_title="ATP & Challenger Matches - April 25, 2026",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 ATP & Challenger Tennis Matches")
st.markdown("**Date:** Saturday, April 25, 2026")

# Initialize session state
if "matches" not in st.session_state:
    st.session_state.matches = []

# Sidebar
with st.sidebar:
    st.header("📅 Tournaments")
    st.markdown("""
    ### ATP Tour
    - **Madrid ATP** (Clay) - 16 matches
    
    ### Challenger Tour  
    - **Gwangju Challenger** (Hard) - 2 matches
    - **Rome Challenger** (Clay) - 1 match
    - **Shymkent Challenger** (Clay) - 1 match
    
    ### Total: 20 matches
    """)
    
    if st.button("🔄 Load Matches", type="primary", use_container_width=True):
        matches_data = get_confirmed_matches()
        st.session_state.matches = matches_data
        st.rerun()

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.matches:
        st.success(f"✅ Loaded {len(st.session_state.matches)} matches")
        
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
                if "time" in match and match["time"]:
                    st.write(f"  🕐 {match['time']} - **{match['player1']}** vs **{match['player2']}**")
                else:
                    st.write(f"  • **{match['player1']}** vs **{match['player2']}**")
            st.divider()
        
        # Display as dataframe
        with st.expander("📊 View as Table"):
            df = pd.DataFrame(st.session_state.matches)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("👈 Click 'Load Matches' in the sidebar to see all ATP and Challenger matches for April 25, 2026")

with col2:
    if st.session_state.matches:
        txt_content = export_to_txt(st.session_state.matches)
        
        st.metric("Total Matches", len(st.session_state.matches))
        st.metric("ATP Tour", sum(1 for m in st.session_state.matches if "ATP" in m["tournament"]))
        st.metric("Challenger", sum(1 for m in st.session_state.matches if "Challenger" in m["tournament"]))
        
        st.download_button(
            label="📥 Download TXT File",
            data=txt_content,
            file_name=f"tennis_matches_20260425.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        with st.expander("📄 Preview Export"):
            st.code(txt_content, language="text", line_numbers=True)

# Show preview if no matches loaded
if not st.session_state.matches:
    with st.expander("📅 Preview of matches for April 25, 2026"):
        st.code("""
Madrid ATP (Clay)
10:00 - Cerundolo F. vs Hanfmann Y.
10:00 - Davidovich Fokina A. vs Carreno-Busta P.
10:00 - Khachanov K. vs Walton A.
11:30 - Cerundolo J. M. vs Darderi L.
11:30 - Damm M. vs Mensik J.
11:30 - Munar J. vs Ruud C.
13:00 - Gaubas V. vs Auger-Aliassime F.
13:00 - Humbert U. vs Atmane T.
13:00 - Nakashima B. vs Blockx A.
14:30 - Tien L. vs Vallejo D.
14:30 - Ugo Carabelli C. vs Cobolli F.
15:00 - Navone M. vs Zverev A.
16:00 - Budkov Kjaer N. vs Shapovalov D.
16:00 - Merida Aguilar D. vs Moutet C.
18:00 - Medvedev D. vs Marozsan F.
20:30 - Bublik A. vs Tsitsipas S.

Gwangju Challenger (Hard)
03:00 - Holmgren A. vs Riedi L.
03:00 - Kwon S. vs Hsu Y. H.

Rome Challenger (Clay)
12:30 - Svrcina D. vs Vasami J.

Shymkent Challenger (Clay)
10:00 - Skatov T. vs Fomin S.
        """, language="text")
