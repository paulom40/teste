import streamlit as st
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from datetime import datetime, timedelta
import time
import re

# --- Scraping Function using Chromium ---
@st.cache_data(ttl=3600, show_spinner=False)
def scrape_tennis24_matches():
    """
    Scrapes ATP and Challenger matches from Tennis24.com for tomorrow
    """
    matches = []
    driver = None
    
    try:
        # Configure Chromium options for headless operation
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # Initialize driver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to Tennis24
        driver.get("https://www.tennis24.com/")
        
        # Wait for page to load
        wait = WebDriverWait(driver, 20)
        time.sleep(5)  # Allow initial JavaScript load
        
        # Try to find and click "Upcoming" tab for tomorrow's matches
        try:
            # Look for Upcoming button/link
            upcoming_buttons = driver.find_elements(By.XPATH, "//*[contains(text(), 'Upcoming')]")
            for btn in upcoming_buttons:
                if btn.is_displayed() and btn.is_enabled():
                    btn.click()
                    time.sleep(3)
                    break
        except:
            pass
        
        # Find all match containers - Tennis24 uses specific structure
        # Look for tournament sections
        tournament_sections = driver.find_elements(By.CSS_SELECTOR, "div[class*='tournament'], div[class*='event']")
        
        current_tournament = "Unknown Tournament"
        current_surface = "Hard"
        
        # If tournament sections found, parse them
        if tournament_sections:
            for section in tournament_sections:
                section_text = section.text
                
                # Detect tournament name (contains ATP or Challenger)
                if "ATP" in section_text or "CHALLENGER" in section_text.upper():
                    current_tournament = section_text.split('\n')[0]
                    
                    # Determine surface from tournament name
                    if "Clay" in current_tournament:
                        current_surface = "Clay"
                    elif "Grass" in current_tournament:
                        current_surface = "Grass"
                    else:
                        current_surface = "Hard"
                    
                    # Find matches within this tournament section
                    match_elements = section.find_elements(By.CSS_SELECTOR, "div[class*='match'], div[class*='row']")
                    
                    for match in match_elements:
                        match_text = match.text
                        
                        # Look for player vs player pattern
                        if " vs " in match_text:
                            players = match_text.split(" vs ")
                            if len(players) >= 2:
                                player1 = players[0].strip()
                                player2 = players[1].strip()
                                
                                # Clean up player names (remove scores if present)
                                player1 = re.sub(r'\s+\d+.*$', '', player1)
                                player2 = re.sub(r'\s+\d+.*$', '', player2)
                                
                                if player1 and player2 and len(player1) > 1 and len(player2) > 1:
                                    matches.append({
                                        "tournament": current_tournament,
                                        "surface": current_surface,
                                        "player1": player1,
                                        "player2": player2
                                    })
        
        # Alternative: Parse all match rows from the main page
        if not matches:
            # Find all rows that contain match information
            all_rows = driver.find_elements(By.CSS_SELECTOR, "div[class*='row'], tr[class*='match']")
            
            for row in all_rows:
                row_text = row.text
                
                # Look for ATP or Challenger in the same context
                if "ATP" in row_text or "CHALLENGER" in row_text.upper():
                    lines = row_text.split('\n')
                    
                    # Find tournament name
                    tournament = "Unknown"
                    for line in lines:
                        if "ATP" in line or "CHALLENGER" in line.upper():
                            tournament = line
                            break
                    
                    # Find player names with vs pattern
                    for line in lines:
                        if " vs " in line:
                            players = line.split(" vs ")
                            if len(players) >= 2:
                                player1 = re.sub(r'\s+\d+.*$', '', players[0].strip())
                                player2 = re.sub(r'\s+\d+.*$', '', players[1].strip())
                                
                                # Determine surface
                                surface = "Hard"
                                if "Clay" in tournament:
                                    surface = "Clay"
                                elif "Grass" in tournament:
                                    surface = "Grass"
                                
                                matches.append({
                                    "tournament": tournament,
                                    "surface": surface,
                                    "player1": player1,
                                    "player2": player2
                                })
        
        # Remove duplicates
        unique_matches = []
        seen = set()
        for match in matches:
            key = f"{match['player1']} vs {match['player2']}"
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        return unique_matches
        
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return []
    
    finally:
        if driver:
            driver.quit()

def export_to_txt(matches):
    """Convert matches to the required txt format"""
    if not matches:
        return "No matches found for tomorrow.\n\nCheck Tennis24.com for schedule."
    
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
    page_title="ATP & Challenger Matches - Tennis24",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 ATP & Challenger Tennis Matches")
st.markdown("Scraping real matches from Tennis24.com using Chromium")

# Initialize session state
if "matches" not in st.session_state:
    st.session_state.matches = []

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This scraper fetches **real ATP and Challenger matches** from Tennis24.com.
    
    **Source:** Tennis24.com
    **Data:** Live and upcoming matches
    **Filter:** ATP Tour & Challenger events only
    
    **Note:** Tennis24 updates matches in real-time.
    """)

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔍 Fetch Matches from Tennis24", type="primary", use_container_width=True):
        with st.spinner("Scraping Tennis24.com with Chromium... This may take 10-15 seconds..."):
            matches_data = scrape_tennis24_matches()
            
            if matches_data and len(matches_data) > 0:
                st.session_state.matches = matches_data
                st.success(f"✅ Found {len(matches_data)} ATP/Challenger matches!")
                
                # Display matches in a table
                df = pd.DataFrame(matches_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Show sample of what was found
                st.subheader("📋 Match List")
                for i, match in enumerate(matches_data[:10], 1):
                    st.write(f"{i}. **{match['player1']}** vs **{match['player2']}** *({match['tournament']})*")
            else:
                st.warning("No ATP or Challenger matches found on Tennis24 right now.")
                st.info("💡 Try again during active tournament hours (typically 10:00 - 20:00 CET)")
                st.session_state.matches = []

with col2:
    if st.session_state.matches and len(st.session_state.matches) > 0:
        txt_content = export_to_txt(st.session_state.matches)
        
        st.metric("Total Matches Found", len(st.session_state.matches))
        
        st.download_button(
            label="📥 Download as TXT File",
            data=txt_content,
            file_name=f"tennis_matches_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        with st.expander("📄 Preview Export"):
            st.code(txt_content, language="text", line_numbers=True)

# Display message when no matches
if not st.session_state.matches:
    st.info("👈 Click the button above to fetch current ATP and Challenger matches from Tennis24.com")

# Instructions for deployment
with st.expander("🚀 Deployment on Streamlit Cloud"):
    st.markdown("""
    **Create `packages.txt`:**
