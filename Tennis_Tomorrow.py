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
    Scrapes ATP and Challenger matches from Tennis24.com
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
        time.sleep(5)
        
        # Find all match elements directly using Tennis24's structure
        # Look for elements with match information
        match_elements = driver.find_elements(By.CSS_SELECTOR, "div[class*='event__match'], div[class*='matchRow'], tr[class*='match']")
        
        current_tournament = None
        
        for element in match_elements:
            try:
                # Get the text content
                text = element.text
                
                # Look for tournament names in parent or previous elements
                parent = element.find_element(By.XPATH, "..")
                parent_text = parent.text
                
                # Find tournament name if not set
                if "ATP" in parent_text or "Challenger" in parent_text:
                    lines = parent_text.split('\n')
                    for line in lines:
                        if ("ATP" in line or "Challenger" in line) and len(line) < 100:
                            current_tournament = line.strip()
                            break
                
                # Check if this element contains a match (has vs or - with player names)
                if " vs " in text or " - " in text:
                    # Split to get potential player names
                    if " vs " in text:
                        parts = text.split(" vs ")
                    else:
                        parts = text.split(" - ")
                    
                    if len(parts) >= 2:
                        player1 = parts[0].strip()
                        player2 = parts[1].strip()
                        
                        # Clean up - remove scores, odds, and extra numbers
                        player1 = re.sub(r'\s+[\d\.]+\s*$', '', player1)
                        player1 = re.sub(r'\s+\d+:\d+', '', player1)
                        player1 = re.sub(r'^\d+\s+', '', player1)
                        
                        player2 = re.sub(r'\s+[\d\.]+\s*$', '', player2)
                        player2 = re.sub(r'\s+\d+:\d+', '', player2)
                        player2 = re.sub(r'^\d+\s+', '', player2)
                        
                        # Validate that these look like real player names
                        # Real names have letters and are not generic terms
                        if (player1 and player2 and 
                            len(player1) > 2 and len(player2) > 2 and
                            not any(term in player1.upper() for term in ['CHALLENGER', 'ATP', 'WTA', 'ITF', 'SINGLES', 'DOUBLES', 'RACE']) and
                            not any(term in player2.upper() for term in ['CHALLENGER', 'ATP', 'WTA', 'ITF', 'SINGLES', 'DOUBLES', 'RACE'])):
                            
                            # Determine surface from tournament name
                            surface = "Hard"
                            if current_tournament and "Clay" in current_tournament:
                                surface = "Clay"
                            elif current_tournament and "Grass" in current_tournament:
                                surface = "Grass"
                            
                            matches.append({
                                "tournament": current_tournament if current_tournament else "ATP Tournament",
                                "surface": surface,
                                "player1": player1,
                                "player2": player2
                            })
            except:
                continue
        
        # Alternative approach: Get all text and parse line by line with context
        if len(matches) < 3:
            page_text = driver.find_element(By.TAG_NAME, "body").text
            lines = page_text.split('\n')
            
            for i, line in enumerate(lines):
                # Look for actual player names (two words, capital letters)
                if " vs " in line and len(line) < 100:
                    players = line.split(" vs ")
                    if len(players) == 2:
                        player1 = players[0].strip()
                        player2 = players[1].strip()
                        
                        # Check if these look like real names (not category headers)
                        if (re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$', player1) and
                            re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$', player2)):
                            
                            # Look for tournament name in surrounding lines
                            tournament = "ATP Challenger"
                            for j in range(max(0, i-5), min(len(lines), i+5)):
                                if "ATP" in lines[j] or "Challenger" in lines[j]:
                                    if len(lines[j]) < 100 and " vs " not in lines[j]:
                                        tournament = lines[j].strip()
                                        break
                            
                            matches.append({
                                "tournament": tournament,
                                "surface": "Hard",
                                "player1": player1,
                                "player2": player2
                            })
        
        # Remove duplicates and filter out invalid matches
        unique_matches = []
        seen = set()
        for match in matches:
            key = f"{match['player1']} vs {match['player2']}"
            if key not in seen and len(match['player1']) > 2 and len(match['player2']) > 2:
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
        return "No real matches found for tomorrow.\n\nTry again during active tournament times."
    
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
    st.markdown("This scraper fetches real ATP and Challenger matches from Tennis24.com")
    st.markdown("**Source:** Tennis24.com")
    st.markdown("**Filter:** Only real player matches (no category headers)")
    st.markdown("---")
    st.markdown("**Current date:** " + datetime.now().strftime("%Y-%m-%d"))

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔍 Fetch Matches from Tennis24", type="primary", use_container_width=True):
        with st.spinner("Scraping Tennis24.com with Chromium... This may take 10-15 seconds..."):
            matches_data = scrape_tennis24_matches()
            
            if matches_data and len(matches_data) > 0:
                st.session_state.matches = matches_data
                st.success(f"Found {len(matches_data)} real ATP/Challenger matches!")
                
                # Display matches in a table
                df = pd.DataFrame(matches_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Show match list
                st.subheader("Match List")
                for i, match in enumerate(matches_data, 1):
                    st.write(f"{i}. **{match['player1']}** vs **{match['player2']}**")
                    st.caption(f"   Tournament: {match['tournament']} ({match['surface']})")
            else:
                st.warning("No real matches found on Tennis24 right now.")
                st.info("Tips:\n- Try during active tournament hours (10:00 - 20:00 CET)\n- Check if tournaments are currently running\n- The scraper filters out category headers and only shows real player matches")
                st.session_state.matches = []

with col2:
    if st.session_state.matches and len(st.session_state.matches) > 0:
        txt_content = export_to_txt(st.session_state.matches)
        
        st.metric("Total Matches Found", len(st.session_state.matches))
        
        st.download_button(
            label="Download as TXT File",
            data=txt_content,
            file_name=f"tennis_matches_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        with st.expander("Preview Export"):
            st.code(txt_content, language="text", line_numbers=True)

# Display message when no matches
if not st.session_state.matches:
    st.info("Click the button above to fetch current ATP and Challenger matches from Tennis24.com")
    st.markdown("---")
    st.markdown("**Note:** The scraper now filters out category headers like 'CHALLENGER MEN - SINGLES' and only shows real matches with actual player names.")

# Deployment instructions
with st.expander("Deployment on Streamlit Cloud"):
    st.markdown("**Create packages.txt:**")
    st.code("chromium-browser", language="text")
    
    st.markdown("**Create requirements.txt:**")
    st.code("""
streamlit>=1.28.0
pandas>=2.0.0
selenium>=4.15.0
    """, language="text")
