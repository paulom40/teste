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
def scrape_sofascore_tomorrow():
    """
    Scrapes ATP and Challenger matches from Sofascore for tomorrow using Chromium
    """
    matches = []
    driver = None
    
    try:
        # Configure Chromium options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # Initialize driver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Calculate tomorrow's date
        tomorrow_date = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow_date.strftime('%Y-%m-%d')
        
        # Go directly to tomorrow's tennis schedule
        driver.get(f"https://www.sofascore.com/tennis/{tomorrow_str}")
        
        # Wait for page to load
        wait = WebDriverWait(driver, 20)
        time.sleep(8)  # Let JavaScript render
        
        # Get all text from the page
        page_text = driver.find_element(By.TAG_NAME, "body").text
        lines = page_text.split('\n')
        
        # Parse matches
        current_tournament = None
        current_surface = "Hard"
        
        for i, line in enumerate(lines):
            line_upper = line.upper()
            
            # Detect tournament (contains ATP or CHALLENGER)
            if "ATP" in line_upper or "CHALLENGER" in line_upper:
                current_tournament = line.strip()
                
                # Determine surface from tournament name
                if "CLAY" in line_upper:
                    current_surface = "Clay"
                elif "GRASS" in line_upper:
                    current_surface = "Grass"
                elif "HARD" in line_upper:
                    current_surface = "Hard"
                else:
                    current_surface = "Hard"  # Default
                    
                # Look for matches in the next lines
                for j in range(i+1, min(i+15, len(lines))):
                    match_line = lines[j]
                    match_line_clean = match_line.strip()
                    
                    # Check if this line contains a match
                    if match_line_clean and not any(x in match_line_clean.upper() for x in ["ATP", "CHALLENGER", "WTA", "ITF"]):
                        # Look for player vs player pattern
                        players = None
                        
                        # Pattern 1: Player1 vs Player2
                        if " vs " in match_line_clean:
                            players = match_line_clean.split(" vs ")
                        # Pattern 2: Player1 - Player2
                        elif " - " in match_line_clean:
                            parts = match_line_clean.split(" - ")
                            if len(parts) == 2 and len(parts[0].split()) >= 2 and len(parts[1].split()) >= 2:
                                players = parts
                        # Pattern 3: Look for two names with scores
                        elif re.search(r'[A-Z][a-z]+\s+[A-Z]\.?\s+\d+', match_line_clean):
                            # Skip if it has scores (finished match)
                            continue
                        # Pattern 4: Just two names
                        else:
                            import re
                            name_pattern = r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?)'
                            name_matches = re.findall(name_pattern, match_line_clean)
                            if len(name_matches) >= 1:
                                players = [name_matches[0][0], name_matches[0][1]]
                        
                        # If we found players, add match
                        if players and len(players) >= 2:
                            matches.append({
                                "tournament": current_tournament,
                                "surface": current_surface,
                                "player1": players[0].strip(),
                                "player2": players[1].strip()
                            })
                            break  # Move to next tournament
        
        # If no matches found, try alternative URL
        if not matches:
            driver.get("https://www.sofascore.com/tennis")
            time.sleep(5)
            
            # Try to click on tomorrow's date
            try:
                date_picker = driver.find_element(By.CSS_SELECTOR, "[class*='DatePicker']")
                date_picker.click()
                time.sleep(1)
                
                # Find tomorrow's date button
                date_buttons = driver.find_elements(By.CSS_SELECTOR, "button, div[role='button']")
                for btn in date_buttons:
                    btn_text = btn.text
                    if btn_text and str(tomorrow_date.day) in btn_text:
                        btn.click()
                        time.sleep(3)
                        break
            except:
                pass
            
            # Re-parse the page
            page_text = driver.find_element(By.TAG_NAME, "body").text
            lines = page_text.split('\n')
            
            current_tournament = None
            current_surface = "Hard"
            
            for i, line in enumerate(lines):
                line_upper = line.upper()
                
                if "ATP" in line_upper or "CHALLENGER" in line_upper:
                    current_tournament = line.strip()
                    
                    if "CLAY" in line_upper:
                        current_surface = "Clay"
                    elif "GRASS" in line_upper:
                        current_surface = "Grass"
                    elif "HARD" in line_upper:
                        current_surface = "Hard"
                    
                    for j in range(i+1, min(i+10, len(lines))):
                        match_line = lines[j].strip()
                        if match_line and " vs " in match_line:
                            players = match_line.split(" vs ")
                            if len(players) >= 2:
                                matches.append({
                                    "tournament": current_tournament,
                                    "surface": current_surface,
                                    "player1": players[0].strip(),
                                    "player2": players[1].strip()
                                })
                                break
        
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
        return "No matches found for tomorrow."
    
    lines = []
    current_tournament = None
    
    for match in matches:
        tourney_name = match["tournament"]
        surface = match["surface"]
        player1 = match["player1"]
        player2 = match["player2"]
        
        # Add tournament header if it's a new tournament
        if current_tournament != tourney_name:
            lines.append(f"{tourney_name} ({surface})")
            current_tournament = tourney_name
        
        # Add match line
        lines.append(f"{player1} vs {player2}")
    
    return "\n".join(lines)

# --- Streamlit UI ---
st.set_page_config(
    page_title="ATP & Challenger Matches - Sofascore Scraper",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 ATP & Challenger Matches for Tomorrow")
st.markdown("Real-time data from Sofascore - ATP Tour and Challenger events only")

# Initialize session state
if "matches" not in st.session_state:
    st.session_state["matches"] = []

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔍 Fetch Tomorrow's Matches from Sofascore", type="primary", use_container_width=True):
        with st.spinner("Scraping Sofascore with Chromium... This may take 15-20 seconds..."):
            matches_data = scrape_sofascore_tomorrow()
            
            if matches_data:
                st.session_state["matches"] = matches_data
                st.success(f"✅ Found {len(matches_data)} ATP/Challenger matches for tomorrow!")
                
                # Display matches in a nice table
                if matches_data:
                    df = pd.DataFrame(matches_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No matches found for tomorrow")
            else:
                st.error("❌ No matches found. Sofascore might have no matches scheduled for tomorrow or the site structure changed.")
                st.session_state["matches"] = []

with col2:
    if st.session_state.get("matches") and len(st.session_state["matches"]) > 0:
        txt_content = export_to_txt(st.session_state["matches"])
        
        st.metric("Total Matches", len(st.session_state["matches"]))
        st.download_button(
            label="📥 Export as TXT File",
            data=txt_content,
            file_name=f"tennis_matches_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Show preview
        with st.expander("📄 Preview TXT Format"):
            st.code(txt_content, language="text", line_numbers=True)

# Instructions
if not st.session_state.get("matches"):
    st.info("👈 Click the button above to fetch tomorrow's ATP and Challenger matches directly from Sofascore")

# Deployment requirements
with st.expander("📦 Deployment Configuration for Streamlit Cloud"):
    st.markdown("**Create a file called `packages.txt`:**")
    st.code("chromium-browser", language="text")
    
    st.markdown("**Create a file called `requirements.txt`:**")
    st.code("""
streamlit>=1.28.0
selenium>=4.15.0
pandas>=2.0.0
    """, language="text")
    
    st.markdown("**Note:** This scraper only fetches real matches from Sofascore. No demo data is used.")
