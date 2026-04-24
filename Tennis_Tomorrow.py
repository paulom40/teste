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
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Configure Chromium options for Streamlit Cloud / local
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    matches = []
    driver = None
    
    try:
        # Initialize driver (Streamlit Cloud uses chromium)
        driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to Sofascore Tennis page
        driver.get("https://www.sofascore.com/tennis")
        
        # Wait for page to load
        wait = WebDriverWait(driver, 15)
        time.sleep(5)  # Allow initial JS load
        
        # Try to find tournament filter and select ATP/Challenger
        try:
            # Look for filter buttons
            filter_buttons = driver.find_elements(By.CSS_SELECTOR, "button[class*='filter'], div[class*='Filter']")
            for btn in filter_buttons:
                if "ATP" in btn.text or "Challenger" in btn.text:
                    btn.click()
                    time.sleep(2)
        except:
            pass
        
        # Find all match containers
        match_elements = driver.find_elements(By.CSS_SELECTOR, 
            "div[class*='event'], div[class*='match'], div[data-testid*='match']")
        
        for match in match_elements[:100]:  # Limit to first 100 matches
            try:
                # Get all text from match element
                match_text = match.text
                
                # Only process ATP and Challenger matches
                if "ATP" in match_text or "CHALLENGER" in match_text.upper():
                    
                    # Extract lines from text
                    lines = match_text.split('\n')
                    
                    # Find tournament name (usually first line with ATP/Challenger)
                    tournament = "Unknown Tournament"
                    for line in lines:
                        if "ATP" in line or "CHALLENGER" in line.upper():
                            tournament = line
                            break
                    
                    # Determine surface
                    surface = "Hard"  # Default
                    if "Clay" in tournament or "CLAY" in tournament.upper():
                        surface = "Clay"
                    elif "Grass" in tournament:
                        surface = "Grass"
                    
                    # Find player names using common patterns
                    players = []
                    for line in lines:
                        # Look for player names (usually lines with vs, -, or two names)
                        if " vs " in line or " - " in line or re.search(r'[A-Z][a-z]+ [A-Z]\.?\s*[vV][sS]', line):
                            # Split by common separators
                            if " vs " in line:
                                parts = line.split(" vs ")
                            elif " - " in line:
                                parts = line.split(" - ")
                            else:
                                # Use regex to find player names
                                player_match = re.findall(r'([A-Z][a-z]+ [A-Z]\.?)', line)
                                parts = player_match if len(player_match) >= 2 else []
                            
                            if len(parts) >= 2:
                                players = [parts[0].strip(), parts[1].strip()]
                                break
                    
                    # If we found at least 2 players, add the match
                    if len(players) >= 2:
                        matches.append({
                            "tournament": tournament,
                            "surface": surface,
                            "player1": players[0],
                            "player2": players[1]
                        })
                        
            except Exception as e:
                continue
        
        # If still no matches, try alternative approach
        if not matches:
            # Try searching for specific match patterns
            page_text = driver.find_element(By.TAG_NAME, "body").text
            lines = page_text.split('\n')
            
            for i, line in enumerate(lines):
                if ("ATP" in line or "CHALLENGER" in line.upper()) and i+1 < len(lines):
                    tournament = line
                    
                    # Look for match line in next few lines
                    for j in range(i+1, min(i+10, len(lines))):
                        if " vs " in lines[j] or " - " in lines[j]:
                            match_line = lines[j]
                            if " vs " in match_line:
                                players = match_line.split(" vs ")
                            elif " - " in match_line:
                                players = match_line.split(" - ")
                            else:
                                continue
                            
                            if len(players) >= 2:
                                matches.append({
                                    "tournament": tournament,
                                    "surface": "Hard",
                                    "player1": players[0].strip(),
                                    "player2": players[1].strip()
                                })
                            break
                    
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return []
    
    finally:
        if driver:
            driver.quit()
    
    # Remove duplicates
    unique_matches = []
    seen = set()
    for match in matches:
        key = f"{match['player1']} vs {match['player2']}"
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)
    
    return unique_matches

# Fallback demo data if scraping fails
def get_demo_matches():
    """Returns demo match data for testing"""
    demo_matches = [
        {"tournament": "Mutua Madrid Open ATP", "surface": "Clay", "player1": "Carlos Alcaraz", "player2": "Jannik Sinner"},
        {"tournament": "Mutua Madrid Open ATP", "surface": "Clay", "player1": "Novak Djokovic", "player2": "Casper Ruud"},
        {"tournament": "BNP Paribas Challenger", "surface": "Hard", "player1": "Dominic Thiem", "player2": "Andy Murray"},
        {"tournament": "Rome ATP Masters", "surface": "Clay", "player1": "Daniil Medvedev", "player2": "Alexander Zverev"},
        {"tournament": "Oeiras Challenger", "surface": "Clay", "player1": "Richard Gasquet", "player2": "Stan Wawrinka"},
        {"tournament": "Oeiras Challenger", "surface": "Clay", "player1": "Joao Sousa", "player2": "Ben Shelton"},
    ]
    return demo_matches

def export_to_txt(matches):
    """Convert matches to the required txt format"""
    if not matches:
        return "No matches found for tomorrow."
    
    lines = []
    current_tournament = None
    match_counter = 0
    
    for match in matches:
        tourney_name = match["tournament"]
        surface = match["surface"]
        player1 = match["player1"]
        player2 = match["player2"]
        
        # Add tournament header if it's a new tournament
        if current_tournament != tourney_name:
            lines.append(f"{tourney_name} ({surface})")
            current_tournament = tourney_name
            match_counter = 0
        
        # Add match line
        lines.append(f"{player1} vs {player2}")
        match_counter += 1
    
    return "\n".join(lines)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Tennis Scraper - ATP Challenger Matches",
    page_icon="🎾",
    layout="wide"
)

st.title("🎾 ATP & Challenger Matches Scraper")
st.markdown("Scrapes tomorrow's tennis matches from Sofascore using Chromium")

# Initialize session state
if "matches" not in st.session_state:
    st.session_state["matches"] = []

# Sidebar for options
with st.sidebar:
    st.header("⚙️ Settings")
    method = st.radio(
        "Scraping Method",
        ["Chromium WebDriver (Sofascore)", "Demo Data (For Testing)"],
        help="Chromium method tries to scrape live from Sofascore. Use demo data if live scraping fails."
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool scrapes:")
    st.markdown("- ✅ ATP Tour matches")
    st.markdown("- ✅ Challenger matches")
    st.markdown("- ✅ Extracts tournament name & surface")
    st.markdown("- ✅ Exports to TXT format")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔄 Fetch Tomorrow's Matches", type="primary", use_container_width=True):
        with st.spinner("Scraping Sofascore with Chromium... This may take 10-15 seconds..."):
            if method == "Chromium WebDriver (Sofascore)":
                matches_data = scrape_sofascore_tomorrow()
                
                # If scraping fails or returns empty, use fallback
                if not matches_data:
                    st.warning("⚠️ Live scraping returned no data. Using demo data.")
                    matches_data = get_demo_matches()
            else:
                matches_data = get_demo_matches()
            
            if matches_data:
                st.session_state["matches"] = matches_data
                st.success(f"✅ Found {len(matches_data)} matches for tomorrow!")
                
                # Display matches in a nice table
                df = pd.DataFrame(matches_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.error("❌ No matches found.")
                st.session_state["matches"] = []

with col2:
    if st.session_state.get("matches"):
        txt_content = export_to_txt(st.session_state["matches"])
        
        st.metric("Total Matches", len(st.session_state["matches"]))
        st.download_button(
            label="📥 Download TXT File",
            data=txt_content,
            file_name=f"tennis_matches_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Show preview
        with st.expander("📄 Preview Export Format"):
            st.code(txt_content, language="text", line_numbers=True)

# Display instructions
if not st.session_state.get("matches"):
    st.info("👈 Click 'Fetch Tomorrow's Matches' to start scraping")

# Requirements for deployment
with st.expander("📦 Deployment Requirements (for Streamlit Cloud)"):
    st.markdown("""
    **Create a `packages.txt` file:**
